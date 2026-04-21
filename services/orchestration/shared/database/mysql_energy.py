"""
MySQL Energy DB — External meter readings integration.

Connects to the external MySQL database that stores energy readings from
physical meters via MQTT. Used to calculate power consumed per vehicle
charging session by correlating plug_time / plug_out_time with total_kwh.

Table: energy_readings
  org_id          VARCHAR
  controller_id   INT
  sensor_id       INT
  total_kwh       FLOAT   — cumulative kWh counter from meter
  total_power_kw  FLOAT
  power_factor    FLOAT
  frequency       FLOAT
  received_time   DATETIME
  created_at      DATETIME

Primary:  energy = total_kwh(plug_out) − total_kwh(plug_in)
Fallback: if plug times are absent, use car in_time / out_time with ±2 min
          buffer to find the closest meter readings.

Environment Variables (override defaults):
  MYSQL_ENERGY_HOST      default: 13.235.19.21
  MYSQL_ENERGY_USER      default: mqtt_user
  MYSQL_ENERGY_PASSWORD  default: StrongPassword123!
  MYSQL_ENERGY_DB        default: mqtt_db
  MYSQL_ENERGY_PORT      default: 3306
"""
import os
import logging
from datetime import datetime, timedelta, timezone

_log = logging.getLogger(__name__)

# ── Connection config ────────────────────────────────────────────────────────
MYSQL_HOST     = os.getenv("MYSQL_ENERGY_HOST",     "13.235.19.21")
MYSQL_USER     = os.getenv("MYSQL_ENERGY_USER",     "mqtt_user")
MYSQL_PASSWORD = os.getenv("MYSQL_ENERGY_PASSWORD", "StrongPassword123!")
MYSQL_DB       = os.getenv("MYSQL_ENERGY_DB",       "mqtt_db")
MYSQL_PORT     = int(os.getenv("MYSQL_ENERGY_PORT", "3306"))

# Grace window used when falling back to car in/out times
FALLBACK_BUFFER_MINUTES = 2


def _connect():
    """Open a new PyMySQL connection. Caller must close it."""
    import pymysql
    return pymysql.connect(
        host=MYSQL_HOST,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
        port=MYSQL_PORT,
        connect_timeout=5,
        cursorclass=pymysql.cursors.DictCursor,
    )


def _to_naive_utc(dt) -> datetime | None:
    """
    Normalise a datetime-like value to a naive UTC datetime for MySQL queries.
    MySQL DATETIME columns have no timezone — we strip tzinfo after converting.
    """
    if dt is None:
        return None
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except (ValueError, TypeError):
            return None
    if isinstance(dt, datetime):
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    return None


def get_energy_consumed(
    plug_time=None,
    plug_out_time=None,
    in_time=None,
    out_time=None,
    controller_id: int | None = None,
    sensor_id: int | None = None,
) -> float | None:
    """
    Return kWh consumed for a charging session.

    Primary (uses exact gun plug timestamps):
      - t_in  = plug_time
      - t_out = plug_out_time  (None → use latest reading for live sessions)

    Fallback (when plug_time is missing, uses car arrival/departure times):
      - t_in  = in_time  − FALLBACK_BUFFER_MINUTES  (2 min before car arrived)
      - t_out = out_time + FALLBACK_BUFFER_MINUTES  (2 min after car left)
      The buffer accounts for the gap between car arrival and charging start.

    Returns None if no usable start time is available or DB is unreachable.
    """
    buf = timedelta(minutes=FALLBACK_BUFFER_MINUTES)

    # ── Resolve effective in/out times ──────────────────────────────────────
    t_in_raw  = _to_naive_utc(plug_time)
    t_out_raw = _to_naive_utc(plug_out_time)

    using_fallback = False

    if t_in_raw is None:
        # Primary missing — try car in/out times with buffer
        t_in_fb = _to_naive_utc(in_time)
        if t_in_fb is None:
            return None  # No usable start time at all
        t_in_raw  = t_in_fb - buf          # look 2 min before car arrived
        t_out_raw = _to_naive_utc(out_time)
        if t_out_raw is not None:
            t_out_raw = t_out_raw + buf    # look 2 min after car left
        using_fallback = True
        _log.debug(
            f"[MYSQL_ENERGY] Using fallback times: in={t_in_raw} out={t_out_raw}"
        )

    t_in  = t_in_raw
    t_out = t_out_raw  # None = session still active, use latest reading

    # ── Optional DB filters ─────────────────────────────────────────────────
    filters = ""
    params_base: list = []
    if controller_id is not None:
        filters += " AND controller_id = %s"
        params_base.append(controller_id)
    if sensor_id is not None:
        filters += " AND sensor_id = %s"
        params_base.append(sensor_id)

    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                # Reading at or just before t_in
                cur.execute(
                    f"SELECT total_kwh FROM energy_readings "
                    f"WHERE received_time <= %s{filters} "
                    f"ORDER BY received_time DESC LIMIT 1",
                    [t_in] + params_base,
                )
                row_in = cur.fetchone()

                if t_out is not None:
                    # Reading at or just after t_out
                    cur.execute(
                        f"SELECT total_kwh FROM energy_readings "
                        f"WHERE received_time >= %s{filters} "
                        f"ORDER BY received_time ASC LIMIT 1",
                        [t_out] + params_base,
                    )
                    row_out = cur.fetchone()
                else:
                    # Session still active — latest reading after t_in
                    cur.execute(
                        f"SELECT total_kwh FROM energy_readings "
                        f"WHERE received_time >= %s{filters} "
                        f"ORDER BY received_time DESC LIMIT 1",
                        [t_in] + params_base,
                    )
                    row_out = cur.fetchone()

        finally:
            conn.close()

    except Exception as exc:
        _log.warning(f"[MYSQL_ENERGY] Query failed: {exc}")
        return None

    if row_in is None or row_out is None:
        _log.debug(
            f"[MYSQL_ENERGY] No readings found: t_in={t_in} t_out={t_out} "
            f"fallback={using_fallback}"
        )
        return None

    kwh_in  = float(row_in["total_kwh"])
    kwh_out = float(row_out["total_kwh"])
    energy  = kwh_out - kwh_in

    if energy < 0:
        _log.warning(
            f"[MYSQL_ENERGY] Negative energy={energy:.3f} kwh_in={kwh_in} "
            f"kwh_out={kwh_out} t_in={t_in} t_out={t_out} fallback={using_fallback} — skipping"
        )
        return None

    return round(energy, 3)


def test_mysql_connection() -> bool:
    """Return True if the MySQL energy DB is reachable."""
    try:
        conn = _connect()
        conn.close()
        return True
    except Exception as exc:
        _log.warning(f"[MYSQL_ENERGY] Connection test failed: {exc}")
        return False
