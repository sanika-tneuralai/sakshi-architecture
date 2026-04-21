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

Energy consumed = total_kwh(at plug_out) - total_kwh(at plug_in)
For live/active sessions: total_kwh(latest) - total_kwh(at plug_in)

Environment Variables (override defaults):
  MYSQL_ENERGY_HOST      default: 13.235.19.21
  MYSQL_ENERGY_USER      default: mqtt_user
  MYSQL_ENERGY_PASSWORD  default: StrongPassword123!
  MYSQL_ENERGY_DB        default: mqtt_db
  MYSQL_ENERGY_PORT      default: 3306
"""
import os
import logging
from datetime import datetime, timezone

_log = logging.getLogger(__name__)

# ── Connection config ────────────────────────────────────────────────────────
MYSQL_HOST     = os.getenv("MYSQL_ENERGY_HOST",     "13.235.19.21")
MYSQL_USER     = os.getenv("MYSQL_ENERGY_USER",     "mqtt_user")
MYSQL_PASSWORD = os.getenv("MYSQL_ENERGY_PASSWORD", "StrongPassword123!")
MYSQL_DB       = os.getenv("MYSQL_ENERGY_DB",       "mqtt_db")
MYSQL_PORT     = int(os.getenv("MYSQL_ENERGY_PORT", "3306"))


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
            # Convert to UTC then make naive
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    return None


def get_energy_consumed(
    plug_time,
    plug_out_time,
    controller_id: int | None = None,
    sensor_id: int | None = None,
) -> float | None:
    """
    Return the kWh consumed between plug_time and plug_out_time.

    Logic:
      - kwh_in  = total_kwh of the reading closest to (and at/before) plug_time
      - kwh_out = total_kwh of the reading closest to (and at/after) plug_out_time
                  If plug_out_time is None (session still active), use the latest reading.
      - result  = kwh_out - kwh_in  (rounded to 3 dp)

    Returns None if:
      - plug_time is None (session not started)
      - no matching meter readings found
      - DB unreachable (silent — dashboard just shows '—')

    Parameters
    ----------
    plug_time      : ISO string or datetime — gun plug-in timestamp
    plug_out_time  : ISO string or datetime or None — gun plug-out timestamp
    controller_id  : filter by controller (None = any)
    sensor_id      : filter by sensor (None = any)
    """
    t_in = _to_naive_utc(plug_time)
    if t_in is None:
        return None  # Session hasn't started charging yet

    t_out = _to_naive_utc(plug_out_time)  # None means still active

    # Build optional controller/sensor filters
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
                # Reading at or just before plug_in
                cur.execute(
                    f"SELECT total_kwh FROM energy_readings "
                    f"WHERE received_time <= %s{filters} "
                    f"ORDER BY received_time DESC LIMIT 1",
                    [t_in] + params_base,
                )
                row_in = cur.fetchone()

                if t_out is not None:
                    # Reading at or just after plug_out
                    cur.execute(
                        f"SELECT total_kwh FROM energy_readings "
                        f"WHERE received_time >= %s{filters} "
                        f"ORDER BY received_time ASC LIMIT 1",
                        [t_out] + params_base,
                    )
                    row_out = cur.fetchone()
                else:
                    # Session still active — use latest reading
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
            f"[MYSQL_ENERGY] No readings found: plug_in={t_in} plug_out={t_out}"
        )
        return None

    kwh_in  = float(row_in["total_kwh"])
    kwh_out = float(row_out["total_kwh"])

    energy = kwh_out - kwh_in
    if energy < 0:
        # Meter rollover or bad data — return None rather than negative kWh
        _log.warning(
            f"[MYSQL_ENERGY] Negative energy={energy:.3f} kwh_in={kwh_in} kwh_out={kwh_out} "
            f"plug_in={t_in} plug_out={t_out} — skipping"
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
