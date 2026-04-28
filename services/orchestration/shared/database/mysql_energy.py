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

Primary (plug times present):
  kwh_in  = reading within plug_time  ± 1 min  (closest to plug-in moment)
  kwh_out = reading within plug_out_time ± 1 min (closest to plug-out moment)

Fallback (plug times missing, use car arrival/departure):
  kwh_in  = first reading at or AFTER  in_time  (car arrives → gun plugged → meter records)
  kwh_out = last  reading at or BEFORE out_time  (gun unplugged → car leaves)

Environment Variables (override defaults):
  MYSQL_ENERGY_HOST      default: 13.235.19.21
  MYSQL_ENERGY_USER      default: mqtt_user
  MYSQL_ENERGY_PASSWORD  default: StrongPassword123!
  MYSQL_ENERGY_DB        default: mqtt_db
  MYSQL_ENERGY_PORT      default: 3306
"""
import os
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

_log = logging.getLogger(__name__)

# MySQL stores received_time in IST — all queries must use IST naive datetimes
_IST = ZoneInfo("Asia/Kolkata")

# ── Connection config ────────────────────────────────────────────────────────
MYSQL_HOST     = os.getenv("MYSQL_ENERGY_HOST",     "13.235.19.21")
MYSQL_USER     = os.getenv("MYSQL_ENERGY_USER",     "mqtt_user")
MYSQL_PASSWORD = os.getenv("MYSQL_ENERGY_PASSWORD", "StrongPassword123!")
MYSQL_DB       = os.getenv("MYSQL_ENERGY_DB",       "mqtt_db")
MYSQL_PORT     = int(os.getenv("MYSQL_ENERGY_PORT", "3306"))

PLUG_BUFFER_MINUTES = 2   # ±2 min window around plug_time / plug_out_time


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


def _to_naive_ist(dt) -> datetime | None:
    """
    Normalise a datetime-like value to a naive IST datetime for MySQL queries.
    MySQL stores received_time as IST strings with no tzinfo — we must convert
    PostgreSQL UTC timestamps to IST before querying, otherwise we search the
    wrong 5h30m-shifted time window.
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
            # Convert to IST, then strip tzinfo for MySQL comparison
            dt = dt.astimezone(_IST).replace(tzinfo=None)
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

    Primary — plug times present:
      kwh_in  : reading closest to plug_time     within ±1 min window
      kwh_out : reading closest to plug_out_time within ±1 min window
                (None → latest reading after plug_time for live sessions)

    Fallback — plug_time missing, use car times:
      kwh_in  : first reading at or AFTER  in_time  (energy starts after car arrives)
      kwh_out : last  reading at or BEFORE out_time  (energy ends before car leaves)
                (out_time None → latest reading after in_time for live sessions)

    Returns None if no usable start time or DB is unreachable.
    """
    buf = timedelta(minutes=PLUG_BUFFER_MINUTES)

    t_plug_in  = _to_naive_ist(plug_time)
    t_plug_out = _to_naive_ist(plug_out_time)
    t_in       = _to_naive_ist(in_time)
    t_out      = _to_naive_ist(out_time)

    using_fallback = t_plug_in is None

    # Must have at least one start time
    if t_plug_in is None and t_in is None:
        return None

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

                if not using_fallback:
                    # ── PRIMARY: plug times with ±1 min buffer ───────────
                    _log.debug(f"[MYSQL_ENERGY] Primary: plug_in={t_plug_in} plug_out={t_plug_out}")

                    # kwh_in: closest reading within [plug_time-1min, plug_time+1min]
                    cur.execute(
                        f"SELECT total_kwh FROM energy_readings "
                        f"WHERE received_time BETWEEN %s AND %s{filters} "
                        f"ORDER BY ABS(TIMESTAMPDIFF(SECOND, received_time, %s)) ASC LIMIT 1",
                        [t_plug_in - buf, t_plug_in + buf] + params_base + [t_plug_in],
                    )
                    row_in = cur.fetchone()

                    if t_plug_out is not None:
                        # kwh_out: closest reading within [plug_out-1min, plug_out+1min]
                        cur.execute(
                            f"SELECT total_kwh FROM energy_readings "
                            f"WHERE received_time BETWEEN %s AND %s{filters} "
                            f"ORDER BY ABS(TIMESTAMPDIFF(SECOND, received_time, %s)) ASC LIMIT 1",
                            [t_plug_out - buf, t_plug_out + buf] + params_base + [t_plug_out],
                        )
                        row_out = cur.fetchone()
                    else:
                        # Live session — latest reading after plug_in
                        cur.execute(
                            f"SELECT total_kwh FROM energy_readings "
                            f"WHERE received_time >= %s{filters} "
                            f"ORDER BY received_time DESC LIMIT 1",
                            [t_plug_in] + params_base,
                        )
                        row_out = cur.fetchone()

                else:
                    # ── FALLBACK: car in/out times, no buffer ────────────
                    _log.debug(f"[MYSQL_ENERGY] Fallback: in_time={t_in} out_time={t_out}")

                    # kwh_in: first reading AT or AFTER car arrived
                    cur.execute(
                        f"SELECT total_kwh FROM energy_readings "
                        f"WHERE received_time >= %s{filters} "
                        f"ORDER BY received_time ASC LIMIT 1",
                        [t_in] + params_base,
                    )
                    row_in = cur.fetchone()

                    if t_out is not None:
                        # kwh_out: last reading AT or BEFORE car left
                        cur.execute(
                            f"SELECT total_kwh FROM energy_readings "
                            f"WHERE received_time <= %s{filters} "
                            f"ORDER BY received_time DESC LIMIT 1",
                            [t_out] + params_base,
                        )
                        row_out = cur.fetchone()
                    else:
                        # Live session — latest reading after in_time
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
            f"[MYSQL_ENERGY] No readings found — fallback={using_fallback} "
            f"plug_in={t_plug_in} plug_out={t_plug_out} in={t_in} out={t_out}"
        )
        return None

    kwh_in  = float(row_in["total_kwh"])
    kwh_out = float(row_out["total_kwh"])
    energy  = kwh_out - kwh_in

    if energy < 0:
        _log.warning(
            f"[MYSQL_ENERGY] Negative energy={energy:.3f} kwh_in={kwh_in} kwh_out={kwh_out} "
            f"fallback={using_fallback} — skipping"
        )
        return None

    return round(energy, 3)


def fetch_readings_window(start_dt: datetime, end_dt: datetime) -> list[dict]:
    """
    Bulk-load every energy_readings row whose received_time falls in
    [start_dt, end_dt]. Returns a list of dicts ordered by received_time ASC.

    Used by the energy-comparison upload path so we make ONE MySQL roundtrip
    for the whole batch instead of two queries per session (which over a
    multi-hundred-row Excel was taking minutes).

    Both bounds must be naive IST datetimes. The caller is responsible for
    extending the window slightly past the first plug_in / last plug_out so
    the ±2-min buffer used downstream still has data to find.

    Returns [] on connection failure (so the caller degrades to "no energy"
    rather than crashing the upload).
    """
    if start_dt is None or end_dt is None or start_dt > end_dt:
        return []

    try:
        conn = _connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT received_time, total_kwh "
                    "FROM energy_readings "
                    "WHERE received_time BETWEEN %s AND %s "
                    "ORDER BY received_time ASC",
                    [start_dt, end_dt],
                )
                rows = cur.fetchall()
        finally:
            conn.close()
    except Exception as exc:
        _log.warning(f"[MYSQL_ENERGY] Bulk fetch failed: {exc}")
        return []

    _log.info(
        f"[MYSQL_ENERGY] Bulk fetch: window=[{start_dt}, {end_dt}] rows={len(rows)}"
    )
    return rows


def compute_kwh_from_readings(
    readings: list[dict],
    plug_time=None,
    plug_out_time=None,
    in_time=None,
    out_time=None,
) -> float | None:
    """
    Pure-Python equivalent of get_energy_consumed() that reads from a
    pre-loaded `readings` list (output of fetch_readings_window) instead of
    issuing a fresh MySQL query. Mirrors the same primary/fallback logic.

    `readings` must be sorted ASC by received_time and each row must contain
    {'received_time': datetime, 'total_kwh': float}.

    Returns None when no usable kwh_in / kwh_out can be located, or when the
    computed energy would be negative (clock skew / out-of-order readings).
    """
    if not readings:
        return None

    buf = timedelta(minutes=PLUG_BUFFER_MINUTES)

    t_plug_in  = _to_naive_ist(plug_time)
    t_plug_out = _to_naive_ist(plug_out_time)
    t_in       = _to_naive_ist(in_time)
    t_out      = _to_naive_ist(out_time)

    using_fallback = t_plug_in is None
    if t_plug_in is None and t_in is None:
        return None

    def _closest_in_window(target: datetime, window: timedelta):
        """Reading closest to `target` whose received_time is within ±window."""
        lo, hi = target - window, target + window
        best = None
        best_delta = None
        for row in readings:
            rt = row["received_time"]
            if rt < lo:
                continue
            if rt > hi:
                break
            delta = abs((rt - target).total_seconds())
            if best_delta is None or delta < best_delta:
                best, best_delta = row, delta
        return best

    def _first_at_or_after(target: datetime):
        for row in readings:
            if row["received_time"] >= target:
                return row
        return None

    def _last_at_or_before(target: datetime):
        last = None
        for row in readings:
            if row["received_time"] > target:
                break
            last = row
        return last

    def _latest():
        return readings[-1] if readings else None

    if not using_fallback:
        # Primary: plug times with ±2 min buffer
        row_in = _closest_in_window(t_plug_in, buf)
        if t_plug_out is not None:
            row_out = _closest_in_window(t_plug_out, buf)
        else:
            # Live session — latest reading after plug_in
            after_plug = [r for r in readings if r["received_time"] >= t_plug_in]
            row_out = after_plug[-1] if after_plug else None
    else:
        # Fallback: car in/out, no buffer
        row_in = _first_at_or_after(t_in)
        if t_out is not None:
            row_out = _last_at_or_before(t_out)
        else:
            after_in = [r for r in readings if r["received_time"] >= t_in]
            row_out = after_in[-1] if after_in else None

    if row_in is None or row_out is None:
        return None

    kwh_in  = float(row_in["total_kwh"])
    kwh_out = float(row_out["total_kwh"])
    energy  = kwh_out - kwh_in

    if energy < 0:
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
