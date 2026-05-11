"""
Energy-loss comparison between client-provided OCPP Excel and our CCTV sessions.

The Excel is authoritative for per-gun energy (`Units Consumed`). Our CCTV
sessions + MySQL meter are unreliable per-gun because:
  - Number plate (VRN) is often missing (Gemini Vision fails frequently)
  - Gun detection is noisy (slot_id → gun_number mapping is not trustworthy)
  - Meter only gives cumulative kWh across both guns

Strategy: weighted scoring across every signal available. Strong signals
(VRN, OCPP-start vs CCTV plug_time, duration) dominate when present; weak
signals (gun, model) keep pulling in the right direction when strong ones
are missing.

Gate: a candidate must clear the numeric score threshold AND have at least
one strong signal (exact VRN, a duration band, or a time band). Without
this, model + gun + same-day alone is enough to score 4 and produces
spurious matches between unrelated sessions of the same common model
(Tata TIAGO on Connector 1, etc.).

Excel "OCPP Start/End Time" — full timestamps when present. We compare
them to CCTV plug_time / in_time (both normalised to naive IST).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from openpyxl import load_workbook


# ── Config ──────────────────────────────────────────────────────────────────

WEIGHT_VRN_MATCH        = 3
WEIGHT_VRN_CONFLICT     = -5
WEIGHT_MODEL_MATCH      = 2
WEIGHT_GUN_MATCH        = 1
WEIGHT_DURATION_TIGHT   = 2   # within ±3 min
WEIGHT_DURATION_LOOSE   = 1   # within ±10 min
WEIGHT_DURATION_CONFLICT = -3 # > 10 min apart — likely different sessions
WEIGHT_TIME_TIGHT       = 3   # OCPP start within ±5 min of CCTV plug_time
WEIGHT_TIME_LOOSE       = 2   # within ±15 min
WEIGHT_TIME_CONFLICT    = -3  # > 30 min apart
WEIGHT_SAME_DATE        = 1
WEIGHT_DATE_CONFLICT    = -3

SCORE_THRESHOLD         = 3   # below → UNMATCHED

DURATION_TIGHT_SECONDS  = 3 * 60
DURATION_LOOSE_SECONDS  = 10 * 60
TIME_TIGHT_SECONDS      = 5 * 60
TIME_LOOSE_SECONDS      = 15 * 60
TIME_CONFLICT_SECONDS   = 30 * 60

# Station-specific slot→connector mapping. Flip this constant if wiring differs.
SLOT_TO_CONNECTOR: dict[str, int] = {
    "ROI_1": 1,
    "ROI_2": 2,
}


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class ExcelRow:
    """One OCPP transaction from the client Excel."""
    row_index: int
    transaction_id: str | None
    session_id_ocpp: str | None
    start_dt: datetime | None      # OCPP Start Time (full timestamp, naive IST)
    end_dt: datetime | None        # OCPP End Time   (full timestamp, naive IST)
    duration_seconds: int | None   # from Session Duration(hh:mm:ss)
    connector_id: int | None
    vrn_raw: str | None
    vrn_norm: str | None
    make: str | None
    model: str | None
    units_kwh: float | None        # Units Consumed(kWh) — authoritative
    meter_start: float | None
    meter_end: float | None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def date(self) -> datetime | None:
        """Day (midnight) of the OCPP start timestamp, for day-level pre-filter."""
        if self.start_dt is None:
            return None
        return self.start_dt.replace(hour=0, minute=0, second=0, microsecond=0)


@dataclass
class CctvSession:
    """Subset of our ChargingSession needed for matching."""
    session_id: int
    camera_id: str | None
    slot_id: str | None
    gun_number: str | None
    car_number: str | None
    car_model: str | None
    in_time: datetime | None
    plug_time: datetime | None
    plug_out_time: datetime | None
    out_time: datetime | None
    energy_kwh: float | None       # our meter-derived estimate

    @property
    def duration_seconds(self) -> int | None:
        """Prefer plug-based duration; fall back to car arrival/exit."""
        a, b = self.plug_time, self.plug_out_time
        if a is None or b is None:
            a, b = self.in_time, self.out_time
        if a is None or b is None:
            return None
        return int((b - a).total_seconds())

    @property
    def anchor_date(self) -> datetime | None:
        """The date this session belongs to (IST)."""
        t = self.plug_time or self.in_time
        if t is None:
            return None
        return _to_ist_date(t)


# ── Normalizers ─────────────────────────────────────────────────────────────

_VRN_STRIP = re.compile(r"[\s\-]+")

def normalize_vrn(v: Any) -> str | None:
    """KL 21 Z 1314 → KL21Z1314. Drops obvious junk like '1234'."""
    if v is None:
        return None
    s = str(v).strip().upper()
    s = _VRN_STRIP.sub("", s)
    if not s or s in {"-", "NA", "N/A", "NONE"}:
        return None
    # Plate must contain at least one letter to be considered valid for matching.
    if not any(c.isalpha() for c in s):
        return None
    return s


def normalize_model_text(v: Any) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def model_matches(cctv_model: str | None, excel_make: str | None, excel_model: str | None) -> bool:
    """Fuzzy: any token overlap between CCTV free-text and Excel Make+Model."""
    cctv = normalize_model_text(cctv_model)
    parts = normalize_model_text(f"{excel_make or ''} {excel_model or ''}")
    if not cctv or not parts:
        return False
    cctv_tokens = set(cctv.split())
    excel_tokens = set(parts.split())
    # Need at least one shared token ≥ 3 chars to avoid matching on "ev", "e".
    return any(t in cctv_tokens for t in excel_tokens if len(t) >= 3)


def parse_duration(s: Any) -> int | None:
    """hh:mm:ss → seconds. Also accepts timedelta (openpyxl sometimes gives that)."""
    if s is None:
        return None
    if isinstance(s, timedelta):
        return int(s.total_seconds())
    txt = str(s).strip()
    m = re.match(r"^(\d+):(\d+):(\d+)$", txt)
    if m:
        h, mnt, sec = map(int, m.groups())
        return h * 3600 + mnt * 60 + sec
    return None


def parse_ocpp_datetime(v: Any) -> datetime | None:
    """Excel 'OCPP Start/End Time' — accepts datetime objects (openpyxl) or
    strings like 'dd/mm/yyyy HH:MM:SS' / 'dd/mm/yyyy'. Returns a naive datetime
    in IST (the client export is already IST)."""
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v.replace(tzinfo=None) if v.tzinfo else v
    txt = str(v).strip()
    fmts = (
        "%d/%m/%Y %H:%M:%S", "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
        "%d-%m-%Y %H:%M:%S", "%d-%m-%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y",
    )
    for fmt in fmts:
        try:
            return datetime.strptime(txt, fmt)
        except ValueError:
            continue
    return None


def _to_ist_naive(dt: datetime | None) -> datetime | None:
    """Normalise a (possibly tz-aware UTC) datetime to a naive IST datetime."""
    if dt is None:
        return None
    from zoneinfo import ZoneInfo
    if dt.tzinfo is not None:
        dt = dt.astimezone(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    return dt


def _to_ist_date(dt: datetime) -> datetime:
    """Date of `dt` expressed in IST, as a naive midnight datetime for comparison."""
    naive = _to_ist_naive(dt)
    if naive is None:
        return None  # type: ignore[return-value]
    return naive.replace(hour=0, minute=0, second=0, microsecond=0)


# ── Parser ──────────────────────────────────────────────────────────────────

# Header → ExcelRow field. Tolerates capitalization / whitespace drift.
_HEADER_MAP = {
    "transaction id":        "transaction_id",
    "session id":            "session_id_ocpp",
    "ocpp start time":       "_start",
    "ocpp end time":         "_end",
    "session duration":      "_duration",          # matches "Session Duration(hh:mm:ss)"
    "connector id":          "_connector",
    "vrn":                   "_vrn",
    "make":                  "make",
    "model":                 "model",
    "units consumed":        "_units",             # matches "Units Consumed(kWh)"
    "meter start":           "_meter_start",
    "meter end":             "_meter_end",
}


def _header_key(h: Any) -> str:
    """Strip parenthetical suffix so 'Units Consumed(kWh)' → 'units consumed'."""
    if h is None:
        return ""
    s = str(h).strip().lower()
    s = re.sub(r"\s*\(.*?\)\s*", "", s)   # drop (kWh), (hh:mm:ss)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_excel(file_bytes: bytes) -> list[ExcelRow]:
    """Parse the client OCPP export xlsx into ExcelRow records."""
    from io import BytesIO
    wb = load_workbook(BytesIO(file_bytes), data_only=True, read_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)

    headers = next(rows_iter, None)
    if not headers:
        return []
    col_idx: dict[str, int] = {}
    for i, h in enumerate(headers):
        key = _header_key(h)
        if key in _HEADER_MAP:
            col_idx[_HEADER_MAP[key]] = i

    parsed: list[ExcelRow] = []
    for row_num, row in enumerate(rows_iter, start=2):
        if row is None or all(c is None or c == "" for c in row):
            continue

        def get(field_name: str):
            i = col_idx.get(field_name)
            if i is None or i >= len(row):
                return None
            return row[i]

        vrn_raw = get("_vrn")
        duration_s = parse_duration(get("_duration"))
        start_dt = parse_ocpp_datetime(get("_start"))
        end_dt = parse_ocpp_datetime(get("_end"))
        connector_raw = get("_connector")
        try:
            connector_id = int(connector_raw) if connector_raw not in (None, "") else None
        except (TypeError, ValueError):
            connector_id = None

        def _to_float(v):
            if v in (None, "", "-"):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        parsed.append(ExcelRow(
            row_index=row_num,
            transaction_id=str(get("transaction_id")) if get("transaction_id") is not None else None,
            session_id_ocpp=str(get("session_id_ocpp")) if get("session_id_ocpp") is not None else None,
            start_dt=start_dt,
            end_dt=end_dt,
            duration_seconds=duration_s,
            connector_id=connector_id,
            vrn_raw=str(vrn_raw).strip() if vrn_raw else None,
            vrn_norm=normalize_vrn(vrn_raw),
            make=str(get("make")).strip() if get("make") else None,
            model=str(get("model")).strip() if get("model") else None,
            units_kwh=_to_float(get("_units")),
            meter_start=_to_float(get("_meter_start")),
            meter_end=_to_float(get("_meter_end")),
        ))
    return parsed


# ── Matcher ─────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    excel_row: ExcelRow
    cctv_session: CctvSession | None
    score: int
    reasons: list[str]
    confidence: str  # HIGH | MEDIUM | LOW | UNMATCHED

    @property
    def client_kwh(self) -> float | None:
        return self.excel_row.units_kwh

    @property
    def meter_kwh(self) -> float | None:
        return self.cctv_session.energy_kwh if self.cctv_session else None

    @property
    def loss_kwh(self) -> float | None:
        c, m = self.client_kwh, self.meter_kwh
        if c is None or m is None:
            return None
        return round(c - m, 3)


def _score_pair(excel: ExcelRow, cctv: CctvSession) -> tuple[int, list[str]]:
    score = 0
    reasons: list[str] = []

    # Date — pre-filter but also scored for clarity in the reasons trail.
    excel_date = excel.date
    cctv_date = cctv.anchor_date
    if excel_date and cctv_date:
        if excel_date == cctv_date:
            score += WEIGHT_SAME_DATE
            reasons.append("date")
        else:
            score += WEIGHT_DATE_CONFLICT
            reasons.append("date-conflict")

    # VRN — strongest signal when both present and valid.
    if excel.vrn_norm and cctv.car_number:
        cctv_vrn = normalize_vrn(cctv.car_number)
        if cctv_vrn:
            if cctv_vrn == excel.vrn_norm:
                score += WEIGHT_VRN_MATCH
                reasons.append("VRN")
            else:
                score += WEIGHT_VRN_CONFLICT
                reasons.append("VRN-conflict")

    # Model — fuzzy token overlap.
    if model_matches(cctv.car_model, excel.make, excel.model):
        score += WEIGHT_MODEL_MATCH
        reasons.append("model")

    # Gun ↔ connector.
    if excel.connector_id is not None and cctv.slot_id is not None:
        expected_connector = SLOT_TO_CONNECTOR.get(cctv.slot_id)
        if expected_connector is not None and expected_connector == excel.connector_id:
            score += WEIGHT_GUN_MATCH
            reasons.append("gun")

    # Duration proximity (or penalty when both are known and far apart).
    cctv_dur = cctv.duration_seconds
    if excel.duration_seconds is not None and cctv_dur is not None:
        diff = abs(excel.duration_seconds - cctv_dur)
        if diff <= DURATION_TIGHT_SECONDS:
            score += WEIGHT_DURATION_TIGHT
            reasons.append(f"duration±3m({diff}s)")
        elif diff <= DURATION_LOOSE_SECONDS:
            score += WEIGHT_DURATION_LOOSE
            reasons.append(f"duration±10m({diff}s)")
        else:
            score += WEIGHT_DURATION_CONFLICT
            reasons.append(f"duration-conflict({diff}s)")

    # Wall-clock proximity: OCPP start vs CCTV plug_time (fallback in_time).
    # Strongest single non-VRN signal — same gun at the same minute is decisive.
    excel_start = _to_ist_naive(excel.start_dt)
    cctv_anchor = _to_ist_naive(cctv.plug_time) or _to_ist_naive(cctv.in_time)
    if excel_start is not None and cctv_anchor is not None:
        time_diff = abs(int((excel_start - cctv_anchor).total_seconds()))
        if time_diff <= TIME_TIGHT_SECONDS:
            score += WEIGHT_TIME_TIGHT
            reasons.append(f"time±5m({time_diff}s)")
        elif time_diff <= TIME_LOOSE_SECONDS:
            score += WEIGHT_TIME_LOOSE
            reasons.append(f"time±15m({time_diff}s)")
        elif time_diff > TIME_CONFLICT_SECONDS:
            score += WEIGHT_TIME_CONFLICT
            reasons.append(f"time-conflict({time_diff}s)")

    return score, reasons


def _has_strong_signal(reasons: list[str]) -> bool:
    """A candidate must clear at least one of: exact VRN, a duration band, or
    a time band. Without any of these, model+gun+date alone is too weak —
    that combination produced the spurious LOW matches in the client sample."""
    if "VRN" in reasons:
        return True
    for r in reasons:
        if r.startswith("duration±") or r.startswith("time±"):
            return True
    return False


def _confidence(score: int, reasons: list[str]) -> str:
    if score < SCORE_THRESHOLD:
        return "UNMATCHED"
    has_vrn = "VRN" in reasons
    has_time_tight = any(r.startswith("time±5m") for r in reasons)
    has_dur_tight = any(r.startswith("duration±3m") for r in reasons)
    if has_vrn and (has_time_tight or has_dur_tight):
        return "HIGH"
    if has_time_tight and has_dur_tight:
        return "HIGH"
    if score >= 6:
        return "HIGH"
    if score >= 4:
        return "MEDIUM"
    return "LOW"


def match_excel_to_cctv(excel_rows: list[ExcelRow], cctv_sessions: list[CctvSession]) -> list[MatchResult]:
    """Greedy highest-score-first assignment.

    Each CCTV session is consumed at most once. If no candidate clears the
    SCORE_THRESHOLD for an Excel row, it is reported as UNMATCHED (typical for
    blind-spot parking or a camera outage during that session).
    """
    # Build candidate scores (excel_index, cctv_index, score, reasons).
    candidates: list[tuple[int, int, int, list[str]]] = []
    for ei, ex in enumerate(excel_rows):
        for ci, cc in enumerate(cctv_sessions):
            score, reasons = _score_pair(ex, cc)
            # Two-stage filter: numeric threshold + at least one strong signal.
            # Without the strong-signal gate, common-model + same-gun + same-day
            # bundles score 4 and produce nonsense pairings (e.g. an 11-second
            # OCPP cancel matched to an unrelated half-hour TIAGO session).
            if score >= SCORE_THRESHOLD and _has_strong_signal(reasons):
                candidates.append((ei, ci, score, reasons))

    candidates.sort(key=lambda t: (-t[2], t[0]))

    used_excel: set[int] = set()
    used_cctv: set[int] = set()
    results_by_ei: dict[int, MatchResult] = {}

    for ei, ci, score, reasons in candidates:
        if ei in used_excel or ci in used_cctv:
            continue
        used_excel.add(ei)
        used_cctv.add(ci)
        results_by_ei[ei] = MatchResult(
            excel_row=excel_rows[ei],
            cctv_session=cctv_sessions[ci],
            score=score,
            reasons=reasons,
            confidence=_confidence(score, reasons),
        )

    # Emit one result per Excel row, in original order.
    results: list[MatchResult] = []
    for ei, ex in enumerate(excel_rows):
        if ei in results_by_ei:
            results.append(results_by_ei[ei])
        else:
            results.append(MatchResult(
                excel_row=ex,
                cctv_session=None,
                score=0,
                reasons=[],
                confidence="UNMATCHED",
            ))
    return results


def result_to_dict(r: MatchResult) -> dict[str, Any]:
    """Serializable shape for the dashboard."""
    ex = r.excel_row
    cc = r.cctv_session
    return {
        "excel_row": ex.row_index,
        "transaction_id": ex.transaction_id,
        "ocpp_date": ex.date.strftime("%Y-%m-%d") if ex.date else None,
        "ocpp_start_time": ex.start_dt.strftime("%Y-%m-%d %H:%M:%S") if ex.start_dt else None,
        "ocpp_end_time":   ex.end_dt.strftime("%Y-%m-%d %H:%M:%S")   if ex.end_dt   else None,
        "duration_seconds": ex.duration_seconds,
        "connector_id": ex.connector_id,
        "vrn": ex.vrn_raw,
        "make": ex.make,
        "model": ex.model,
        "client_kwh": ex.units_kwh,
        "meter_kwh": r.meter_kwh,
        "loss_kwh": r.loss_kwh,
        "matched_session_id": cc.session_id if cc else None,
        "matched_slot_id": cc.slot_id if cc else None,
        "matched_car_number": cc.car_number if cc else None,
        "matched_car_model": cc.car_model if cc else None,
        "match_score": r.score,
        "match_reasons": r.reasons,
        "confidence": r.confidence,
    }
