"""
Mobile-push alert channel.

The alert service owns the *decision* that an alert should fire
(process_pipeline_alerts); this module is one delivery channel it fans out to
— a sibling of alert/telegram.py.

Client requirement: every time a wrong-parking violation is detected, the event
must be pushed to the agency that builds GoEC's consumer mobile app. That team
owns the app and decides how the notification is surfaced / prioritised on
*their* end, so this channel delivers structured *data* (a JSON POST), not a
formatted human message. We hand them the full violation payload and stay out
of presentation.

Mirrors the Telegram channel's contract exactly: opt-in and self-disabling. If
MOBILE_PUSH_ENABLED is false, or MOBILE_PUSH_URL is missing, send_mobile_push()
is a guarded no-op so the rest of the pipeline runs unchanged. Delivery is
fire-and-forget — any failure is logged and swallowed; the agency endpoint
being slow or down must never break alert processing or the HTTP response.
(No retry/queue yet — channels are best-effort.)

Required env to actually send:
  MOBILE_PUSH_ENABLED=true
  MOBILE_PUSH_URL=<agency push-ingest endpoint>

Optional env:
  MOBILE_PUSH_AUTH_TOKEN=<token>      -> sent as "Authorization: Bearer <token>"
  MOBILE_PUSH_HEADERS={"X-API-Key":"..."}   extra headers as a JSON object
  MOBILE_PUSH_TIMEOUT=5.0             HTTP timeout (seconds) for the POST
  MOBILE_PUSH_USECASES=parking_compliance   comma-separated allowlist;
                                             "" or "*" => every fired alert
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

MOBILE_PUSH_ENABLED = os.getenv("MOBILE_PUSH_ENABLED", "false").lower() in ("1", "true", "yes")
MOBILE_PUSH_URL = os.getenv("MOBILE_PUSH_URL", "").strip()
MOBILE_PUSH_AUTH_TOKEN = os.getenv("MOBILE_PUSH_AUTH_TOKEN", "").strip()
MOBILE_PUSH_TIMEOUT = float(os.getenv("MOBILE_PUSH_TIMEOUT", "5.0"))

# Comma-separated allowlist of usecase_ids that should be pushed to the mobile
# app. Defaults to parking_compliance — the usecase that carries wrong /
# unauthorized / non-EV parking violations — so the channel does exactly what
# the client asked for out of the box. Empty / "*" => every fired alert.
_raw_allow = os.getenv("MOBILE_PUSH_USECASES", "parking_compliance").strip()
MOBILE_PUSH_USECASES = (
    None if _raw_allow in ("", "*")
    else {u.strip() for u in _raw_allow.split(",") if u.strip()}
)


def _extra_headers() -> Dict[str, str]:
    """Parse MOBILE_PUSH_HEADERS (a JSON object) into a headers dict.

    Malformed JSON is logged once and ignored — a bad header string must not
    stop the push from being attempted.
    """
    raw = os.getenv("MOBILE_PUSH_HEADERS", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): str(v) for k, v in parsed.items()}
        logger.warning("[MOBILE_PUSH] MOBILE_PUSH_HEADERS is not a JSON object — ignoring")
    except Exception as e:
        logger.warning(f"[MOBILE_PUSH] MOBILE_PUSH_HEADERS parse error ({e}) — ignoring")
    return {}


def is_mobile_push_target(usecase_id: str) -> bool:
    """True if this usecase should be pushed to the mobile app, per the allowlist."""
    if not MOBILE_PUSH_ENABLED:
        return False
    if MOBILE_PUSH_USECASES is None:
        return True
    return usecase_id in MOBILE_PUSH_USECASES


def build_mobile_payload(
    camera_id: str,
    usecase_id: str,
    alert_type: str,
    message: str,
    timestamp: str,
    matched_count: int,
    extras: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the structured JSON body pushed to the mobile app.

    The agency decides presentation/prioritisation, so we send the raw event
    data: the violation list (each with its reason, description, bbox, ROIs and
    confidence) plus enough context to identify camera + time. A snapshot URL is
    included when the upstream result carried one so the app can show the frame.
    """
    extras = extras or {}
    violations: List[Dict[str, Any]] = extras.get("violations") or []
    snapshot_url = (
        extras.get("snapshot_url")
        or extras.get("screenshot_url")
        or extras.get("snapshot_path")
    )

    return {
        # Stable top-level type the app can key on for routing/priority.
        "event": "parking_violation",
        "usecase_id": usecase_id,
        "alert_type": alert_type,
        "camera_id": camera_id,
        "timestamp": timestamp,
        "message": message,
        "violation_count": len(violations) or matched_count,
        "violations": violations,
        "snapshot_url": snapshot_url,
    }


def send_mobile_push(payload: Dict[str, Any]) -> bool:
    """POST a structured alert payload to the mobile-app push endpoint.

    Returns True on a 2xx response, False otherwise (disabled, unconfigured, or
    a delivery error). Never raises.
    """
    if not MOBILE_PUSH_ENABLED:
        return False
    if not MOBILE_PUSH_URL:
        logger.warning("[MOBILE_PUSH] enabled but MOBILE_PUSH_URL not set — skipping")
        return False

    headers = {"Content-Type": "application/json"}
    headers.update(_extra_headers())
    if MOBILE_PUSH_AUTH_TOKEN:
        headers["Authorization"] = f"Bearer {MOBILE_PUSH_AUTH_TOKEN}"

    try:
        resp = requests.post(
            MOBILE_PUSH_URL,
            json=payload,
            headers=headers,
            timeout=MOBILE_PUSH_TIMEOUT,
        )
        if 200 <= resp.status_code < 300:
            logger.info(
                "[MOBILE_PUSH] pushed %s for camera=%s (%d violation(s))",
                payload.get("event"), payload.get("camera_id"),
                payload.get("violation_count", 0),
            )
            return True
        logger.warning(f"[MOBILE_PUSH] push failed: {resp.status_code} {resp.text[:200]}")
        return False
    except Exception as e:
        logger.warning(f"[MOBILE_PUSH] push error: {e}")
        return False
