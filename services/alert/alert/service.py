"""
Alert service for processing and sending alerts.

Generic design: any usecase that sets triggered=True fires an alert.
Rule-specific extras (events, violations, vehicle_details, etc.) are
passed through unchanged so the caller always gets the full picture.

Special threshold rules (usecases that need a minimum count before
alerting) are listed in THRESHOLD_RULES below.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from alert.schemas import (
    AlertDetail,
    AlertRequest,
    AlertResponse,
    PipelineAlertRequest,
    PipelineAlertResponse,
)


# ---------------------------------------------------------------------------
# Usecases that require a minimum matched_objects count before alerting.
# All other triggered usecases fire unconditionally.
# ---------------------------------------------------------------------------
THRESHOLD_RULES: Dict[str, int] = {
    "crowd_in_roi":   3,
    "people_counter": 5,   # alert only when occupancy exceeds 5
}


def _build_alert_type(usecase_id: str) -> str:
    """Derive a human-readable alert type string from the usecase ID."""
    return f"{usecase_id}_triggered"


def _build_message(usecase_id: str, matched_count: int, extras: Optional[Dict[str, Any]]) -> str:
    """Build a human-readable alert message."""
    base = f"[{usecase_id}] {matched_count} object(s) detected"

    # Enrich message with the most relevant extras field
    if extras:
        if "events" in extras and extras["events"]:
            event_types = list({e.get("event_type", "") for e in extras["events"]})
            base += f" | events: {', '.join(event_types)}"
        elif "violations" in extras and extras["violations"]:
            reasons = list({v.get("metadata", {}).get("reason", "") for v in extras["violations"]})
            base += f" | violations: {', '.join(r for r in reasons if r)}"
        elif "vehicle_details" in extras:
            vd = extras["vehicle_details"]
            if isinstance(vd, list):
                parts = [f"plate: {v.get('car_number', 'N/A')}  model: {v.get('car_model', 'N/A')}" for v in vd]
                base += " | " + " | ".join(parts)
            elif isinstance(vd, dict):
                base += f" | plate: {vd.get('car_number', 'N/A')}  model: {vd.get('car_model', 'N/A')}"

    return base


def process_pipeline_alerts(request: PipelineAlertRequest) -> PipelineAlertResponse:
    """
    Process multiple usecase results and send appropriate alerts.

    For each result:
    - Skip if triggered=False
    - Skip if the usecase has a threshold and matched count is below it
    - Otherwise fire an alert, persist to DB, and include in response

    All fields from the usecase result (extras, snapshot_b64, timestamp)
    flow through to both the response and the DB row.
    """
    print(f"\n[SERVICE] process_pipeline_alerts | camera={request.camera_id} "
          f"| results={len(request.usecase_results)}")

    alerts_sent: List[AlertDetail] = []
    fired_at = datetime.now(timezone.utc).isoformat()

    for result in request.usecase_results:
        usecase_id    = result.get("usecase_id") or result.get("usecase_name", "unknown")
        triggered     = result.get("triggered", False)
        matched_objects = result.get("matched_objects", [])
        matched_count = len(matched_objects) if matched_objects else result.get("matched_count", 0)
        detection_id  = result.get("detection_id")
        screenshot_path = result.get("screenshot_path")
        extras        = result.get("extras") or {}
        # Use timestamp from the result if available, otherwise use now
        timestamp     = result.get("timestamp") or fired_at

        print(f"[SERVICE]  usecase={usecase_id} triggered={triggered} count={matched_count}")

        if not triggered:
            continue

        # Check threshold rules
        min_count = THRESHOLD_RULES.get(usecase_id)
        if min_count is not None and matched_count < min_count:
            print(f"[SERVICE]  {usecase_id}: count {matched_count} below threshold {min_count}, skipping")
            continue

        alert_type = _build_alert_type(usecase_id)
        message    = _build_message(usecase_id, matched_count, extras)

        print(f"[ALERT] ⚠  ALERT | camera={request.camera_id} usecase={usecase_id} "
              f"type={alert_type} count={matched_count}")
        print(f"[ALERT]    message : {message}")
        if extras:
            print(f"[ALERT]    extras  : {extras}")

        alerts_sent.append(AlertDetail(
            usecase_id=usecase_id,
            alert_type=alert_type,
            alert_count=matched_count,
            message=message,
            timestamp=timestamp,
            extras=extras if extras else None,
        ))

    print(f"[SERVICE] process_pipeline_alerts done | alerts_sent={len(alerts_sent)}\n")

    return PipelineAlertResponse(
        camera_id=request.camera_id,
        total_alerts_sent=len(alerts_sent),
        alerts_sent=alerts_sent,
    )


def process_alert(request: AlertRequest) -> AlertResponse:
    """
    Process a single alert request (legacy endpoint).
    Kept for backward compatibility.
    """
    print(f"[SERVICE] process_alert | camera={request.camera_id} usecase={request.usecase_id}")

    alert_sent = False
    message = "Alert conditions not met. No alert sent."

    if request.alert_required:
        alert_sent = True
        message = f"Alert sent for {request.usecase_id}. Count: {request.alert_count}"
        print(f"[ALERT] ⚠  ALERT | camera={request.camera_id} type={request.alert_type}")

    return AlertResponse(
        camera_id=request.camera_id,
        alert_sent=alert_sent,
        alert_type=request.alert_type,
        alert_count=request.alert_count,
        message=message,
    )
