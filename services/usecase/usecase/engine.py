import logging
from typing import Any, Dict, List

from usecase.rules import get_usecase_rule
from usecase.schemas import UsecaseResult


logger = logging.getLogger(__name__)

def _slim_detection(detetion: Dict[str, Any]):
    """
    Strip a detection dict down to only the fields usecase rules need.
    ROI has been removed from the detection service — rules now evaluate
    against all detections regardless of position.
    """
    slim = {
        "class_name": detetion.get("class_name"),
        "confidence": detetion.get("confidence"),
        "bbox": detetion.get("bbox", {})
    }
    # preserve track_id if present — required by parking_detection centroid tracker
    if "track_id" in detetion:
        slim["track_id"] = detetion["track_id"]
    return slim

def build_slim_payload(detection_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a slim version of the detection output for queueing.
    snapshot_url is a short string so it travels directly in the payload —
    no Redis detour needed (the old Redis pattern existed only to keep large
    base64 blobs out of RabbitMQ messages).
    """
    camera_id = detection_output.get("camera_id")
    raw_detections = detection_output.get("detections", [])
    return {
        "detections": [_slim_detection(d) for d in raw_detections],
        "first_detection_id": detection_output.get("first_detection_id"),
        "camera_id": camera_id,
        "rois": detection_output.get("rois"),
        "snapshot_url": detection_output.get("snapshot_url"),
        # Frame capture timestamp — parking_detection's _event_ts() reads this
        # so DB rows are stamped with capture time, not rule-fire time.
        "timestamp": detection_output.get("timestamp"),
        # Per-slot CV occupancy signal from camera-detection's logo-occlusion
        # check. parking_detection ORs this with YOLO occupancy so YOLO-miss
        # frames still drive parking_intime / parking_outtime.
        "logo_occluded": detection_output.get("logo_occluded"),
    }

def evaluate_single_usecase(usecase_id: str, slim_payload: Dict[str, Any], camera_id: str) -> UsecaseResult:
    """
    Evaluate one detection aganist the slim detection payload.  
    Each usecase is an independent unit. 
    Parallelism: 15 usecases -> 15 celery tasks -> can be processed by worker pool simultaneously instead of sequentally.
    Fault Isolation: If one of the usecase fails rest of them works unaffected
    payload reduction: Each task only carries it's own result.
    Reusability: This function is called identically from :
        - workers/tasks.py(celery worker context)
        - usecase/service.py(direct api call)
        -tests(unit testing)

    Args:
    usecase_id: the id of the usecase to evaluate, e.g. "person_in_roi"
    slim_payload: the slimmed down detection payload
    camera_id: the id of the camera that captured the detection

    Returns:
    UsecaseResult object containing the result of the evaluation
    """
    logger.info(f"[ENGINE] Evaluating usecase '{usecase_id}' for camera '{camera_id}' with slim payload: {slim_payload}")

    try:
        rule = get_usecase_rule(usecase_id)

        # snapshot_url is included in slim_payload directly — no Redis detour needed
        evaluation = rule.evaluate(slim_payload)
        print(f"[ENGINE DEBUG] usecase={usecase_id} | evaluation keys={list(evaluation.keys())}")
        matched = evaluation.get("matched_objects", [])

        slim_matched = [_slim_detection(d) for d in matched]

        # Collect every rule-specific key beyond the standard ones into extras.
        # This is pluggable — any field a rule returns flows through automatically.
        _standard_keys = {"triggered", "matched_objects"}
        extras = {k: v for k, v in evaluation.items() if k not in _standard_keys}

        result = UsecaseResult(
            usecase_id=usecase_id,
            triggered=evaluation.get("triggered", False),
            matched_count=len(matched),
            matched_objects=slim_matched,
            detection_id=slim_payload.get("first_detection_id"),
            snapshot_url=slim_payload.get("snapshot_url"),
            extras=extras,
        )
        logger.info(
            f"[ENGINE] Usecase '{usecase_id}' evaluation completed. Triggered: {result.triggered}, Matched Count: {result.matched_count}, Matched Objects: {result.matched_objects}"
        )
        return result
    
    except Exception as e:
        logger.exception(f"[ENGINE] Error evaluating usecase '{usecase_id}': {e}")
        raise


def evaluate_all_usecases(
        camera_id: str,
        detection_output: Dict[str, Any], usecases: List[str],
        task_id: str = None) -> List[UsecaseResult]:

        """
        Evaluate ALL usecases for ONE frame of ONE camera, sequentially.

        This is the single unit of work for both execution paths:
        - direct path (service.py): called inline in the API process.
        - queue path (workers/tasks.evaluate_frame_task): called inside a
          per-camera Celery worker (one whole-frame task, NOT one task per
          usecase). Running every usecase for a frame in this one function —
          in order, in one process — is what keeps ChargingSession correct:
          parking_detection runs first and hands `tracked_cars` to
          gun_detection / vehicle_extraction (see the injection below), and
          the caller receives the complete, ordered result list to assemble
          one ChargingSession row from.

        Each usecase is wrapped in try/except and continues on failure, so one
        bad usecase never drops the rest of the frame's results.

        Args:
                camera_id: camera_identifier
                detection_output: full detection API response
                usecases: list of usecase IDs
                task_id: Celery task id (queue path only). Injected as
                        slim["_task_id"] so rules can dedupe event publishes
                        across task retries. None in the direct path.
            Returns:
                List of UsecaseResult objects

        """
        slim = build_slim_payload(detection_output)
        # Idempotency key for the queue path: a Celery retry reuses the same
        # task id, so rules keyed on _task_id won't re-publish the same event.
        if task_id is not None:
            slim["_task_id"] = task_id
        results = []

        for usecase_id in usecases:
            try:
                result = evaluate_single_usecase(usecase_id=usecase_id, slim_payload=slim, camera_id=camera_id)
                results.append(result)
                _persist_result_direct(camera_id, result)

                # After parking_detection runs:
                # 1. Backfill track_ids into slim["detections"] (existing behaviour).
                # 2. Inject slim["tracked_cars"] so gun_detection and vehicle_extraction
                #    read the canonical tracked list without running their own trackers.
                if usecase_id == "parking_detection":
                    tracked_by_bbox = {
                        (obj.get("bbox", {}).get("x1"), obj.get("bbox", {}).get("y1"),
                         obj.get("bbox", {}).get("x2"), obj.get("bbox", {}).get("y2")): obj.get("track_id")
                        for obj in result.matched_objects
                        if obj.get("track_id") and obj.get("bbox")
                    }
                    for det in slim["detections"]:
                        b = det.get("bbox", {})
                        key = (b.get("x1"), b.get("y1"), b.get("x2"), b.get("y2"))
                        if key in tracked_by_bbox:
                            det["track_id"] = tracked_by_bbox[key]
                    # Canonical tracked car list for downstream rules
                    slim["tracked_cars"] = result.matched_objects

            except Exception as e:
                logger.error(f"[ENGINE] Failed to evaluate usecase '{usecase_id}' for camera '{camera_id}' with error: {e}")
                continue
        return results


def _persist_result_direct(camera_id: str, result: "UsecaseResult") -> None:
    """Persist result to DB in direct (non-queue) mode. Mirrors tasks._persist_result."""
    try:
        from shared.database.connection import SessionLocal
        if SessionLocal is None:
            logger.debug(f"[ENGINE] DATABASE_URL not configured — skipping DB persistence for {camera_id}/{result.usecase_id}")
            return
        from shared.database.persistence import persist_usecase_result
        persist_usecase_result(
            camera_id=camera_id,
            usecase_name=result.usecase_id,
            triggered=result.triggered,
            detection_id=result.detection_id,
        )
        logger.debug(f"[ENGINE] persisted result for {camera_id}/{result.usecase_id}")
    except Exception as e:
        logger.error(f"[ENGINE] DB persist failed for {camera_id}/{result.usecase_id}: {e}")