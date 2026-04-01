import logging
from typing import Any, Dict, List

from usecase.rules import get_usecase_rule
from usecase.schemas import UsecaseResult
from workers.redis_state import set_snapshot, get_snapshot


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
    snapshot_b64 is stored in Redis separately (via set_snapshot) and NOT
    included in the task payload — base64 images are 500KB-1MB each and
    bloat RabbitMQ messages causing tasks to fail/timeout when sent 5× per frame.
    Workers retrieve the snapshot via get_snapshot(camera_id) instead.
    """
    camera_id = detection_output.get("camera_id")
    snapshot_b64 = detection_output.get("snapshot_b64")

    # Store snapshot in Redis so workers can fetch it without it travelling
    # through RabbitMQ inside every task message.
    if camera_id and snapshot_b64:
        set_snapshot(camera_id, snapshot_b64)

    raw_detections = detection_output.get("detections", [])
    return {
        "detections": [_slim_detection(d) for d in raw_detections],
        "first_detection_id": detection_output.get("first_detection_id"),
        "camera_id": camera_id,
        "rois": detection_output.get("rois"),
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

        # Fetch snapshot from Redis (stored by build_slim_payload, not in task payload)
        snapshot_b64 = get_snapshot(camera_id)

        # Inject snapshot into the payload so rules like vehicle_extraction can use it
        payload_with_snapshot = dict(slim_payload)
        if snapshot_b64:
            payload_with_snapshot["snapshot_b64"] = snapshot_b64

        evaluation = rule.evaluate(payload_with_snapshot)
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
            snapshot_b64=snapshot_b64,
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
        detection_output: Dict[str, Any], usecases: List[str]) -> List[UsecaseResult]:

        """
        Evaluate multiple usecases sequentially(no celery)

        For the legacy API endpoint, testing and small deployyment where RabbitMQ isn't running. It uses the same evaluate_single_usecase() function - so behavior is identical to the worker path.

        -celery path:worker calls evaluate_single_usecase() per task
        -direct path: this funnction calls evaluate_single_usecase() in a loop
        -Tests: both paths produce identical results because same engine

        Args:
                camera_id: camera_identifier
                detection_output: full detection API response
                usecases: list of usecase IDs
            Returns:
                List of UsecaseResult objects

        """
        slim = build_slim_payload(detection_output)
        results = []

        for usecase_id in usecases:
            try:
                result = evaluate_single_usecase(usecase_id=usecase_id, slim_payload=slim, camera_id=camera_id)
                results.append(result)
            except Exception as e:
                logger.error(f"[ENGINE] Failed to evaluate usecase '{usecase_id}' for camera '{camera_id}' with error: {e}")
                continue
        return results