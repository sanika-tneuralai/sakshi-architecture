import logging
from typing import Any, Dict, List

from usecase.rules import get_usecase_rule
from usecase.schemas import UsecaseResult


logger = logging.getLogger(__name__)

def _slim_detection(detetion: Dict[str, Any]):
    """
    strip a detection dict down to only the fields usecase rules need.
    bcoz the detection output contains bbox, coord, raw tensors, and other data that rules never use.

    Rules need : class_name, confidence, in_roi, bbox is included for dashboard/alert display purpose.
    """
    return {
        "class_name": detetion.get("class_name"),
        "confidence": detetion.get("confidence"),
        "in_roi": detetion.get("in_roi", False),
        "bbox": detetion.get("bbox", {})
    }

def build_slim_payload(detection_output:Dict[str, Any]) -> Dict[str, Any]:
    """
   Build a slim version of the detection ouptut for queueing.
   The orchestrator or API calls this once before submitting the tasks.
   Each celery task then receives this slim payload instead of the full detection output.
   You build it once, reuse across all 15 tasks.
    
    Args: detection_output: The full detection output from the camera. which may contain many fields and data that are not relevant for usecase evaluation.

    Returns: slim dict with only what usecase rules need.
    """
    raw_detections = detection_output.get("detections", [])
    return{
        "detections": [_slim_detection(d) for d in raw_detections],
        "screenshot_path": detection_output.get("screenshot_path"),
        "first_detection_id": detection_output.get("first_detection_id"),
        "camera_id": detection_output.get("camera_id")
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
        evaluation = rule.evaluate(slim_payload)
        matched = evaluation.get("matched_objects", [])

        slim_matched = [_slim_detection(d) for d in matched]

        result = UsecaseResult(
            usecase_id = usecase_id,
            triggered = evaluation.get("triggered", False),
            matched_count=len(matched),
            matched_objects=slim_matched,
            detection_id=slim_payload.get("first_detection_id"),
            screenshot_path=slim_payload.get("screenshot_path"),
        )
        logger.info(
            f"[ENGINE] Usecase '{usecase_id}' evaluation completed. Triggered: {result.triggered}, Matched Count: {result.matched_count}, Matched Objects: {result.matched_objects}"
        )
        return result
    
    except Exception as e:
        logger.error(f"[ENGINE] Unknown usecase '{usecase_id}': {e}")
        return UsecaseResult(
            usecase_id=usecase_id,
            triggered=False,
            matched_count=0,
            matched_objects=[],
            detection_id=slim_payload.get("first_detection_id"),
            screenshot_path=slim_payload.get("screenshot_path"),
        )
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