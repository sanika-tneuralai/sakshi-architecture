import logging 
from typing import Any, Dict

from .celery_app import celery_app
from usecase.engine import evaluate_single_usecase
from usecase.schemas import UsecaseResult

logger = logging.getLogger(__name__)

@celery_app.task(name="workers.tasks.evaluate_usecase_task",
                 max_retries=3, #max retries =3
                 default_retry_delay=2, #wait 2 seconds after first retry
                 queue="usecase_queue", #which queue to publish to(defined in celery_app.py)
                 soft_time_limit=60,
                 time_limit=90,
                 )

def evaluate_usecase_task(
    usecase_id: str,
    slim_payload: Dict[str,Any],
    camera_id: str,
) -> Dict[str, Any]:
    """"
    Celery task:  Evaluate one usecase for one camera's detection output.
    This is done as the celery serializes the arguments to json when publishing to RabbitMQ.
    This means arguments  must be json-serializable(dicts, lists,strings).
    We can't pass UsecaseResultobject directly they get converted to dict and returned as dict, then the caller converts back to usecaeResult.
     
     Args:
     usecase_id: eg "person_in_roi"-which rules to run
     slim_payload: the slimmed down detection output that contains only what usecase rules need, built by build_slim_payload() in engine.py
     camera_id: "cam_1", for logging and DB persistence.

     Returns:
        dict representation of UsecaseResult(JSON-serializable)
     """
    logger.info(f'[Task] worker picked up task| usecase_id={usecase_id} | camera_id={camera_id}')

    try:
        #call the pure engine function, same function used in direct API path
        result: UsecaseResult = evaluate_single_usecase(
            usecase_id=usecase_id,
            slim_payload=slim_payload,
            camera_id=camera_id
        )

        #persist to DB inside the worker(not in the API process), DB writes are I/O bound. Doing them in the worker means the API returns immediately without waiting for DB. worker handle persistence.
        _persist_result(camera_id, result)

        result_dict = result.model_dump() #return as dict (celery stores this as redis json)
        logger.info(
            f'[Task] Completed | camera{camera_id} | usecase={usecase_id}| triggered{result.triggered}'
        )
        return result_dict
    
    except Exception as exc:
        logger.error(f'[Task] Failed | camera{camera_id} | usecase={usecase_id} | error={str(exc)} | retrying...')
        raise evaluate_usecase_task.retry(
            exc=exc,
            countdown=2
        ) # Exponential backoff: retry 1 waits 2s, retry 2 waits 4s, retry 3 waits 8s
    

def _persist_result(camera_id: str, result: UsecaseResult):
    """
    Persist useccase result to DB.
    Seperation of concerns. The task functions handles celery mechanics
    (retry, serialization, logging) This function handles DB concerns.
    If you swap databases, you change only this function. So if DB persistence fails, we can log and continue without failing the entire task."""
    try:
        from shared.database.persistence import persist_usecase_result
        persist_usecase_result(
            camera_id=camera_id,
            usecase_name=result.usecase_id,
            triggered=result.triggered,
            detection_id=result.detection_id,
        )
        logger.debug(f'[Task] persisted result for {camera_id}/{result.usecase_id}')
    except Exception as e:
        logger.error(f'[Task] DB persist failed for {camera_id}/{result.usecase_id}: {e}')