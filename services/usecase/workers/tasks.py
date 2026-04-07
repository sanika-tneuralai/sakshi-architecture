import logging
from typing import Any, Dict

from .celery_app import celery_app
from usecase.engine import evaluate_single_usecase
from usecase.schemas import UsecaseResult

logger = logging.getLogger(__name__)

@celery_app.task(name="workers.tasks.evaluate_usecase_task",
                 bind=True,
                 max_retries=3,
                 default_retry_delay=2,
                 soft_time_limit=60,
                 time_limit=90,
                 )

def evaluate_usecase_task(
    self,
    usecase_id: str,
    slim_payload: Dict[str,Any],
    camera_id: str,
) -> Dict[str, Any]:
    """"
    Celery task:  Evaluate one usecase for one camera's detection output.
    This is done as the celery serializes the arguments to json when publishing to RabbitMQ.
    This means arguments  must be json-serializable(dicts, lists,strings).
    We can't pass UsecaseResultobject directly they get converted to dict and returned as dict, then the caller converts back to usecaeResult.

    bind=True gives access to self.request.id (the Celery task ID). We inject
    this as slim_payload["_task_id"] so rules can use it as an idempotency key
    to avoid re-publishing events if the task retries.

     Args:
     usecase_id: eg "person_in_roi"-which rules to run
     slim_payload: the slimmed down detection output that contains only what usecase rules need, built by build_slim_payload() in engine.py
     camera_id: "cam_1", for logging and DB persistence.

     Returns:
        dict representation of UsecaseResult(JSON-serializable)
     """
    logger.info(f'[Task] worker picked up task| usecase_id={usecase_id} | camera_id={camera_id}')

    # Inject task_id for idempotency — rules use this to avoid re-publishing
    # events when the task retries after a transient failure.
    slim_payload = {**slim_payload, "_task_id": self.request.id}

    try:
        #call the pure engine function, same function used in direct API path
        result: UsecaseResult = evaluate_single_usecase(
            usecase_id=usecase_id,
            slim_payload=slim_payload,
            camera_id=camera_id
        )

        #persist to DB inside the worker(not in the API process), DB writes are I/O bound. Doing them in the worker means the API returns immediately without waiting for DB. worker handle persistence.
        _persist_result(camera_id, result)

        result_dict = result.model_dump()  # snapshot_url is a short string, safe to include in Celery result
        logger.info(
            f'[Task] Completed | camera{camera_id} | usecase={usecase_id}| triggered{result.triggered}'
        )
        return result_dict
    
    except Exception as exc:
        logger.error(f'[Task] Failed | camera={camera_id} | usecase={usecase_id} | attempt={self.request.retries + 1}/{self.max_retries + 1} | error={str(exc)} | retrying...')
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)  # 1s, 2s, 4s backoff
    

def _persist_result(camera_id: str, result: UsecaseResult):
    """
    Persist useccase result to DB.
    Seperation of concerns. The task functions handles celery mechanics
    (retry, serialization, logging) This function handles DB concerns.
    If you swap databases, you change only this function. So if DB persistence fails, we can log and continue without failing the entire task."""
    try:
        from shared.database.connection import SessionLocal
        if SessionLocal is None:
            logger.debug(f'[Task] DATABASE_URL not configured — skipping DB persistence for {camera_id}/{result.usecase_id}')
            return
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