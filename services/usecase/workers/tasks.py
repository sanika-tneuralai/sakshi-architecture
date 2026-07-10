import logging
from typing import Any, Dict, List

from .celery_app import celery_app
from .redis_state import release_inflight
from usecase.engine import evaluate_all_usecases
from usecase.schemas import UsecaseResult

logger = logging.getLogger(__name__)


@celery_app.task(name="workers.tasks.evaluate_frame_task",
                 bind=True,
                 max_retries=3,
                 default_retry_delay=2,
                 soft_time_limit=90,
                 time_limit=120,
                 )
def evaluate_frame_task(
    self,
    camera_id: str,
    detection_output: Dict[str, Any],
    usecases: List[str],
) -> List[Dict[str, Any]]:
    """
    Celery task: evaluate ALL usecases for ONE frame of ONE camera.

    This is the unit of work for the camera-sharded queue. Each camera is
    routed to exactly one shard queue drained by a single-slot worker
    (concurrency=1), so:
      * frames of a camera are processed strictly in order (frame N commits
        its Redis slot state before frame N+1 starts) — the lockless
        get/mutate/set in redis_state stays race-free without a global
        concurrency=1-per-usecase bottleneck;
      * usecases WITHIN a frame run in dependency order via
        evaluate_all_usecases (parking_detection first, then it hands
        `tracked_cars` to gun_detection / vehicle_extraction);
      * different cameras run fully in parallel across separate shard workers.

    The returned list is the complete, ordered set of results for the frame,
    so the orchestrator assembles one correct ChargingSession row from it.

    Args must be JSON-serializable (Celery serializes to JSON for RabbitMQ).

    Returns:
        List of dict representations of UsecaseResult (JSON-serializable).
    """
    logger.info(f'[Task] worker picked up FRAME | camera_id={camera_id} | usecases={usecases}')
    try:
        # self.request.id is stable across retries -> rules dedupe event
        # publishes via slim["_task_id"].
        results: List[UsecaseResult] = evaluate_all_usecases(
            camera_id=camera_id,
            detection_output=detection_output,
            usecases=usecases,
            task_id=self.request.id,
        )
        triggered = sum(1 for r in results if r.triggered)
        logger.info(
            f'[Task] Completed FRAME | camera={camera_id} | '
            f'results={len(results)} | triggered={triggered}'
        )
        # Success -> free the camera so its next frame can be submitted.
        release_inflight(camera_id)
        return [r.model_dump() for r in results]

    except Exception as exc:
        # NOTE: we deliberately do NOT release the in-flight guard on a retry.
        # Holding it makes the producer skip newer frames for this camera
        # during the retry backoff, so the retried (older) frame can never be
        # overtaken by a newer frame on the shard queue — ordering is
        # preserved. The guard is released only on terminal outcomes; the TTL
        # backstops a worker crash.
        if self.request.retries >= self.max_retries:
            logger.error(
                f'[Task] FRAME failed permanently | camera={camera_id} | '
                f'retries exhausted ({self.max_retries}) | error={str(exc)}'
            )
            release_inflight(camera_id)
            raise
        logger.error(
            f'[Task] FRAME failed | camera={camera_id} | '
            f'attempt={self.request.retries + 1}/{self.max_retries + 1} | error={str(exc)} | retrying...'
        )
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
