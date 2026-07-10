"""
Usecase/service.py - Orchestrator. Decides: direct path or queue path?

Two execution paths, both run the SAME engine (evaluate_all_usecases), so
behaviour is identical — the only difference is WHERE the frame is evaluated.

1. Queue path (production, USE_WORKER_QUEUE=true):
   submit_frame_task() -> RabbitMQ (camera's shard queue) -> Celery worker
   (concurrency=1) runs evaluate_all_usecases -> Redis -> await one result.
   ONE whole-frame task per (camera, frame), routed by camera_id. Cameras run
   in parallel across shard workers; each camera's frames stay ordered on its
   own lane. This is what makes 5 (and later 500) streams run in parallel
   without corrupting the per-camera slot state / ChargingSession.

2. Direct path (dev/test, USE_WORKER_QUEUE=false):
   evaluate_all_usecases() runs inline in the API process. Simple, no broker,
   but all cameras serialize on the one event loop — fine for local dev.

The path is selected by the USE_WORKER_QUEUE env variable, so you can develop
locally without Docker/RabbitMQ and switch to queue mode in production with a
single env change.

Note on granularity: we submit ONE task per frame (all usecases together),
NOT one task per usecase. Fanning out per usecase would run parking/gun/
vehicle in parallel and break the within-frame data dependency
(parking_detection must run first and hand `tracked_cars` downstream), which
is what previously produced incomplete/garbled ChargingSession rows.
"""

import os
import logging
from typing import Dict, Any, List

from usecase.schemas import UsecaseResult, UsecaseResponse
from usecase.engine import evaluate_all_usecases

logger = logging.getLogger(__name__)

USE_WORKER_QUEUE = os.getenv("USE_WORKER_QUEUE", "false").lower() == "true"

async def evaluate_usecases_service(
        camera_id: str,
        detection_output: Dict[str, Any],
        usecases: List[str],
        
) -> UsecaseResponse:
    """
    Main service entry point. Routes to queue or direct path.
    
    The async signature even for direct is because the FastAPI endpoints are async. Having an async service function means  the API endpoint doesn't need to know which path is taken. The direct path runs synchronously inside the async function-thats fine for small loads. For heavy loadss, the queue path properly uses asyncio primitives.
    
    Args:
        camera_id: the id of the camera that captured the detection
        detection_output: the full detection output from the camera, which may contain many fields and data that are not relevant for usecase evaluation.
        usecases: a list of usecase ID to evaluate.

    Returns:
        UsecaseResponse schema object
    """

    logger.info(f"[SERVICE] Evaluating usecases for camera_id={camera_id}, usecases={usecases}, queue_mode={USE_WORKER_QUEUE}")

    if USE_WORKER_QUEUE:
        results = await _queue_path(camera_id, detection_output, usecases)
    else:
        results = _direct_path(camera_id, detection_output, usecases)
    return UsecaseResponse(
        camera_id=camera_id,
        results=results)


async def _queue_path(
        camera_id: str,
        detection_output: Dict[str, Any],
        usecases: List[str],) -> List[UsecaseResult]:
    """
    Submit ONE whole-frame task to the camera's shard queue and await it.
    Lazy import: if RabbitMQ/Redis aren't running, importing workers.queue at
    module level would crash the service on startup. Lazy import lets the
    service start and only fails when the queue is actually used.

    A None handle means backpressure kicked in (a frame for this camera is
    still being processed) — we return [] so the caller simply skips this
    cycle. The in-flight frame will produce the next ChargingSession update.
    """
    from workers.queue import submit_frame_task, await_frame_result

    handle = submit_frame_task(
        camera_id=camera_id, detection_output=detection_output, usecases=usecases
    )
    if handle is None:
        logger.info(
            f"[SERVICE] camera={camera_id}: previous frame still in flight — skipping this cycle"
        )
        return []

    return await await_frame_result(handle=handle, camera_id=camera_id)

def _direct_path(
        camera_id: str,
        detection_output: Dict[str, Any],
        usecases: List[str],) -> List[UsecaseResult]:
    """
    Evaluate usecases directly in the API process (no queue).

    Both paths call the SAME engine function, evaluate_all_usecases():
      - direct path: this function calls it inline;
      - queue path: workers.tasks.evaluate_frame_task calls it inside a worker.

    Same engine -> identical behaviour -> no divergence between dev and
    production, so bugs can't hide in a path that only runs in one environment.
    """
    return evaluate_all_usecases(
        camera_id = camera_id,
        detection_output = detection_output,
        usecases = usecases,
    )


  

                                          

