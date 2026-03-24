"""
This is the bridge between the FastAPI layer and the celery/rabbitmq layer.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from celery.result import AsyncResult
from workers.celery_app import celery_app
from workers.tasks import evaluate_usecase_task
from usecase.engine import build_slim_payload
from usecase.schemas import UsecaseResult

logger = logging.getLogger(__name__)

TAKE_RESULT_TIMEOUT = int(30)

def submit_usecase_tasks(
        camera_id: str,
        detection_output : Dict[str, Any],
        usecases: List[str],
) -> Dict[str, AsyncResult]:
    """
    Submit one celery task per usecase. Returns task handles immediately.

    If you submit 15 separate tasks, the worker pool distributes them. with 8 workers and 15 usecases: first 8 runs parallel,
    then the remaining 7 tasks start as soon as a worker becomes available.

    build_slim_payload() strips detection output to only what rules need. 
    we call it once and pass the same slim_payload to all 15 tasks. That means each task message in rabbitmq carries minimal data.

    without this: 15 tasks * full_detection_payload = 15* the bandwidth.
    with_this: 15 tasks * slim_payload = much smaller,identical per task.

    Args:
        camera_id: the id of the camera that captured the detection
        detection_output: tfull detection api response.
        usecases: a list of usecase ID to evaluate.

    Returns:
        Dict mapping usecase_id->celery AsyncResult handle. The caller can use these handles to check task status or get results later.
    """

    slim_payload = build_slim_payload(detection_output)
    task_handles: Dict[str, AsyncResult] = {}
    for usecase_id in usecases:
        #.delay() is shorthand for .apply_async()
        # It serializes the args to JSON, publishes to rabbitmq,returns immediately
        # The actual work happens in a worker process, not here
        handle = evaluate_usecase_task.delay(
            usecase_id=usecase_id,
            slim_payload=slim_payload,
            camera_id=camera_id,
        )
        task_handles[usecase_id] = handle
        logger.debug(
            f'[Queue] Submitted task for usecase_id={usecase_id}, camera_id={camera_id}, task_id={handle.id}'
        )
    return task_handles
    
async def await_usecase_results(
        task_handles: Dict[str, AsyncResult],
        camera_id: str,
        timeeout: int = TAKE_RESULT_TIMEOUT,
) -> List[UsecaseResult]:
    
    """ 
    collect results from all the submitted tasks. waits up to 'timeouts' seconds per tasks.
    FasAPI endpoints are async. We need to await results without blocking the event loop (which would prevent other requests from being served).
    asyncio.to_thread() runs thee blocking celery.get() in a thread pool.
    
    Args:
        task_handles: dict of usecase_id->celery AsyncResult handle returned by submit_usecase_tasks()
        camera_id: for logging
        timeout: how long to wait for each task result before giving up

    Returns:
        List of UsecaseResult objects(one per usecase, in submission order)
    """
    results: List[UsecaseResult] = []

    #Collect all tasks concurrently using asyncio gather
    # Each task waits independently- a slow task doesn't delay others
    async def collect_one(usecase_id: str, handle:AsyncResult) -> UsecaseResult:
        try:
            result_dict = await asyncio.to_thread(
                handle.get,
                timeout=timeeout,
                propagate=False,
            )

            if isinstance(result_dict,Exception):
                logger.error(f'[Queue] Task for usecase_id={usecase_id} failed with exception: {result_dict}')
                return _safe_default_result(usecase_id, handle.id)
            
            return UsecaseResult(**result_dict)
        
        except Exception as e:
            logger.error(f'[Queue] Failed to collect {camera_id}:{usecase_id} with task_id={handle.id} due to exception: {e}')
            return _safe_default_result(usecase_id, handle.id)

    coroutines = [
        collect_one(usecase_id, handle) for usecase_id, handle in task_handles.items()

    ]
    results = list(await asyncio.gather(*coroutines))

    triggered_count = sum(1 for r in results if r.triggered)

    logger.info(f'[Queue] Collected {len(results)} results for camera={camera_id}, tiggered={triggered_count}/{len(results)}')

    return results



def _safe_default_result(usecase_id:str, task_id: str) -> UsecaseResult:
    """
    Return a safe non-triggered result when a task fails or times out.
    Failing one usecase should never break the entire pipeline.
    The orchestrator and alert service expect a result for every usecase that was requested. We return triggered=False as a safe fallback so the pipeline continues normally"""

    logger.warning(
        f'[Queue] Returning safe default result for usecase_id={usecase_id}, task_id={task_id}'
    )
    return UsecaseResult(
        usecase_id=usecase_id,
        triggered=False,
        matched_count=0,
        matched_objects=[],
        detection_id=None,
        snapshot_b64=None)
    
def get_task_status(task_id: str) -> Dict[str, Any]:
    """
    Get the current status of a celery task by ID.
    For long-running or queued tasks, callers can poll this endpoint to check progress without waiting. Useful for a dashboard showing live worker queue depth.
    Returns dict with: task_id, status, result(if done)"""

    result = AsyncResult(task_id, app=celery_app)
    response = {
        'task_id': task_id,
        'status': result.status, #PENDING, STARTED, SUCCESS, FAILURE, RETRY
    }

    if result.successful():
        response["result"] = result.result
    elif result.failed():
        response["error"] = str(result.result)
    return response

