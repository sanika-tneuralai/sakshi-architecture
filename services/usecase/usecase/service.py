"""
Usecase/service.py - Orchestrator. Decides: direct path or queue path?

Two Execution paths:
1.Async path(default, production):
submit_usecase_tasks() -> RabbitMQ -> Celery Worker -> Redis -> await results
Best for 100 camereas, high throughput, parallel execution

2. Direct path(fallback, testing):
evaluate_all_usecases() -> sequential evaluation in the API process
Best fir: development without RabbitMQ, unit tests, small deployments

The path is selected by the USE_WORKER_QUEUE env variable. 
This means you can develop locally without Docker/RabbbitMQ and 
switch to queue mode in production with a simple environment variable change.

100 cmaeras * 15 usecases = 1500 tasks per poll cycle(every one second)

Async path: 1500 tasks distributed across N workers (camera processing time: slowest usecases)
Direct path: 1500 evaluations in the API process(sequential per camera)(camera processing time: sum of all usecases, much slower, not scalable, but simple for testing and development)
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
    Submit tasks to RabbitMQ workers and wait for results.
    If RabbitMQ/Redis aren't running, importing workers.queue at module level would crash the entire service on startup. Lazy import means the service starts fine and only fails when actually trying to use the queue. This make development without docker much easier.
    """
    from workers.queue import submit_usecase_tasks, await_usecase_results

    task_handles = submit_usecase_tasks(camera_id=camera_id, detection_output=detection_output, usecases=usecases)

    results = await await_usecase_results(task_handles = task_handles,
    camera_id=camera_id)
    return results

def _direct_path(
        camera_id: str,
        detection_output: Dict[str, Any],
        usecases: List[str],) -> List[UsecaseResult]:
    """
    Evaluate usecases directly in the API process (no queue). 

    Why this uses the same engine as the queue path. 
    evaluate_all_usecases() calls the evaluate_single_uusecase() in a loop.
    evaluate_usecase_task(the celery task) also calls evaluate_single_usecase().

    same engine function -> identical behavior-> no behavioral divergence between dev and production. If you have different logic in the direct path vs the queue path, you might end up with bugs that only appear in production and are hard to debug. By using the same engine function for both paths, you ensure that the core logic is consistent regardless of how it's executed.
    """
    return evaluate_all_usecases(
        camera_id = camera_id,
        detection_output = detection_output,
        usecases = usecases,
    )


  

                                          

