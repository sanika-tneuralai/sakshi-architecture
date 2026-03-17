"""
usecase/api.py - FastAPI router. thin HTTP layer only.

This API validate HTTP request -> call service -> return HTTP response,
All business logic lives in service.py and engine.py

This follows the principle:
     Router - HTTP concerns (status codes, request parsing, response models)
     services - Application concerns (which path to take, coordination)
     engine - Buisness logic (evaluation, rules)
     Tasks - Worker concerns(retry, persistence, serialization)

ENDPOINTS:
POST /evaluate-usecases: main endpoint (queue or direct)
GET /usecase/task/{task_id}: check status od a specific celery task
GET /usecase/queue-status: check if queue mode is active
"""

import os
import logging
from fastapi import APIRouter, HTTPException

from usecase.schemas import UsecaseRequest, UsecaseResponse
from usecase.service import evaluate_usecases_service
logger = logging.getLogger(__name__)

router = APIRouter(prefix='/usecase', tags=['usecase'])

USE_WORKER_QUEUE = os.getenv("USE_WORKER_QUEUE", "false").lower() == "true"

@router.post('/evaluate', response_model=UsecaseResponse)
async def evaluate_usecase(request: UsecaseRequest):
    """
    Evaluate multiple usecases against detection output.
    
    when USE_WORKER_QUEUE is true, this endpoint submits tasks to the worker queue and waits for results asynchronously. The API process is not blocked while waiting, allowing it to handle other requests concurrently.
    
    when USE_WORKER_QUEUE is false, this endpoint evaluates usecases directly in the API process. This is simpler and has lower latency for small loads, but can block the API if evaluation takes a long time or if there are many concurrent requests.
    """

    if not request.usecases:
        raise HTTPException(status_code = 400, detail='At least one usecase must be specicied')

    try:
        result = await evaluate_usecases_service(
            camera_id=request.camera_id,
            detection_output=request.detection_output,
            usecases=request.usecases,
        )
        logger.info(
           f'[API] Successfully evaluated usecases for camera_id={request.camera_id}, usecases={request.usecases}, queue_mode={USE_WORKER_QUEUE}, results = {len(result.results)}' 
        )
        return result
    
    except Exception as e:
        logger.exception(f'[API] Error evaluating usecases for camera_id={request.camera_id}, usecases={request.usecases}, queue_mode={USE_WORKER_QUEUE}: Error: {e}')
        raise HTTPException(status_code=500, detail=f'Usecase evaluation failed: {str(e)}')
    
@router.get('/task/{task_id}')
def get_task_status(task_id: str):
    """
    Check the status of a specific celery task.
    
    In the queue mode, tasks are processed asynchronously. For debugging, monitoring, or a dashboard, you need to inspect individual task state.
    states: pending (not yet picked up by worker), started (currently being processed), success (completed successfully), failure (raised an exception), retry (scheduled for retry after failure)  PENDING -> STARTED -> SUCCESS/FAILURE/RETRY

    useful for:
    -dashboard: how manyy tasks are currently queued for processing?
    -debugging: why did usecases X not trigger for camera Y
    -Monitoring: show worker queue depth in real time

    """
    if not USE_WORKER_QUEUE:
        raise HTTPException(status_code=400, detail='Task status endpoint only avaialble in queue mode. set USE_WORKER_QUEUE=true')
    from workers.queue import get_task_status
    return get_task_status(task_id)


@router.get('/queue-status')
def get_queue_status():
    """
    Return current execution mode and queue health.
    
    Operation need to know:
    1. is the service running in queue mode or direct mode?
    2. If queue mode, can it reach RabbitMQ and Redis?
    Without this you'd have to check env vars, manually on the server.
    """

    status = {
        'mode': 'queue' if USE_WORKER_QUEUE else 'direct',
        'queue_enabled': USE_WORKER_QUEUE,
    }

    if USE_WORKER_QUEUE:

        try:
            from workers.celery_app import celery_app
            inspect = celery_app.control.inspect(timeout=2.0)
            active = inspect.active()
            status['broker_reachable'] = True
            status['active_workers'] = len(active) if active else 0
            status['worker_details'] = active or {}

        except Exception as e:
            status['broker_reachable'] = False
            status['broker_error'] = str(e)

    return status
