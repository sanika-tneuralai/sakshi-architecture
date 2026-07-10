"""
This is the bridge between the FastAPI layer and the celery/rabbitmq layer.
"""

import asyncio
import logging
import os
import threading
import zlib
from typing import Dict, Any, List, Optional

from celery.result import AsyncResult
from workers.celery_app import celery_app
from workers.tasks import evaluate_frame_task
from workers.redis_state import try_acquire_inflight, release_inflight, get_or_assign_shard
from usecase.schemas import UsecaseResult

logger = logging.getLogger(__name__)

TAKE_RESULT_TIMEOUT = int(30)

# Number of camera shards (= number of ordered lanes / single-slot workers).
# A camera is pinned to shard crc32(camera_id) % N_SHARDS, so its frames are
# always processed in order by one worker while different cameras run in
# parallel across shards. Scale by raising N_SHARDS and adding shard workers;
# the producer and the workers MUST agree on this value.
N_SHARDS = int(os.getenv("N_SHARDS", "5"))

# How long the per-camera in-flight guard survives a worker crash. Slightly
# above the frame task's time_limit (120s) so a live-but-slow frame is never
# evicted mid-flight.
INFLIGHT_TTL = int(os.getenv("INFLIGHT_TTL", "130"))

# redis-py connections are not thread-safe. asyncio.gather runs concurrent
# handle.get() calls on multiple threads, causing interleaved reads and
# Protocol Errors. This lock serializes Redis reads to prevent that.
_redis_read_lock = threading.Lock()


def _shard_for(camera_id: str) -> int:
    """
    Shard index for a camera. Prefers sticky, balanced assignment via Redis
    (get_or_assign_shard) so a small fleet spreads one-camera-per-lane instead
    of colliding. Falls back to stateless crc32 hashing if Redis is
    unavailable — still stable per camera, just not collision-balanced.
    """
    shard = get_or_assign_shard(camera_id, N_SHARDS)
    if shard is None:
        shard = zlib.crc32(camera_id.encode("utf-8")) % N_SHARDS
    return shard


def submit_frame_task(
        camera_id: str,
        detection_output: Dict[str, Any],
        usecases: List[str],
) -> Optional[AsyncResult]:
    """
    Submit ONE whole-frame task for a camera, routed to that camera's shard.

    This is the production path: one task per (camera, frame) carrying every
    usecase, NOT one task per usecase. The task runs evaluate_all_usecases in
    a single-slot shard worker, so usecases run in dependency order and the
    frame's results come back complete and ordered (correct ChargingSession).

    Backpressure: if a frame for this camera is still being processed we skip
    submitting a new one and return None — the caller treats that as "no
    results this cycle". This stops a slow camera (e.g. an LLM frame > poll
    interval) from piling unbounded frames onto its shard queue.

    Returns the AsyncResult handle, or None if skipped by backpressure.
    """
    if not try_acquire_inflight(camera_id, ttl_seconds=INFLIGHT_TTL):
        logger.info(
            f'[Queue] camera={camera_id} still has a frame in flight — '
            f'skipping this frame (backpressure)'
        )
        return None

    shard = _shard_for(camera_id)
    queue_name = f"usecase_shard_{shard}"
    try:
        handle = evaluate_frame_task.apply_async(
            kwargs={
                "camera_id": camera_id,
                "detection_output": detection_output,
                "usecases": usecases,
            },
            queue=queue_name,
        )
    except Exception:
        # Enqueue failed after acquiring the guard — release it now so the
        # camera isn't wedged until the TTL expires. (The worker releases it
        # on the normal path; here the task never reached a worker.)
        release_inflight(camera_id)
        raise

    logger.debug(
        f'[Queue] Submitted FRAME task camera_id={camera_id} -> {queue_name}, task_id={handle.id}'
    )
    return handle


async def await_frame_result(
        handle: AsyncResult,
        camera_id: str,
        timeout: int = TAKE_RESULT_TIMEOUT,
) -> List[UsecaseResult]:
    """
    Wait for a whole-frame task's result without blocking the event loop.

    Returns the full list of UsecaseResult for the frame, or [] on
    timeout/failure (a safe "no results this cycle" — the worker still owns
    the in-flight guard until it finishes, so no newer frame is submitted in
    the meantime).
    """
    def get_with_lock():
        with _redis_read_lock:
            return handle.get(timeout=timeout, propagate=False)

    try:
        result_list = await asyncio.to_thread(get_with_lock)
    except Exception as e:
        logger.error(
            f'[Queue] Failed to collect FRAME result for camera={camera_id}, '
            f'task_id={handle.id}: {e}'
        )
        return []

    if isinstance(result_list, Exception):
        logger.error(
            f'[Queue] FRAME task for camera={camera_id} failed: {result_list}'
        )
        return []

    if not result_list:
        return []

    results = [UsecaseResult(**d) for d in result_list]
    triggered_count = sum(1 for r in results if r.triggered)
    logger.info(
        f'[Queue] Collected {len(results)} results for camera={camera_id}, '
        f'triggered={triggered_count}/{len(results)}'
    )
    return results


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

