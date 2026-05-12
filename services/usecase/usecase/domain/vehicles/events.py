"""
Vehicle domain event queues and helpers.

Queues:
    parking_events   — parking_intime, parking_outtime
    violation_events — unauthorized_parking, wrong_parking, non_ev_parking, multiple_cars_in_roi
    gun_events       — gun_plugin, gun_plugout

Adding a new vehicle event type: just publish to the relevant queue with
the appropriate event_type string. No schema changes required.

Idempotency:
    publish_sync() accepts an optional task_id. When provided, it records a
    Redis key ``usecase:event_published:<task_id>:<event_type>:<slot_id>``
    with a short TTL. On task retry, duplicate events are silently dropped.
"""
import asyncio
import logging
import os
from typing import Dict, List, Optional

import redis as _redis

logger = logging.getLogger(__name__)

QUEUES: Dict[str, asyncio.Queue] = {
    "parking_events": asyncio.Queue(),
    "violation_events": asyncio.Queue(),
    "gun_events": asyncio.Queue(),
}

_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_idempotency_client: _redis.Redis = _redis.from_url(_REDIS_URL, decode_responses=True)
# TTL covers the full retry window: 3 retries × 90s hard limit = 270s, rounded up.
_IDEMPOTENCY_TTL = 300  # 5 minutes


def _idempotency_key(task_id: str, event_type: str, slot_id: str) -> str:
    return f"usecase:event_published:{task_id}:{event_type}:{slot_id}"


def _already_published(task_id: Optional[str], event_type: str, slot_id: str) -> bool:
    """Return True if this task already published this event (i.e. we're in a retry)."""
    if not task_id:
        return False
    try:
        key = _idempotency_key(task_id, event_type, slot_id)
        return bool(_idempotency_client.exists(key))
    except Exception:
        # On Redis error, allow the publish — safer to risk a duplicate than to drop
        return False


def _mark_published(task_id: Optional[str], event_type: str, slot_id: str) -> None:
    """Record that this task published this event so retries can skip it."""
    if not task_id:
        return
    try:
        key = _idempotency_key(task_id, event_type, slot_id)
        _idempotency_client.setex(key, _IDEMPOTENCY_TTL, "1")
    except Exception:
        pass  # non-critical — worst case a retry publishes a duplicate


def build_event(
    event_type: str,
    camera_id: str,
    timestamp: str,
    track_id: str,
    metadata: dict,
) -> dict:
    """Return a standard event dict."""
    return {
        "event_type": event_type,
        "camera_id": camera_id,
        "timestamp": timestamp,
        "track_id": track_id,
        "metadata": metadata,
    }


async def publish(queue_name: str, event: dict) -> None:
    """Put *event* on the named queue. Logs warning and drops if queue unknown."""
    if queue_name not in QUEUES:
        logger.warning("[VEHICLE EVENTS] Unknown queue '%s' — event dropped: %s", queue_name, event)
        return
    await QUEUES[queue_name].put(event)
    logger.debug("[VEHICLE EVENTS] Published to '%s': %s", queue_name, event.get("event_type"))


def publish_sync(queue_name: str, event: dict, task_id: Optional[str] = None) -> None:
    """
    Fire-and-forget publish from a synchronous context (inside evaluate()).
    Schedules on the running FastAPI event loop. Falls back to
    loop.run_until_complete() when no loop is running (unit tests).

    task_id: Celery task ID injected by the worker. When provided, duplicate
    publishes caused by task retries are silently dropped via a Redis
    idempotency check keyed on (task_id, event_type, slot_id).
    """
    event_type = event.get("event_type", "")
    slot_id = event.get("metadata", {}).get("slot_id") or event.get("track_id", "")

    if _already_published(task_id, event_type, slot_id):
        logger.warning(
            "[VEHICLE EVENTS] Idempotency: skipping duplicate '%s' slot=%s task_id=%s",
            event_type, slot_id, task_id,
        )
        return

    _mark_published(task_id, event_type, slot_id)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(publish(queue_name, event))
        else:
            loop.run_until_complete(publish(queue_name, event))
    except Exception as exc:
        logger.error("[VEHICLE EVENTS] Failed to publish to '%s': %s", queue_name, exc)


async def drain_queue(queue_name: str, max_items: int = 100) -> List[dict]:
    """Non-blocking drain — returns up to *max_items* events from the queue."""
    if queue_name not in QUEUES:
        logger.warning("[VEHICLE EVENTS] Unknown queue '%s'", queue_name)
        return []
    q = QUEUES[queue_name]
    items: List[dict] = []
    while not q.empty() and len(items) < max_items:
        try:
            items.append(q.get_nowait())
        except asyncio.QueueEmpty:
            break
    return items


def get_queue_sizes() -> Dict[str, int]:
    return {name: q.qsize() for name, q in QUEUES.items()}
