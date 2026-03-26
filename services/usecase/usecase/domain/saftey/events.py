"""
Safety domain event queues and helpers.

Queues:
    safety_events — safety_alert (fire, smoke)

Adding a new safety event type: just publish to safety_events with
the appropriate event_type string.
"""
import asyncio
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

QUEUES: Dict[str, asyncio.Queue] = {
    "safety_events": asyncio.Queue(),
}


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
        logger.warning("[SAFETY EVENTS] Unknown queue '%s' — event dropped: %s", queue_name, event)
        return
    await QUEUES[queue_name].put(event)
    logger.debug("[SAFETY EVENTS] Published to '%s': %s", queue_name, event.get("event_type"))


def publish_sync(queue_name: str, event: dict) -> None:
    """
    Fire-and-forget publish from a synchronous context (inside evaluate()).
    Schedules on the running FastAPI event loop. Falls back to
    loop.run_until_complete() when no loop is running (unit tests).
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(publish(queue_name, event))
        else:
            loop.run_until_complete(publish(queue_name, event))
    except Exception as exc:
        logger.error("[SAFETY EVENTS] Failed to publish to '%s': %s", queue_name, exc)


async def drain_queue(queue_name: str, max_items: int = 100) -> List[dict]:
    """Non-blocking drain — returns up to *max_items* events from the queue."""
    if queue_name not in QUEUES:
        logger.warning("[SAFETY EVENTS] Unknown queue '%s'", queue_name)
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
