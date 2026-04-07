"""
Redis State Helper
==================
Provides a simple, domain-agnostic get/set interface for JSON-serializable
state keyed by a string. Intended for use by any usecase rule that needs to
persist state across Celery worker processes (separate OS processes that do
not share in-process memory).

All keys are namespaced under the prefix ``usecase:state:`` to avoid
collisions with Celery result keys stored in the same Redis instance.

Usage::

    from workers.redis_state import get_state, set_state

    # Works for any domain — vehicles, stores, safety, …
    state = get_state("parking:cam_01")
    state["counter"] += 1
    set_state("parking:cam_01", state)

Connection errors are handled gracefully:
- ``get_state`` logs a warning and returns an empty dict on failure.
- ``set_state`` logs a warning and silently skips on failure so a single
  Redis hiccup never crashes a worker task.
"""
import json
import logging
import os

import redis

logger = logging.getLogger(__name__)

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
_KEY_PREFIX = "usecase:state:"

# Module-level client — created once per worker process, reused across calls.
# decode_responses=True so we receive str from Redis, not raw bytes.
_client: redis.Redis = redis.from_url(REDIS_URL, decode_responses=True)


def _full_key(key: str) -> str:
    return f"{_KEY_PREFIX}{key}"


def get_state(key: str) -> dict:
    """
    Retrieve the JSON-encoded state stored under *key*.

    Returns an empty dict if the key does not exist or if any error occurs
    (Redis connection failure, corrupt data, etc.).
    """
    try:
        raw = _client.get(_full_key(key))
        if raw is None:
            return {}
        return json.loads(raw)
    except redis.RedisError as exc:
        logger.warning("[redis_state] get_state failed for key=%s: %s", key, exc)
        return {}
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "[redis_state] Failed to deserialize state for key=%s: %s", key, exc
        )
        return {}


def set_state(key: str, value: dict, ttl_seconds: int = 43200) -> None:
    """
    Persist *value* as JSON under *key* with an expiry of *ttl_seconds*.

    Default TTL is 12 hours (43200 s). This covers EV charging sessions that
    run through a full working shift. The previous 1-hour default caused
    tracker state to expire mid-session, resetting track_ids and creating
    duplicate sessions for cars still physically parked.

    State for offline cameras is still evicted automatically after 12 hours.

    Silently skips (logs a warning) on any Redis or serialization error.
    """
    try:
        serialized = json.dumps(value)
        _client.setex(_full_key(key), ttl_seconds, serialized)
    except redis.RedisError as exc:
        logger.warning("[redis_state] set_state failed for key=%s: %s", key, exc)
    except (TypeError, ValueError) as exc:
        logger.warning(
            "[redis_state] Failed to serialize state for key=%s: %s", key, exc
        )
