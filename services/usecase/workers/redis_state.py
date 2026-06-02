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


# ---------------------------------------------------------------------------
# Slot state — one Redis key per (camera_id, slot_id)
# Key format: slot:{camera_id}:{slot_id}
# ---------------------------------------------------------------------------

def _default_slot_state() -> dict:
    """
    Canonical default for a slot state object.
    Every field matches the locked design exactly.
    """
    return {
        # Parking
        "occupied": False,
        "in_time": None,          # ISO str, set once on parking_intime
        "track_id": None,         # informational only
        "car_absent_since": None, # ISO str, first frame the car went missing; None while present

        # Vehicle extraction
        "extracted": False,                # True once consensus reached OR budget spent
        "extraction_attempts": 0,          # reads spent so far, 0.._MAX_READS
        "extraction_backoff_until": 0,     # frame_counter value — do not retry before this
        "plate_reads": [],                 # ballot: every readable plate across reads
        "model_reads": [],                 # ballot: every readable model across reads
        "is_ev_reads": [],                 # ballot: every concrete ev/non_ev verdict
        "vehicle_type_reads": [],          # ballot: every concrete two_wheeler/four_wheeler verdict
        "car_number": None,                # running per-character consensus of plate_reads
        "car_model": None,                 # running plurality consensus of model_reads
        "is_ev": None,            # "ev" | "non_ev" | "unknown" — consensus EV verdict
        "vehicle_type": None,     # "two_wheeler" | "four_wheeler" | "unknown" — consensus type
        "parking_quality": None,  # "proper" | "across_line" | ... — latest LLM parking verdict

        # Gun
        "gun_present_frames": 0,  # consecutive frames with gun in slot
        "gun_absent_frames": 0,   # consecutive frames without gun (post-plugin only)
        "plugin_logged": False,
        "plugout_logged": False,
        "plug_time": None,        # ISO str, set once on gun_plugin
        "plug_out_time": None,    # ISO str, set once on gun_plugout
        "gun_name": None,

        # Inferred plug-in: when the rule synthesizes gun_plugin 2 min after
        # parking_intime without an actual gun detection. We still expect to
        # see the gun afterwards; if we never do, the inference is rolled back.
        "plug_time_inferred": False,
        "gun_seen_after_plugin": False,   # ever observed a real gun frame since plug
        "last_gun_seen_at": None,         # ISO ts of most recent real gun frame

        # Plug-out debounce state machine. The naive "gun absent N frames"
        # check fires too early when a person walks in front of the gun.
        # Instead: enter MAYBE_OUT after sustained absence; if the gun
        # reappears we exit MAYBE_OUT entirely (occlusion was a false alarm).
        # Only commit gun_plugout once the gun stays absent past both the
        # grace and confirmation windows without returning.
        "gun_maybe_out_since": None,        # ISO ts when MAYBE_OUT entered

        # Provisional plug-out (YOLO backend). When the absence debounce
        # completes we do NOT emit gun_plugout immediately — we mark it
        # PENDING. If the gun reappears on the same car (slot never reset, so
        # no new car arrived) before the finalize window elapses, the plug-out
        # is cancelled silently: it was a person occluding the gun, not a real
        # unplug. Only after the finalize window without a return do we commit
        # and emit gun_plugout. Nothing is published while pending, so an
        # occlusion never produces a (first-write-wins) wrong plug_out_time.
        "plugout_pending_since": None,      # ISO ts when plug-out became provisional
        "plugout_pending_time": None,       # candidate plug_out_time to emit on commit

        # Same idea for the car: don't fire parking_outtime on a single
        # missing-frame stretch. Wait for sustained absence past both
        # the grace and confirmation windows so a tracker hiccup doesn't
        # fragment one visit into multiple ChargingSession rows.
        "car_maybe_gone_since": None,        # ISO ts when MAYBE_GONE entered

        # LLM-driven gun detection (GUN_DETECTION_BACKEND=llm). We poll the
        # LLM at a tight cadence while hunting plug-in (1 min, 2-of-N
        # confirmation) and a loose cadence while watching plug-out
        # (5 min, one-shot). All three fields are wall-clock based so they
        # are robust to orchestration poll_interval changes.
        "last_gun_check_at": None,                 # ISO ts of most recent LLM call for this slot
        "gun_check_attempts": 0,                   # plug-in attempts; capped by GUN_LLM_PLUGIN_MAX_POLLS
        "gun_consecutive_pluggedin_count": 0,      # 2-of-N counter for plug-in confirmation
        "gun_consecutive_notpluggedin_count": 0,   # 2-of-N counter for plug-out confirmation

        # Monotonically increasing frame counter — survives restarts via Redis
        "frame_counter": 0,
    }


def get_slot_state(camera_id: str, slot_id: str) -> dict:
    """
    Return the slot state for (camera_id, slot_id).
    Merges any missing fields from _default_slot_state() so callers always
    receive a complete object even after schema additions.
    """
    key = f"slot:{camera_id}:{slot_id}"
    stored = get_state(key)
    if not stored:
        return _default_slot_state()
    # Forward-compatible: fill in any fields added after the key was first written
    defaults = _default_slot_state()
    for field, default_value in defaults.items():
        stored.setdefault(field, default_value)
    return stored


def set_slot_state(camera_id: str, slot_id: str, slot: dict, ttl_seconds: int = 43200) -> None:
    """
    Persist the slot state for (camera_id, slot_id).
    TTL matches the standard 12-hour session window.
    """
    key = f"slot:{camera_id}:{slot_id}"
    set_state(key, slot, ttl_seconds=ttl_seconds)


def reset_slot_state(camera_id: str, slot_id: str) -> None:
    """
    Reset a slot to default state (called on parking_outtime / confirmed car exit).
    Preserves frame_counter so backoff arithmetic stays monotonic across car lifecycles.
    """
    key = f"slot:{camera_id}:{slot_id}"
    current = get_state(key)
    fresh = _default_slot_state()
    fresh["frame_counter"] = current.get("frame_counter", 0)
    set_state(key, fresh)
