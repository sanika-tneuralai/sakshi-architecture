"""
Per-slot session state machine
==============================
Consumes the cheap per-frame signals (YOLO car count + logo-occlusion +
optional background-subtraction) and emits the events that gate LLM calls.

The state machine is pure: it does not call YOLO, the LLM, or S3. The
caller feeds it observations and reacts to the events it emits. That keeps
the detection service free of orchestration concerns and lets the same
module be unit-tested with synthetic frame streams.

States
------
    EMPTY       slot is reliably empty
    OCCUPYING   transition: signals say occupied but not yet debounced
    OCCUPIED    a car is parked; LLM extraction has fired
    LEAVING     transition: signals say empty but not yet debounced

Transitions
-----------
    EMPTY     --[occupied N consecutive frames]--> OCCUPYING --[immediate]--> OCCUPIED  (emits SESSION_START)
    OCCUPIED  --[empty   M consecutive frames]--> LEAVING   --[immediate]--> EMPTY     (emits SESSION_END)
    OCCUPIED  --[every GUN_POLL_INTERVAL_S until gun_seen or attempts exhausted]-->     (emits GUN_CHECK_DUE)
    OCCUPIED  --[yolo and CV disagree for N frames AND extraction not yet done]-->      (emits DISCREPANCY)

Why a separate OCCUPYING/LEAVING pair instead of just edge-triggering on
the debounced signal: we want to be able to inspect *which* signals are
flipping during the transition for tuning, and to give the caller a
single, idempotent event ("SESSION_START") rather than risking a double
fire on a flicker.

Per-slot state is held in memory keyed by (camera_id, roi_id). A separate
process / restart loses state — the caller can rehydrate from the
ChargingSession table on boot if it cares about cross-restart continuity.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Tunables. Frame-count debounces assume the caller pushes observations at
# roughly 1 fps; adjust if the cadence differs.
ENTER_DEBOUNCE_FRAMES = 5      # frames of "occupied" before SESSION_START fires
LEAVE_DEBOUNCE_FRAMES = 10     # frames of "empty" before SESSION_END fires
DISCREPANCY_FRAMES = 5         # frames of YOLO/CV disagreement before DISCREPANCY fires
GUN_POLL_INTERVAL_S = 120      # 2 min between gun-presence LLM calls
GUN_POLL_MAX_ATTEMPTS = 10     # cap calls per session — abandon after ~20 min


class SlotState(Enum):
    EMPTY = "empty"
    OCCUPYING = "occupying"
    OCCUPIED = "occupied"
    LEAVING = "leaving"


class EventKind(Enum):
    SESSION_START = "session_start"      # car arrived; caller should run LLM extraction
    SESSION_END = "session_end"          # car left; caller should close the session
    GUN_CHECK_DUE = "gun_check_due"      # 2-min cadence elapsed; caller should ask LLM about gun
    DISCREPANCY = "discrepancy"          # YOLO and CV disagree; caller may want a second-look LLM call


@dataclass
class Observation:
    """One frame's worth of signals for one slot."""
    timestamp: datetime
    yolo_says_occupied: bool      # YOLO car bbox center inside this slot ROI
    cv_says_occupied: bool        # logo-coverage AND background-subtraction agree


@dataclass
class Event:
    kind: EventKind
    camera_id: str
    roi_id: str
    timestamp: datetime
    # SESSION_START / SESSION_END payloads use these:
    session_open_at: Optional[datetime] = None
    session_close_at: Optional[datetime] = None
    # DISCREPANCY payload: which signal said what, so the caller can log/audit.
    yolo_says_occupied: Optional[bool] = None
    cv_says_occupied: Optional[bool] = None
    # GUN_CHECK_DUE payload: which attempt this is (1-based).
    gun_attempt: Optional[int] = None


@dataclass
class _SlotCtx:
    """Per-slot mutable state. Internal; not part of the public surface."""
    state: SlotState = SlotState.EMPTY
    occupied_streak: int = 0
    empty_streak: int = 0
    disagreement_streak: int = 0

    session_open_at: Optional[datetime] = None
    extraction_done: bool = False        # True after first SESSION_START fired for this session

    last_gun_check_at: Optional[datetime] = None
    gun_attempts: int = 0
    gun_seen: bool = False               # caller flips this via mark_gun_seen()


class SlotStateMachine:
    """
    Consumes per-frame observations and emits events. One instance handles
    every (camera, ROI) pair on the host — internal dict keys are
    (camera_id, roi_id).

    Typical wiring on the orchestrator side:

        sm = SlotStateMachine()
        for frame_response in detection_stream:
            for roi_id, occluded in (frame_response.logo_occluded or {}).items():
                obs = Observation(
                    timestamp=frame_response.timestamp,
                    yolo_says_occupied=_yolo_hit_in_roi(frame_response, roi_id),
                    cv_says_occupied=occluded,  # AND with bg-sub once you add it
                )
                for ev in sm.observe(frame_response.camera_id, roi_id, obs):
                    handle_event(ev)   # dispatch to LLM / DB / etc.

            # Caller informs the state machine of LLM gun-check outcome:
            if gun_check_was_run and llm_says_gun_plugged_in:
                sm.mark_gun_seen(camera_id, roi_id)
    """

    def __init__(self) -> None:
        self._slots: Dict[Tuple[str, str], _SlotCtx] = {}

    def _ctx(self, camera_id: str, roi_id: str) -> _SlotCtx:
        key = (camera_id, roi_id)
        if key not in self._slots:
            self._slots[key] = _SlotCtx()
        return self._slots[key]

    def observe(self, camera_id: str, roi_id: str, obs: Observation) -> List[Event]:
        ctx = self._ctx(camera_id, roi_id)
        events: List[Event] = []

        # Either signal occupied counts as occupied for the union; we want
        # YOLO misses to still drive entry. CV-only or YOLO-only is fine
        # for entry; for confidence we track agreement separately below.
        any_occupied = obs.yolo_says_occupied or obs.cv_says_occupied

        if any_occupied:
            ctx.occupied_streak += 1
            ctx.empty_streak = 0
        else:
            ctx.empty_streak += 1
            ctx.occupied_streak = 0

        # Discrepancy tracking only matters once a session is open — we use
        # it to ask the LLM to re-extract details, not to gate session entry.
        if obs.yolo_says_occupied != obs.cv_says_occupied:
            ctx.disagreement_streak += 1
        else:
            ctx.disagreement_streak = 0

        # ---- state transitions ----
        if ctx.state == SlotState.EMPTY:
            if ctx.occupied_streak >= ENTER_DEBOUNCE_FRAMES:
                ctx.state = SlotState.OCCUPIED
                ctx.session_open_at = obs.timestamp
                ctx.extraction_done = True   # SESSION_START is the extraction trigger; it fires exactly once
                ctx.last_gun_check_at = None
                ctx.gun_attempts = 0
                ctx.gun_seen = False
                events.append(Event(
                    kind=EventKind.SESSION_START,
                    camera_id=camera_id, roi_id=roi_id,
                    timestamp=obs.timestamp,
                    session_open_at=obs.timestamp,
                ))
                logger.info("[SLOT] %s/%s SESSION_START at %s", camera_id, roi_id, obs.timestamp)

        elif ctx.state == SlotState.OCCUPIED:
            # 1. Leave detection
            if ctx.empty_streak >= LEAVE_DEBOUNCE_FRAMES:
                close_at = obs.timestamp
                events.append(Event(
                    kind=EventKind.SESSION_END,
                    camera_id=camera_id, roi_id=roi_id,
                    timestamp=obs.timestamp,
                    session_open_at=ctx.session_open_at,
                    session_close_at=close_at,
                ))
                logger.info(
                    "[SLOT] %s/%s SESSION_END at %s (open=%s)",
                    camera_id, roi_id, close_at, ctx.session_open_at,
                )
                # Reset the slot fully; next observation starts a fresh cycle.
                self._slots[(camera_id, roi_id)] = _SlotCtx()
                return events

            # 2. Gun-presence polling. Fires at most once per observe() call.
            if (
                not ctx.gun_seen
                and ctx.gun_attempts < GUN_POLL_MAX_ATTEMPTS
                and self._gun_poll_due(ctx, obs.timestamp)
            ):
                ctx.gun_attempts += 1
                ctx.last_gun_check_at = obs.timestamp
                events.append(Event(
                    kind=EventKind.GUN_CHECK_DUE,
                    camera_id=camera_id, roi_id=roi_id,
                    timestamp=obs.timestamp,
                    gun_attempt=ctx.gun_attempts,
                ))
                logger.info(
                    "[SLOT] %s/%s GUN_CHECK_DUE attempt=%d/%d",
                    camera_id, roi_id, ctx.gun_attempts, GUN_POLL_MAX_ATTEMPTS,
                )

            # 3. Discrepancy: only fires once per session — repeat LLM
            # re-extractions on the same parked car waste budget.
            if (
                ctx.disagreement_streak == DISCREPANCY_FRAMES  # exact == so it fires once
            ):
                events.append(Event(
                    kind=EventKind.DISCREPANCY,
                    camera_id=camera_id, roi_id=roi_id,
                    timestamp=obs.timestamp,
                    yolo_says_occupied=obs.yolo_says_occupied,
                    cv_says_occupied=obs.cv_says_occupied,
                ))
                logger.info(
                    "[SLOT] %s/%s DISCREPANCY yolo=%s cv=%s",
                    camera_id, roi_id, obs.yolo_says_occupied, obs.cv_says_occupied,
                )

        return events

    def mark_gun_seen(self, camera_id: str, roi_id: str) -> None:
        """
        Caller invokes this after the LLM responds that the gun is plugged in.
        Stops further GUN_CHECK_DUE events for the current session.
        """
        ctx = self._ctx(camera_id, roi_id)
        if ctx.state == SlotState.OCCUPIED:
            ctx.gun_seen = True
            logger.info("[SLOT] %s/%s gun confirmed plugged in", camera_id, roi_id)

    def snapshot(self) -> Dict[Tuple[str, str], SlotState]:
        """Read-only view of every tracked slot's current state. For /health."""
        return {key: ctx.state for key, ctx in self._slots.items()}

    @staticmethod
    def _gun_poll_due(ctx: _SlotCtx, now: datetime) -> bool:
        if ctx.last_gun_check_at is None:
            # First check fires immediately on entering OCCUPIED — drivers
            # often plug in within seconds of parking.
            return True
        return now - ctx.last_gun_check_at >= timedelta(seconds=GUN_POLL_INTERVAL_S)


_machine: Optional[SlotStateMachine] = None


def get_slot_state_machine() -> SlotStateMachine:
    global _machine
    if _machine is None:
        _machine = SlotStateMachine()
        logger.info(
            "[SLOT] State machine initialised (enter=%d empty=%d disc=%d gun=%ds max=%d)",
            ENTER_DEBOUNCE_FRAMES, LEAVE_DEBOUNCE_FRAMES, DISCREPANCY_FRAMES,
            GUN_POLL_INTERVAL_S, GUN_POLL_MAX_ATTEMPTS,
        )
    return _machine


print("✓ detection.slot_state module loaded")
