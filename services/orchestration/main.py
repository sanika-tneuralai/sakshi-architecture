"""
Orchestration Service - Async Pipeline Controller

High-performance async orchestration for video surveillance pipelines.

SCALING ARCHITECTURE:
- 1 stream: Single asyncio task, minimal overhead (~10MB RAM)
- 10 streams: 10 concurrent tasks, semaphores prevent downstream service overload
- 50-100 streams: Semaphore limits (CAMERA_DETECTION_CONCURRENCY, etc.) become the 
  primary tuning knob. No code changes needed, just adjust environment variables.
- 100+ streams: Only env var tuning required (CONCURRENCY limits, poll_interval per camera)
- Cross-machine scaling (500+ streams): Replace asyncio.Queue with Redis/RabbitMQ 
  task queue as a drop-in replacement. The producer-consumer interface stays identical.

ARCHITECTURE PATTERN: Producer → Consumer Pipeline
Each camera runs as an independent asyncio task:
  frame_producer → detection_consumer/producer → usecase_consumer/producer → alert_consumer

All stages run concurrently across all cameras using asyncio.gather().
Each stage is rate-limited with asyncio.Semaphore to prevent overwhelming downstream services.

This service does NOT contain camera/detection/usecase logic itself.
"""

import sys
import os
from dotenv import load_dotenv

# Load .env before any os.getenv() calls
load_dotenv()

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging
from logging.handlers import RotatingFileHandler
import uvicorn
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
from shared.database.persistence import get_camera_rois, get_camera_usecases, get_class_thresholds, upsert_charging_session, persist_alerts_from_results, close_stale_sessions
from shared.database.connection import SessionLocal

# ---------------------------------------------------------------------------
# Logging
#
# Three sinks on the root logger so every getLogger(__name__) inherits:
#   - stdout                        INFO+   operational (systemd / docker logs)
#   - logs/orchestration.log        INFO+   bounded operational history
#   - logs/orchestration_debug.log  DEBUG+  full pipeline trace for post-mortem
#
# Format includes file:line so each line points at the call site (e.g.
# main.py:1390 vs persistence.py:412) — fastest path to "where it broke."
#
# RotatingFileHandler caps each file at 10 MB × 5 backups (~50 MB per file,
# ~100 MB total) so a long-running container can't blow up the disk.
# ---------------------------------------------------------------------------
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(filename)s:%(lineno)d - %(message)s"
_LOG_FORMATTER = logging.Formatter(_LOG_FORMAT)

_root = logging.getLogger()
_root.setLevel(logging.DEBUG)
for _h in list(_root.handlers):
    _root.removeHandler(_h)

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(_LOG_FORMATTER)
_root.addHandler(_console)

_info_file = RotatingFileHandler(
    LOG_DIR / "orchestration.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_info_file.setLevel(logging.INFO)
_info_file.setFormatter(_LOG_FORMATTER)
_root.addHandler(_info_file)

_debug_file = RotatingFileHandler(
    LOG_DIR / "orchestration_debug.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_debug_file.setLevel(logging.DEBUG)
_debug_file.setFormatter(_LOG_FORMATTER)
_root.addHandler(_debug_file)

# Quiet noisy third-party libs so the debug file stays readable.
for _noisy in ("urllib3", "httpx", "httpcore", "PIL", "matplotlib", "asyncio"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info("Logging configured: dir=%s (info+debug, rotating)", LOG_DIR.resolve())

# =============================================================================
# CONFIGURATION FROM ENVIRONMENT VARIABLES
# =============================================================================

# Service URLs

CAMERA_DETECTION_URL = os.getenv("CAMERA_DETECTION_URL", "http://100.123.244.59:8004")
USECASE_SERVICE_URL = os.getenv("USECASE_SERVICE_URL", "http://100.112.71.40:8001")
ALERT_SERVICE_URL = os.getenv("ALERT_SERVICE_URL", "http://100.112.71.40:8002")
ANALYTICS_SERVICE_URL = os.getenv("ANALYTICS_SERVICE_URL", "http://100.112.71.40:8003")

# Request timeouts. Detection is the slowest hop (Pi inference over Tailscale)
# but we want to fail fast on real hangs — 30s previously meant a single bad
# request stalled the camera for 30s. Per-stage timeouts make slow detections
# visible without freezing the loop.
REQUEST_TIMEOUT          = float(os.getenv("REQUEST_TIMEOUT", "30.0"))     # legacy default
DETECTION_TIMEOUT        = float(os.getenv("DETECTION_TIMEOUT", "8.0"))    # Pi inference call
USECASE_TIMEOUT          = float(os.getenv("USECASE_TIMEOUT", "5.0"))      # local-network call
ALERT_TIMEOUT            = float(os.getenv("ALERT_TIMEOUT", "5.0"))        # local-network call
RETRY_ATTEMPTS           = int(os.getenv("RETRY_ATTEMPTS", "1"))           # was 3 — compound delays

# Static-config cache TTL. ROIs, usecases, and per-class thresholds change only
# when an admin updates them; refetching every iteration was 3 DB roundtrips of
# wasted work. 60s is the staleness window before a config edit takes effect.
CONFIG_CACHE_TTL_SECONDS = float(os.getenv("CONFIG_CACHE_TTL_SECONDS", "60"))

# Concurrency limits per service (semaphore limits)
CAMERA_DETECTION_CONCURRENCY = int(os.getenv("CAMERA_DETECTION_CONCURRENCY", "10"))
USECASE_CONCURRENCY = int(os.getenv("USECASE_CONCURRENCY", "20"))
ALERT_CONCURRENCY = int(os.getenv("ALERT_CONCURRENCY", "30"))

# Default poll interval between pipeline iterations
DEFAULT_POLL_INTERVAL = float(os.getenv("DEFAULT_POLL_INTERVAL", "1.0"))

# httpx client configuration
MAX_CONNECTIONS = int(os.getenv("MAX_CONNECTIONS", "100"))
MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("MAX_KEEPALIVE_CONNECTIONS", "50"))

# Global shared httpx client and semaphores (initialized in lifespan)
http_client: Optional[httpx.AsyncClient] = None
camera_detection_semaphore: Optional[asyncio.Semaphore] = None
usecase_semaphore: Optional[asyncio.Semaphore] = None
alert_semaphore: Optional[asyncio.Semaphore] = None


# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class CameraStats:
    """Per-camera runtime statistics"""
    camera_id: str
    running: bool = True
    iterations: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    last_run: Optional[datetime] = None
    last_error: Optional[str] = None
    latencies: List[float] = field(default_factory=list)  # Keep last 100
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency in milliseconds"""
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)
    
    def add_latency(self, latency_ms: float):
        """Add latency sample, keep last 100"""
        self.latencies.append(latency_ms)
        if len(self.latencies) > 100:
            self.latencies.pop(0)


class CameraConfig(BaseModel):
    """Configuration for a single camera pipeline"""
    camera_id: str = Field(..., description="Unique camera identifier")
    usecases: List[str] = Field(
        default=["person_in_roi", "crowd_in_roi", "restricted_zone_breach"],
        description="List of usecases to evaluate"
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold"
    )
    poll_interval: float = Field(
        default=DEFAULT_POLL_INTERVAL,
        gt=0.0,
        description="Seconds between pipeline iterations"
    )
    max_errors_before_pause: int = Field(
        default=5,
        ge=1,
        description="Pause camera after N consecutive errors"
    )


class BatchStartRequest(BaseModel):
    """Request to start multiple camera pipelines"""
    cameras: List[CameraConfig] = Field(..., description="List of camera configurations")


class PipelineStatus(BaseModel):
    """Status response for a single camera pipeline"""
    camera_id: str
    running: bool
    iterations: int
    errors: int
    last_run: Optional[datetime]
    avg_latency_ms: float
    last_error: Optional[str]


class PipelineRequest(BaseModel):
    """Legacy single-shot pipeline execution request"""
    camera_id: Optional[str] = Field(None, description="Camera identifier (optional - if not provided, processes all active cameras)")
    usecases: Optional[List[str]] = Field(
        None,
        description="List of usecases to evaluate. If None, uses defaults."
    )
    confidence_threshold: Optional[float] = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Detection confidence threshold"
    )


# =============================================================================
# ASYNC HTTP CLIENT WITH RETRY LOGIC
# =============================================================================

def create_retry_decorator():
    """Create retry decorator for HTTP calls.

    RETRY_ATTEMPTS=1 by default (no retries on top of the initial try): in a
    real-time detection loop, retrying a slow hop just compounds latency —
    skipping a frame is preferable to stalling the whole camera. Bump via
    env var if/when a hop is genuinely flaky.
    """
    return retry(
        stop=stop_after_attempt(RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=0.5, min=1, max=10),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.TimeoutException)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )


retry_on_failure = create_retry_decorator()


# =============================================================================
# STATIC CONFIG CACHE
# =============================================================================
# ROIs, usecases, and per-class thresholds change rarely (admin edits only).
# Refetching all three from Postgres on every iteration was 3 sequential DB
# roundtrips of pure waste. Cache per camera with a short TTL so a config edit
# still propagates promptly.
_config_cache: Dict[str, Dict[str, Any]] = {}


def _get_camera_config_cached(camera_id: str) -> Dict[str, Any]:
    """Return {rois, usecases, class_thresholds} for camera_id, cached for TTL.

    Single-threaded by virtue of the asyncio loop — no lock needed for the
    in-process dict. The DB calls themselves still run in the executor (they're
    blocking), but only when the cache is cold or stale.
    """
    now = datetime.now().timestamp()
    entry = _config_cache.get(camera_id)
    if entry and (now - entry["fetched_at"]) < CONFIG_CACHE_TTL_SECONDS:
        return entry

    rois             = get_camera_rois(camera_id)
    db_usecases      = get_camera_usecases(camera_id)
    class_thresholds = get_class_thresholds(camera_id)
    entry = {
        "rois":             rois,
        "usecases":         db_usecases,
        "class_thresholds": class_thresholds,
        "fetched_at":       now,
    }
    _config_cache[camera_id] = entry
    return entry


@retry_on_failure
async def fetch_frame(camera_id: str) -> dict:
    """Fetch frame from camera-detection service with retry"""
    async with camera_detection_semaphore:
        response = await http_client.get(
            f"{CAMERA_DETECTION_URL}/camera/frame/{camera_id}",
            timeout=DETECTION_TIMEOUT
        )
        response.raise_for_status()
        return response.json()


@retry_on_failure
async def run_detection(camera_id: str, confidence_threshold: float, class_thresholds: Optional[Dict[str, float]] = None) -> dict:
    """Run detection on frame with retry"""
    async with camera_detection_semaphore:
        payload: Dict[str, Any] = {
            "camera_id": camera_id,
            "confidence_threshold": confidence_threshold,
        }
        if class_thresholds:
            payload["class_thresholds"] = class_thresholds
        response = await http_client.post(
            f"{CAMERA_DETECTION_URL}/detection/detect",
            json=payload,
            timeout=DETECTION_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        # Full body only to debug file; INFO summary is logged by the loop.
        logger.debug(f"[{camera_id}] detection response: {data}")
        return data


@retry_on_failure
async def evaluate_usecases(
    camera_id: str,
    detection_data: dict,
    usecases: List[str],
    rois: List[dict],
) -> dict:
    """Evaluate usecases with retry.

    Merges ROI definitions (fetched from DB by the caller) into detection_output
    so stateful rules (parking_detection, restricted_area, people_counter, etc.)
    have the geometry they need.
    """
    async with usecase_semaphore:
        detection_output = dict(detection_data)
        # Convert DB list-of-dicts → {"ROI_1": [[x,y], ...], "ROI_2": ...}
        # Rules (parking_detection, gun_detection, etc.) expect this dict format.
        detection_output["rois"] = {
            r["roi_id"]: r["points"] for r in rois
        }
        # Request payload (full ROI polygons + detections) is large — debug file only.
        logger.debug(
            f"[{camera_id}] usecase request: usecases={usecases} | "
            f"rois={detection_output['rois']} | "
            f"detections={detection_output.get('detections', [])}"
        )
        response = await http_client.post(
            f"{USECASE_SERVICE_URL}/usecase/evaluate",
            json={
                "camera_id": camera_id,
                "detection_output": detection_output,
                "usecases": usecases,
            },
            timeout=USECASE_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        logger.debug(f"[{camera_id}] usecase response: {data}")
        return data


@retry_on_failure
async def send_alerts(camera_id: str, usecase_results: List[dict]) -> dict:
    """Send alerts with retry (failures are non-critical)"""
    async with alert_semaphore:
        response = await http_client.post(
            f"{ALERT_SERVICE_URL}/alert/send",
            json={
                "camera_id": camera_id,
                "usecase_results": usecase_results
            },
            timeout=ALERT_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()
        logger.debug(f"[{camera_id}] alert response: {data}")
        return data


# =============================================================================
# PIPELINE MANAGER
# =============================================================================

class PipelineManager:
    """
    Manages async pipeline tasks for multiple cameras.
    
    Each camera runs as an independent asyncio task with graceful error handling.
    Failed cameras do not affect other cameras.
    """
    
    def __init__(self):
        self.pipelines: Dict[str, asyncio.Task] = {}
        self.stop_events: Dict[str, asyncio.Event] = {}
        self.configs: Dict[str, CameraConfig] = {}
        self.stats: Dict[str, CameraStats] = {}
        self.lock = asyncio.Lock()
    
    async def start_pipeline(self, config: CameraConfig) -> dict:
        """
        Start async pipeline for a camera.
        
        Returns:
            dict with status: 'started' or 'already_running'
        """
        async with self.lock:
            if config.camera_id in self.pipelines and not self.pipelines[config.camera_id].done():
                logger.warning(f"[{config.camera_id}] Pipeline already running")
                return {
                    "status": "already_running",
                    "camera_id": config.camera_id,
                    "message": f"Pipeline already active for {config.camera_id}"
                }
            
            # Create stop event and stats
            stop_event = asyncio.Event()
            self.stop_events[config.camera_id] = stop_event
            self.configs[config.camera_id] = config
            self.stats[config.camera_id] = CameraStats(camera_id=config.camera_id)
            
            # Start pipeline task
            task = asyncio.create_task(
                self._run_camera_pipeline(config, stop_event),
                name=f"pipeline-{config.camera_id}"
            )
            self.pipelines[config.camera_id] = task
            
            logger.info(f"[{config.camera_id}] Started continuous pipeline | usecases={config.usecases} | confidence={config.confidence_threshold}")
            
            return {
                "status": "started",
                "camera_id": config.camera_id,
                "message": f"Continuous pipeline started for {config.camera_id}",
                "configuration": config.model_dump()
            }
    
    async def stop_pipeline(self, camera_id: str) -> dict:
        """
        Stop pipeline for a specific camera.
        
        Returns:
            dict with status: 'stopped', 'stopping', or 'not_running'
        """
        async with self.lock:
            if camera_id not in self.pipelines:
                return {
                    "status": "not_running",
                    "camera_id": camera_id,
                    "message": f"No active pipeline found for {camera_id}"
                }
            
            # Signal stop
            self.stop_events[camera_id].set()
            
            # Cancel task if still running
            task = self.pipelines[camera_id]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup
            del self.pipelines[camera_id]
            del self.stop_events[camera_id]
            if camera_id in self.stats:
                self.stats[camera_id].running = False
            
            logger.info(f"[{camera_id}] Pipeline stopped")
            
            return {
                "status": "stopped",
                "camera_id": camera_id,
                "message": f"Pipeline stopped for {camera_id}"
            }
    
    async def stop_all(self) -> dict:
        """Stop all active pipelines"""
        async with self.lock:
            camera_ids = list(self.pipelines.keys())
            
            if not camera_ids:
                return {
                    "status": "no_pipelines",
                    "count": 0,
                    "message": "No active pipelines to stop"
                }
            
            # Signal all to stop
            for camera_id in camera_ids:
                self.stop_events[camera_id].set()
            
            # Cancel all tasks
            tasks = list(self.pipelines.values())
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for all to finish
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Cleanup
            self.pipelines.clear()
            self.stop_events.clear()
            for camera_id in camera_ids:
                if camera_id in self.stats:
                    self.stats[camera_id].running = False
            
            logger.info(f"Stopped all {len(camera_ids)} pipelines: {camera_ids}")
            
            return {
                "status": "stopped_all",
                "count": len(camera_ids),
                "camera_ids": camera_ids,
                "message": f"Stopped {len(camera_ids)} pipeline(s)"
            }
    
    async def get_status(self, camera_id: Optional[str] = None) -> dict:
        """Get status of pipelines"""
        async with self.lock:
            if camera_id:
                # Single camera status
                if camera_id not in self.stats:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No pipeline found for camera {camera_id}"
                    )
                
                stats = self.stats[camera_id]
                return PipelineStatus(
                    camera_id=stats.camera_id,
                    running=stats.running,
                    iterations=stats.iterations,
                    errors=stats.errors,
                    last_run=stats.last_run,
                    avg_latency_ms=stats.avg_latency_ms,
                    last_error=stats.last_error
                ).model_dump()
            else:
                # All pipelines status
                pipelines_status = []
                for camera_id, stats in self.stats.items():
                    pipelines_status.append(PipelineStatus(
                        camera_id=stats.camera_id,
                        running=stats.running,
                        iterations=stats.iterations,
                        errors=stats.errors,
                        last_run=stats.last_run,
                        avg_latency_ms=stats.avg_latency_ms,
                        last_error=stats.last_error
                    ).model_dump())
                
                return {
                    "active_pipelines": len([s for s in self.stats.values() if s.running]),
                    "total_tracked": len(self.stats),
                    "pipelines": pipelines_status
                }
    
    async def _run_camera_pipeline(self, config: CameraConfig, stop_event: asyncio.Event):
        """
        Main pipeline loop for a single camera.
        
        Runs continuously until stop_event is set or max consecutive errors reached.
        Each iteration: fetch_frame → detection → usecase → alerts
        """
        camera_id = config.camera_id
        stats = self.stats[camera_id]

        logger.info(f"[{camera_id}] Pipeline worker started | poll_interval={config.poll_interval}s")

        loop = asyncio.get_event_loop()

        while not stop_event.is_set():
            # Use loop.time() (monotonic) for pacing — datetime.now() can jump
            # if the system clock is adjusted, which would break the deadline
            # arithmetic at the bottom of the loop.
            tick_start = loop.time()
            iteration_start = datetime.now()

            try:
                stats.iterations += 1
                iteration = stats.iterations
                tag = f"[{camera_id}][iter={iteration}]"   # grep-friendly

                logger.debug(f"{tag} starting")

                # STEP 1: Static config (cached). Cold/stale path runs the 3
                # blocking DB calls in the executor; warm path is in-memory.
                t0 = loop.time()
                cfg = await loop.run_in_executor(None, _get_camera_config_cached, camera_id)
                rois             = cfg["rois"]
                db_usecases      = cfg["usecases"]
                class_thresholds = cfg["class_thresholds"]
                t_cfg = (loop.time() - t0) * 1000
                # Fall back to config usecases if none configured in DB
                active_usecases = db_usecases if db_usecases else config.usecases

                # STEP 2: Run detection with per-class thresholds when configured
                t0 = loop.time()
                detection_data = await run_detection(
                    camera_id,
                    config.confidence_threshold,
                    class_thresholds or None,
                )
                t_det = (loop.time() - t0) * 1000
                total_det = detection_data.get('total_detections_count', 0)

                # Cache snapshot URL if detections found
                if total_det > 0 and detection_data.get('snapshot_url'):
                    detection_snapshots[camera_id] = {
                        "camera_id": camera_id,
                        "snapshot_url": detection_data['snapshot_url'],
                        "timestamp": detection_data.get('timestamp'),
                        "detections": detection_data.get('detections', []),
                        "detection_count": total_det
                    }

                # STEP 3: Evaluate usecases
                t0 = loop.time()
                usecase_data = await evaluate_usecases(camera_id, detection_data, active_usecases, rois)
                t_uc = (loop.time() - t0) * 1000
                results = usecase_data.get('results', [])
                triggered = [r for r in results if r.get('triggered')]

                # On a triggered iteration we want full evidence in the INFO
                # log (debug file always has it). On a quiet iteration we keep
                # the line short so the log stays readable at high cadence.
                if triggered:
                    import json as _json
                    for r in triggered:
                        uid = r.get('usecase_id', r.get('usecase_name'))
                        logger.info(
                            f"{tag} TRIGGERED usecase={uid} | "
                            f"extras={r.get('extras')} | events={r.get('events')} | "
                            f"vehicle_details={r.get('vehicle_details')}"
                        )
                    logger.debug(f"{tag} raw results = {_json.dumps(results, default=str)}")

                # STEP 3.5: Persist charging session data
                t0 = loop.time()
                try:
                    await loop.run_in_executor(
                        None, upsert_charging_session, camera_id, results
                    )
                except Exception as e:
                    logger.error(f"{tag} charging session update failed: {e!r}", exc_info=True)
                t_sess = (loop.time() - t0) * 1000

                # STEP 3.6: Persist triggered alerts to DB (for dashboard)
                t0 = loop.time()
                alerts_written = 0
                try:
                    alerts_written = await loop.run_in_executor(
                        None, persist_alerts_from_results, camera_id, results
                    )
                except Exception as e:
                    logger.warning(f"{tag} alert persistence failed (non-critical): {e!r}")
                t_alerts = (loop.time() - t0) * 1000

                # STEP 3.7: Periodically close stale open sessions (Issue #8)
                # Every 60 iterations (~1 min at 1fps) sweep for sessions open > 4h.
                if iteration % 60 == 0:
                    try:
                        await loop.run_in_executor(
                            None, close_stale_sessions
                        )
                    except Exception as e:
                        logger.warning(f"{tag} stale session sweep failed (non-critical): {e!r}")

                # STEP 4: Send dashboard alerts for parking_compliance and safety_monitoring only
                _DASHBOARD_USECASES = {"parking_compliance", "safety_monitoring"}
                dashboard_results = [
                    r for r in results
                    if r.get("usecase_id") in _DASHBOARD_USECASES or r.get("usecase_name") in _DASHBOARD_USECASES
                ]
                t_send = 0.0
                if dashboard_results:
                    t0 = loop.time()
                    try:
                        await send_alerts(camera_id, dashboard_results)
                    except Exception as e:
                        logger.warning(f"{tag} dashboard alert send failed (non-critical): {e!r}")
                    t_send = (loop.time() - t0) * 1000

                # Update stats
                iteration_time = (datetime.now() - iteration_start).total_seconds() * 1000
                stats.add_latency(iteration_time)
                stats.last_run = datetime.now()
                stats.consecutive_errors = 0  # Reset on success

                # Single structured INFO line per iteration: per-stage timing
                # makes it instantly clear which hop is slow next time. Full
                # payloads are in the debug file at the same iter= tag.
                logger.info(
                    f"{tag} done | total={iteration_time:.0f}ms "
                    f"cfg={t_cfg:.0f} det={t_det:.0f} uc={t_uc:.0f} "
                    f"sess={t_sess:.0f} alerts={t_alerts:.0f} send={t_send:.0f} | "
                    f"detections={total_det} triggered={len(triggered)} "
                    f"alerts_written={alerts_written}"
                )

            except asyncio.CancelledError:
                logger.info(f"[{camera_id}] Pipeline cancelled")
                break

            except Exception as e:
                stats.errors += 1
                stats.consecutive_errors += 1
                stats.last_error = str(e)

                logger.error(
                    f"[{camera_id}][iter={stats.iterations}] error | "
                    f"consecutive_errors={stats.consecutive_errors} | "
                    f"error={e!r}"
                )

                # Pause camera if too many consecutive errors
                if stats.consecutive_errors >= config.max_errors_before_pause:
                    logger.error(
                        f"[{camera_id}] Too many consecutive errors ({stats.consecutive_errors}), "
                        f"pausing for 30s"
                    )
                    await asyncio.sleep(30)
                    stats.consecutive_errors = 0  # Reset after pause
                else:
                    await asyncio.sleep(5)  # Short backoff

                continue

            # Deadline-based pacing: aim for one tick every poll_interval
            # regardless of how long the work took. If work took longer than
            # the interval, fire the next tick immediately (catch-up) instead
            # of compounding the delay with another full poll_interval sleep.
            elapsed = loop.time() - tick_start
            sleep_for = config.poll_interval - elapsed
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                logger.warning(
                    f"[{camera_id}][iter={stats.iterations}] iter exceeded poll_interval "
                    f"({elapsed:.1f}s > {config.poll_interval}s) — running next tick immediately"
                )
        
        stats.running = False
        logger.info(
            f"[{camera_id}] Pipeline worker stopped | "
            f"total_iterations={stats.iterations} | "
            f"total_errors={stats.errors}"
        )


# Global pipeline manager
pipeline_manager: Optional[PipelineManager] = None

# In-memory detection snapshot store: {camera_id: {snapshot_url, timestamp, detections, detection_count}}
detection_snapshots: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: initialize shared resources on startup, cleanup on shutdown.
    """
    global http_client, camera_detection_semaphore, usecase_semaphore, alert_semaphore, pipeline_manager
    
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Async Orchestration Service")
    logger.info("=" * 60)
    logger.info(f"Camera-Detection URL: {CAMERA_DETECTION_URL}")
    logger.info(f"Usecase Service URL: {USECASE_SERVICE_URL}")
    logger.info(f"Alert Service URL: {ALERT_SERVICE_URL}")
    logger.info(f"Timeouts (s): detection={DETECTION_TIMEOUT} usecase={USECASE_TIMEOUT} alert={ALERT_TIMEOUT} (legacy={REQUEST_TIMEOUT})")
    logger.info(f"Retry attempts per HTTP call: {RETRY_ATTEMPTS}")
    logger.info(f"Config cache TTL: {CONFIG_CACHE_TTL_SECONDS}s")
    logger.info(f"Concurrency Limits:")
    logger.info(f"  Camera-Detection: {CAMERA_DETECTION_CONCURRENCY}")
    logger.info(f"  Usecase: {USECASE_CONCURRENCY}")
    logger.info(f"  Alert: {ALERT_CONCURRENCY}")
    logger.info(f"Default Poll Interval: {DEFAULT_POLL_INTERVAL}s")
    logger.info(f"Max Connections: {MAX_CONNECTIONS}")
    logger.info("=" * 60)
    
    # Initialize httpx AsyncClient.
    # Connection pooling is DISABLED here. The 15s keepalive we tried previously
    # was still long enough that pooled sockets to the Pi over Tailscale silently
    # went stale — measured RTT to /detection/health is ~150ms, but reusing a
    # dead pooled connection would stall in TCP retransmit for 30-60s before
    # the OS gave up, surfacing as `det=48000ms` in the iteration log even
    # though the network is healthy. Forcing a fresh handshake per request
    # costs ~70ms (TCP only, no TLS) and eliminates the failure mode entirely.
    # Bring keepalive back if/when we move off Tailscale or set up a proper
    # TCP keepalive policy on both sides.
    http_client = httpx.AsyncClient(
        limits=httpx.Limits(
            max_connections=MAX_CONNECTIONS,
            max_keepalive_connections=0,
            keepalive_expiry=0.0,
        ),
        # Per-call timeout= overrides this for run_detection / evaluate_usecases /
        # send_alerts. The client-level read timeout is just a safety ceiling for
        # ad-hoc requests elsewhere in main.py (e.g. /pipeline/once handlers).
        timeout=httpx.Timeout(
            connect=5.0,
            read=DETECTION_TIMEOUT,
            write=10.0,
            pool=5.0
        )
    )
    logger.info("✓ HTTP client initialized (keepalive disabled — fresh connection per request)")
    
    # Initialize semaphores for rate limiting
    camera_detection_semaphore = asyncio.Semaphore(CAMERA_DETECTION_CONCURRENCY)
    usecase_semaphore = asyncio.Semaphore(USECASE_CONCURRENCY)
    alert_semaphore = asyncio.Semaphore(ALERT_CONCURRENCY)
    logger.info("✓ Semaphores initialized for rate limiting")
    
    # Initialize pipeline manager
    pipeline_manager = PipelineManager()
    logger.info("✓ Pipeline manager initialized")
    
    logger.info("Async Orchestration Service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Async Orchestration Service")
    
    # Stop all pipelines gracefully
    if pipeline_manager:
        stop_result = await pipeline_manager.stop_all()
        logger.info(f"Stopped {stop_result.get('count', 0)} pipeline(s)")
    
    # Close HTTP client
    if http_client:
        await http_client.aclose()
        logger.info("✓ HTTP client closed")
    
    logger.info("Async Orchestration Service shut down successfully")


# Create FastAPI application
app = FastAPI(
    title="Async Orchestration Service",
    description="""
    ## High-Performance Async Pipeline Orchestration
    
    Coordinates continuous detection pipelines using async/await for maximum concurrency.
    
    ### Architecture:
    - **Async I/O**: All HTTP calls use httpx.AsyncClient (non-blocking)
    - **Concurrency**: Each camera runs as independent asyncio task
    - **Rate Limiting**: Semaphores prevent overwhelming downstream services
    - **Connection Pooling**: Shared HTTP client with keepalive
    - **Graceful Degradation**: Per-camera error handling, failures don't cascade
    
    ### Scaling:
    - **1-10 cameras**: Single instance, minimal resources
    - **10-50 cameras**: Tune semaphore limits via env vars
    - **50-100 cameras**: Adjust poll intervals per camera
    - **100+ cameras**: Consider horizontal scaling with Redis task queue
    
    ### Pipeline Flow:
    Each camera: `frame → detection → usecase → alerts` (continuous loop)
    
    ### Services:
    - **Camera-Detection**: Frame extraction + object detection
    - **Usecase**: Business rule evaluation
    - **Alert**: Notification generation
    """,
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/pipeline/start/{camera_id}", tags=["pipeline"], response_model=dict)
async def start_camera_pipeline(camera_id: str, config: CameraConfig):
    """
    Start continuous async pipeline for a single camera.
    
    The pipeline will run continuously in the background:
    - Fetch frames from camera
    - Run object detection
    - Evaluate usecases
    - Send alerts if triggered
    
    **Parameters:**
    - **camera_id**: Camera identifier (path parameter)
    - **config**: Camera configuration (request body)
    
    **Example:**
    ```json
    {
      "camera_id": "s1_cam_1",
      "usecases": ["person_in_roi", "crowd_in_roi"],
      "confidence_threshold": 0.5,
      "poll_interval": 1.0,
      "max_errors_before_pause": 5
    }
    ```
    """
    # Override camera_id from config with path parameter
    config.camera_id = camera_id
    return await pipeline_manager.start_pipeline(config)


@app.post("/pipeline/start-batch", tags=["pipeline"], response_model=dict)
async def start_batch_pipelines(request: BatchStartRequest):
    """
    Start continuous async pipelines for multiple cameras at once.
    
    All cameras start in parallel using asyncio.gather().
    
    **Example:**
    ```json
    {
      "cameras": [
        {
          "camera_id": "s1_cam_1",
          "usecases": ["person_in_roi"],
          "confidence_threshold": 0.5,
          "poll_interval": 1.0
        },
        {
          "camera_id": "s1_cam_2",
          "usecases": ["crowd_in_roi"],
          "confidence_threshold": 0.6,
          "poll_interval": 2.0
        }
      ]
    }
    ```
    """
    results = []
    for config in request.cameras:
        result = await pipeline_manager.start_pipeline(config)
        results.append(result)
    
    started = len([r for r in results if r.get("status") == "started"])
    already_running = len([r for r in results if r.get("status") == "already_running"])
    
    return {
        "status": "batch_complete",
        "total": len(request.cameras),
        "started": started,
        "already_running": already_running,
        "results": results
    }


@app.post("/pipeline/stop/{camera_id}", tags=["pipeline"], response_model=dict)
async def stop_camera_pipeline(camera_id: str):
    """
    Stop continuous pipeline for a specific camera.
    
    The pipeline task will be cancelled gracefully.
    
    **Parameters:**
    - **camera_id**: Camera identifier
    
    **Example:**
    ```
    POST /pipeline/stop/s1_cam_1
    ```
    """
    return await pipeline_manager.stop_pipeline(camera_id)


@app.post("/pipeline/stop-all", tags=["pipeline"], response_model=dict)
async def stop_all_pipelines():
    """
    Stop all active pipelines.
    
    All pipeline tasks will be cancelled gracefully in parallel.
    """
    return await pipeline_manager.stop_all()


@app.get("/pipeline/status", tags=["pipeline"], response_model=dict)
async def get_all_pipelines_status():
    """
    Get status of all active pipelines.
    
    **Returns:**
    - List of all pipelines with statistics (iterations, errors, latency, etc.)
    """
    return await pipeline_manager.get_status()


@app.get("/pipeline/status/{camera_id}", tags=["pipeline"], response_model=PipelineStatus)
async def get_camera_pipeline_status(camera_id: str):
    """
    Get status of a specific camera pipeline.
    
    **Parameters:**
    - **camera_id**: Camera identifier
    
    **Returns:**
    - Pipeline statistics for the specified camera
    """
    return await pipeline_manager.get_status(camera_id)


@app.post("/pipeline/execute", tags=["pipeline - legacy"], response_model=dict)
async def execute_pipeline_once(request: PipelineRequest):
    """
    Execute pipeline once for all active cameras (batch processing).
    
    **LEGACY ENDPOINT** - Kept for backward compatibility and testing.
    For production use, prefer `/pipeline/start/{camera_id}` for continuous execution.
    
    This endpoint runs the complete pipeline once and returns results immediately:
    1. Get all active cameras (or use specified camera_id)
    2. Run detection on all cameras (parallel)
    3. Evaluate usecases for each camera (parallel)
    4. Send alerts for each camera (parallel)
    5. Return aggregated results for all cameras
    
    **Example (all cameras):**
    ```json
    {
      "confidence_threshold": 0.5,
      "usecases": ["person_in_roi", "crowd_in_roi"]
    }
    ```
    
    **Example (single camera):**
    ```json
    {
      "camera_id": "s1_cam_1",
      "confidence_threshold": 0.5,
      "usecases": ["person_in_roi", "crowd_in_roi"]
    }
    ```
    """
    usecases = request.usecases or ["person_in_roi", "crowd_in_roi", "restricted_zone_breach"]
    confidence_threshold = request.confidence_threshold
    
    try:
        # STEP 1: Get all active cameras (or use provided camera_id)
        if request.camera_id:
            camera_ids = [request.camera_id]
            logger.info(f"Batch pipeline execution for single camera: {request.camera_id}")
        else:
            # Call GET /camera/list to get all active cameras
            response = await http_client.get(
                f"{CAMERA_DETECTION_URL}/camera/list",
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            camera_list_data = response.json()
            
            # DEBUG: Log the raw response
            logger.info(f"DEBUG: Camera list response: {camera_list_data}")
            
            # Extract camera IDs - handle different response formats
            camera_ids = []
            
            # Format 1: cameras array with camera objects
            if 'cameras' in camera_list_data and isinstance(camera_list_data['cameras'], list):
                camera_ids = [cam['camera_id'] for cam in camera_list_data['cameras'] if 'camera_id' in cam]
            
            # Format 2: single_stream_cameras and multi_stream_cameras lists
            elif 'single_stream_cameras' in camera_list_data or 'multi_stream_cameras' in camera_list_data:
                if 'single_stream_cameras' in camera_list_data:
                    camera_ids.extend(camera_list_data['single_stream_cameras'])
                if 'multi_stream_cameras' in camera_list_data:
                    camera_ids.extend(camera_list_data['multi_stream_cameras'])
            
            # DEBUG: Log extracted camera IDs
            logger.info(f"DEBUG: Extracted camera_ids: {camera_ids}")
            
            if not camera_ids:
                logger.warning("No active cameras found in response")
                return {
                    "status": "no_active_cameras",
                    "message": "No active cameras found",
                    "results": []
                }
            
            logger.info(f"Batch pipeline execution for {len(camera_ids)} active cameras: {camera_ids}")
        
        # STEP 2: Run detection on all cameras in parallel
        async def process_single_camera(camera_id: str) -> dict:
            """Process pipeline for a single camera"""
            try:
                logger.info(f"DEBUG: Starting pipeline for camera {camera_id}")
                
                # Fetch ROIs, usecases, and per-class thresholds from DB
                rois = await asyncio.get_event_loop().run_in_executor(
                    None, get_camera_rois, camera_id
                )
                db_usecases = await asyncio.get_event_loop().run_in_executor(
                    None, get_camera_usecases, camera_id
                )
                class_thresholds = await asyncio.get_event_loop().run_in_executor(
                    None, get_class_thresholds, camera_id
                )
                active_usecases = db_usecases if db_usecases else usecases

                # Detection
                logger.info(f"DEBUG: [{camera_id}] Calling run_detection...")
                detection_data = await run_detection(camera_id, confidence_threshold, class_thresholds or None)
                total_det = detection_data.get('total_detections_count', 0)
                logger.info(f"DEBUG: [{camera_id}] Detection completed: {total_det} detections")

                # Cache snapshot URL if detections found
                if total_det > 0 and detection_data.get('snapshot_url'):
                    detection_snapshots[camera_id] = {
                        "camera_id": camera_id,
                        "snapshot_url": detection_data['snapshot_url'],
                        "timestamp": detection_data.get('timestamp'),
                        "detections": detection_data.get('detections', []),
                        "detection_count": total_det
                    }
                logger.info(f"DEBUG: [{camera_id}] DB config | rois={len(rois)} usecases={active_usecases}")

                # Evaluate usecases
                logger.info(f"DEBUG: [{camera_id}] Calling evaluate_usecases...")
                usecase_data = await evaluate_usecases(camera_id, detection_data, active_usecases, rois)
                logger.info(f"DEBUG: [{camera_id}] Usecases evaluated: {len(usecase_data.get('results', []))} results")

                # Persist all results to DB; send dashboard alerts only for parking_compliance + safety_monitoring
                results = usecase_data.get('results', [])
                try:
                    alerts_written = await asyncio.get_event_loop().run_in_executor(
                        None, persist_alerts_from_results, camera_id, results
                    )
                    logger.info(f"DEBUG: [{camera_id}] Alerts persisted to DB | count={alerts_written}")
                except Exception as e:
                    logger.warning(f"[{camera_id}] Alert persistence failed (non-critical): {str(e)}")

                _DASHBOARD_USECASES = {"parking_compliance", "safety_monitoring"}
                dashboard_results = [
                    r for r in results
                    if r.get("usecase_id") in _DASHBOARD_USECASES or r.get("usecase_name") in _DASHBOARD_USECASES
                ]
                if dashboard_results:
                    try:
                        await send_alerts(camera_id, dashboard_results)
                    except Exception as e:
                        logger.warning(f"[{camera_id}] Dashboard alert send failed (non-critical): {str(e)}")

                triggered_usecases = [r for r in usecase_data.get('results', []) if r.get('triggered')]
                
                return {
                    "status": "success",
                    "camera_id": camera_id,
                    "pipeline_results": {
                        "detection": {
                            "total_detections": detection_data.get('total_detections_count'),
                            "processing_time_ms": detection_data.get('processing_time_ms')
                        },
                        "usecases": {
                            "evaluated": len(usecase_data.get('results', [])),
                            "triggered": len(triggered_usecases),
                            "results": usecase_data.get('results', [])
                        },
                        "alerts": {
                            "sent": len(alert_data.get('alerts_sent', [])),
                            "details": alert_data.get('alerts_sent', [])
                        }
                    }
                }
            except Exception as e:
                logger.error(f"[{camera_id}] Pipeline failed: {str(e)}")
                return {
                    "status": "failed",
                    "camera_id": camera_id,
                    "error": str(e)
                }
        
        # Process all cameras in parallel
        logger.info(f"DEBUG: Starting parallel processing for {len(camera_ids)} cameras...")
        results = await asyncio.gather(
            *[process_single_camera(camera_id) for camera_id in camera_ids],
            return_exceptions=False
        )
        logger.info(f"DEBUG: Parallel processing completed. Got {len(results)} results")
        
        # Aggregate results
        successful_results = [r for r in results if r.get('status') == 'success']
        failed_results = [r for r in results if r.get('status') == 'failed']
        
        logger.info(f"DEBUG: Aggregation - {len(successful_results)} successful, {len(failed_results)} failed")
        
        return {
            "status": "completed",
            "total_cameras": len(camera_ids),
            "successful": len(successful_results),
            "failed": len(failed_results),
            "results": results
        }
        
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error: {e.response.status_code}")
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.ConnectError as e:
        logger.error(f"Connection error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Service connection failed: {str(e)}")
    except httpx.TimeoutException as e:
        logger.error(f"Timeout error: {str(e)}")
        raise HTTPException(status_code=504, detail=f"Service timeout: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {str(e)}")


@app.get("/snapshots/{camera_id}", tags=["snapshots"])
async def get_detection_snapshot(camera_id: str):
    """
    Get the latest detection snapshot for a camera.

    Returns the most recent frame (base64 JPEG) where detections were found.
    Returns 404 if no detection has occurred yet for this camera.

    **Used by dashboard to display detection screenshots.**
    """
    snap = detection_snapshots.get(camera_id)
    if not snap:
        raise HTTPException(
            status_code=404,
            detail=f"No detection snapshot available for camera {camera_id}"
        )
    return snap


@app.get("/snapshots", tags=["snapshots"])
async def list_detection_snapshots():
    """
    List all cameras that have detection snapshots cached.
    """
    return {
        "total": len(detection_snapshots),
        "cameras": [
            {
                "camera_id": cam_id,
                "timestamp": snap.get("timestamp"),
                "detection_count": snap.get("detection_count")
            }
            for cam_id, snap in detection_snapshots.items()
        ]
    }


@app.get("/alert/list", tags=["dashboard"])
async def list_alerts(
    camera_id: Optional[str] = None,
    usecase_name: Optional[str] = None,
    limit: int = 100,
):
    """
    Return saved alert rows for the dashboard.
    Query params: camera_id, usecase_name, limit (default 100).
    """
    db = SessionLocal()
    try:
        from shared.database.models import Alert
        q = db.query(Alert)
        if camera_id:
            q = q.filter(Alert.camera_id == camera_id)
        if usecase_name:
            q = q.filter(Alert.usecase_name == usecase_name)
        rows = q.order_by(Alert.timestamp.desc()).limit(limit).all()
        return {
            "alerts": [
                {
                    "alert_id":    r.alert_id,
                    "camera_id":   r.camera_id,
                    "usecase_name": r.usecase_name,
                    "alert_type":  r.alert_type,
                    "message":     r.message,
                    "timestamp":   r.timestamp.isoformat() if r.timestamp else None,
                    "status":      r.status,
                    "snapshot_url": r.snapshot_url,
                    "extras":      r.extras,
                }
                for r in rows
            ]
        }
    except Exception as e:
        logger.warning(f"[DB] Failed to fetch alerts: {e}")
        return {"alerts": []}
    finally:
        db.close()


@app.get("/alert/{alert_id}/snapshot", tags=["dashboard"])
async def get_alert_snapshot(alert_id: int):
    """
    Return the S3 snapshot URL for a specific alert.
    Call this on demand when displaying an alert's image in the dashboard.
    """
    db = SessionLocal()
    try:
        from shared.database.models import Alert
        row = db.query(Alert).filter(Alert.alert_id == alert_id).first()
        if not row:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        return {"alert_id": alert_id, "snapshot_url": row.snapshot_url}
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"[DB] Failed to fetch snapshot for alert {alert_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch snapshot")
    finally:
        db.close()


@app.get("/charging-sessions", tags=["dashboard"])
async def list_charging_sessions(
    camera_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
):
    """
    Return charging session rows for the dashboard.
    Query params: camera_id, status (active/charging/completed/incomplete/discarded), limit.
    """
    db = SessionLocal()
    try:
        from shared.database.models import ChargingSession
        q = db.query(ChargingSession)
        if camera_id:
            q = q.filter(ChargingSession.camera_id == camera_id)
        if status:
            q = q.filter(ChargingSession.session_status == status)
        else:
            # Hide discarded (sub-floor noise) from default listings. Callers
            # who want to audit discarded rows can pass status='discarded'.
            q = q.filter(ChargingSession.session_status != "discarded")
        rows = q.order_by(ChargingSession.created_at.desc()).limit(limit).all()
        return {
            "sessions": [
                {
                    "session_id":     r.session_id,
                    "camera_id":      r.camera_id,
                    "gun_number":     r.gun_number,
                    "car_number":     r.car_number,
                    "car_model":      r.car_model,
                    "in_time":        r.in_time.isoformat() if r.in_time else None,
                    "plug_time":      r.plug_time.isoformat() if r.plug_time else None,
                    "plug_out_time":  r.plug_out_time.isoformat() if r.plug_out_time else None,
                    "out_time":       r.out_time.isoformat() if r.out_time else None,
                    "session_status": r.session_status,
                    "created_at":     r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
        }
    except Exception as e:
        logger.warning(f"[DB] Failed to fetch charging sessions: {e}")
        return {"sessions": []}
    finally:
        db.close()


@app.get("/dashboard/sessions", tags=["dashboard"])
def dashboard_sessions(
    camera_id: Optional[str] = None,
    gun_number: Optional[str] = None,
    car_number: Optional[str] = None,
    status: Optional[str] = None,
    date: Optional[str] = None,   # YYYY-MM-DD in IST — filters by in_time date
    limit: int = 100,
):
    """
    Primary dashboard session table — all 7 fields per vehicle session.

    Gun Number | Car Number | Car Model | In Time |
    Plug In Time | Plug Out Time | Car Out Time

    Proxies to the analytics service /analytics/charging/sessions.
    Fields populate automatically as the vehicle moves through the facility:
    - in_time       — car enters parking ROI         (parking_detection)
    - plug_time     — charging gun plugged in         (gun_detection)
    - plug_out_time — charging gun unplugged          (gun_detection)
    - out_time      — car exits parking ROI           (parking_detection)
    - gun_number    — derived from ROI name           (gun_detection)
    - car_number    — license plate via Gemini Vision (vehicle_extraction)
    - car_model     — make/model via Gemini Vision    (vehicle_extraction)

    Query params: camera_id, gun_number, car_number,
                  status (active/charging/completed/incomplete/discarded), limit.
    """
    db = SessionLocal()
    try:
        from shared.database.models import ChargingSession
        from shared.database.mysql_energy import get_energy_consumed
        q = db.query(ChargingSession)
        if camera_id:
            q = q.filter(ChargingSession.camera_id == camera_id)
        if gun_number:
            q = q.filter(ChargingSession.gun_number == gun_number)
        if car_number:
            q = q.filter(ChargingSession.car_number == car_number)
        if status:
            q = q.filter(ChargingSession.session_status == status)
        else:
            # Hide discarded (sub-floor noise) from default listings. Callers
            # who want to audit discarded rows can pass status='discarded'.
            q = q.filter(ChargingSession.session_status != "discarded")
        if date:
            from datetime import date as date_type
            from zoneinfo import ZoneInfo
            import datetime as _dt
            _IST = ZoneInfo("Asia/Kolkata")
            try:
                d = date_type.fromisoformat(date)
                day_start_utc = _dt.datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=_IST).astimezone(_dt.timezone.utc)
                day_end_utc   = _dt.datetime(d.year, d.month, d.day, 23, 59, 59, tzinfo=_IST).astimezone(_dt.timezone.utc)
                q = q.filter(ChargingSession.in_time >= day_start_utc,
                             ChargingSession.in_time <= day_end_utc)
            except (ValueError, TypeError):
                pass
        rows = q.order_by(ChargingSession.created_at.desc()).limit(limit).all()
        sessions = []
        for r in rows:
            # Energy consumed: primary uses plug times; falls back to car in/out
            # times (±2 min buffer) when plug times are not yet recorded.
            energy_kwh = get_energy_consumed(
                plug_time=r.plug_time,
                plug_out_time=r.plug_out_time,
                in_time=r.in_time,
                out_time=r.out_time,
            )
            sessions.append({
                "session_id":    r.session_id,
                "camera_id":     r.camera_id,
                "slot_id":       r.slot_id,
                "gun_number":    r.gun_number,
                "car_number":    r.car_number,
                "car_model":     r.car_model,
                "in_time":       r.in_time.isoformat() if r.in_time else None,
                "plug_time":     r.plug_time.isoformat() if r.plug_time else None,
                "plug_out_time": r.plug_out_time.isoformat() if r.plug_out_time else None,
                "out_time":      r.out_time.isoformat() if r.out_time else None,
                "session_status": r.session_status,
                "energy_kwh":    energy_kwh,
                "created_at":    r.created_at.isoformat() if r.created_at else None,
                "updated_at":    r.updated_at.isoformat() if r.updated_at else None,
            })
        return {"sessions": sessions, "total": len(sessions)}
    except Exception as e:
        logger.warning(f"[DASHBOARD] Failed to fetch sessions: {e}")
        return {"sessions": [], "total": 0}
    finally:
        db.close()


@app.post("/dashboard/energy-comparison/upload", tags=["dashboard"])
def dashboard_energy_comparison_upload(
    file: UploadFile = File(...),
    camera_id: Optional[str] = "camera_01",
    days: int = 14,
):
    """
    Accept the client OCPP Excel export and return per-row energy-loss analysis.

    Matching uses weighted scoring across VRN, car model, slot→connector, session
    duration, and date (see shared/energy_comparison.py). No per-gun meter query
    is possible — the physical meter is cumulative across both connectors — so
    `meter_kwh` is our existing per-session estimate (which may over-count when
    two guns were active simultaneously). The client Excel value is authoritative.

    Performance: the previous implementation called get_energy_consumed() per
    CCTV session, which opened a fresh MySQL connection and ran 2 queries per
    row. With ~hundreds of sessions over a 14-day window that turned uploads
    into multi-minute calls. We now do ONE bulk fetch of all energy readings
    in the date window and match in Python. Endpoint is also `def` (FastAPI
    runs it in a thread pool) so its sync DB work doesn't block the asyncio
    loop / starve the live detection pipeline.
    """
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xlsm")):
        raise HTTPException(status_code=400, detail="Upload an .xlsx file")

    try:
        # Sync read since we're now a `def` handler — file.file is the raw
        # SpooledTemporaryFile under the hood.
        contents = file.file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read upload: {e}")

    from shared.energy_comparison import (
        parse_excel,
        match_excel_to_cctv,
        result_to_dict,
        CctvSession,
    )
    try:
        excel_rows = parse_excel(contents)
    except Exception as e:
        logger.warning(f"[ENERGY_COMPARE] Excel parse failed: {e}")
        raise HTTPException(status_code=400, detail=f"Could not parse Excel: {e}")

    if not excel_rows:
        return {"results": [], "summary": {"total": 0, "matched": 0, "unmatched": 0}}

    # Constrain CCTV lookup to the date range covered by the Excel so we don't
    # load thousands of irrelevant sessions.
    excel_dates = [r.date for r in excel_rows if r.date is not None]
    date_floor = min(excel_dates) - timedelta(days=1) if excel_dates else None
    date_ceiling = max(excel_dates) + timedelta(days=2) if excel_dates else None

    db = SessionLocal()
    try:
        from shared.database.models import ChargingSession
        from shared.database.mysql_energy import (
            fetch_readings_window,
            compute_kwh_from_readings,
            _to_naive_ist,
        )

        q = db.query(ChargingSession)
        if camera_id:
            q = q.filter(ChargingSession.camera_id == camera_id)
        if date_floor is not None and date_ceiling is not None:
            q = q.filter(ChargingSession.created_at >= date_floor)
            q = q.filter(ChargingSession.created_at <= date_ceiling)
        else:
            q = q.filter(ChargingSession.created_at >= datetime.now(timezone.utc) - timedelta(days=days))

        # Sub-floor visits ('discarded') are tracker fragmentation or non-events;
        # never match them against meter readings — the kWh would be garbage.
        q = q.filter(ChargingSession.session_status != "discarded")

        rows = q.order_by(ChargingSession.created_at.desc()).limit(2000).all()

        # Single bulk MySQL fetch covering the whole CCTV window, padded to
        # cover the ±2-min buffer compute_kwh_from_readings uses internally.
        # Fall back to a full-day pad if the row set is empty so we don't hit
        # MySQL with a degenerate window.
        if rows:
            session_times: list[datetime] = []
            for r in rows:
                for t in (r.plug_time, r.plug_out_time, r.in_time, r.out_time):
                    naive = _to_naive_ist(t)
                    if naive is not None:
                        session_times.append(naive)
            if session_times:
                window_start = min(session_times) - timedelta(minutes=5)
                window_end   = max(session_times) + timedelta(minutes=5)
                readings = fetch_readings_window(window_start, window_end)
            else:
                readings = []
        else:
            readings = []

        cctv_sessions = []
        for r in rows:
            energy_kwh = compute_kwh_from_readings(
                readings,
                plug_time=r.plug_time,
                plug_out_time=r.plug_out_time,
                in_time=r.in_time,
                out_time=r.out_time,
            )
            cctv_sessions.append(CctvSession(
                session_id=r.session_id,
                camera_id=r.camera_id,
                slot_id=r.slot_id,
                gun_number=r.gun_number,
                car_number=r.car_number,
                car_model=r.car_model,
                in_time=r.in_time,
                plug_time=r.plug_time,
                plug_out_time=r.plug_out_time,
                out_time=r.out_time,
                energy_kwh=energy_kwh,
            ))
    except Exception as e:
        logger.warning(f"[ENERGY_COMPARE] CCTV session fetch failed: {e}")
        cctv_sessions = []
    finally:
        db.close()

    results = match_excel_to_cctv(excel_rows, cctv_sessions)
    payload = [result_to_dict(r) for r in results]

    matched = sum(1 for r in results if r.cctv_session is not None)
    total_loss = sum(
        r.loss_kwh for r in results
        if r.loss_kwh is not None
    )
    total_client_kwh = sum(
        r.excel_row.units_kwh for r in results
        if r.excel_row.units_kwh is not None
    )

    return {
        "results": payload,
        "summary": {
            "total": len(results),
            "matched": matched,
            "unmatched": len(results) - matched,
            "total_client_kwh": round(total_client_kwh, 3),
            "total_loss_kwh": round(total_loss, 3),
            "cctv_pool_size": len(cctv_sessions),
        },
    }


@app.post("/dashboard/energy-analysis", tags=["dashboard"])
def dashboard_energy_analysis(request: dict):
    """
    Accept matched energy-comparison results and use OpenAI to generate
    a natural-language insight report identifying:
      - Cars consuming the most energy
      - Cars with the highest energy loss vs client OCPP data
      - Unusual or repeated patterns
      - Summary recommendations

    Request body: { "results": [...], "summary": {...} }
    Returns: { "insight": "<markdown text>" }
    """
    import os
    import json

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured on server")

    results  = request.get("results", [])
    summary  = request.get("summary", {})

    if not results:
        raise HTTPException(status_code=400, detail="No results provided for analysis")

    # Build a compact data table for the prompt
    table_lines = ["VRN | Make | Model | Connector | Duration(min) | Client kWh | Meter kWh | Loss kWh | Match"]
    table_lines.append("----|------|-------|-----------|---------------|------------|-----------|----------|------")
    for r in results:
        vrn      = r.get("vrn") or "—"
        make     = r.get("make") or "—"
        model    = r.get("model") or "—"
        conn     = r.get("connector_id") or "—"
        dur_min  = round(r.get("duration_seconds", 0) / 60, 1) if r.get("duration_seconds") else "—"
        c_kwh    = r.get("client_kwh")
        m_kwh    = r.get("meter_kwh")
        loss     = r.get("loss_kwh")
        conf     = r.get("confidence", "UNMATCHED")
        table_lines.append(
            f"{vrn} | {make} | {model} | {conn} | {dur_min} | "
            f"{c_kwh if c_kwh is not None else '—'} | "
            f"{m_kwh if m_kwh is not None else '—'} | "
            f"{loss if loss is not None else '—'} | {conf}"
        )

    prompt = f"""You are an EV charging station energy analyst. Analyse the charging session data below and return ONLY a valid JSON object — no markdown, no code fences, no extra text.

## Summary
- Total sessions: {summary.get('total', '?')}
- Matched: {summary.get('matched', '?')}, Unmatched: {summary.get('unmatched', '?')}
- Total client kWh: {summary.get('total_client_kwh', '?')}
- Total energy loss (client − meter): {summary.get('total_loss_kwh', '?')} kWh

## Session Data
{chr(10).join(table_lines)}

## Required JSON shape (return exactly this, filled in):
{{
  "top_consumer": "VRN or car that consumed the most — one sentence",
  "highest_loss": "VRN or car with biggest client−meter gap — one sentence",
  "patterns": ["bullet 1", "bullet 2", "bullet 3"],
  "unmatched_note": "one sentence about the unmatched sessions",
  "recommendations": ["action 1", "action 2", "action 3"],
  "risk_level": "LOW | MEDIUM | HIGH"
}}

Be concise. Use actual VRNs and numbers from the data.
"""

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content.strip()
        structured = json.loads(raw)
    except json.JSONDecodeError:
        structured = {"raw": raw}
    except Exception as e:
        logger.warning(f"[OPENAI] Energy analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

    return {"insight": structured}


@app.get("/dashboard/parking-compliance", tags=["dashboard"])
def dashboard_parking_compliance(
    camera_id: Optional[str] = None,
    limit: int = 100,
):
    """
    Parking Compliance dashboard section.

    Returns parking compliance violations from the alert service:
    - unauthorized_parking  — car detected outside all defined ROIs
    - wrong_parking         — car centroid inside multiple ROIs simultaneously
    - multiple_cars_in_roi  — more than one car occupying the same ROI

    Each violation row includes alert_type, message, timestamp, extras
    (bbox, confidence, roi name), and snapshot_url.

    Query params: camera_id, limit.
    """
    db = SessionLocal()
    try:
        from shared.database.models import Alert
        q = db.query(Alert).filter(Alert.usecase_name == "parking_compliance")
        if camera_id:
            q = q.filter(Alert.camera_id == camera_id)
        rows = q.order_by(Alert.timestamp.desc()).limit(limit).all()
        alerts = [
            {
                "alert_id":    r.alert_id,
                "camera_id":   r.camera_id,
                "usecase_name": r.usecase_name,
                "alert_type":  r.alert_type,
                "message":     r.message,
                "timestamp":   r.timestamp.isoformat() if r.timestamp else None,
                "status":      r.status,
                "snapshot_url": r.snapshot_url,
                "extras":      r.extras,
            }
            for r in rows
        ]
        return {"violations": alerts, "total": len(alerts)}
    except Exception as e:
        logger.warning(f"[DASHBOARD] Failed to fetch parking compliance: {e}")
        return {"violations": [], "total": 0}
    finally:
        db.close()


@app.get("/dashboard/safety-monitoring", tags=["dashboard"])
def dashboard_safety_monitoring(
    camera_id: Optional[str] = None,
    limit: int = 100,
):
    """
    Safety Monitoring dashboard section.

    Returns safety alerts from the alert service:
    - fire  — fire detected in camera frame
    - smoke — smoke detected in camera frame

    Each alert row includes alert_type, message, timestamp, extras
    (hazard_type, confidence, bbox), and snapshot_url.

    Query params: camera_id, limit.
    """
    db = SessionLocal()
    try:
        from shared.database.models import Alert
        q = db.query(Alert).filter(Alert.usecase_name == "safety_monitoring")
        if camera_id:
            q = q.filter(Alert.camera_id == camera_id)
        rows = q.order_by(Alert.timestamp.desc()).limit(limit).all()
        alerts = [
            {
                "alert_id":    r.alert_id,
                "camera_id":   r.camera_id,
                "usecase_name": r.usecase_name,
                "alert_type":  r.alert_type,
                "message":     r.message,
                "timestamp":   r.timestamp.isoformat() if r.timestamp else None,
                "status":      r.status,
                "snapshot_url": r.snapshot_url,
                "extras":      r.extras,
            }
            for r in rows
        ]
        return {"safety_alerts": alerts, "total": len(alerts)}
    except Exception as e:
        logger.warning(f"[DASHBOARD] Failed to fetch safety alerts: {e}")
        return {"safety_alerts": [], "total": 0}
    finally:
        db.close()


@app.get("/dashboard/station", tags=["dashboard"])
def dashboard_station(
    camera_id: str = "camera_01",
    station_id: str = "station_01",
):
    """
    Real-time slot-oriented station view for the EV charging station monitor.

    Returns the two active slots (ROI_1, ROI_2) with the most-recent
    open/active session for each slot. Compliance violations are surfaced
    via a separate popup driven by /dashboard/active-violations.

    Response shape:
    {
      "station_id": "station_01",
      "slots": {
        "ROI_1": {
          "car_number": "KL87G345",
          "car_model": "Tata Tiago EV",
          "car_intime": "22:56:44",
          "car_outtime": null,
          "gun_number": "Gun 2",
          "gun_plugin_time": "22:58:53",
          "gun_plugout_time": null,
          "status": "charging"
        },
        "ROI_2": { "status": "empty" }
      }
    }
    """
    db = SessionLocal()
    try:
        from shared.database.models import ChargingSession
        from shared.database.mysql_energy import get_energy_consumed

        SLOTS = ["ROI_1", "ROI_2"]
        slots_out = {}

        for slot_id in SLOTS:
            # Only consider genuinely live sessions: no out_time AND status is
            # active or charging. incomplete means the stale-cleanup job already
            # declared the session abandoned — the slot is physically empty.
            session = (
                db.query(ChargingSession)
                .filter(
                    ChargingSession.camera_id == camera_id,
                    ChargingSession.slot_id == slot_id,
                    ChargingSession.out_time.is_(None),
                    ChargingSession.session_status.in_(("active", "charging")),
                )
                .order_by(ChargingSession.created_at.desc())
                .first()
            )

            if session is None:
                slots_out[slot_id] = {"status": "empty"}
                continue

            def fmt_time(dt):
                if dt is None:
                    return None
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()

            # Live energy: kWh consumed so far (plug_out_time=None for active sessions)
            energy_kwh = get_energy_consumed(
                plug_time=session.plug_time,
                plug_out_time=session.plug_out_time,
                in_time=session.in_time,
                out_time=session.out_time,
            )

            slots_out[slot_id] = {
                "car_number":      session.car_number,
                "car_model":       session.car_model,
                "car_intime":      fmt_time(session.in_time),
                "car_outtime":     fmt_time(session.out_time),
                "gun_number":      session.gun_number,
                "gun_plugin_time": fmt_time(session.plug_time),
                "gun_plugout_time": fmt_time(session.plug_out_time),
                "status":          session.session_status or "active",
                "energy_kwh":      energy_kwh,
            }

        return {"station_id": station_id, "slots": slots_out}

    except Exception as e:
        logger.warning(f"[DASHBOARD] /dashboard/station failed: {e}")
        return {
            "station_id": station_id,
            "slots": {"ROI_1": {"status": "empty"}, "ROI_2": {"status": "empty"}},
        }
    finally:
        db.close()


@app.get("/dashboard/active-violations", tags=["dashboard"])
def dashboard_active_violations(
    camera_id: str = "camera_01",
    station_id: str = "station_01",
    window_seconds: int = 30,
):
    """
    Currently-active parking_compliance violations (within the last
    `window_seconds`). Designed to drive a popup/toast on the dashboard
    that appears while a violation is fresh and disappears once it stops
    being re-emitted.

    Response:
    {
      "station_id": "station_01",
      "violations": [
        {
          "alert_id": 42,
          "type": "wrong_parking",
          "slots": ["ROI_1", "ROI_2"],   # affected slots, [] if none
          "timestamp": "2026-04-27T...",
          "snapshot_url": "..."
        }
      ]
    }
    """
    db = SessionLocal()
    try:
        from shared.database.models import Alert

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)
        rows = (
            db.query(Alert)
            .filter(
                Alert.camera_id    == camera_id,
                Alert.usecase_name == "parking_compliance",
                Alert.timestamp    >= cutoff,
            )
            .order_by(Alert.timestamp.desc())
            .all()
        )

        # Dedupe: one entry per alert_type, keeping the most recent.
        seen = {}
        for r in rows:
            if r.alert_type in seen:
                continue
            extras = r.extras or {}
            viols  = extras.get("violations") or []
            meta   = (viols[0] or {}).get("metadata", {}) if viols else {}
            slots  = meta.get("overlapping_rois") or (
                [r.slot_id] if r.slot_id else []
            )
            ts = r.timestamp
            if ts is not None and ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            seen[r.alert_type] = {
                "alert_id":     r.alert_id,
                "type":         r.alert_type,
                "slots":        slots,
                "timestamp":    ts.isoformat() if ts else None,
                "snapshot_url": r.snapshot_url,
            }

        return {"station_id": station_id, "violations": list(seen.values())}

    except Exception as e:
        logger.warning(f"[DASHBOARD] /dashboard/active-violations failed: {e}")
        return {"station_id": station_id, "violations": []}
    finally:
        db.close()


@app.get("/dashboard/compliance-violations", tags=["dashboard"])
def dashboard_compliance_violations(
    camera_id: str = "camera_01",
    station_id: str = "station_01",
    limit: int = 100,
):
    """
    Parking compliance violations shaped for the station monitor dashboard.

    Returns violations with: station, car_model, violation_type, duration.

    Violation types:
      - unauthorized_parking  — car centroid outside all ROIs
      - wrong_parking         — car centroid inside more than one ROI simultaneously
      - multiple_cars_in_roi  — more than one car in the same ROI

    car_model is joined from the most-recent ChargingSession that shares
    the same camera_id and was active around the violation timestamp.

    Response:
    {
      "violations": [
        {
          "alert_id": 1,
          "station": "station_01",
          "car_model": "Tata Tiago EV",
          "violation_type": "wrong_parking",
          "timestamp": "2026-04-12T10:30:00",
          "duration": "5m 12s"      # time since violation, null if unknown
        }
      ],
      "total": 1
    }
    """
    db = SessionLocal()
    try:
        from shared.database.models import Alert, ChargingSession
        from datetime import datetime, timezone

        rows = (
            db.query(Alert)
            .filter(Alert.camera_id == camera_id)
            .filter(Alert.usecase_name == "parking_compliance")
            .order_by(Alert.timestamp.desc())
            .limit(limit)
            .all()
        )

        def fmt_duration(in_dt, out_dt):
            """Format parking duration from session in_time → out_time (or now if active)."""
            if not in_dt:
                return None
            try:
                if in_dt.tzinfo is None:
                    in_dt = in_dt.replace(tzinfo=timezone.utc)
                end = out_dt if out_dt else datetime.now(timezone.utc)
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                secs = int((end - in_dt).total_seconds())
                if secs < 0:
                    return None
                h = secs // 3600
                m = (secs % 3600) // 60
                s = secs % 60
                if h > 0:
                    return f"{h}h {m}m"
                if m > 0:
                    return f"{m}m {s}s"
                return f"{s}s"
            except Exception:
                return None

        def resolve_violation_type(row):
            """Extract the specific violation type from alert_type or extras."""
            atype = row.alert_type or ""
            if atype in ("multiple_cars_in_roi", "unauthorized_parking", "wrong_parking"):
                return atype
            extras = row.extras or {}
            for v in extras.get("violations", []):
                et = v.get("event_type", "")
                if et in ("unauthorized_parking", "wrong_parking", "multiple_cars_in_roi"):
                    return et
            return atype or "unknown"

        def extract_track_id(row):
            """Pull track_id from the alert extras."""
            extras = row.extras or {}
            for v in extras.get("violations", []):
                tid = v.get("track_id")
                if tid:
                    return str(tid)
            for e in extras.get("events", []):
                tid = e.get("track_id")
                if tid:
                    return str(tid)
            return None

        def resolve_session(row):
            """Return the ChargingSession matched by track_id only (no cross-car fallback)."""
            track_id = extract_track_id(row)
            if not track_id:
                return None
            return (
                db.query(ChargingSession)
                .filter(
                    ChargingSession.camera_id == camera_id,
                    ChargingSession.track_id  == track_id,
                )
                .order_by(ChargingSession.created_at.desc())
                .first()
            )

        violations = []
        for row in rows:
            session = resolve_session(row)
            car_number = session.car_number if session else None
            car_model  = session.car_model  if session else None
            duration   = fmt_duration(
                session.in_time  if session else None,
                session.out_time if session else None,
            )
            violations.append({
                "alert_id":       row.alert_id,
                "station":        station_id,
                "slot_id":        row.slot_id,
                "track_id":       extract_track_id(row),
                "car_number":     car_number,
                "car_model":      car_model,
                "violation_type": resolve_violation_type(row),
                "timestamp":      row.timestamp.isoformat() if row.timestamp else None,
                "duration":       duration,
                "snapshot_url":   row.snapshot_url,
            })

        return {"violations": violations, "total": len(violations)}

    except Exception as e:
        logger.warning(f"[DASHBOARD] /dashboard/compliance-violations failed: {e}")
        return {"violations": [], "total": 0}
    finally:
        db.close()


@app.get("/health", tags=["health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns the health status of the orchestration service and connectivity 
    to dependent services.
    """
    health_status = {
        "status": "healthy",
        "service": "async-orchestration",
        "version": "2.0.0",
        "architecture": "async/await",
        "services": {}
    }
    
    # Check connectivity to dependent services
    services = {
        "camera_detection": CAMERA_DETECTION_URL,
        "usecase": USECASE_SERVICE_URL,
        "alert": ALERT_SERVICE_URL,
        "analytics": ANALYTICS_SERVICE_URL,
    }
    
    for service_name, service_url in services.items():
        try:
            response = await http_client.get(f"{service_url}/health", timeout=5.0)
            health_status["services"][service_name] = {
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "url": service_url,
                "status_code": response.status_code
            }
        except Exception as e:
            health_status["services"][service_name] = {
                "status": "unreachable",
                "url": service_url,
                "error": str(e)
            }
    
    # Overall health is degraded if any service is down
    if any(s["status"] != "healthy" for s in health_status["services"].values()):
        health_status["status"] = "degraded"
    
    # Add pipeline statistics
    if pipeline_manager:
        pipeline_status = await pipeline_manager.get_status()
        health_status["pipelines"] = {
            "active": pipeline_status.get("active_pipelines", 0),
            "total_tracked": pipeline_status.get("total_tracked", 0)
        }
    
    return health_status


@app.get("/", tags=["root"])
async def root():
    """API Root - Welcome and Quick Links"""
    status = await pipeline_manager.get_status() if pipeline_manager else {"active_pipelines": 0}
    active_count = status.get("active_pipelines", 0)
    
    return {
        "message": "Async Orchestration Service - High-Performance Pipeline Controller",
        "version": "2.0.0",
        "architecture": "async/await with httpx",
        "status": "operational",
        "active_pipelines": active_count,
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc",
            "openapi_json": "/openapi.json"
        },
        "endpoints": {
            "pipeline": {
                "start_single": "POST /pipeline/start/{camera_id}",
                "start_batch": "POST /pipeline/start-batch",
                "stop_single": "POST /pipeline/stop/{camera_id}",
                "stop_all": "POST /pipeline/stop-all",
                "status_all": "GET /pipeline/status",
                "status_single": "GET /pipeline/status/{camera_id}",
                "execute_once": "POST /pipeline/execute (legacy)",
            },
            "dashboard": {
                "sessions": "GET /dashboard/sessions — all 7 session fields (Gun Number | Car Number | Car Model | In Time | Plug In Time | Plug Out Time | Car Out Time)",
                "parking_compliance": "GET /dashboard/parking-compliance — unauthorized parking, wrong parking, multiple cars in ROI",
                "safety_monitoring": "GET /dashboard/safety-monitoring — fire and smoke alerts",
                "charging_sessions_legacy": "GET /charging-sessions — direct DB session query (legacy)",
                "alerts_legacy": "GET /alert/list — direct DB alert query (legacy)",
            },
            "snapshots": {
                "snapshot_single": "GET /snapshots/{camera_id}",
                "snapshot_list": "GET /snapshots",
            },
            "health": "GET /health"
        },
        "configured_services": {
            "camera_detection": CAMERA_DETECTION_URL,
            "usecase": USECASE_SERVICE_URL,
            "alert": ALERT_SERVICE_URL,
            "analytics": ANALYTICS_SERVICE_URL,
        },
        "concurrency_limits": {
            "camera_detection": CAMERA_DETECTION_CONCURRENCY,
            "usecase": USECASE_CONCURRENCY,
            "alert": ALERT_CONCURRENCY
        },
        "scaling_notes": {
            "current": "Single instance with async I/O",
            "1-10_cameras": "Minimal resources, default config",
            "10-50_cameras": "Tune semaphore limits",
            "50-100_cameras": "Adjust poll intervals per camera",
            "100+_cameras": "Consider Redis task queue for horizontal scaling"
        }
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )
