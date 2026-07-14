import sys
import os
from dotenv import load_dotenv

# Load .env file before anything else reads environment variables
load_dotenv()

# Add DeepStream path FIRST, before any other imports
deepstream_path = os.getenv('DEEPSTREAM_PATH', '/opt/nvidia/deepstream/deepstream-6.4/lib')
if deepstream_path not in sys.path:
    sys.path.insert(0, deepstream_path)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import uvicorn


# ---------------------------------------------------------------------------
# Logging
#
# Three sinks on the root logger so every getLogger(__name__) inherits:
#   - stdout                              INFO+   operational view
#   - logs/camera_detection.log           INFO+   bounded operational history
#   - logs/camera_detection_debug.log     DEBUG+  full pipeline trace
#
# Format includes file:line so each line points at the call site (e.g.
# camera/service.py:123 vs detection/yolo.py:88).
#
# RotatingFileHandler caps each file at 10 MB × 5 backups so a long-running
# container can't blow up the disk.
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
    LOG_DIR / "camera_detection.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_info_file.setLevel(logging.INFO)
_info_file.setFormatter(_LOG_FORMATTER)
_root.addHandler(_info_file)

_debug_file = RotatingFileHandler(
    LOG_DIR / "camera_detection_debug.log",
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Import here to avoid circular imports
    from camera.service import camera_manager
    from shared.common.utils import log_system_info
    from shared.database.connection import init_db
    
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Camera-Detection Service")
    logger.info("=" * 60)
    log_system_info()
    
    # Initialize database (non-fatal — service can run without DB)
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning(f"Database initialization failed (continuing without DB): {e}")
    
    # Log GStreamer info
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        logger.info(f"GStreamer version: {Gst.version_string()}")
    except Exception as e:
        logger.error(f"GStreamer not available: {str(e)}")
    
    logger.info("Camera-Detection service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Camera-Detection Service")
    try:
        await camera_manager.stop_all()
        logger.info("All cameras stopped")
    except Exception as e:
        logger.error(f"Error stopping cameras during shutdown: {str(e)}")
    
    logger.info("Camera-Detection service shut down successfully")


# Create FastAPI application
app = FastAPI(
    title="Camera-Detection Service",
    description="""
    ## Camera Management & Object Detection Service
    
    This service provides:
    - **Camera Management**: RTSP stream handling with ROI support
    - **Object Detection**: YOLO-based detection with ROI filtering
    - **Single Camera Mode**: For < 10 cameras with independent processing
    - **Multi-Stream Mode**: For 100+ cameras with batched GPU processing
    - **Hardware Acceleration**: NVIDIA DeepStream for optimal performance
    
    ### Camera Modes:
    
    #### Single Camera Mode
    - Best for: < 10 cameras
    - Each camera runs independently
    - Lower latency per camera
    - Use `/camera/start` endpoint
    
    #### Multi-Stream Mode (Recommended for 100+ cameras)
    - Best for: 100+ cameras
    - Batched GPU processing
    - Optimal resource utilization
    - Use `/camera/start-multi` endpoint
    
    ### Quick Start:
    1. Start a camera: `POST /camera/start`
    2. Check status: `GET /camera/status/{camera_id}`
    3. Get frame: `GET /camera/frame/{camera_id}`
    4. Run detection: `POST /detection/detect`
    5. Stop camera: `DELETE /camera/stop/{camera_id}`
    
    **Note**: This service is designed to be called by an orchestrator service.
    It does NOT include usecase evaluation, alerting, or analytics logic.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc)
        }
    )


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint for orchestrator"""
    return {
        "status": "healthy",
        "service": "camera-detection",
        "version": "1.0.0"
    }


# ---------------------------------------------------------------------------
# System metrics — consumed by orchestration /dashboard/system-metrics to
# populate the dashboard System page (this is the GPU box, so it reports GPU%).
# ---------------------------------------------------------------------------
import psutil

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _NVML_OK = False


def _gpu_percent():
    """GPU utilization % of device 0, or None if NVML is unavailable."""
    if not _NVML_OK:
        return None
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        return round(float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu), 1)
    except Exception:
        return None


def _camera_perf():
    """Per-camera perf from the live DeepStream pipeline (best-effort, defensive).

    Surfaces what the pipeline actually tracks: frame_count, configured fps, and
    the current detection count. queue/dropped/inference/tracking are NOT measured
    by the pipeline yet, so they're reported null (dashboard shows "—") until the
    nvinfer/nvstreammux probes are instrumented.
    """
    out = {}
    try:
        from camera.service import camera_manager
        pipe = getattr(camera_manager, "ds_pipeline", None)
        if pipe is None:
            return out
        for cam_id in pipe.list_cameras():
            try:
                st = pipe.get_status(cam_id) or {}
                latest = pipe.get_latest(cam_id)
                out[cam_id] = {
                    "frame_count":    st.get("frame_count", 0),
                    "fps":            st.get("fps"),
                    "detections":     len(latest.detections) if latest else 0,
                    "queue_size":     None,
                    "dropped_frames": None,
                    "inference_ms":   None,
                    "tracking_ms":    None,
                }
            except Exception:
                continue
    except Exception:
        pass
    return out


# Plain `def` → FastAPI runs it in a threadpool; cpu_percent() blocks 0.2s.
@app.get("/system/metrics", tags=["system"])
def system_metrics():
    """Host CPU/RAM + GPU% + per-camera DeepStream perf for the dashboard System page."""
    return {
        "cpu": round(psutil.cpu_percent(interval=0.2), 1),
        "ram": round(psutil.virtual_memory().percent, 1),
        "gpu": _gpu_percent(),
        "cameras": _camera_perf(),
    }


# Import and include routers
from camera.api import router as camera_router
from detection.api import router as detection_router

app.include_router(camera_router)
app.include_router(detection_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8004,
        reload=False,
        log_level="info"
    )
