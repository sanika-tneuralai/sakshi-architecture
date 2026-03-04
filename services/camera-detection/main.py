import sys
import os

# Add DeepStream path FIRST, before any other imports
deepstream_path = os.getenv('DEEPSTREAM_PATH', '/opt/nvidia/deepstream/deepstream-6.4/lib')
if deepstream_path not in sys.path:
    sys.path.insert(0, deepstream_path)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import uvicorn


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('camera_detection_api.log')
    ]
)

logger = logging.getLogger(__name__)


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
    
    # Initialize database
    init_db()
    logger.info("Database initialized")
    
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


# Import and include routers
from camera.api import router as camera_router
from detection.api import router as detection_router

app.include_router(camera_router)
app.include_router(detection_router)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
