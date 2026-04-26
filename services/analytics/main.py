"""
Analytics Service - Standalone API for analytics and reporting.

This service provides:
- Scheduled daily aggregation of analytics data
- API endpoints for daily and monthly analytics reports
- Detection and alert analytics
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


# ---------------------------------------------------------------------------
# Logging
#
# Three sinks on the root logger so every getLogger(__name__) inherits:
#   - stdout                       INFO+   operational view
#   - logs/analytics.log           INFO+   bounded operational history
#   - logs/analytics_debug.log     DEBUG+  full pipeline + scheduler trace
#
# Format includes file:line so each line points at the call site.
# RotatingFileHandler caps each file at 10 MB × 5 backups.
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
    LOG_DIR / "analytics.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_info_file.setLevel(logging.INFO)
_info_file.setFormatter(_LOG_FORMATTER)
_root.addHandler(_info_file)

_debug_file = RotatingFileHandler(
    LOG_DIR / "analytics_debug.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_debug_file.setLevel(logging.DEBUG)
_debug_file.setFormatter(_LOG_FORMATTER)
_root.addHandler(_debug_file)

# Quiet noisy third-party libs so the debug file stays readable. APScheduler
# is on this list because the scheduler logs every job tick at INFO by default.
for _noisy in ("urllib3", "httpx", "httpcore", "PIL", "matplotlib", "asyncio", "apscheduler"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info("Logging configured: dir=%s (info+debug, rotating)", LOG_DIR.resolve())

# Imports that may emit logs at import time go AFTER logging is configured.
from analytics.api import router as analytics_router
from analytics.scheduler import start_scheduler, stop_scheduler


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("Starting Analytics Service...")
    logger.info(f"Database URL: {os.getenv('DATABASE_URL', 'Not set')}")
    
    # Start analytics scheduler
    start_scheduler()
    logger.info("Analytics scheduler started")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Analytics Service...")
    stop_scheduler()
    logger.info("Analytics scheduler stopped")


# Create FastAPI app
app = FastAPI(
    title="Analytics Service",
    description="Standalone analytics and reporting service for GOEC",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(analytics_router)


@app.get("/")
def root():
    """Root endpoint."""
    return {
        "service": "Analytics Service",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "analytics",
        "scheduler": "running"
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8003))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting Analytics Service on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("RELOAD", "false").lower() == "true"
    )
