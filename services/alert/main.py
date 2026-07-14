"""
Alert Service
A standalone microservice for processing and sending alerts based on usecase evaluation results.
"""
import sys
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn


# ---------------------------------------------------------------------------
# Logging
#
# Three sinks on the root logger so every getLogger(__name__) inherits:
#   - stdout                     INFO+   operational view
#   - logs/alert_service.log     INFO+   bounded operational history
#   - logs/alert_debug.log       DEBUG+  full alert dispatch trace
#
# Format includes file:line so each line points at the call site (e.g.
# alert/dispatcher.py:88 vs alert/api.py:42).
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
    LOG_DIR / "alert_service.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_info_file.setLevel(logging.INFO)
_info_file.setFormatter(_LOG_FORMATTER)
_root.addHandler(_info_file)

_debug_file = RotatingFileHandler(
    LOG_DIR / "alert_debug.log",
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
    from shared.database.connection import init_db
    
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Alert Service")
    logger.info("=" * 60)
    
    # Initialize database
    try:
        init_db()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {str(e)}")
    
    # Log alert channel configuration
    email_enabled = os.getenv('ALERT_EMAIL_ENABLED', 'false').lower() == 'true'
    sms_enabled = os.getenv('ALERT_SMS_ENABLED', 'false').lower() == 'true'
    webhook_enabled = os.getenv('ALERT_WEBHOOK_ENABLED', 'false').lower() == 'true'
    mobile_push_enabled = os.getenv('MOBILE_PUSH_ENABLED', 'false').lower() == 'true'

    logger.info("Alert Channels Configuration:")
    logger.info(f"  - Email: {'Enabled' if email_enabled else 'Disabled'}")
    logger.info(f"  - SMS: {'Enabled' if sms_enabled else 'Disabled'}")
    logger.info(f"  - Webhook: {'Enabled' if webhook_enabled else 'Disabled'}")
    logger.info(f"  - Mobile Push: {'Enabled' if mobile_push_enabled else 'Disabled'}")

    if email_enabled:
        smtp_host = os.getenv('ALERT_EMAIL_SMTP_HOST', 'Not configured')
        logger.info(f"  - SMTP Host: {smtp_host}")

    if webhook_enabled:
        webhook_url = os.getenv('ALERT_WEBHOOK_URL', 'Not configured')
        logger.info(f"  - Webhook URL: {webhook_url}")

    if mobile_push_enabled:
        mobile_push_url = os.getenv('MOBILE_PUSH_URL', 'Not configured')
        mobile_push_usecases = os.getenv('MOBILE_PUSH_USECASES', 'parking_compliance')
        logger.info(f"  - Mobile Push URL: {mobile_push_url}")
        logger.info(f"  - Mobile Push usecases: {mobile_push_usecases}")
    
    logger.info("✓ Alert service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Alert Service")


# Create FastAPI app
app = FastAPI(
    title="Alert Service",
    description="Standalone microservice for processing and sending alerts",
    version="1.0.0",
    lifespan=lifespan
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )


# Health check endpoint
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "alert",
        "version": "1.0.0"
    }


# Root endpoint
@app.get("/")
def root():
    """Root endpoint with service information"""
    return {
        "service": "Alert Service",
        "version": "1.0.0",
        "description": "Standalone microservice for processing and sending alerts",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "alert_send": "/alert/send",
            "alert_send_single": "/alert/send-single",
            "alert_list": "/alert/list"
        }
    }


# Import and include alert router
from alert.api import router as alert_router
app.include_router(alert_router)


# Main entry point
if __name__ == "__main__":
    port = int(os.getenv("ALERT_SERVICE_PORT", "8002"))
    host = os.getenv("ALERT_SERVICE_HOST", "0.0.0.0")
    
    logger.info(f"Starting Alert Service on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )
