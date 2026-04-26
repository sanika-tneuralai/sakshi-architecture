"""
Usecase Evaluation Service
A standalone microservice for evaluating business rules against detection data.
"""
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
# Configure the root logger once so every `logging.getLogger(__name__)` in
# the codebase inherits the same handlers. Three sinks:
#   - stdout                     INFO+   operational view (systemd / docker logs)
#   - logs/usecase_service.log   INFO+   bounded operational history
#   - logs/usecase_debug.log     DEBUG+  full state-machine transitions for
#                                        post-mortem ("where did it break?")
#
# Format includes file:line so a log line tells you exactly which rule
# emitted it (e.g. gun_detection.py:204 vs parking_detection.py:160) —
# that's the single biggest payoff per character when chasing bugs.
#
# Both files use RotatingFileHandler so a long-running container can't
# blow up the disk. Limits are conservative (10 MB × 5 backups = ~50 MB
# per file ceiling, ~100 MB total for the service).
# ---------------------------------------------------------------------------
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

_LOG_FORMAT = "%(asctime)s %(levelname)-7s %(filename)s:%(lineno)d - %(message)s"
_LOG_FORMATTER = logging.Formatter(_LOG_FORMAT)

_root = logging.getLogger()
_root.setLevel(logging.DEBUG)         # root captures everything; handlers filter
# Drop any handlers a previous import attached (uvicorn reloads, tests, etc.)
for h in list(_root.handlers):
    _root.removeHandler(h)

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(_LOG_FORMATTER)
_root.addHandler(_console)

_info_file = RotatingFileHandler(
    LOG_DIR / "usecase_service.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_info_file.setLevel(logging.INFO)
_info_file.setFormatter(_LOG_FORMATTER)
_root.addHandler(_info_file)

_debug_file = RotatingFileHandler(
    LOG_DIR / "usecase_debug.log",
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8",
)
_debug_file.setLevel(logging.DEBUG)
_debug_file.setFormatter(_LOG_FORMATTER)
_root.addHandler(_debug_file)

# Quiet down third-party libraries that flood DEBUG (otherwise the debug
# file is unreadable). Add more here if needed.
for noisy in ("urllib3", "httpx", "httpcore", "PIL", "matplotlib"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.info("Logging configured: dir=%s (info+debug, rotating)", LOG_DIR.resolve())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Import here to avoid circular imports
    from shared.database.connection import init_db
    
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Usecase Evaluation Service")
    logger.info("=" * 60)
    
    # Initialize database (optional — only if DATABASE_URL is configured)
    if os.getenv("DATABASE_URL"):
        try:
            init_db()
            logger.info("✓ Database initialized")
        except Exception as e:
            logger.error(f"✗ Database initialization failed: {str(e)}")
    else:
        logger.info("⚠ DATABASE_URL not set — skipping database initialization")
    
    logger.info("✓ Usecase service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Usecase Evaluation Service")
    logger.info("✓ Usecase service shut down successfully")


# Create FastAPI application
app = FastAPI(
    title="Usecase Evaluation Service",
    description="""
    ## Usecase Evaluation Microservice
    
    ### Purpose:
    Evaluates business rules (usecases) against object detection data.
    
    ### Features:
    - **Multi-Usecase Evaluation**: Evaluate multiple business rules in a single request
    - **Rule-Based Logic**: Flexible rule engine for custom detection scenarios
    - **Database Integration**: Store evaluation results for analytics
    - **Standalone Service**: Independent deployment, no camera/detection dependencies
    
    ### Available Usecases:
    Usecases are auto-discovered at startup from the `usecase/rules/` directory.
    Call `GET /usecase/list` to see all currently registered usecase IDs.

    ### How It Works:
    1. Receives detection output from detection service
    2. Evaluates requested usecase rules against detection data
    3. Stores results in database
    4. Returns evaluation results
    
    ### Example Flow:
    ```
    Detection Service → Usecase Service → Results
                              ↓
                        Database Storage
    ```
    
    ### Quick Start:
    1. Ensure database is configured via DATABASE_URL env variable
    2. Send detection output: `POST /usecase/evaluate`
    3. Check evaluation results in response
    
    ### Notes:
    - This service does NOT trigger alerts (that's the alert service's job)
    - This service does NOT perform detection (receives detection data)
    - Each usecase is evaluated independently
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
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "usecase-evaluation",
        "version": "1.0.0"
    }


# Import and include usecase router
from usecase.api import router as usecase_router
app.include_router(usecase_router)


if __name__ == "__main__":
    # Get configuration from environment variables
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8001))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )
