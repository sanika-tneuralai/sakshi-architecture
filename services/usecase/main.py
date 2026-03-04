"""
Usecase Evaluation Service
A standalone microservice for evaluating business rules against detection data.
"""
import sys
import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('usecase_service.log')
    ]
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Import here to avoid circular imports
    from shared.database.connection import init_db
    
    # Startup
    logger.info("=" * 60)
    logger.info("Starting Usecase Evaluation Service")
    logger.info("=" * 60)
    
    # Initialize database
    try:
        init_db()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {str(e)}")
    
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
    - **person_in_roi**: Triggers when any person is detected in ROI
    - **crowd_in_roi**: Triggers when 3+ persons are detected in ROI 
    - **restricted_zone_breach**: Triggers when any vehicle is detected in ROI
    
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
