"""
Analytics Service - Standalone API for analytics and reporting.

This service provides:
- Scheduled daily aggregation of analytics data
- API endpoints for daily and monthly analytics reports
- Detection and alert analytics
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from analytics.api import router as analytics_router
from analytics.scheduler import start_scheduler, stop_scheduler
from shared.common.logger import setup_logger

# Setup logger
logger = setup_logger("analytics-service")


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
