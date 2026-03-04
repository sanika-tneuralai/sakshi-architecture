"""
Shared Database Package

This package provides standalone database models and connection utilities
that can be copied to all service branches.

Main Components:
    - connection: Database connection management (engine, session, init_db)
    - models: All database models (Camera, Detection, UsecaseResult, Alert, AnalyticsDaily)

Quick Start:
    1. Set DATABASE_URL environment variable
    2. Import models and connection utilities
    3. Initialize database with init_db()
    
Example Usage:
    
    # In your service's main.py or startup code:
    from shared.database.connection import init_db, get_db
    from shared.database.models import Camera, Detection
    
    # Initialize database tables
    init_db()
    
    # Use in FastAPI endpoints
    from fastapi import Depends
    from sqlalchemy.orm import Session
    
    @app.get("/cameras")
    def get_cameras(db: Session = Depends(get_db)):
        return db.query(Camera).all()

Service-Specific Imports:
    
    Camera-Detection Service:
        from shared.database.models import Camera, Detection
    
    Usecase Service:
        from shared.database.models import Detection, UsecaseResult
    
    Alert Service:
        from shared.database.models import Alert, UsecaseResult
    
    Analytics Service:
        from shared.database.models import Detection, Alert, AnalyticsDaily

Environment Variables:
    DATABASE_URL: PostgreSQL connection string
        Format: postgresql://user:password@host:port/database
        Default: postgresql://postgres:postgres@localhost:5432/goec
"""

# Import connection utilities
from shared.database.connection import (
    engine,
    SessionLocal,
    Base,
    get_db,
    init_db,
    test_connection,
    DATABASE_URL
)

# Import all models
from shared.database.models import (
    Camera,
    Detection,
    UsecaseResult,
    Alert,
    AnalyticsDaily
)

# Define what gets exported when using "from shared.database import *"
__all__ = [
    # Connection utilities
    'engine',
    'SessionLocal',
    'Base',
    'get_db',
    'init_db',
    'test_connection',
    'DATABASE_URL',
    
    # Models
    'Camera',
    'Detection',
    'UsecaseResult',
    'Alert',
    'AnalyticsDaily',
]

# Package metadata
__version__ = '1.0.0'
__author__ = 'GOEC Team'
__description__ = 'Shared database models and connection utilities for GOEC services'
