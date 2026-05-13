"""
Shared Database Connection Management

This module provides database connection utilities that can be used across all services.
Configure the DATABASE_URL environment variable to connect to your PostgreSQL database.

Environment Variables:
    DATABASE_URL: PostgreSQL connection string (default: postgresql://postgres:postgres@localhost:5432/goec)

Example:
    export DATABASE_URL=postgresql://user:password@hostname:5432/dbname
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool

# Database URL from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/goec")

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
    echo=False
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """
    Get database session.

    Usage:
        from shared.database.connection import get_db

        # In FastAPI:
        @app.get("/items")
        def read_items(db: Session = Depends(get_db)):
            return db.query(Item).all()

    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database tables.

    Creates all tables defined in the models (checkfirst=True skips existing tables).
    Should be called once during application startup.

    Raises:
        Exception: If database connection or table creation fails
    """
    from shared.database.models import (
        Camera, Detection, UsecaseResult, Alert, AnalyticsDaily,
        ROIConfig, CameraUsecase, ChargingSession,
    )

    print(f"[DATABASE] Initializing database...")
    print(f"[DATABASE] Database URL: {DATABASE_URL.split('@')[-1]}")  # Hide credentials

    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        print(f"[DATABASE] Tables created successfully")

        with engine.connect() as conn:
            print(f"[DATABASE] Connection test successful")
    except Exception as e:
        print(f"[DATABASE] Error initializing database: {str(e)}")
        raise


def ensure_schema():
    """Apply idempotent column additions on an already-deployed schema.

    `Base.metadata.create_all` only creates *missing tables*; it never alters
    existing ones. Production already has the `camera` table populated, so new
    columns added to the Camera model (rtsp_url, fps) have to be applied with
    explicit ALTER TABLE statements. Postgres' `ADD COLUMN IF NOT EXISTS` keeps
    this safe to run on every startup.

    Add new ALTERs to the list below as the schema evolves; do not remove old
    ones — they're cheap on a column that already exists.
    """
    from sqlalchemy import text
    statements = [
        "ALTER TABLE camera ADD COLUMN IF NOT EXISTS rtsp_url TEXT",
        "ALTER TABLE camera ADD COLUMN IF NOT EXISTS fps INTEGER NOT NULL DEFAULT 5",
    ]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def test_connection():
    """
    Test database connection.

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"[DATABASE] Connection test failed: {str(e)}")
        return False
