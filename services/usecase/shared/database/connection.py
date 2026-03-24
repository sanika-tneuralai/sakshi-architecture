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
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Base class for models
Base = declarative_base()

# Only create engine if DATABASE_URL is provided
engine = None
SessionLocal = None

if DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        echo=False
    )
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


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
    
    This will create all tables defined in the models.
    Should be called once during application startup.
    
    Usage:
        from shared.database.connection import init_db
        
        if __name__ == "__main__":
            init_db()
    
    Raises:
        Exception: If database connection or table creation fails
    """
    # Import models to register them with Base.metadata
    from shared.database.models import Camera, Detection, UsecaseResult, Alert, AnalyticsDaily
    
    print(f"[DATABASE] Initializing database...")
    print(f"[DATABASE] Database URL: {DATABASE_URL.split('@')[-1]}")  # Hide credentials
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print(f"[DATABASE] Tables created successfully")
        
        # Test connection
        with engine.connect() as conn:
            print(f"[DATABASE] Connection test successful")
    except Exception as e:
        print(f"[DATABASE] Error initializing database: {str(e)}")
        raise


def test_connection():
    """
    Test database connection.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        print(f"[DATABASE] Connection test failed: {str(e)}")
        return False
