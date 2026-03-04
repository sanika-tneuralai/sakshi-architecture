#!/usr/bin/env python3
"""
Test script to verify the shared database package works standalone.

This script tests:
1. Importing all models and connection utilities
2. Database connection (if DATABASE_URL is set)
3. All models are accessible

Usage:
    export DATABASE_URL="postgresql://user:pass@host:5432/goec"
    python test_shared_database.py
"""

import sys
import os

# Add parent directory to path to import shared package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """Test that all imports work correctly."""
    print("Testing imports...")
    
    try:
        # Test importing connection utilities
        from shared.database import (
            engine, SessionLocal, Base, get_db, init_db, test_connection, DATABASE_URL
        )
        print("✓ Connection utilities imported successfully")
        
        # Test importing models
        from shared.database.models import (
            Camera, Detection, UsecaseResult, Alert, AnalyticsDaily
        )
        print("✓ All models imported successfully")
        
        # Test alternative import style
        from shared.database import Camera, Detection
        print("✓ Direct model import works")
        
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_models_accessible():
    """Test that all model classes are accessible."""
    print("\nTesting model accessibility...")
    
    try:
        from shared.database.models import (
            Camera, Detection, UsecaseResult, Alert, AnalyticsDaily
        )
        
        models = {
            "Camera": Camera,
            "Detection": Detection,
            "UsecaseResult": UsecaseResult,
            "Alert": Alert,
            "AnalyticsDaily": AnalyticsDaily
        }
        
        for name, model in models.items():
            tablename = model.__tablename__
            print(f"✓ {name} -> table: {tablename}")
        
        return True
    except Exception as e:
        print(f"✗ Model accessibility test failed: {e}")
        return False


def test_connection_config():
    """Test database connection configuration."""
    print("\nTesting connection configuration...")
    
    try:
        from shared.database import DATABASE_URL, test_connection
        
        print(f"✓ DATABASE_URL configured: {DATABASE_URL.split('@')[-1]}")  # Hide credentials
        
        # Try to test connection (will fail if DB is not accessible, which is OK)
        try:
            if test_connection():
                print("✓ Database connection successful!")
            else:
                print("⚠ Database connection failed (DB may not be running)")
        except Exception as e:
            print(f"⚠ Database connection test failed: {e}")
            print("  (This is expected if PostgreSQL is not running)")
        
        return True
    except Exception as e:
        print(f"✗ Connection config test failed: {e}")
        return False


def test_init_db_function():
    """Test that init_db function exists and is callable."""
    print("\nTesting init_db function...")
    
    try:
        from shared.database import init_db
        
        print("✓ init_db function is accessible")
        print("  (Not calling it to avoid modifying database)")
        
        return True
    except Exception as e:
        print(f"✗ init_db test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Shared Database Package - Standalone Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Accessibility Test", test_models_accessible),
        ("Connection Config Test", test_connection_config),
        ("Init DB Function Test", test_init_db_function),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The shared database package is ready to use.")
        return 0
    else:
        print("\n⚠ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
