"""
Shared common utilities package.

This package provides standalone utilities that can be used across all services:
- Logger: Centralized logging configuration
- Config: Environment-aware configuration management
- Utils: Common utility functions for frame processing and validation
"""

from .logger import setup_logger, get_logger
from .config import Config

__all__ = [
    'setup_logger',
    'get_logger',
    'Config',
]
