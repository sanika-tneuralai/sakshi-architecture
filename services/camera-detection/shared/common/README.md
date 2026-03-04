# Shared Common Utilities Package

A standalone package providing common utilities, configuration management, and logging for all GOEC services.

## Overview

This package is designed to be copied into each service (camera-detection, orchestration, usecase, alert, analytics) to provide consistent:
- **Logging**: Centralized logging configuration
- **Configuration**: Environment-aware configuration management
- **Utilities**: Common functions for frame processing and validation

## Directory Structure

```
shared/common/
├── __init__.py          # Package initialization
├── logger.py            # Logging configuration
├── config.py            # Environment-aware configuration
├── utils.py             # Common utility functions
└── README.md            # This file
```

---

## 📦 Components

### 1. Logger (`logger.py`)

Environment-aware logging configuration that supports file and console output.

#### Usage

```python
from shared.common import setup_logger, get_logger

# Setup logger with defaults (reads from ENV)
logger = setup_logger(__name__)

# Or with custom parameters
logger = setup_logger(
    name="my_service",
    log_file="logs/my_service.log",
    level="DEBUG"
)

# Get existing logger
logger = get_logger(__name__)

# Use logger
logger.info("Service started")
logger.debug("Debug information")
logger.error("Error occurred")
```

#### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `INFO` |
| `LOG_FILE` | Path to log file | `app.log` |
| `LOG_FORMAT` | Custom log format string | Standard format |

---

### 2. Config (`config.py`)

Environment-aware configuration management using a static class pattern.

#### Usage

```python
from shared.common import Config

# Get configuration values
database_url = Config.get_database_url()
api_port = Config.get_api_port()
use_gpu = Config.use_gpu()

# Get service URLs (for orchestration)
camera_url = Config.get_camera_detection_url()
usecase_url = Config.get_usecase_service_url()

# Validate and create required directories
Config.validate_paths()

# Debug: Print all configuration
Config.print_config()
```

#### Environment Variables

##### Base Directories
- `BASE_DIR`: Base application directory
- `MODELS_DIR`: Directory for model files
- `SCREENSHOTS_DIR`: Directory for screenshots

##### Camera Settings
- `DEFAULT_FPS`: Default frames per second (default: `5`)
- `MAX_FPS`: Maximum FPS allowed (default: `30`)
- `MIN_FPS`: Minimum FPS allowed (default: `1`)
- `MAX_CAMERAS_SINGLE_MODE`: Max cameras in single-stream mode (default: `50`)
- `DEFAULT_CAMERA_TIMEOUT`: Camera connection timeout in seconds (default: `30`)

##### Detection Settings
- `DEFAULT_CONFIDENCE_THRESHOLD`: Detection confidence threshold (default: `0.5`)
- `DEFAULT_IOU_THRESHOLD`: IOU threshold for NMS (default: `0.45`)
- `YOLO_MODEL_PATH`: Path to YOLO model file
- `USE_GPU`: Use GPU for inference (default: `true`)

##### API Settings
- `API_HOST`: API host address (default: `0.0.0.0`)
- `API_PORT`: API port number (default: `8000`)
- `API_RELOAD`: Enable auto-reload for development (default: `false`)

##### ROI Settings
- `ROI_COLOR`: ROI color in BGR format as comma-separated (default: `0,255,255`)
- `ROI_THICKNESS`: ROI line thickness (default: `2`)
- `ROI_FILL_ALPHA`: ROI fill transparency 0.0-1.0 (default: `0.3`)

##### Frame Processing
- `MAX_FRAME_WIDTH`: Maximum frame width (default: `1920`)
- `MAX_FRAME_HEIGHT`: Maximum frame height (default: `1080`)
- `JPEG_QUALITY`: JPEG compression quality 0-100 (default: `85`)

##### Multi-Stream (DeepStream)
- `MULTI_STREAM_BATCH_SIZE`: Batch size for multi-stream (default: `4`)
- `MULTI_STREAM_WIDTH`: Multi-stream width (default: `1280`)
- `MULTI_STREAM_HEIGHT`: Multi-stream height (default: `720`)
- `DEEPSTREAM_PATH`: Path to DeepStream installation (optional)

##### Database
- `DATABASE_URL`: PostgreSQL connection URL (format: `postgresql://user:pass@host:port/db`)

##### Service URLs (for Orchestration Service)
- `CAMERA_DETECTION_URL`: URL of camera-detection service (default: `http://localhost:8000`)
- `USECASE_SERVICE_URL`: URL of usecase service (default: `http://localhost:8001`)
- `ALERT_SERVICE_URL`: URL of alert service (default: `http://localhost:8002`)
- `ANALYTICS_SERVICE_URL`: URL of analytics service (default: `http://localhost:8003`)

---

### 3. Utils (`utils.py`)

Common utility functions for frame processing, ROI handling, and validation.

#### Usage

```python
from shared.common.utils import (
    create_roi_mask,
    apply_roi_to_frame,
    validate_rtsp_url,
    calculate_fps,
    draw_roi_on_frame,
    resize_frame,
    PerformanceMonitor
)

# ROI operations
roi_points = [[100, 100], [200, 100], [200, 200], [100, 200]]
mask = create_roi_mask(roi_points, (720, 1280))
masked_frame = apply_roi_to_frame(frame, mask)
frame_with_roi = draw_roi_on_frame(frame, roi_points)

# Validation
is_valid = validate_rtsp_url("rtsp://camera.example.com:554/stream")

# Frame processing
resized = resize_frame(frame, 640, 480, keep_aspect_ratio=True)

# Performance monitoring
monitor = PerformanceMonitor(window_size=100)
monitor.add_frame()
fps = monitor.get_fps()
stats = monitor.get_stats()
```

#### Available Functions

| Function | Description |
|----------|-------------|
| `create_roi_mask()` | Create binary mask from ROI polygon points |
| `apply_roi_to_frame()` | Apply ROI mask to frame |
| `validate_rtsp_url()` | Validate RTSP URL format |
| `calculate_fps()` | Calculate FPS from frame count and time |
| `draw_roi_on_frame()` | Draw ROI polygon on frame for visualization |
| `resize_frame()` | Resize frame with optional aspect ratio preservation |
| `preprocess_frame_for_detection()` | Preprocess frame for object detection |
| `format_timestamp()` | Format timestamp to ISO format |
| `calculate_grid_layout()` | Calculate optimal grid layout for tiling |
| `get_frame_metadata()` | Extract frame metadata |
| `validate_roi_points()` | Validate ROI points within frame boundaries |
| `log_system_info()` | Log system information for debugging |

#### PerformanceMonitor Class

Track and monitor performance metrics:

```python
monitor = PerformanceMonitor(window_size=100)

# During processing
monitor.add_frame()

# Get metrics
fps = monitor.get_fps()
uptime = monitor.get_uptime()
stats = monitor.get_stats()  # {'fps': 25.5, 'uptime_seconds': 120.3, ...}

# Reset if needed
monitor.reset()
```

---

## 🚀 Integration Guide

### For Each Service

1. **Copy this package** to your service directory:
   ```bash
   cp -r shared/ /path/to/your-service/
   ```

2. **Update imports** in your service code:
   ```python
   # Instead of:
   # from common.logger import setup_logger
   # from common.config import Config
   
   # Use:
   from shared.common import setup_logger, get_logger, Config
   from shared.common.utils import create_roi_mask, PerformanceMonitor
   ```

3. **Create `.env` file** with required environment variables:
   ```bash
   # Database
   DATABASE_URL=postgresql://postgres:password@localhost:5432/goec
   
   # API
   API_HOST=0.0.0.0
   API_PORT=8000
   
   # Logging
   LOG_LEVEL=INFO
   LOG_FILE=logs/service.log
   
   # Service-specific settings
   # ... add more as needed
   ```

4. **Initialize in your service**:
   ```python
   from shared.common import setup_logger, Config
   
   # Setup logging
   logger = setup_logger(__name__)
   
   # Validate configuration
   Config.validate_paths()
   Config.print_config()  # Optional: for debugging
   
   # Use configuration
   db_url = Config.get_database_url()
   api_port = Config.get_api_port()
   ```

---

## 📋 Example .env File

```bash
# ==============================================
# Database Configuration
# ==============================================
DATABASE_URL=postgresql://postgres:password@localhost:5432/goec

# ==============================================
# API Configuration
# ==============================================
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=false

# ==============================================
# Logging Configuration
# ==============================================
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
LOG_FORMAT=%(asctime)s - %(name)s - %(levelname)s - %(message)s

# ==============================================
# Camera Settings (for camera-detection service)
# ==============================================
DEFAULT_FPS=5
MAX_FPS=30
MAX_CAMERAS_SINGLE_MODE=50
DEFAULT_CAMERA_TIMEOUT=30

# ==============================================
# Detection Settings (for camera-detection service)
# ==============================================
DEFAULT_CONFIDENCE_THRESHOLD=0.5
DEFAULT_IOU_THRESHOLD=0.45
YOLO_MODEL_PATH=/app/models/yolo11n.pt
USE_GPU=true

# ==============================================
# DeepStream (for camera-detection service only)
# ==============================================
DEEPSTREAM_PATH=/opt/nvidia/deepstream/deepstream-6.4/lib
MULTI_STREAM_BATCH_SIZE=4

# ==============================================
# Service URLs (for orchestration service only)
# ==============================================
CAMERA_DETECTION_URL=http://server1:8000
USECASE_SERVICE_URL=http://server3:8001
ALERT_SERVICE_URL=http://server3:8002
ANALYTICS_SERVICE_URL=http://server3:8003
```

---

## 🔧 Dependencies

This package requires:
- `numpy`
- `opencv-python` (cv2)
- `psutil` (optional, for system info logging)

Add to your service's `requirements.txt`:
```
numpy>=1.24.0
opencv-python>=4.8.0
psutil>=5.9.0
```

---

## 🧪 Testing

Test the package in your service:

```python
# test_common_package.py
from shared.common import setup_logger, Config
from shared.common.utils import validate_rtsp_url, PerformanceMonitor

def test_logger():
    logger = setup_logger("test")
    logger.info("Logger works!")
    print("✓ Logger test passed")

def test_config():
    Config.print_config()
    assert Config.get_api_port() > 0
    print("✓ Config test passed")

def test_utils():
    assert validate_rtsp_url("rtsp://test.com/stream") == True
    assert validate_rtsp_url("http://test.com") == False
    
    monitor = PerformanceMonitor()
    monitor.add_frame()
    stats = monitor.get_stats()
    assert 'fps' in stats
    print("✓ Utils test passed")

if __name__ == "__main__":
    test_logger()
    test_config()
    test_utils()
    print("\n✅ All tests passed!")
```

---

## 📝 Notes

### Service-Specific Variables

Not all services need all environment variables. For example:

- **Camera-Detection**: Needs `YOLO_MODEL_PATH`, `DEEPSTREAM_PATH`, `USE_GPU`
- **Orchestration**: Needs `*_SERVICE_URL` variables
- **Usecase/Alert/Analytics**: Only need `DATABASE_URL`, `API_PORT`, logging configs

### Updating Shared Code

When updating this shared package:
1. Update in one place (e.g., `backend/shared/common/`)
2. Copy to all service directories
3. Test each service independently
4. Commit to respective service branches

### Best Practices

1. **Always use environment variables** instead of hardcoded configs
2. **Set sensible defaults** in Config class methods
3. **Validate configuration** on service startup using `Config.validate_paths()`
4. **Use type hints** for better IDE support
5. **Log appropriately**: DEBUG for details, INFO for normal flow, ERROR for issues

---

## 🆘 Troubleshooting

### Import Errors

If you get import errors:
```python
# Make sure the path is correct
from shared.common import setup_logger  # ✓ Correct
from common import setup_logger         # ✗ Wrong (old import)
```

### Configuration Not Applied

Make sure environment variables are loaded:
```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load .env file
from shared.common import Config
```

### Logger Not Writing to File

Check file permissions and directory exists:
```python
import os
log_dir = os.path.dirname(Config.get_log_file())
os.makedirs(log_dir, exist_ok=True)
```

---

## 📚 Additional Resources

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Environment Variables Best Practices](https://12factor.net/config)

---

**Version**: 1.0.0  
**Last Updated**: March 2026  
**Maintainer**: GOEC Team
