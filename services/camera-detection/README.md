# Camera-Detection Service

## Overview

The Camera-Detection Service is a standalone microservice that provides RTSP camera stream management and real-time object detection capabilities using NVIDIA DeepStream and YOLO models.

**This service is designed to be called by an orchestrator** and does NOT include usecase evaluation, alerting, or analytics logic.

### Key Features

- **RTSP Camera Management**: Handle multiple RTSP camera streams
- **ROI Support**: Define regions of interest for focused detection
- **Object Detection**: YOLO-based real-time detection with GPU acceleration
- **Two Operation Modes**:
  - **Single Camera Mode**: Best for < 10 cameras
  - **Multi-Stream Mode**: Optimized for 100+ cameras with batched processing
- **Hardware Acceleration**: NVIDIA DeepStream for optimal performance
- **Database Persistence**: Store camera configurations and detection results
- **REST API**: Easy integration with orchestrator services

---

## Architecture

```
┌─────────────────────────────────────────┐
│   Camera-Detection Service              │
├─────────────────────────────────────────┤
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Camera Management                │ │
│  │  - RTSP Stream Handling           │ │
│  │  - ROI Configuration              │ │
│  │  - Frame Extraction               │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Object Detection                 │ │
│  │  - YOLO Model Inference           │ │
│  │  - ROI Filtering                  │ │
│  │  - Bounding Box Generation        │ │
│  └───────────────────────────────────┘ │
│                                         │
│  ┌───────────────────────────────────┐ │
│  │  Shared Components                │ │
│  │  - Database Models                │ │
│  │  - Configuration                  │ │
│  │  - Logging Utilities              │ │
│  └───────────────────────────────────┘ │
│                                         │
└─────────────────────────────────────────┘
           │                    │
           ▼                    ▼
    PostgreSQL DB         RTSP Cameras
```

---

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with compute capability 6.0+ (required for DeepStream)
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 10GB+ for models and screenshots

### Software Requirements

- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **NVIDIA Docker Runtime**: For GPU support
- **NVIDIA DeepStream SDK**: 6.4+ (included in base image)

---

## Installation

### 1. Clone the Repository

```bash
cd services/camera-detection
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit configuration
nano .env
```

**Key configurations:**

```env
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/goec
DEEPSTREAM_PATH=/opt/nvidia/deepstream/deepstream-6.4/lib
YOLO_MODEL_PATH=/app/models/yolo11n.pt
DEFAULT_CONFIDENCE_THRESHOLD=0.5
```

### 3. Place YOLO Models

```bash
# Ensure YOLO model is in the models directory
ls -lh models/
# Should see: yolo11n.pt (or your model file)
```

### 4. Verify NVIDIA Docker Runtime

```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

---

## Deployment

### Option 1: Docker Compose (Recommended)

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f camera-detection

# Check health
curl http://localhost:8000/health
```

### Option 2: Docker Build & Run

```bash
# Build image
docker build -t camera-detection:latest .

# Run container
docker run -d \
  --name camera-detection \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/screenshots:/app/screenshots \
  --env-file .env \
  camera-detection:latest
```

### Option 3: Local Development

```bash
# Install dependencies
pip install -r requirements.txt
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Set environment variables
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/goec"
export DEEPSTREAM_PATH="/opt/nvidia/deepstream/deepstream-6.4/lib"
export YOLO_MODEL_PATH="./models/yolo11n.pt"

# Run service
python main.py
```

---

## API Documentation

Once deployed, access interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Health Check
```bash
GET /health
```

#### Camera Management

**Start Single Camera:**
```bash
POST /camera/start
Content-Type: application/json

{
  "camera_id": "queue_cam_1",
  "rtsp_url": "rtsp://admin:admin@192.168.1.100:554/stream",
  "fps": 5,
  "roi_points": [[100, 100], [500, 100], [500, 400], [100, 400]]
}
```

**Get Camera Status:**
```bash
GET /camera/status/{camera_id}
```

**Get Current Frame:**
```bash
GET /camera/frame/{camera_id}
```

**Stop Camera:**
```bash
DELETE /camera/stop/{camera_id}
```

#### Object Detection

**Run Detection:**
```bash
POST /detection/detect
Content-Type: application/json

{
  "camera_id": "queue_cam_1",
  "confidence": 0.5,
  "classes": ["person"],
  "save_screenshot": true
}
```

**Get Recent Detections:**
```bash
GET /detection/recent?camera_id=queue_cam_1&limit=10
```

---

## Configuration

### Camera Modes

#### Single Camera Mode
- Best for: < 10 cameras
- Lower latency per camera
- Independent processing
- Use: `/camera/start` endpoint

#### Multi-Stream Mode
- Best for: 100+ cameras
- Batched GPU processing
- Optimal resource utilization
- Use: `/camera/start-multi` endpoint

### Detection Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| confidence | 0.5 | 0.0-1.0 | Minimum confidence for detections |
| iou_threshold | 0.45 | 0.0-1.0 | IoU threshold for NMS |
| fps | 5 | 1-30 | Frame extraction rate |

### ROI (Region of Interest)

Define custom regions for detection:

```json
{
  "roi_points": [
    [x1, y1],
    [x2, y2],
    [x3, y3],
    [x4, y4]
  ]
}
```

- Points define a polygon
- Only detections within this region are returned
- Coordinates are in pixels (image space)

---

## Database Schema

### Tables Used

- **cameras**: Camera configurations
- **detections**: Detection results with bounding boxes

### Models

Shared database models are in `shared/database/models.py`:

- `Camera`: RTSP camera configuration
- `Detection`: Object detection results
- `BoundingBox`: Detection bounding boxes

---

## Monitoring

### Health Checks

```bash
# Service health
curl http://localhost:8000/health

# Container health
docker ps | grep camera-detection
```

### Logs

```bash
# Docker Compose
docker-compose logs -f camera-detection

# Docker
docker logs -f camera-detection

# Log files
tail -f logs/camera_detection_api.log
```

### Metrics

Monitor:
- GPU utilization: `nvidia-smi`
- Memory usage: `docker stats camera-detection`
- API latency: Check `/docs` endpoint timings
- Database connections: PostgreSQL logs

---

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected

```bash
# Verify NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Check Docker daemon.json
cat /etc/docker/daemon.json
# Should have: "default-runtime": "nvidia"
```

#### 2. DeepStream Import Errors

```bash
# Verify DeepStream path
docker exec camera-detection ls -la /opt/nvidia/deepstream/deepstream-6.4/lib

# Check environment variable
docker exec camera-detection printenv DEEPSTREAM_PATH
```

#### 3. Database Connection Failed

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Test connection
docker exec camera-detection pg_isready -h postgres -U postgres
```

#### 4. RTSP Stream Issues

```bash
# Test RTSP URL manually
ffplay "rtsp://admin:admin@192.168.1.100:554/stream"

# Check camera logs
docker-compose logs camera-detection | grep -i "rtsp"
```

#### 5. Model Not Found

```bash
# Verify model file exists
docker exec camera-detection ls -lh /app/models/

# Check model path in .env
grep YOLO_MODEL_PATH .env
```

---

## Performance Tuning

### For < 10 Cameras

```env
# Use single camera mode
# Lower FPS for reduced CPU load
DEFAULT_CAMERA_FPS=5
WORKER_THREADS=2
```

### For 100+ Cameras

```env
# Use multi-stream mode
# Batch processing on GPU
DEFAULT_CAMERA_FPS=3
WORKER_THREADS=8
MAX_QUEUE_SIZE=200
```

### GPU Optimization

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Adjust batch size in DeepStream config
# See camera/streams/multi_stream.py
```

---

## Integration with Orchestrator

This service is designed to be called by an orchestrator service:

```python
# Example orchestrator call
import requests

# Start camera
response = requests.post(
    "http://camera-detection:8000/camera/start",
    json={
        "camera_id": "cam1",
        "rtsp_url": "rtsp://...",
        "fps": 5
    }
)

# Run detection
response = requests.post(
    "http://camera-detection:8000/detection/detect",
    json={
        "camera_id": "cam1",
        "confidence": 0.5
    }
)

detections = response.json()["detections"]
```

---

## Development

### Running Tests

```bash
# TODO: Add test suite
# pytest tests/
```

### Adding New Detection Classes

Edit YOLO model or update detection service:

```python
# detection/service.py
ALLOWED_CLASSES = ["person", "car", "truck", "bus"]
```

### Modifying ROI Logic

See `detection/roi.py` for ROI filtering implementation.

---

## Deployment Checklist

- [ ] NVIDIA GPU available
- [ ] Docker and NVIDIA Docker runtime installed
- [ ] `.env` file configured
- [ ] YOLO model file in `models/` directory
- [ ] PostgreSQL database accessible
- [ ] RTSP cameras accessible from server
- [ ] Firewall allows port 8000
- [ ] GPU drivers and CUDA installed
- [ ] DeepStream SDK available (via Docker image)

---

## Security Considerations

1. **Database Credentials**: Use strong passwords, never commit `.env`
2. **RTSP URLs**: Secure camera authentication credentials
3. **API Access**: Add authentication middleware for production
4. **CORS**: Configure `CORS_ORIGINS` for specific domains
5. **Network**: Use private networks for camera streams

---

## Maintenance

### Updating Models

```bash
# Place new model in models/
cp yolo11m.pt models/

# Update .env
YOLO_MODEL_PATH=/app/models/yolo11m.pt

# Restart service
docker-compose restart camera-detection
```

### Database Migrations

```bash
# TODO: Add Alembic migrations
# alembic upgrade head
```

### Logs Rotation

```bash
# Configure log rotation in docker-compose.yml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

---

## Support & Contact

For issues or questions:
- Check logs: `docker-compose logs camera-detection`
- Review API docs: http://localhost:8000/docs
- See main documentation: [../../docs/](../../docs/)

---

## License

[Your License Here]

---

## Version History

- **v1.0.0** (2024-03-03): Initial standalone service release
  - Camera management API
  - Object detection API
  - DeepStream integration
  - Database persistence
