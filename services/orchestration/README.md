# Orchestration Service 🎯

Pipeline controller for coordinating the complete detection pipeline across distributed services.

## Overview

The Orchestration Service acts as the central coordinator that orchestrates the complete detection pipeline by making HTTP calls to independent services:

- **Camera-Detection Service**: Frame extraction and object detection
- **Usecase Service**: Usecase evaluation and rule processing  
- **Alert Service**: Alert generation and notifications

**Key Principle**: This service contains **NO** camera, detection, or usecase logic. It purely orchestrates HTTP calls between services.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Orchestration Service                  │
│                  (Pipeline Controller)                  │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Camera-    │  │   Usecase    │  │    Alert     │
│  Detection   │  │   Service    │  │   Service    │
│   Service    │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
```

## Pipeline Flow

1. **Camera Frame Extraction** → GET `/camera/frame/{camera_id}`
2. **Object Detection** → POST `/detection/detect`
3. **Usecase Evaluation** → POST `/usecase/evaluate`
4. **Alert Generation** → POST `/alert/send`

## Features

✅ **Pipeline Orchestration**: Coordinates complete detection pipeline  
✅ **Error Handling**: Robust error handling with automatic retries  
✅ **Service Communication**: HTTP-based inter-service communication  
✅ **Configurable**: Service URLs configured via environment variables  
✅ **Health Checks**: Monitor connectivity to all dependent services  
✅ **Logging**: Detailed logging of pipeline execution  
✅ **Lightweight**: Minimal dependencies (FastAPI + requests)

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- Access to Camera-Detection, Usecase, and Alert services

### 1. Local Development

```bash
# Clone/navigate to service directory
cd services/orchestration

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Edit .env with your service URLs
nano .env

# Run the service
python main.py
```

The service will start on `http://localhost:8000`

### 2. Docker Deployment

```bash
# Build the image
docker build -t orchestration-service .

# Run with docker-compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop service
docker-compose down
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `CAMERA_DETECTION_URL` | Camera-Detection service URL | `http://localhost:8001` | ✅ Yes |
| `USECASE_SERVICE_URL` | Usecase service URL | `http://localhost:8002` | ✅ Yes |
| `ALERT_SERVICE_URL` | Alert service URL | `http://localhost:8003` | ✅ Yes |
| `REQUEST_TIMEOUT` | HTTP request timeout (seconds) | `30` | No |
| `MAX_RETRIES` | Max retry attempts for failed requests | `3` | No |
| `RETRY_BACKOFF_FACTOR` | Backoff factor for retries | `0.5` | No |
| `PORT` | Service port | `8000` | No |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `DATABASE_URL` | Database connection (optional) | - | No |

### Deployment Scenarios

#### Scenario 1: Local Development (Single Machine)
```bash
CAMERA_DETECTION_URL=http://localhost:8001
USECASE_SERVICE_URL=http://localhost:8002
ALERT_SERVICE_URL=http://localhost:8003
```

#### Scenario 2: Docker Compose (Container Names)
```bash
CAMERA_DETECTION_URL=http://camera-detection:8001
USECASE_SERVICE_URL=http://usecase:8002
ALERT_SERVICE_URL=http://alert:8003
```

#### Scenario 3: Multi-Server Production
```bash
# Server 1: Camera-Detection (GPU server)
CAMERA_DETECTION_URL=http://192.168.1.100:8001

# Server 3: Usecase + Alert (Business logic server)
USECASE_SERVICE_URL=http://192.168.1.102:8002
ALERT_SERVICE_URL=http://192.168.1.102:8003
```

## API Endpoints

### Execute Pipeline

**POST** `/pipeline/execute`

Execute the complete detection pipeline.

**Request Body:**
```json
{
  "camera_id": "s1_cam_1",
  "confidence_threshold": 0.5,
  "usecases": ["person_in_roi", "crowd_in_roi"]
}
```

**Response:**
```json
{
  "status": "success",
  "camera_id": "s1_cam_1",
  "pipeline_results": {
    "camera": {
      "status": "success",
      "backend": "deepstream"
    },
    "detection": {
      "total_detections": 5,
      "roi_detections": 3,
      "processing_time_ms": 45.2
    },
    "usecases": {
      "evaluated": 2,
      "triggered": 1,
      "results": [...]
    },
    "alerts": {
      "sent": 1,
      "details": [...]
    }
  }
}
```

**Example cURL:**
```bash
curl -X POST http://localhost:8000/pipeline/execute \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "s1_cam_1",
    "confidence_threshold": 0.5,
    "usecases": ["person_in_roi", "crowd_in_roi"]
  }'
```

### Health Check

**GET** `/health`

Check service health and connectivity to dependent services.

**Response:**
```json
{
  "status": "healthy",
  "service": "orchestration",
  "version": "1.0.0",
  "services": {
    "camera_detection": {
      "status": "healthy",
      "url": "http://localhost:8001"
    },
    "usecase": {
      "status": "healthy",
      "url": "http://localhost:8002"
    },
    "alert": {
      "status": "healthy",
      "url": "http://localhost:8003"
    }
  }
}
```

### Root

**GET** `/`

API information and quick links.

## Error Handling

The orchestration service implements comprehensive error handling:

### Automatic Retries

Failed requests are automatically retried with exponential backoff:
- **Max Retries**: 3 (configurable)
- **Retry on Status Codes**: 429, 500, 502, 503, 504
- **Backoff**: 0.5s, 1s, 2s (configurable)

### Error Types

| Error | Status Code | Description |
|-------|-------------|-------------|
| Service Unreachable | 503 | Cannot connect to service |
| Service Timeout | 504 | Request timeout exceeded |
| Service Error | Varies | Service returned error response |
| Pipeline Error | 500 | Unexpected error during pipeline execution |

### Non-Critical Failures

Alert API failures are logged as warnings but don't fail the entire pipeline, as alerts are considered non-critical.

## Logging

The service provides detailed logging at each pipeline stage:

```
[ORCHESTRATOR] PIPELINE EXECUTION STARTED
[ORCHESTRATOR] Camera ID: s1_cam_1
[ORCHESTRATOR] STEP 1/4: Calling Camera API
[ORCHESTRATOR] Camera API Success
[ORCHESTRATOR] STEP 2/4: Calling Detection API
[ORCHESTRATOR] Detection API Success
  - Total detections: 5
  - ROI detections: 3
[ORCHESTRATOR] STEP 3/4: Calling Usecase API
[ORCHESTRATOR] Usecase API Success
  - person_in_roi: ✓ TRIGGERED
  - crowd_in_roi: ✗ Not triggered
[ORCHESTRATOR] STEP 4/4: Calling Alert API
[ORCHESTRATOR] Alert API Success
[ORCHESTRATOR] PIPELINE EXECUTION COMPLETED
```

Logs are written to:
- **Console**: stdout
- **File**: `orchestration.log`

## Testing

### Test Pipeline Execution

```bash
# Test with default usecases
curl -X POST http://localhost:8000/pipeline/execute \
  -H "Content-Type: application/json" \
  -d '{"camera_id": "s1_cam_1"}'

# Test with specific usecases
curl -X POST http://localhost:8000/pipeline/execute \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "s1_cam_1",
    "confidence_threshold": 0.6,
    "usecases": ["person_in_roi", "restricted_zone_breach"]
  }'
```

### Test Health Check

```bash
curl http://localhost:8000/health
```

## Deployment

### Server 2 Deployment (Recommended)

The orchestration service is recommended to run on its own server (Server 2):

```
Server 1: Camera-Detection Service (GPU-intensive)
Server 2: Orchestration Service (Lightweight, coordinates all)
Server 3: Usecase + Alert + Analytics Services (Business logic)
```

### Production Checklist

- [ ] Configure all service URLs in `.env`
- [ ] Set appropriate timeouts and retry settings
- [ ] Configure logging level (`INFO` for production)
- [ ] Enable health check endpoints
- [ ] Set up monitoring/alerting
- [ ] Configure database for pipeline logging (optional)
- [ ] Test connectivity to all services
- [ ] Set up backup/redundancy

## Monitoring

### Health Check Endpoint

Monitor service health via `/health` endpoint:

```bash
# Check every 30 seconds
watch -n 30 curl http://localhost:8000/health
```

### Docker Health Check

The Docker container includes built-in health checks:

```bash
# Check container health
docker ps

# View health check logs
docker inspect orchestration-service | grep Health
```

## Troubleshooting

### Issue: Service Connection Failed

**Error**: `Service connection failed: Connection refused`

**Solutions**:
1. Verify service URLs in `.env` are correct
2. Check that dependent services are running
3. Test connectivity: `curl http://<service-url>/health`
4. Check firewall/network settings

### Issue: Request Timeout

**Error**: `Service timeout: Read timed out`

**Solutions**:
1. Increase `REQUEST_TIMEOUT` in `.env`
2. Check network latency between services
3. Verify dependent services are responding

### Issue: Service Returning Errors

**Error**: Various HTTP error codes from services

**Solutions**:
1. Check logs of the failing service
2. Verify request payload format
3. Test service endpoint directly
4. Check service health: `curl http://<service-url>/health`

## Project Structure

```
services/orchestration/
├── main.py                 # Orchestration logic
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container image
├── docker-compose.yml     # Docker Compose config
├── .env.example           # Environment template
├── README.md              # This file
└── shared/                # Shared components
    ├── database/          # Database models
    └── common/            # Common utilities
```

## Dependencies

Minimal dependencies for lightweight orchestration:

- **FastAPI**: Web framework
- **Uvicorn**: ASGI server
- **Requests**: HTTP client
- **urllib3**: Retry logic
- **SQLAlchemy**: Database (optional)
- **Pydantic**: Data validation

## Development

### Running Tests

```bash
# Test import and syntax
python -c "import main; print('✓ Import successful')"

# Test service startup
python main.py &
sleep 5
curl http://localhost:8000/health
kill %1
```

### Adding New Pipeline Steps

To add new steps to the pipeline:

1. Add new service URL to `.env.example`
2. Update `main.py` lifespan to log new service
3. Add call to new service in `execute_pipeline()`
4. Update error handling
5. Update response schema
6. Update documentation

## License

Part of the GOEC (Graphical Object Edge Computing) project.

## Support

For issues or questions:
- Check the main project documentation
- Review logs in `orchestration.log`
- Check health status of all services
- Verify environment configuration

---

**Version**: 1.0.0  
**Last Updated**: 2026-03-03
