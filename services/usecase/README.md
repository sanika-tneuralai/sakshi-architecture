# Usecase Evaluation Service

A standalone microservice for evaluating business rules (usecases) against object detection data.

## 📋 Overview

The Usecase Evaluation Service is part of the GOEC (General Operations Edge Computing) modularized architecture. It receives detection data and evaluates custom business rules to determine if specific conditions are met.

### Key Features

- **Multi-Usecase Evaluation**: Evaluate multiple business rules in a single request
- **Rule-Based Architecture**: Flexible, extensible rule engine
- **Database Integration**: Store evaluation results for analytics and audit
- **Standalone Deployment**: No dependencies on camera or detection services
- **RESTful API**: Simple HTTP interface for integration
- **Lightweight**: Minimal resource footprint (CPU/RAM only, no GPU required)

## 🏗️ Architecture

```
┌──────────────────┐
│ Detection Service│
└────────┬─────────┘
         │ Detection Data
         ▼
┌─────────────────────┐
│ Usecase Service     │
│ ┌─────────────────┐ │
│ │ Rule Engine     │ │
│ │ - person_in_roi │ │
│ │ - crowd_in_roi  │ │
│ │ - restricted_   │ │
│ │   zone_breach   │ │
│ └─────────────────┘ │
└──────────┬──────────┘
           │ Results
           ▼
    ┌──────────────┐
    │   Database   │
    └──────────────┘
```

### What This Service Does

✅ Receives detection output (objects detected, coordinates, confidence)
✅ Evaluates multiple usecase rules against the detection data
✅ Stores evaluation results in database
✅ Returns structured evaluation results

### What This Service Does NOT Do

❌ Perform object detection (receives detection data)
❌ Manage cameras or video streams
❌ Trigger alerts (that's the alert service's job)
❌ Generate analytics reports (that's the analytics service's job)

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose (recommended)
- PostgreSQL database (shared with other services)
- Python 3.11+ (for local development)

### Using Docker (Recommended)

1. **Clone and navigate to the service directory:**

```bash
cd services/usecase
```

2. **Configure environment variables:**

```bash
cp .env.example .env
# Edit .env with your database configuration
```

3. **Start the service:**

```bash
docker-compose up -d
```

4. **Verify the service is running:**

```bash
curl http://localhost:8001/health
```

### Local Development

1. **Install dependencies:**

```bash
pip install -r requirements.txt
```

2. **Set environment variables:**

```bash
export DATABASE_URL="postgresql://goec:goec@localhost:5432/goec"
export PORT=8001
```

3. **Run the service:**

```bash
python main.py
```

4. **Access API documentation:**

Open your browser to `http://localhost:8001/docs` for interactive API documentation.

## 📚 Available Usecases

### 1. Person in ROI (`person_in_roi`)

**Description:** Triggers when any person is detected inside the Region of Interest (ROI).

**Use Cases:**
- Queue monitoring (is anyone in line?)
- Restricted area monitoring
- Presence detection

**Trigger Condition:**
```python
class_name == "person" AND in_roi == true
```

**Example Scenario:** Detect when any person enters a restricted area.

---

### 2. Crowd in ROI (`crowd_in_roi`)

**Description:** Triggers when 3 or more persons are detected inside the ROI simultaneously.

**Use Cases:**
- Crowd management
- Social distancing monitoring
- Capacity monitoring
- Queue congestion detection

**Trigger Condition:**
```python
count(class_name == "person" AND in_roi == true) >= 3
```

**Configuration:**
- `CROWD_THRESHOLD`: Minimum number of persons (default: 3)

**Example Scenario:** Alert when too many people gather in a lobby area.

---

### 3. Restricted Zone Breach (`restricted_zone_breach`)

**Description:** Triggers when any vehicle is detected inside the ROI (restricted zone).

**Use Cases:**
- Parking violation detection
- No-vehicle zone monitoring
- Pedestrian area protection

**Trigger Condition:**
```python
class_name in ["car", "truck", "bus", "motorcycle", "bicycle"] AND in_roi == true
```

**Monitored Vehicle Types:**
- `car`
- `truck`
- `bus`
- `motorcycle`
- `bicycle`

**Example Scenario:** Detect vehicles in pedestrian-only zones.

---

## 🔧 API Usage

### Evaluate Usecases

**Endpoint:** `POST /usecase/evaluate`

**Purpose:** Evaluate one or more usecase rules against detection data.

#### Request Body

```json
{
  "camera_id": "s1_cam_1",
  "detection_output": {
    "camera_id": "s1_cam_1",
    "detections": [
      {
        "class_name": "person",
        "confidence": 0.92,
        "in_roi": true,
        "bbox": [100, 150, 200, 350]
      },
      {
        "class_name": "person",
        "confidence": 0.88,
        "in_roi": true,
        "bbox": [300, 160, 400, 360]
      },
      {
        "class_name": "car",
        "confidence": 0.85,
        "in_roi": false,
        "bbox": [500, 200, 700, 400]
      }
    ],
    "screenshot_path": "/screenshots/s1_cam_1_1234567890.jpg",
    "first_detection_id": 1001
  },
  "usecases": ["person_in_roi", "crowd_in_roi"]
}
```

#### Response

```json
{
  "camera_id": "s1_cam_1",
  "results": [
    {
      "usecase_id": "person_in_roi",
      "triggered": true,
      "matched_count": 2,
      "matched_objects": [
        {
          "class_name": "person",
          "confidence": 0.92,
          "in_roi": true,
          "bbox": [100, 150, 200, 350]
        },
        {
          "class_name": "person",
          "confidence": 0.88,
          "in_roi": true,
          "bbox": [300, 160, 400, 360]
        }
      ],
      "detection_id": 1001,
      "screenshot_path": "/screenshots/s1_cam_1_1234567890.jpg"
    },
    {
      "usecase_id": "crowd_in_roi",
      "triggered": false,
      "matched_count": 2,
      "matched_objects": [
        {
          "class_name": "person",
          "confidence": 0.92,
          "in_roi": true,
          "bbox": [100, 150, 200, 350]
        },
        {
          "class_name": "person",
          "confidence": 0.88,
          "in_roi": true,
          "bbox": [300, 160, 400, 360]
        }
      ],
      "detection_id": 1001,
      "screenshot_path": "/screenshots/s1_cam_1_1234567890.jpg"
    }
  ]
}
```

#### Response Fields

- **camera_id**: Camera identifier
- **results**: List of evaluation results (one per requested usecase)
  - **usecase_id**: Usecase identifier
  - **triggered**: Boolean indicating if the usecase condition was met
  - **matched_count**: Number of objects that matched the rule
  - **matched_objects**: List of detection objects that matched
  - **detection_id**: Associated detection record ID
  - **screenshot_path**: Path to detection screenshot

### Health Check

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "usecase-evaluation",
  "version": "1.0.0"
}
```

## 🗄️ Database Schema

The service stores evaluation results in the `usecase_results` table:

```sql
CREATE TABLE usecase_results (
    id SERIAL PRIMARY KEY,
    camera_id VARCHAR(100) NOT NULL,
    usecase_name VARCHAR(100) NOT NULL,
    triggered BOOLEAN NOT NULL,
    detection_id INTEGER REFERENCES detections(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id)
);
```

### Indexes
- `camera_id` (for queries by camera)
- `usecase_name` (for queries by usecase type)
- `timestamp` (for time-based analytics)

## 🔌 Integration with Other Services

### With Detection Service (Upstream)

The detection service sends detection output to the usecase service:

```python
import requests

detection_output = {
    "camera_id": "s1_cam_1",
    "detections": [...],
    "screenshot_path": "...",
    "first_detection_id": 1001
}

response = requests.post(
    "http://usecase-service:8001/usecase/evaluate",
    json={
        "camera_id": "s1_cam_1",
        "detection_output": detection_output,
        "usecases": ["person_in_roi", "crowd_in_roi"]
    }
)

results = response.json()
```

### With Alert Service (Downstream)

The orchestrator or caller can check which usecases triggered and send to alert service:

```python
for result in results["results"]:
    if result["triggered"]:
        # Send to alert service
        requests.post(
            "http://alert-service:8002/alert/trigger",
            json={
                "camera_id": camera_id,
                "usecase_id": result["usecase_id"],
                "matched_count": result["matched_count"],
                "screenshot_path": result["screenshot_path"]
            }
        )
```

### With Orchestration Service

The orchestration service coordinates the flow:

```
Orchestrator → Detection Service → Get detections
             ↓
             → Usecase Service → Evaluate rules
             ↓
             → Alert Service (if triggered) → Send alerts
```

## 🛠️ Adding Custom Usecases

The rule engine is extensible. To add a new usecase:

### 1. Create a New Rule File

Create `usecase/rules/your_new_rule.py`:

```python
from typing import Dict, Any
from usecase.rules.base import BaseUsecaseRule

class YourNewRule(BaseUsecaseRule):
    """
    Description of your usecase.
    """
    
    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate your custom logic.
        
        Args:
            detection_output: Detection API response
            
        Returns:
            Evaluation result with triggered status and matched objects
        """
        detections = self.get_detections(detection_output)
        matched_objects = []
        
        for detection in detections:
            # Your custom logic here
            if self._matches_condition(detection):
                matched_objects.append(detection)
        
        triggered = len(matched_objects) > 0
        
        return {
            "triggered": triggered,
            "matched_objects": matched_objects
        }
    
    def _matches_condition(self, detection: Dict[str, Any]) -> bool:
        """Your custom matching logic"""
        # Example: Check if object is a specific class and has high confidence
        return (
            detection.get("class_name") == "person" and
            detection.get("confidence", 0) > 0.8
        )
```

### 2. Register the Rule

Update `usecase/rules/__init__.py`:

```python
from usecase.rules.your_new_rule import YourNewRule

USECASE_RULES = {
    "person_in_roi": PersonInROIRule,
    "crowd_in_roi": CrowdInROIRule,
    "restricted_zone_breach": RestrictedZoneRule,
    "your_new_rule": YourNewRule,  # Add your rule
}
```

### 3. Use the New Rule

```bash
curl -X POST http://localhost:8001/usecase/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "s1_cam_1",
    "detection_output": {...},
    "usecases": ["your_new_rule"]
  }'
```

## 📊 Logging and Debugging

The service provides detailed logging for each evaluation:

```
[ORCHESTRATOR] ============================================================
[ORCHESTRATOR] USECASE EVALUATION ORCHESTRATOR
[ORCHESTRATOR] Camera ID: s1_cam_1
[ORCHESTRATOR] Number of usecases to evaluate: 2
[ORCHESTRATOR] Usecases: person_in_roi, crowd_in_roi
[ORCHESTRATOR] ============================================================

[RULE:person_in_roi] ========== EVALUATION START ==========
[RULE:person_in_roi] Processing 3 detections
[RULE:person_in_roi] Detection 1: class='person', in_roi=True, conf=0.92
[RULE:person_in_roi] ✓✓✓ MATCH: Person in ROI (confidence: 0.92)
[RULE:person_in_roi] Triggered: True
[RULE:person_in_roi] Matched: 2 objects
```

Logs are written to:
- `stdout` (Docker logs)
- `usecase_service.log` (persistent file in logs/ directory)

## 🔒 Security Considerations

### Production Recommendations

1. **Database Security:**
   - Use strong passwords
   - Enable SSL for database connections
   - Limit database user permissions to necessary tables

2. **API Security:**
   - Add authentication (JWT, API keys)
   - Enable HTTPS/TLS
   - Configure CORS properly (restrict origins)
   - Add rate limiting

3. **Network Security:**
   - Deploy on private network
   - Use firewall rules
   - Enable network encryption between services

4. **Environment Variables:**
   - Never commit .env files
   - Use secret management tools (e.g., HashiCorp Vault)
   - Rotate credentials regularly

## 📈 Performance Considerations

### Resource Requirements

- **CPU**: 1-2 cores recommended
- **Memory**: 512MB - 1GB
- **Storage**: Minimal (logs only, database is separate)
- **Network**: Standard HTTP traffic

### Scaling

The service is stateless and can be horizontally scaled:

```yaml
# docker-compose.yml
services:
  usecase-service:
    deploy:
      replicas: 3
```

Load balancer can distribute requests across instances.

### Performance Tips

1. **Database Indexing**: Ensure indexes on `camera_id`, `usecase_name`, `timestamp`
2. **Connection Pooling**: SQLAlchemy handles this automatically
3. **Async Processing**: For high-throughput, consider async/await patterns
4. **Caching**: Cache rule instances (already implemented)

## 🐛 Troubleshooting

### Common Issues

#### 1. Database Connection Fails

**Error:** `sqlalchemy.exc.OperationalError: could not connect to server`

**Solution:**
- Check `DATABASE_URL` in .env
- Ensure PostgreSQL is running
- Verify network connectivity
- Check database credentials

```bash
# Test database connection
psql -h localhost -U goec -d goec
```

#### 2. Service Won't Start

**Error:** `Port 8001 is already allocated`

**Solution:**
- Change PORT in .env file
- Stop conflicting service: `docker-compose down`

#### 3. Usecase Not Found

**Error:** `ValueError: Unknown usecase: my_usecase`

**Solution:**
- Check usecase ID spelling
- Verify usecase is registered in `usecase/rules/__init__.py`
- Use one of: `person_in_roi`, `crowd_in_roi`, `restricted_zone_breach`

#### 4. No Results Returned

**Possible causes:**
- Detection output format incorrect
- No objects match rule conditions
- Check logs for detailed evaluation trace

```bash
# View logs
docker-compose logs -f usecase-service
```

## 📝 Environment Variables Reference

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `HOST` | Service host | `0.0.0.0` | No |
| `PORT` | Service port | `8001` | No |
| `DATABASE_URL` | PostgreSQL connection string | - | Yes |
| `LOG_LEVEL` | Logging level | `INFO` | No |
| `PERSON_IN_ROI_MIN_CONFIDENCE` | Min confidence threshold | `0.5` | No |
| `CROWD_IN_ROI_MIN_COUNT` | Min persons for crowd | `3` | No |
| `RESTRICTED_ZONE_MIN_CONFIDENCE` | Min confidence threshold | `0.5` | No |

## 🧪 Testing

### Manual Testing

```bash
# Test health endpoint
curl http://localhost:8001/health

# Test usecase evaluation
curl -X POST http://localhost:8001/usecase/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "test_cam",
    "detection_output": {
      "camera_id": "test_cam",
      "detections": [
        {
          "class_name": "person",
          "confidence": 0.9,
          "in_roi": true,
          "bbox": [100, 100, 200, 200]
        }
      ],
      "screenshot_path": "/test.jpg",
      "first_detection_id": 1
    },
    "usecases": ["person_in_roi"]
  }'
```

### Expected Response

```json
{
  "camera_id": "test_cam",
  "results": [
    {
      "usecase_id": "person_in_roi",
      "triggered": true,
      "matched_count": 1,
      "matched_objects": [...]
    }
  ]
}
```

## 📦 Deployment

### Standalone Deployment

Deploy this service independently:

```bash
cd services/usecase
docker-compose up -d
```

### Multi-Service Deployment (Server 3)

On Server 3, deploy with alert and analytics services:

```yaml
# Combined docker-compose.yml
version: '3.8'

services:
  usecase-service:
    build: ./usecase
    ports:
      - "8001:8001"
    environment:
      DATABASE_URL: postgresql://goec:goec@postgres:5432/goec
  
  alert-service:
    build: ./alert
    ports:
      - "8002:8002"
  
  analytics-service:
    build: ./analytics
    ports:
      - "8003:8003"

  postgres:
    image: postgres:15-alpine
    ...
```

## 📖 Additional Resources

- [API Documentation](http://localhost:8001/docs) - Interactive Swagger UI
- [ReDoc Documentation](http://localhost:8001/redoc) - Alternative API docs
- [GOEC Architecture Overview](../../README.md) - Main architecture documentation
- [Deployment Guide](../../DEPLOYMENT_GUIDE.md) - Full deployment instructions

## 🤝 Contributing

To add new usecases or improve existing rules:

1. Follow the rule structure in `usecase/rules/base.py`
2. Add comprehensive logging
3. Update this README with new usecase documentation
4. Test thoroughly with various detection scenarios
5. Update API documentation

## 📄 License

Part of the GOEC (General Operations Edge Computing) project.

## 📞 Support

For issues or questions:
- Check the [Troubleshooting](#-troubleshooting) section
- Review logs: `docker-compose logs -f usecase-service`
- Consult the main project documentation

---

**Version:** 1.0.0  
**Last Updated:** 2026-03-03  
**Maintained By:** GOEC Team
