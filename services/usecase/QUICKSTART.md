# Usecase Service - Quick Start Guide

Get the Usecase Evaluation Service up and running in 5 minutes!

## 🚀 Quick Start with Docker (Recommended)

### Step 1: Configure Environment

```bash
cd services/usecase
cp .env.example .env
```

Edit `.env` if needed (defaults should work for local testing):
```bash
DATABASE_URL=postgresql://goec:goec@postgres:5432/goec
PORT=8001
```

### Step 2: Start the Service

```bash
docker-compose up -d
```

This will:
- Build the usecase service Docker image
- Start PostgreSQL database (if not already running)
- Initialize database tables
- Start the service on port 8001

### Step 3: Verify Service is Running

```bash
# Check service health
curl http://localhost:8001/health

# Expected response:
# {"status":"healthy","service":"usecase-evaluation","version":"1.0.0"}
```

### Step 4: View API Documentation

Open in your browser:
- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

### Step 5: Test the Service

Run a simple test:

```bash
curl -X POST http://localhost:8001/usecase/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "test_camera",
    "detection_output": {
      "camera_id": "test_camera",
      "detections": [
        {
          "class_name": "person",
          "confidence": 0.92,
          "in_roi": true,
          "bbox": [100, 150, 200, 350]
        }
      ],
      "screenshot_path": "/screenshots/test.jpg",
      "first_detection_id": 1
    },
    "usecases": ["person_in_roi"]
  }'
```

Expected response:
```json
{
  "camera_id": "test_camera",
  "results": [
    {
      "usecase_id": "person_in_roi",
      "triggered": true,
      "matched_count": 1,
      "matched_objects": [
        {
          "class_name": "person",
          "confidence": 0.92,
          "in_roi": true,
          "bbox": [100, 150, 200, 350]
        }
      ],
      "detection_id": 1,
      "screenshot_path": "/screenshots/test.jpg"
    }
  ]
}
```

## 🐍 Local Development Setup (Without Docker)

### Prerequisites

- Python 3.11+
- PostgreSQL database running

### Step 1: Install Dependencies

```bash
cd services/usecase
pip install -r requirements.txt
```

### Step 2: Set Environment Variables

```bash
export DATABASE_URL="postgresql://goec:goec@localhost:5432/goec"
export PORT=8001
```

### Step 3: Initialize Database

Make sure your PostgreSQL database is running and accessible:

```bash
# Create database (if not exists)
createdb goec
```

### Step 4: Run the Service

```bash
python main.py
```

Output should show:
```
INFO:     Starting Usecase Evaluation Service
INFO:     ✓ Database initialized
INFO:     ✓ Usecase service started successfully
INFO:     Uvicorn running on http://0.0.0.0:8001
```

## 🧪 Testing All Usecases

### Test Person in ROI

```bash
curl -X POST http://localhost:8001/usecase/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "test_cam",
    "detection_output": {
      "detections": [
        {"class_name": "person", "confidence": 0.9, "in_roi": true, "bbox": [100, 100, 200, 200]}
      ],
      "screenshot_path": "/test.jpg",
      "first_detection_id": 1
    },
    "usecases": ["person_in_roi"]
  }'
```

**Expected:** `triggered: true` (1 person in ROI)

### Test Crowd in ROI

```bash
curl -X POST http://localhost:8001/usecase/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "test_cam",
    "detection_output": {
      "detections": [
        {"class_name": "person", "confidence": 0.9, "in_roi": true, "bbox": [100, 100, 200, 200]},
        {"class_name": "person", "confidence": 0.85, "in_roi": true, "bbox": [300, 100, 400, 200]},
        {"class_name": "person", "confidence": 0.88, "in_roi": true, "bbox": [500, 100, 600, 200]}
      ],
      "screenshot_path": "/test.jpg",
      "first_detection_id": 2
    },
    "usecases": ["crowd_in_roi"]
  }'
```

**Expected:** `triggered: true` (3 persons = crowd)

### Test Restricted Zone

```bash
curl -X POST http://localhost:8001/usecase/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "test_cam",
    "detection_output": {
      "detections": [
        {"class_name": "car", "confidence": 0.92, "in_roi": true, "bbox": [100, 100, 300, 200]}
      ],
      "screenshot_path": "/test.jpg",
      "first_detection_id": 3
    },
    "usecases": ["restricted_zone_breach"]
  }'
```

**Expected:** `triggered: true` (vehicle in restricted zone)

### Test Multiple Usecases at Once

```bash
curl -X POST http://localhost:8001/usecase/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "test_cam",
    "detection_output": {
      "detections": [
        {"class_name": "person", "confidence": 0.9, "in_roi": true, "bbox": [100, 100, 200, 200]},
        {"class_name": "person", "confidence": 0.85, "in_roi": true, "bbox": [300, 100, 400, 200]},
        {"class_name": "car", "confidence": 0.92, "in_roi": true, "bbox": [500, 100, 700, 200]}
      ],
      "screenshot_path": "/test.jpg",
      "first_detection_id": 4
    },
    "usecases": ["person_in_roi", "crowd_in_roi", "restricted_zone_breach"]
  }'
```

**Expected Results:**
- `person_in_roi`: `triggered: true` (2 persons)
- `crowd_in_roi`: `triggered: false` (only 2 persons, need 3)
- `restricted_zone_breach`: `triggered: true` (1 car)

## 📊 View Logs

### Docker Logs

```bash
# Real-time logs
docker-compose logs -f usecase-service

# Last 100 lines
docker-compose logs --tail=100 usecase-service
```

### Local Logs

Logs are written to:
- Console (stdout)
- `usecase_service.log` file

```bash
tail -f usecase_service.log
```

## 🔧 Common Commands

### Stop the Service

```bash
docker-compose down
```

### Restart the Service

```bash
docker-compose restart
```

### Rebuild After Code Changes

```bash
docker-compose down
docker-compose build
docker-compose up -d
```

### View Database Records

```bash
# Connect to database
docker-compose exec postgres psql -U goec -d goec

# Query usecase results
SELECT * FROM usecase_results ORDER BY timestamp DESC LIMIT 10;

# Exit
\q
```

## 🐛 Troubleshooting

### Port Already in Use

```bash
# Change port in .env
PORT=8005

# Or stop conflicting service
docker-compose down
```

### Database Connection Error

```bash
# Check if database is running
docker-compose ps

# Verify DATABASE_URL in .env
cat .env | grep DATABASE_URL

# Test database connection
docker-compose exec postgres psql -U goec -d goec -c "SELECT 1;"
```

### Service Won't Start

```bash
# View detailed logs
docker-compose logs usecase-service

# Check if all dependencies are installed
docker-compose build --no-cache
```

## 📈 Integration with Other Services

### With Detection Service

The detection service calls the usecase service:

```python
import requests

# Detection service calls usecase service
response = requests.post(
    "http://usecase-service:8001/usecase/evaluate",
    json={
        "camera_id": camera_id,
        "detection_output": detection_data,
        "usecases": ["person_in_roi", "crowd_in_roi"]
    }
)

results = response.json()
print(f"Triggered usecases: {[r['usecase_id'] for r in results['results'] if r['triggered']]}")
```

### With Orchestration Service

The orchestrator coordinates the flow:

```python
# 1. Get detections from detection service
detections = detection_service.detect(camera_id)

# 2. Evaluate usecases
usecase_results = usecase_service.evaluate(camera_id, detections, usecases)

# 3. Trigger alerts if needed
for result in usecase_results['results']:
    if result['triggered']:
        alert_service.send_alert(camera_id, result)
```

## 🎯 Next Steps

1. **Explore the Full README**: See [README.md](README.md) for detailed documentation
2. **Add Custom Rules**: Learn how to create custom usecases
3. **Deploy to Production**: Review deployment best practices
4. **Monitor Performance**: Set up logging and monitoring
5. **Integrate with Other Services**: Connect with detection and alert services

## 📚 Additional Resources

- [Full Documentation](README.md)
- [API Reference](http://localhost:8001/docs)
- [Architecture Overview](../../README.md)
- [Deployment Guide](../../DEPLOYMENT_GUIDE.md)

---

**Need Help?**
- Check logs: `docker-compose logs -f usecase-service`
- Review health: `curl http://localhost:8001/health`
- Consult [README.md](README.md) for detailed troubleshooting

**Ready to Go!** 🎉

Your usecase service is now running and ready to evaluate business rules against detection data!
