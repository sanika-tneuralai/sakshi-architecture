# Analytics Service - Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### 1. Setup Environment
```bash
cd services/analytics
cp .env.example .env
```

### 2. Edit `.env`
```bash
# Update with your PostgreSQL connection
DATABASE_URL=postgresql://user:pass@localhost:5432/goec
```

### 3. Start Service
```bash
# Option A: Docker (recommended)
docker-compose up -d

# Option B: Local
pip install -r requirements.txt
python main.py
```

### 4. Verify Running
```bash
curl http://localhost:8003/health
```

Expected:
```json
{"status": "healthy", "service": "analytics", "scheduler": "running"}
```

## 📊 Quick API Tests

### Get Daily Analytics
```bash
curl http://localhost:8003/analytics/daily
```

### Get Alert Analytics
```bash
curl http://localhost:8003/analytics/alerts
```

### Get Detection Analytics
```bash
curl http://localhost:8003/analytics/detections
```

### Filter by Camera
```bash
curl "http://localhost:8003/analytics/daily?camera_id=cam_001"
```

### Filter by Date Range
```bash
curl "http://localhost:8003/analytics/daily?start_date=2024-03-01&end_date=2024-03-31"
```

## 🔧 Configuration

### Key Environment Variables
```bash
DATABASE_URL=postgresql://user:pass@host:5432/goec
PORT=8003
SCHEDULER_ENABLED=true
SCHEDULER_HOUR=0        # Run at 00:30 UTC
SCHEDULER_MINUTE=30
```

### Disable Scheduler (for testing)
```bash
SCHEDULER_ENABLED=false
```

### Change Schedule
```bash
# Run at 2:00 AM instead
SCHEDULER_HOUR=2
SCHEDULER_MINUTE=0
```

## 🧪 Run All Tests
```bash
chmod +x test_api.sh
./test_api.sh
```

## 📱 Access API Docs
- **Swagger UI**: http://localhost:8003/docs
- **ReDoc**: http://localhost:8003/redoc

## 🐛 Troubleshooting

### Check Logs
```bash
# Docker
docker-compose logs -f analytics

# Local
# See console output
```

### Database Connection Issues
```bash
# Test connection
docker exec analytics-service python -c "from shared.database.connection import SessionLocal; db = SessionLocal(); print('Connected!')"
```

### Scheduler Not Running
Check logs for:
```
[SCHEDULER] Starting analytics scheduler
[SCHEDULER] Scheduler started - Daily aggregation scheduled for 00:30 UTC
```

If missing, check `SCHEDULER_ENABLED=true` in `.env`

## 🛑 Stop Service
```bash
# Docker
docker-compose down

# Local
# Ctrl+C to stop
```

## 📚 Full Documentation
See [README.md](README.md) for complete documentation.

## 🔗 Related Services
- **Usecase Service**: Port 8001
- **Alert Service**: Port 8002
- **Analytics Service**: Port 8003 (this service)
