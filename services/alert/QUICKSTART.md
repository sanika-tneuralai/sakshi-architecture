# Alert Service - Quick Start Guide

Get the Alert Service running in under 5 minutes!

## 🚀 Quick Start

### 1. Navigate to Service Directory

```bash
cd services/alert/
```

### 2. Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit with your settings (at minimum, configure database)
nano .env
```

**Minimal Configuration:**
```bash
# Database (required)
DATABASE_URL=postgresql://goec:goec@localhost:5432/goec

# Service (optional - defaults shown)
ALERT_SERVICE_PORT=8002

# Alert channels (enable at least one)
ALERT_EMAIL_ENABLED=false
ALERT_SMS_ENABLED=false
ALERT_WEBHOOK_ENABLED=false
```

### 3. Start the Service

#### Option A: Docker Compose (Recommended)

```bash
docker-compose up -d
```

#### Option B: Local Python

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Run service
python main.py
```

### 4. Verify Service

```bash
# Health check
curl http://localhost:8002/health

# Service info
curl http://localhost:8002/

# API documentation
# Open in browser: http://localhost:8002/docs
```

## 📡 Send a Test Alert

```bash
# Test person detection alert
curl -X POST http://localhost:8002/alert/send \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "test_cam",
    "usecase_results": [{
      "usecase_id": "person_in_roi",
      "triggered": true,
      "matched_count": 1,
      "matched_objects": ["person"],
      "detection_id": 1,
      "screenshot_path": "/screenshots/test.jpg"
    }]
  }'
```

**Expected Response:**
```json
{
  "camera_id": "test_cam",
  "total_alerts_sent": 1,
  "alerts_sent": [{
    "usecase_id": "person_in_roi",
    "alert_type": "person_detected",
    "alert_count": 1,
    "message": "Person detected inside ROI. Count: 1"
  }]
}
```

## 🔔 Configure Alert Channels

### Email Alerts

```bash
# In .env file
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_SMTP_HOST=smtp.gmail.com
ALERT_EMAIL_SMTP_PORT=587
ALERT_EMAIL_USERNAME=your-email@gmail.com
ALERT_EMAIL_PASSWORD=your-app-password
ALERT_EMAIL_TO=admin@example.com
```

### SMS Alerts (Twilio)

```bash
# In .env file
ALERT_SMS_ENABLED=true
ALERT_SMS_PROVIDER=twilio
ALERT_SMS_ACCOUNT_SID=your-account-sid
ALERT_SMS_AUTH_TOKEN=your-auth-token
ALERT_SMS_FROM=+1234567890
ALERT_SMS_TO=+1987654321
```

### Webhook Alerts

```bash
# In .env file
ALERT_WEBHOOK_ENABLED=true
ALERT_WEBHOOK_URL=https://your-endpoint.com/alerts
ALERT_WEBHOOK_METHOD=POST
ALERT_WEBHOOK_AUTH_TOKEN=your-token
```

## 🧪 Run Test Suite

```bash
# Run all tests
./test_api.sh

# Set custom service URL
ALERT_SERVICE_URL=http://localhost:8002 ./test_api.sh
```

## 📊 View Alert History

```bash
# Get all alerts (last 100)
curl http://localhost:8002/alert/list

# Filter by camera
curl http://localhost:8002/alert/list?camera_id=test_cam

# Filter by usecase
curl http://localhost:8002/alert/list?usecase_name=person_in_roi

# Limit results
curl http://localhost:8002/alert/list?limit=10
```

## 🛑 Stop Service

```bash
# Docker Compose
docker-compose down

# Local Python
# Press Ctrl+C in the terminal running main.py
```

## 📖 Next Steps

- **Full Documentation**: See [README.md](README.md)
- **API Documentation**: http://localhost:8002/docs
- **Configure Alert Channels**: Edit `.env` file
- **Integration**: Connect with orchestration service
- **Production Deployment**: See deployment guide

## 🐛 Troubleshooting

### Service won't start

```bash
# Check if port is in use
lsof -i :8002

# Check Docker logs
docker logs alert-service

# Check local logs
cat alert_service.log
```

### Database connection failed

```bash
# Verify PostgreSQL is running
docker ps | grep postgres

# Test connection
psql -h localhost -U goec -d goec -c "SELECT 1;"
```

### Alerts not sending

1. Check `.env` - ensure at least one channel is enabled
2. Verify credentials are correct
3. Check service logs for errors
4. Test SMTP/API connectivity manually

## 📞 Support

- **Documentation**: [README.md](README.md)
- **API Docs**: http://localhost:8002/docs
- **Test Script**: `./test_api.sh`

---

**Ready in 3 Steps**: Config → Start → Test 🎉
