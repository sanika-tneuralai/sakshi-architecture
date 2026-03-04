# Alert Service

A standalone microservice for processing and sending alerts based on usecase evaluation results.

## 📋 Overview

The Alert Service is part of the GOEC (Generic Object Event Collection) modular architecture. It receives triggered usecase results from the orchestration service and sends alerts through configured channels (Email, SMS, Webhook).

## 🎯 Features

- **Multiple Alert Channels**: Support for Email, SMS, and Webhook notifications
- **Flexible Configuration**: Enable/disable channels via environment variables
- **Alert Persistence**: Store alert records in PostgreSQL database
- **Alert History**: Query past alerts by camera, usecase, or time range
- **RESTful API**: Simple HTTP endpoints for alert processing
- **Health Monitoring**: Built-in health check endpoints
- **Docker Support**: Containerized deployment with docker-compose

## 🏗️ Architecture

```
Alert Service (Port 8002)
├── Receives triggered usecase results
├── Evaluates alert rules
├── Sends alerts via configured channels
│   ├── Email (SMTP)
│   ├── SMS (Twilio)
│   └── Webhook (HTTP POST)
└── Stores alert records in database
```

## 📦 Directory Structure

```
services/alert/
├── main.py                  # FastAPI application entry point
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container image definition
├── docker-compose.yml      # Multi-container orchestration
├── .env.example            # Environment configuration template
├── README.md              # This file
├── shared/                # Shared components
│   ├── database/          # Database models and connection
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── connection.py
│   └── common/            # Common utilities
│       ├── __init__.py
│       ├── config.py
│       ├── logger.py
│       └── utils.py
└── alert/                 # Alert module
    ├── __init__.py
    ├── api.py            # API endpoints
    ├── service.py        # Business logic
    └── schemas.py        # Pydantic models
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- PostgreSQL database (shared across services)
- Python 3.11+ (for local development)

### 1. Configuration

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` and configure your alert channels:

```bash
# Enable at least one alert channel
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_USERNAME=your-email@gmail.com
ALERT_EMAIL_PASSWORD=your-app-password
ALERT_EMAIL_TO=recipient@example.com
```

### 2. Start the Service

Using Docker Compose:

```bash
docker-compose up -d
```

The service will be available at `http://localhost:8002`

### 3. Verify Health

```bash
curl http://localhost:8002/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "alert",
  "version": "1.0.0"
}
```

## 📡 API Endpoints

### Health Check

**GET** `/health`

Returns service health status.

### Root Information

**GET** `/`

Returns service information and available endpoints.

### Send Alerts (Primary)

**POST** `/alert/send`

Process and send alerts based on usecase evaluation results.

**Request Body:**
```json
{
  "camera_id": "cam_001",
  "usecase_results": [
    {
      "usecase_id": "person_in_roi",
      "triggered": true,
      "matched_count": 1,
      "matched_objects": ["person"],
      "detection_id": 123,
      "screenshot_path": "/screenshots/cam_001_20240101_120000.jpg"
    }
  ]
}
```

**Response:**
```json
{
  "camera_id": "cam_001",
  "total_alerts_sent": 1,
  "alerts_sent": [
    {
      "usecase_id": "person_in_roi",
      "alert_type": "person_detected",
      "alert_count": 1,
      "message": "Person detected inside ROI. Count: 1"
    }
  ]
}
```

### Send Single Alert (Legacy)

**POST** `/alert/send-single`

Process and send a single alert.

**Request Body:**
```json
{
  "camera_id": "cam_001",
  "usecase_id": "person_in_roi",
  "alert_required": true,
  "alert_type": "person_detected",
  "alert_objects": [{"class": "person", "confidence": 0.95}],
  "alert_count": 1
}
```

### Get Alert History

**GET** `/alert/list?camera_id=cam_001&limit=100`

Retrieve alert records.

**Query Parameters:**
- `camera_id` (optional): Filter by camera ID
- `usecase_name` (optional): Filter by usecase name
- `limit` (optional): Maximum records to return (default: 100)

**Response:**
```json
{
  "alerts": [
    {
      "alert_id": 1,
      "camera_id": "cam_001",
      "usecase_name": "person_in_roi",
      "alert_type": "person_detected",
      "timestamp": "2024-01-01T12:00:00",
      "status": "sent",
      "screenshot_path": "/screenshots/cam_001_20240101_120000.jpg"
    }
  ],
  "total": 1
}
```

## 🔔 Alert Channels

### Email Alerts

Configure SMTP settings in `.env`:

```bash
ALERT_EMAIL_ENABLED=true
ALERT_EMAIL_SMTP_HOST=smtp.gmail.com
ALERT_EMAIL_SMTP_PORT=587
ALERT_EMAIL_USERNAME=your-email@gmail.com
ALERT_EMAIL_PASSWORD=your-app-password
ALERT_EMAIL_FROM=alerts@goec.com
ALERT_EMAIL_TO=admin@goec.com,security@goec.com
```

**Gmail Setup:**
1. Enable 2-factor authentication
2. Generate App Password: https://myaccount.google.com/apppasswords
3. Use App Password in `ALERT_EMAIL_PASSWORD`

### SMS Alerts

Configure Twilio or other SMS provider:

```bash
ALERT_SMS_ENABLED=true
ALERT_SMS_PROVIDER=twilio
ALERT_SMS_ACCOUNT_SID=your-account-sid
ALERT_SMS_AUTH_TOKEN=your-auth-token
ALERT_SMS_FROM=+1234567890
ALERT_SMS_TO=+1987654321,+1555555555
```

**Twilio Setup:**
1. Sign up: https://www.twilio.com/
2. Get Account SID and Auth Token
3. Purchase a phone number
4. Configure in `.env`

### Webhook Alerts

Send alerts to custom HTTP endpoints:

```bash
ALERT_WEBHOOK_ENABLED=true
ALERT_WEBHOOK_URL=https://your-endpoint.com/alerts
ALERT_WEBHOOK_METHOD=POST
ALERT_WEBHOOK_AUTH_TOKEN=your-bearer-token
```

**Webhook Payload:**
```json
{
  "camera_id": "cam_001",
  "usecase_id": "person_in_roi",
  "alert_type": "person_detected",
  "alert_count": 1,
  "timestamp": "2024-01-01T12:00:00",
  "screenshot_path": "/screenshots/cam_001_20240101_120000.jpg"
}
```

## 🔐 Alert Rules

The service evaluates different rules for each usecase:

### 1. Person in ROI
- **Trigger**: Always when `triggered=True`
- **Alert Type**: `person_detected`
- **Message**: "Person detected inside ROI. Count: {count}"

### 2. Crowd in ROI
- **Trigger**: When `triggered=True` AND `matched_count >= 3`
- **Alert Type**: `crowd_detected`
- **Message**: "Crowd detected inside ROI. Count: {count}"

### 3. Restricted Zone Breach
- **Trigger**: Always when `triggered=True`
- **Alert Type**: `restricted_zone_breach`
- **Message**: "Restricted zone breach detected. Count: {count}"

## 🗄️ Database Schema

The Alert model in the shared database:

```python
class Alert(Base):
    __tablename__ = 'alerts'
    
    alert_id: int (Primary Key)
    camera_id: str
    usecase_name: str
    alert_type: str
    timestamp: datetime
    status: str  # 'sent', 'pending', 'failed'
    detection_id: int (Foreign Key)
    screenshot_path: str
```

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t alert-service:latest .
```

### Run Container

```bash
docker run -d \
  --name alert-service \
  -p 8002:8002 \
  --env-file .env \
  alert-service:latest
```

### Multi-Service Deployment

For production deployment with other services:

```bash
# On Server 3 (with usecase and analytics services)
docker-compose up -d alert-service
```

## 🔧 Development

### Local Setup

1. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
```bash
cp .env.example .env
# Edit .env with your database and alert channel configs
```

4. Run locally:
```bash
python main.py
```

### Testing

Test alert endpoint:

```bash
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

### Logs

View service logs:

```bash
# Docker logs
docker logs -f alert-service

# Local logs
tail -f alert_service.log
```

## 🌐 Integration

### With Orchestration Service

The orchestration service calls the alert endpoint after usecase evaluation:

```python
# Orchestrator sends triggered usecases to alert service
response = requests.post(
    f"{ALERT_SERVICE_URL}/alert/send",
    json={
        "camera_id": camera_id,
        "usecase_results": triggered_results
    }
)
```

### Standalone Usage

Can also be called directly for custom workflows:

```python
import requests

alert_response = requests.post(
    "http://localhost:8002/alert/send",
    json={
        "camera_id": "custom_cam",
        "usecase_results": [...]
    }
)
```

## 📊 Monitoring

### Health Check

```bash
curl http://localhost:8002/health
```

### Service Info

```bash
curl http://localhost:8002/
```

### Database Connectivity

Check if database connection is working by querying alerts:

```bash
curl http://localhost:8002/alert/list?limit=1
```

## 🔒 Security Considerations

1. **Environment Variables**: Never commit `.env` file
2. **API Keys**: Store SMS/Email credentials securely
3. **Database**: Use strong passwords for PostgreSQL
4. **Webhook**: Use HTTPS and authentication tokens
5. **Network**: Restrict port access in production
6. **Secrets**: Use Docker secrets or vault for production

## 🐛 Troubleshooting

### Alert not sending

1. Check alert channel is enabled in `.env`
2. Verify credentials are correct
3. Check service logs for errors
4. Test SMTP/API connectivity manually

### Database connection failed

1. Verify `DATABASE_URL` is correct
2. Ensure PostgreSQL is running
3. Check network connectivity
4. Verify database credentials

### Service not starting

1. Check port 8002 is not in use
2. Verify all environment variables are set
3. Check Docker logs: `docker logs alert-service`
4. Ensure dependencies are installed

## 📈 Future Enhancements

- [ ] Slack/Teams integration
- [ ] Alert templates customization
- [ ] Alert cooldown/rate limiting
- [ ] Alert priority levels
- [ ] Alert aggregation (multiple events)
- [ ] Push notifications (mobile apps)
- [ ] Alert acknowledgment system
- [ ] Alert escalation policies

## 📝 API Documentation

Full interactive API documentation available at:

- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

## 📄 License

Part of the GOEC (Generic Object Event Collection) system.

## 🤝 Support

For issues or questions:
1. Check service logs
2. Refer to the troubleshooting section
3. Contact the development team

---

**Version**: 1.0.0  
**Last Updated**: March 2026  
**Maintainer**: GOEC Team
