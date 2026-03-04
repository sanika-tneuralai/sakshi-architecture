# Analytics Service

Standalone analytics and reporting service for GOEC (Global Operations and Event Correlation).

## Overview

The Analytics Service provides:
- **Scheduled Daily Aggregation**: Automated daily analytics aggregation using APScheduler
- **Analytics API Endpoints**: REST APIs for daily, alert, and detection analytics
- **Independent Operation**: Works standalone, querying shared database

## Features

### 🔄 Scheduled Jobs
- **Daily Aggregation**: Runs at 00:30 UTC (configurable)
- Aggregates detection, alert, and ROI violation data
- Stores results in `analytics_daily` table

### 📊 Analytics Endpoints

#### 1. Daily Analytics (`GET /analytics/daily`)
Get aggregated daily analytics from the `analytics_daily` table.

**Query Parameters:**
- `camera_id` (optional): Filter by specific camera
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)

**Response:**
```json
{
  "data": [
    {
      "date": "2024-03-15",
      "camera_id": "cam_001",
      "total_detections": 150,
      "roi_violations": 12,
      "alerts_sent": 5
    }
  ],
  "total_records": 1
}
```

#### 2. Alert Analytics (`GET /analytics/alerts`)
Get alert analytics aggregated by camera and usecase.

**Query Parameters:**
- `camera_id` (optional): Filter by specific camera
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)

**Response:**
```json
{
  "data": [
    {
      "camera_id": "cam_001",
      "usecase_name": "person_in_roi",
      "total_alerts": 10,
      "alerts_sent": 8,
      "alerts_failed": 2
    }
  ],
  "total_records": 1
}
```

#### 3. Detection Analytics (`GET /analytics/detections`)
Get detection analytics aggregated by camera.

**Query Parameters:**
- `camera_id` (optional): Filter by specific camera
- `start_date` (optional): Start date (YYYY-MM-DD)
- `end_date` (optional): End date (YYYY-MM-DD)

**Response:**
```json
{
  "data": [
    {
      "camera_id": "cam_001",
      "total_detections": 150,
      "roi_detections": 12,
      "non_roi_detections": 138,
      "roi_violation_rate": 8.0
    }
  ],
  "total_records": 1
}
```

## Directory Structure

```
services/analytics/
├── main.py                 # FastAPI app with analytics router
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container configuration
├── docker-compose.yml     # Docker Compose setup
├── .env.example           # Environment template
├── README.md              # This file
├── shared/                # Shared components
│   ├── database/          # Database models and connection
│   │   ├── __init__.py
│   │   ├── models.py      # SQLAlchemy models
│   │   └── connection.py  # Database connection
│   └── common/            # Common utilities
│       ├── __init__.py
│       ├── logger.py      # Logging configuration
│       ├── config.py      # Configuration management
│       └── utils.py       # Utility functions
└── analytics/             # Analytics module
    ├── __init__.py
    ├── api.py             # API endpoints
    ├── service.py         # Business logic
    ├── schemas.py         # Pydantic models
    └── scheduler.py       # APScheduler configuration
```

## Installation & Setup

### Prerequisites
- Python 3.10+
- PostgreSQL database (shared across all services)
- Docker & Docker Compose (optional)

### Option 1: Docker Compose (Recommended)

1. **Copy environment template:**
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your configuration:**
   ```bash
   # For production, use your existing PostgreSQL database
   DATABASE_URL=postgresql://username:password@your-db-host:5432/goec
   ```

3. **Start the service:**
   ```bash
   docker-compose up -d
   ```

4. **Check logs:**
   ```bash
   docker-compose logs -f analytics
   ```

### Option 2: Local Development

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables:**
   ```bash
   export DATABASE_URL="postgresql://user:pass@localhost:5432/goec"
   export PORT=8003
   ```

4. **Run the service:**
   ```bash
   python main.py
   ```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | Required | PostgreSQL connection string |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8003` | Server port |
| `RELOAD` | `false` | Auto-reload on code changes (dev only) |
| `SCHEDULER_ENABLED` | `true` | Enable/disable scheduler |
| `SCHEDULER_HOUR` | `0` | Hour to run daily aggregation (0-23) |
| `SCHEDULER_MINUTE` | `30` | Minute to run daily aggregation (0-59) |
| `SCHEDULER_TIMEZONE` | `UTC` | Timezone for scheduler |

### Scheduler Configuration

The scheduler runs daily aggregation jobs. Default: **00:30 UTC daily**.

To change the schedule:
```bash
# Run at 02:00 UTC
SCHEDULER_HOUR=2
SCHEDULER_MINUTE=0

# Run at 23:30 EST (convert to UTC)
SCHEDULER_HOUR=4
SCHEDULER_MINUTE=30
SCHEDULER_TIMEZONE=UTC
```

To disable the scheduler (for testing):
```bash
SCHEDULER_ENABLED=false
```

## API Testing

### Health Check
```bash
curl http://localhost:8003/health
```

### Get Daily Analytics
```bash
# All data
curl http://localhost:8003/analytics/daily

# Filter by camera
curl "http://localhost:8003/analytics/daily?camera_id=cam_001"

# Filter by date range
curl "http://localhost:8003/analytics/daily?start_date=2024-03-01&end_date=2024-03-31"
```

### Get Alert Analytics
```bash
# All alerts
curl http://localhost:8003/analytics/alerts

# Filter by camera and date
curl "http://localhost:8003/analytics/alerts?camera_id=cam_001&start_date=2024-03-01"
```

### Get Detection Analytics
```bash
# All detections
curl http://localhost:8003/analytics/detections

# Filter by camera
curl "http://localhost:8003/analytics/detections?camera_id=cam_001"
```

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8003/docs
- **ReDoc**: http://localhost:8003/redoc

## Database Schema

The analytics service uses the following tables:

### `analytics_daily`
Aggregated daily analytics (populated by scheduler).

| Column | Type | Description |
|--------|------|-------------|
| `id` | Integer | Primary key |
| `date` | Date | Analytics date |
| `camera_id` | String | Camera identifier |
| `total_detections` | Integer | Total detections |
| `roi_violations` | Integer | ROI violations count |
| `alerts_sent` | Integer | Alerts sent count |

### `detections`
Raw detection records (read-only for analytics).

### `alerts`
Alert records (read-only for analytics).

## Deployment

### Standalone Deployment
Run as an independent service on any server:

```bash
docker-compose up -d
```

### Multi-Service Deployment (Server 3)
Can be deployed alongside other services (usecase, alert) on the same server:

```yaml
# Combined docker-compose.yml for Server 3
services:
  usecase:
    # usecase service config
  
  alert:
    # alert service config
  
  analytics:
    # analytics service config
```

## Monitoring

### Health Check
```bash
curl http://localhost:8003/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "analytics",
  "scheduler": "running"
}
```

### Logs
```bash
# Docker
docker-compose logs -f analytics

# Local
# Check console output
```

### Scheduler Status
Check logs for scheduler messages:
```
[SCHEDULER] Starting analytics scheduler
[SCHEDULER] Scheduler started - Daily aggregation scheduled for 00:30 UTC
[CRON JOB] Daily Analytics Aggregation Started
```

## Troubleshooting

### Issue: Scheduler not running
**Solution:**
1. Check `SCHEDULER_ENABLED=true` in `.env`
2. Check logs: `docker-compose logs analytics`
3. Verify database connection

### Issue: Database connection error
**Solution:**
1. Verify `DATABASE_URL` in `.env`
2. Ensure PostgreSQL is running
3. Check network connectivity
4. Verify database credentials

### Issue: No data in analytics
**Solution:**
1. Check if daily aggregation has run (see logs)
2. Verify detections/alerts exist in database
3. Manually trigger aggregation (see Development section)

### Issue: Port already in use
**Solution:**
```bash
# Change port in .env
PORT=8004

# Or kill existing process
lsof -ti:8003 | xargs kill -9
```

## Development

### Manual Aggregation
Trigger daily aggregation manually (for testing):

```python
from analytics.service import aggregate_daily_analytics
from datetime import date

# Aggregate for yesterday
aggregate_daily_analytics()

# Aggregate for specific date
aggregate_daily_analytics(target_date=date(2024, 3, 15))
```

### Running Tests
```bash
# Create test script: test_api.sh
chmod +x test_api.sh
./test_api.sh
```

### Code Structure
- `main.py`: FastAPI app initialization, scheduler lifecycle
- `analytics/api.py`: API endpoints
- `analytics/service.py`: Business logic, aggregation functions
- `analytics/scheduler.py`: APScheduler configuration
- `analytics/schemas.py`: Pydantic request/response models

## Dependencies

- **FastAPI**: Web framework
- **SQLAlchemy**: Database ORM
- **APScheduler**: Task scheduling
- **Pydantic**: Data validation
- **Uvicorn**: ASGI server

## Integration with Other Services

The Analytics Service is **read-only** for other services' data:

```
┌─────────────┐
│   Camera    │──┐
│  Detection  │  │
└─────────────┘  │
                 ├──> Database ──> Analytics Service
┌─────────────┐  │    (reads data)
│   Usecase   │──┤
│    Alert    │  │
└─────────────┘──┘
```

**Communication:**
- Analytics reads from shared PostgreSQL database
- No direct API calls to other services needed
- Other services don't call analytics (reports are pulled, not pushed)

## Performance

### Scheduler Optimization
- Runs during off-peak hours (00:30 UTC by default)
- Aggregates previous day's data
- Uses database-level aggregation (efficient)

### Query Optimization
- Indexed columns: `date`, `camera_id`, `timestamp`
- Date range queries are optimized
- Results cached at database level

## Security

### Best Practices
1. Never commit `.env` file
2. Use strong database passwords
3. Run with minimal permissions
4. Keep dependencies updated
5. Monitor logs for errors

### Production Checklist
- [ ] Update `DATABASE_URL` with production credentials
- [ ] Set `RELOAD=false`
- [ ] Configure proper `SCHEDULER_TIMEZONE`
- [ ] Set up log rotation
- [ ] Configure firewall rules
- [ ] Enable HTTPS (use reverse proxy like Nginx)
- [ ] Set up monitoring and alerts

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f analytics`
2. Verify environment variables in `.env`
3. Test database connectivity
4. Review API documentation at `/docs`

## License

Part of GOEC (Global Operations and Event Correlation) system.
