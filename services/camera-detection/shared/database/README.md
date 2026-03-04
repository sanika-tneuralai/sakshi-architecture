# Shared Database Package

A standalone database package containing all database models and connection utilities for the GOEC project. This package can be copied to all service branches, allowing each service to import only the models they need while sharing a common database schema.

## 📦 Package Contents

```
shared/database/
├── __init__.py          # Package initialization and exports
├── connection.py        # Database connection management
├── models.py            # All database models
└── README.md            # This file
```

## 🗄️ Database Models

### Camera
Stores camera configuration and metadata.

**Fields:**
- `camera_id` (String, PK): Unique camera identifier
- `name` (String): Human-readable camera name
- `location` (String): Camera location description
- `created_at` (DateTime): Timestamp when camera was added

### Detection
Stores object detection results from camera feeds.

**Fields:**
- `detection_id` (Integer, PK): Auto-incrementing ID
- `camera_id` (String, FK): Reference to cameras table
- `timestamp` (DateTime): When detection occurred
- `object_type` (String): Type of detected object (e.g., 'person', 'vehicle')
- `confidence` (Float): Detection confidence score (0.0 to 1.0)
- `inside_roi` (Boolean): Whether object is inside region of interest
- `screenshot_path` (String): Optional path to detection screenshot

### UsecaseResult
Stores use case evaluation results.

**Fields:**
- `result_id` (Integer, PK): Auto-incrementing ID
- `camera_id` (String, FK): Reference to cameras table
- `usecase_name` (String): Name of the evaluated use case
- `detection_id` (Integer, FK): Optional reference to detections table
- `triggered` (Boolean): Whether the use case was triggered
- `timestamp` (DateTime): When evaluation occurred

### Alert
Stores alert records when use cases are triggered.

**Fields:**
- `alert_id` (Integer, PK): Auto-incrementing ID
- `camera_id` (String, FK): Reference to cameras table
- `usecase_name` (String): Name of the use case that triggered alert
- `alert_type` (String): Type of alert (e.g., 'email', 'sms', 'webhook')
- `timestamp` (DateTime): When alert was triggered
- `status` (String): Alert status ('sent' or 'failed')
- `detection_id` (Integer, FK): Optional reference to detections table
- `screenshot_path` (String): Optional path to alert screenshot

### AnalyticsDaily
Stores daily aggregated analytics data.

**Fields:**
- `id` (Integer, PK): Auto-incrementing ID
- `date` (Date): Date of the analytics record
- `camera_id` (String, FK): Reference to cameras table
- `total_detections` (Integer): Total number of detections for the day
- `roi_violations` (Integer): Number of ROI violations for the day
- `alerts_sent` (Integer): Number of alerts sent for the day

**Constraints:**
- Unique constraint on (`date`, `camera_id`) combination

## ⚙️ Configuration

### Environment Variables

Set the `DATABASE_URL` environment variable to configure the database connection:

```bash
export DATABASE_URL="postgresql://user:password@hostname:5432/database_name"
```

**Default:** `postgresql://postgres:postgres@localhost:5432/goec`

### Connection Parameters

The connection is configured with the following defaults:
- **Pool Size:** 5 connections
- **Max Overflow:** 10 additional connections
- **Pool Pre-Ping:** Enabled (tests connections before use)
- **Echo:** Disabled (set to True for SQL query logging)

## 🚀 Usage

### Basic Setup

1. **Copy the shared database package to your service:**
   ```bash
   cp -r shared/ /path/to/your/service/
   ```

2. **Set the DATABASE_URL environment variable:**
   ```bash
   export DATABASE_URL="postgresql://user:password@host:5432/goec"
   ```

3. **Initialize the database in your service:**
   ```python
   from shared.database import init_db
   
   # Call once at startup
   init_db()
   ```

### Service-Specific Usage

#### Camera-Detection Service
```python
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from shared.database import get_db, init_db
from shared.database.models import Camera, Detection

app = FastAPI()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

# Example endpoint
@app.get("/cameras")
def get_cameras(db: Session = Depends(get_db)):
    return db.query(Camera).all()

@app.post("/detections")
def create_detection(detection_data: dict, db: Session = Depends(get_db)):
    detection = Detection(
        camera_id=detection_data["camera_id"],
        object_type=detection_data["object_type"],
        confidence=detection_data["confidence"],
        inside_roi=detection_data.get("inside_roi", False)
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection
```

#### Usecase Service
```python
from shared.database import get_db, init_db
from shared.database.models import Detection, UsecaseResult

# Query detections
def evaluate_usecase(camera_id: str, db: Session):
    detections = db.query(Detection).filter(
        Detection.camera_id == camera_id,
        Detection.inside_roi == True
    ).all()
    
    # Evaluate and store result
    result = UsecaseResult(
        camera_id=camera_id,
        usecase_name="person_in_roi",
        triggered=len(detections) > 0
    )
    db.add(result)
    db.commit()
    return result
```

#### Alert Service
```python
from shared.database import get_db, init_db
from shared.database.models import Alert, UsecaseResult

# Create alert from triggered usecase
def create_alert(usecase_result: UsecaseResult, db: Session):
    if usecase_result.triggered:
        alert = Alert(
            camera_id=usecase_result.camera_id,
            usecase_name=usecase_result.usecase_name,
            alert_type="email",
            status="sent",
            detection_id=usecase_result.detection_id
        )
        db.add(alert)
        db.commit()
        return alert
```

#### Analytics Service
```python
from datetime import date
from sqlalchemy import func
from shared.database import get_db, init_db
from shared.database.models import Detection, Alert, AnalyticsDaily

# Aggregate daily analytics
def aggregate_daily_analytics(target_date: date, camera_id: str, db: Session):
    total_detections = db.query(func.count(Detection.detection_id)).filter(
        func.date(Detection.timestamp) == target_date,
        Detection.camera_id == camera_id
    ).scalar()
    
    roi_violations = db.query(func.count(Detection.detection_id)).filter(
        func.date(Detection.timestamp) == target_date,
        Detection.camera_id == camera_id,
        Detection.inside_roi == True
    ).scalar()
    
    alerts_sent = db.query(func.count(Alert.alert_id)).filter(
        func.date(Alert.timestamp) == target_date,
        Alert.camera_id == camera_id,
        Alert.status == "sent"
    ).scalar()
    
    # Create or update analytics record
    analytics = AnalyticsDaily(
        date=target_date,
        camera_id=camera_id,
        total_detections=total_detections or 0,
        roi_violations=roi_violations or 0,
        alerts_sent=alerts_sent or 0
    )
    db.merge(analytics)
    db.commit()
    return analytics
```

#### Orchestration Service
```python
from shared.database import get_db, init_db
from shared.database.models import Camera

# Orchestrator typically just needs to read camera config
@app.get("/pipeline/execute")
async def execute_pipeline(camera_id: str, db: Session = Depends(get_db)):
    camera = db.query(Camera).filter(Camera.camera_id == camera_id).first()
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    
    # Make HTTP calls to other services
    # ...
```

### Direct Model Import
```python
# Import only the models you need
from shared.database.models import Camera, Detection
from shared.database.connection import get_db, init_db

# Or import everything
from shared.database import *
```

### Testing Database Connection
```python
from shared.database import test_connection

if test_connection():
    print("Database connection successful!")
else:
    print("Database connection failed!")
```

## 🔧 Database Initialization

The package includes an `init_db()` function that creates all tables:

```python
from shared.database import init_db

# Call this once during application startup
init_db()
```

This will:
1. Connect to the database using DATABASE_URL
2. Create all tables if they don't exist
3. Test the connection
4. Print status messages

## 📋 Requirements

Add these to your service's `requirements.txt`:

```txt
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0
```

For async support (optional):
```txt
asyncpg>=0.29.0
sqlalchemy[asyncio]>=2.0.0
```

## 🔄 Updating the Shared Package

When you update the shared database package:

1. Make changes in one service's `shared/database/` directory
2. Test thoroughly
3. Copy the updated `shared/database/` to all other services:
   ```bash
   # From the service with updates
   cp -r shared/database/ ../other-service/shared/
   ```

## 🗂️ Service Import Matrix

| Service | Models Used |
|---------|-------------|
| **Camera-Detection** | Camera, Detection |
| **Orchestration** | Camera |
| **Usecase** | Detection, UsecaseResult |
| **Alert** | Alert, UsecaseResult |
| **Analytics** | Detection, Alert, AnalyticsDaily |

## 🚨 Important Notes

1. **Single Database:** All services connect to the same PostgreSQL database
2. **Standalone Package:** This package has no dependencies on other backend modules
3. **Model Availability:** All models are available in each service, but import only what you need
4. **Connection Pooling:** Connection pooling is enabled to handle concurrent requests
5. **Thread Safety:** The session factory is thread-safe when used with `get_db()`
6. **Environment First:** Always configure via environment variables, never hardcode credentials

## 🐛 Troubleshooting

### Connection Issues
```python
# Check if DATABASE_URL is set correctly
from shared.database import DATABASE_URL, test_connection

print(f"Database URL: {DATABASE_URL}")
if not test_connection():
    print("Cannot connect to database!")
```

### Import Errors
Make sure `shared/` is in your Python path:
```python
import sys
sys.path.insert(0, '/path/to/your/service')
```

Or use relative imports:
```python
from .shared.database import Camera, Detection
```

### Table Creation Issues
Ensure PostgreSQL is running and accessible:
```bash
psql -h hostname -U username -d database_name -c "SELECT 1"
```

## 📄 License

Part of the GOEC project. Internal use only.
