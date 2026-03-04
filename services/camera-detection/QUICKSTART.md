# Camera-Detection Service - Quick Start Guide

## 🚀 Quick Start (5 Minutes)

### Prerequisites Checklist
- [ ] Docker and Docker Compose installed
- [ ] NVIDIA GPU with drivers installed
- [ ] NVIDIA Docker runtime configured
- [ ] PostgreSQL accessible (or use included container)
- [ ] YOLO model file in `models/` directory

### Step 1: Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (minimal required changes)
nano .env
```

**Required settings:**
```env
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/goec
```

### Step 2: Verify GPU Access
```bash
# Test NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

Expected output: GPU information displayed

### Step 3: Start Services
```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f camera-detection
```

Wait for: `Camera-Detection service started successfully`

### Step 4: Verify Service
```bash
# Health check
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy","service":"camera-detection","version":"1.0.0"}

# Open API docs
xdg-open http://localhost:8000/docs
```

### Step 5: Test Camera & Detection

#### Start a Camera
```bash
curl -X POST "http://localhost:8000/camera/start" \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "test_cam_1",
    "rtsp_url": "rtsp://admin:admin@192.168.1.100:554/stream",
    "fps": 5
  }'
```

#### Check Status
```bash
curl "http://localhost:8000/camera/status/test_cam_1"
```

#### Run Detection
```bash
curl -X POST "http://localhost:8000/detection/detect" \
  -H "Content-Type: application/json" \
  -d '{
    "camera_id": "test_cam_1",
    "confidence": 0.5,
    "save_screenshot": true
  }'
```

#### Stop Camera
```bash
curl -X DELETE "http://localhost:8000/camera/stop/test_cam_1"
```

---

## 🐛 Troubleshooting

### Issue: GPU not detected
```bash
# Check nvidia-docker runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# If fails, verify /etc/docker/daemon.json has:
{
  "default-runtime": "nvidia",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}

# Restart Docker
sudo systemctl restart docker
```

### Issue: Database connection failed
```bash
# Check postgres is running
docker-compose ps postgres

# If not running, start it
docker-compose up -d postgres

# Check logs
docker-compose logs postgres
```

### Issue: Model not found
```bash
# Check model exists
ls -lh models/

# If missing, download YOLO model
# Place yolo11n.pt in models/ directory
```

### Issue: RTSP stream error
```bash
# Test RTSP URL directly
ffplay "rtsp://admin:admin@192.168.1.100:554/stream"

# Check if camera is reachable
ping 192.168.1.100

# Verify credentials and URL format
```

---

## 📊 Monitoring

### View Logs
```bash
# All logs
docker-compose logs -f

# Only camera-detection
docker-compose logs -f camera-detection

# Last 100 lines
docker-compose logs --tail=100 camera-detection
```

### Check Resource Usage
```bash
# GPU usage
watch -n 1 nvidia-smi

# Container stats
docker stats camera-detection

# Disk usage (screenshots)
du -sh screenshots/
```

### Database Queries
```bash
# Connect to database
docker exec -it goec-postgres psql -U postgres -d goec

# Check recent detections
SELECT * FROM detections ORDER BY timestamp DESC LIMIT 10;

# Check cameras
SELECT * FROM cameras;

# Exit
\q
```

---

## 🛑 Stop & Cleanup

### Stop Services
```bash
# Stop all
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Stop and remove images
docker-compose down --rmi all
```

### Clean Logs & Screenshots
```bash
# Remove logs
rm -f logs/*.log
rm -f camera_detection_api.log

# Clean screenshots
rm -f screenshots/*.jpg screenshots/*.png
```

---

## 🔧 Configuration Tips

### For < 10 Cameras
```env
DEFAULT_CAMERA_FPS=5
WORKER_THREADS=2
```

### For 100+ Cameras
```env
DEFAULT_CAMERA_FPS=3
WORKER_THREADS=8
MAX_QUEUE_SIZE=200
```

### High Accuracy Detection
```env
DEFAULT_CONFIDENCE_THRESHOLD=0.7
```

### Fast Detection (More False Positives)
```env
DEFAULT_CONFIDENCE_THRESHOLD=0.3
```

---

## 📖 Next Steps

1. **Read Full Documentation**: [README.md](README.md)
2. **Explore API**: http://localhost:8000/docs
3. **Configure ROI**: See API docs for ROI polygon setup
4. **Integrate with Orchestrator**: This service is designed to be called by an orchestrator
5. **Performance Tuning**: Adjust FPS, confidence, and worker threads

---

## 🆘 Need Help?

- Check logs: `docker-compose logs -f camera-detection`
- Review README: [README.md](README.md)
- Test API: http://localhost:8000/docs
- Check database: `docker exec -it goec-postgres psql -U postgres -d goec`

---

## ✅ Success Checklist

After setup, you should have:
- [ ] Service running: `docker ps | grep camera-detection`
- [ ] Health check passing: `curl localhost:8000/health`
- [ ] API docs accessible: http://localhost:8000/docs
- [ ] Database connected: Check logs for "Database initialized"
- [ ] GPU detected: Check logs for "GStreamer version"
- [ ] Camera can start: Test with `/camera/start`
- [ ] Detection works: Test with `/detection/detect`

**All green? You're ready to go! 🎉**
