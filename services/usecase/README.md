# Usecase Evaluation Service

A standalone microservice that receives object detection results and evaluates them against configurable business rules (usecases). Built with FastAPI, it supports both synchronous evaluation and asynchronous distributed processing via Celery workers.

---

## Table of Contents

- [Overview](#overview)
- [Folder Structure](#folder-structure)
- [Key Directories Explained](#key-directories-explained)
- [How the System Works](#how-the-system-works)
- [Setup & Installation](#setup--installation)
- [Running the Application](#running-the-application)
- [Running Workers](#running-workers)
- [Adding a New Rule](#adding-a-new-rule)
- [Environment Variables](#environment-variables)
- [Important Notes & Assumptions](#important-notes--assumptions)

---

## Overview

The Usecase Evaluation Service sits downstream of a camera detection pipeline. It receives detection payloads — lists of identified objects with their positions and confidence scores — and determines whether any configured business rule has been triggered.

Examples of rules it evaluates:

- A person has entered a restricted zone
- A crowd of three or more people has formed in a region of interest
- A phone, bag, or cash item has been detected
- Dress code compliance has been violated
- Staff presence or mopping activity has been identified

Results are persisted to a shared PostgreSQL database and returned to the caller for further routing to alert or analytics services.

The service supports two execution modes:

- **Direct mode** — evaluations run synchronously within the API process. Suitable for development and low-volume deployments.
- **Queue mode** — evaluations are offloaded to a pool of Celery workers via RabbitMQ, with results collected from Redis. Suitable for production and high-throughput scenarios.

---

## Folder Structure

```
services/usecase/
├── main.py                      # FastAPI application entry point
├── Dockerfile                   # Container definition
├── docker-compose.yml           # Full stack: API, workers, broker, cache
├── requirements.txt             # Python dependencies
├── test_usecase_pipeline.py     # Integration and unit tests
│
├── usecase/
│   ├── api.py                   # HTTP routes (request validation, response formatting)
│   ├── engine.py                # Core evaluation logic
│   ├── service.py               # Routes between direct and queue execution paths
│   ├── schemas.py               # Pydantic request/response models
│   └── rules/
│       ├── __init__.py          # Auto-discovery registry for all rules
│       ├── base.py              # Abstract base class all rules must implement
│       ├── person_in_roi.py
│       ├── crowd_in_roi.py
│       ├── restricted_zone.py
│       ├── restricted_area.py
│       ├── bag_detection.py
│       ├── cash_detection.py
│       ├── dress_code.py
│       ├── heatmap.py
│       ├── mopping_detection.py
│       ├── people_counter.py
│       ├── phone_detection.py
│       ├── smoking_detection.py
│       └── staff_detector.py
│
├── workers/
│   ├── celery_app.py            # Celery configuration (broker, backend, task routing)
│   ├── tasks.py                 # Celery task definitions (wraps evaluation logic)
│   └── queue.py                 # Task submission and result collection bridge
│
└── shared/
    ├── common/
    │   ├── config.py            # Centralised environment-aware configuration
    │   ├── logger.py            # Shared logging setup
    │   └── utils.py             # ROI operations, frame utilities, performance monitoring
    └── database/
        ├── models.py            # SQLAlchemy ORM models (Camera, Detection, UsecaseResult, Alert, AnalyticsDaily)
        ├── connection.py        # Database engine, session factory, table initialisation
        └── persistence.py      # High-level write helpers for each model
```

---

## Key Directories Explained

### `usecase/rules/`

Contains all individual business rule implementations. Each rule is a self-contained Python module that subclasses `BaseUsecaseRule` and declares a unique `USECASE_ID`. The `__init__.py` auto-discovers all rules at startup — no manual registration required. Adding a new rule is as simple as creating a new file in this directory.

### `usecase/`

The core of the service. The engine builds a lightweight detection payload and runs each requested rule against it. The service layer decides whether to run evaluations directly or submit them to the worker queue. The API layer handles HTTP concerns only — validation and response formatting.

### `workers/`

Implements the asynchronous execution path. When queue mode is enabled, the API submits one Celery task per usecase to RabbitMQ. Workers pick up tasks, run the same evaluation logic, persist results to the database, and store results in Redis. The queue bridge then collects all results asynchronously and returns them to the caller.

### `shared/common/`

Shared utilities used across the service: centralised configuration loaded from environment variables, a consistent logging setup, and helper functions for ROI operations, frame resizing, and performance tracking.

### `shared/database/`

Database access layer shared across services. Contains the SQLAlchemy models for all entities (cameras, detections, usecase results, alerts, and daily analytics), the connection and session management, and convenience functions for writing records.

---

## How the System Works

1. An upstream detection service identifies objects in a camera frame and sends a payload to this service via `POST /usecase/evaluate`. The payload includes the camera ID, the detection output, and the list of usecase rules to evaluate.

2. The API layer validates the request and passes it to the service layer.

3. The service layer checks whether queue mode is enabled:
   - **Direct mode:** The engine evaluates each requested rule sequentially within the API process and returns results immediately.
   - **Queue mode:** The engine builds a slim version of the detection payload (stripping unnecessary fields to reduce size) and submits one Celery task per usecase to RabbitMQ. The API then awaits all results asynchronously from Redis.

4. In queue mode, each worker picks up a task, calls the same evaluation function used in direct mode, persists the result to PostgreSQL, and stores the serialised result in Redis.

5. Each rule's `evaluate()` method inspects the detection list, applies its logic, and returns whether it was triggered and which objects matched.

6. The service collects all results and returns a `UsecaseResponse` containing one result per requested usecase, with the camera ID, trigger status, matched object count, and a base64 snapshot if detections were present.

7. If any individual task fails, it returns a safe default (not triggered) so the pipeline continues uninterrupted.

---

## Setup & Installation

### Prerequisites

- Python 3.11 or higher
- PostgreSQL (running and accessible) # need to setup 
- Docker and Docker Compose (for containerised setup)
- RabbitMQ and Redis (required only when running in queue mode)

### Local Setup

1. Navigate to `services/usecase/`.
2. Create and activate a Python virtual environment.
3. Install dependencies from `requirements.txt`.
4. Create a `.env` file and set the required environment variables (see [Environment Variables](#environment-variables)).
5. Ensure PostgreSQL is running. The application will create the required tables on first startup if `DATABASE_URL` is set.

### Docker Setup

The included `docker-compose.yml` starts all components of the full stack:

| Service | Port | Purpose |
|---|---|---|
| usecase | 8001 | Main FastAPI application |
| worker | — | Celery evaluation workers |
| rabbitmq | 5672 / 15672 | Message broker (AMQP / management UI) |
| redis | 6379 | Celery result backend |
| flower | 5555 | Celery monitoring dashboard |

Run the stack from `services/usecase/` using Docker Compose. Workers can be scaled independently.

---

## Running the Application

### Local

Start the FastAPI server using Uvicorn from the `services/usecase/` directory. The application binds to port 8001 by default.

- Health check: `GET /health`
- Evaluate usecases: `POST /usecase/evaluate`
- Queue status: `GET /usecase/queue-status`
- Task status: `GET /usecase/task/{task_id}`

### Docker

Use Docker Compose to build and start all containers from `services/usecase/`. The application will automatically initialise the database tables on startup.

---

## Running Workers (camera-sharded)

The service scales **per camera stream**, not per usecase. Each camera is
pinned to a shard `crc32(camera_id) % N_SHARDS`, and every shard is drained by
one worker running `--concurrency=1`. Consequences:

- A single **whole-frame task** (`workers.tasks.evaluate_frame_task`) carries
  all usecases for one frame and runs them in dependency order via
  `evaluate_all_usecases` — `parking_detection` first, then it hands
  `tracked_cars` to `gun_detection` / `vehicle_extraction`. This is why the
  ChargingSession row assembles correctly (the old per-usecase fan-out ran
  them in parallel and broke that dependency).
- Frames of one camera are processed strictly in order on its shard lane, so
  the per-camera Redis slot state (`slot:{camera_id}:{slot_id}`) never races.
- Different cameras run in parallel across shard workers.
- A per-camera in-flight guard provides backpressure: while a camera's frame
  is being processed, new frames for it are skipped rather than piling up.

**Scaling knob:** `N_SHARDS`. Keep it equal to the number of shard workers you
run, and set it identically for the API and every worker. Rule of thumb: 5 for
~5 cameras, 12–16 for ~50, 32–64 (spread across hosts) for ~500.

- **Docker:** `docker compose up` starts `worker-shard-0..4` plus RabbitMQ,
  Redis, Flower and the API. To change the shard count, add/remove
  `worker-shard-N` services in `docker-compose.yml` and update `N_SHARDS` in
  both the `x-worker-env` anchor and the `usecase` service.
- **Bare metal / systemd:** run `deploy/start_shard_workers.sh` (reads
  `N_SHARDS`, default 5) alongside the API unit.

Set `USE_WORKER_QUEUE=true` to activate queue mode. When disabled, the service
falls back to direct (synchronous) evaluation in the API process — no
RabbitMQ/Redis required (dev/test only; cameras do not run in parallel there).

The Flower dashboard (port 5555 in Docker) shows per-shard worker activity,
task history, queue depth, and retry counts.

---

## Adding a New Rule

1. Create a new Python file inside `services/usecase/usecase/rules/`.
2. Define a class that subclasses `BaseUsecaseRule` from `rules/base.py`.
3. Set a unique `USECASE_ID` class attribute — this is the string callers will use to reference the rule.
4. Implement the `evaluate(detection_output)` method. It receives the detection payload and must return a dictionary with at least a `triggered` boolean and a `matched_objects` list.
5. No further registration is needed. The auto-discovery system in `rules/__init__.py` scans the directory on startup and registers all valid rule classes automatically.

The new rule will be available immediately to any caller that includes its `USECASE_ID` in the `usecases` list of a request.

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:postgres@localhost:5432/goec` |
| `API_HOST` | Host to bind the API server | `0.0.0.0` |
| `API_PORT` | Port for the API server | `8001` |
| `API_RELOAD` | Enable auto-reload in development | `false` |
| `LOG_LEVEL` | Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) | `INFO` |
| `LOG_FILE` | Path to the log output file | _(console only)_ |
| `USE_WORKER_QUEUE` | Enable async Celery processing | `false` |
| `N_SHARDS` | Number of camera shards / single-slot worker lanes. Must match the worker count and be identical for API and workers. | `5` |
| `INFLIGHT_TTL` | Seconds the per-camera in-flight guard survives a worker crash (backstop; > frame `time_limit`). | `130` |
| `RABBITMQ_URL` | RabbitMQ AMQP connection URL | `amqp://guest:guest@localhost:5672//` |
| `REDIS_URL` | Redis connection URL for Celery results | `redis://localhost:6379/0` |
| `DEFAULT_CONFIDENCE_THRESHOLD` | Minimum detection confidence to consider | `0.5` |
| `DEFAULT_IOU_THRESHOLD` | Intersection-over-union threshold | `0.4` |
| `YOLO_MODEL_PATH` | Path to the YOLO model weights file | _(service default)_ |
| `USE_GPU` | Enable GPU acceleration for inference | `false` |
| `CAMERA_DETECTION_URL` | Internal URL of the Camera Detection Service | — |
| `USECASE_SERVICE_URL` | Internal URL of this service | — |
| `ALERT_SERVICE_URL` | Internal URL of the Alert Service | — |
| `ANALYTICS_SERVICE_URL` | Internal URL of the Analytics Service | — |

---

## Important Notes & Assumptions

- **Database is optional at startup.** If `DATABASE_URL` is not set, the service starts without a database connection. Result persistence will be skipped. This allows lightweight deployments without PostgreSQL.

- **Queue mode requires both RabbitMQ and Redis.** If `USE_WORKER_QUEUE=true` and either service is unavailable, task submission will fail. Use direct mode for environments without these dependencies.

- **Persistence happens in workers, not the API.** In queue mode, database writes are performed inside the Celery worker after evaluation completes. This keeps the API non-blocking and allows it to return results without waiting for DB I/O.

- **Failed tasks do not break the pipeline.** If a worker task raises an exception (after retries), the queue layer returns a safe default result — `triggered=false`, `matched_count=0` — for that usecase, so the overall response is always complete.

- **Slim payloads are used in queue mode.** Before submitting tasks to RabbitMQ, the engine strips the detection payload of unnecessary fields (raw tensors, metadata). This reduces message size significantly when evaluating many rules per cycle.

- **ROI filtering has been removed from the detection layer.** All detections are passed to the rules regardless of their position. Rules that historically relied on ROI filtering continue to work via helper methods in `BaseUsecaseRule`, but the filtering is no longer applied at ingestion time.

- **Rules are discovered automatically.** The registry scans the `rules/` directory at import time. Any file containing a valid `BaseUsecaseRule` subclass with a `USECASE_ID` will be registered without any manual changes to the codebase.

- **Workers can be scaled horizontally.** The Celery worker pool is stateless. Run as many worker instances as needed to increase throughput. The Docker Compose setup supports `--scale worker=N`.
