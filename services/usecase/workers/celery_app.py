import os
import sys
from pathlib import Path

# Add parent directory to path so imports work regardless of where celery is run from
PARENT_DIR = str(Path(__file__).parent.parent)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from celery import Celery


# Broker is RabbitMQ
RABBITMQ_URL = os.getenv(
    "RABBITMQ_URL",
    "amqp://guest:guest@localhost:5672//",
)

# Backend = Redis(result storage)
REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0",
)

# create celery app

celery_app = Celery(
    "usecase_worker", #app name that will be shown in logs to monitor which worker is processing the task
    broker=RABBITMQ_URL,
    backend=REDIS_URL
)

#CONFIGURATION
celery_app.conf.update(
    task_serializer = 'json', # How tasks are encoded when sent to RabbitMQ
    result_serializer = 'json', # How results are encoded when stored in Redis
    accept_content = ['json'], # What content types the worker will accept
    timezone = 'UTC',
    enable_utc = True,

    result_expires = 3600, # How long to keep results in Redis (in seconds)

    task_routes = {
        'workers.tasks.evaluate_usecase_task': {'queue': 'usecase_queue'},
    }, # Route specific tasks to specific queues (optional, but good for scaling and organization)

    worker_prefetch_multiplier = 1, # How many tasks a worker prefetches before processing, 1 means a worker only takes ONE task at a time, processes it fully,then takes the next.

    task_acks_late = True, # Acknowledge tasks after they have been processed, not just received. This ensures that if a worker crashes while processing a task, the task will be re-queued and not lost.

    task_ignore_result = False # Don't store results for tasks that raised exceptions, we handle errors explicitly in the task itself.

)

# Explicitly import tasks module after sys.path is configured
# This ensures tasks are registered and imports work correctly
from . import tasks