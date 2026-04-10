import asyncio

from celery import Celery

from app.config import settings

# One event loop per worker process, created on first task, reused forever.
# Never destroyed — worker process lifetime == loop lifetime.
_worker_loop: asyncio.AbstractEventLoop | None = None


def get_worker_loop() -> asyncio.AbstractEventLoop:
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop


celery_app = Celery(
    "driftwatch",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["workers.evaluator", "workers.detection"],
)

celery_app.conf.beat_schedule = {
    "drift-detection-every-10-minutes": {
        "task": "workers.detection.run_drift_detection",
        "schedule": 600.0,
    },
    "cluster-split-check-every-hour": {
        "task": "workers.detection.run_cluster_split",
        "schedule": 3600.0,
    },
    "recover-unclustered-every-5-minutes": {
        "task": "workers.detection.recover_unclustered_responses",
        "schedule": 300.0,
    },
}

celery_app.conf.task_routes = {
    "workers.evaluator.*": {"queue": "evaluator"},
    "workers.detection.*": {"queue": "detection"},
}

celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]

# Acknowledge tasks only after successful completion — if the worker dies mid-task,
# the broker requeues it after visibility_timeout seconds instead of losing it.
celery_app.conf.task_acks_late = True
celery_app.conf.broker_transport_options = {
    "visibility_timeout": 3600,  # 1 hour — must be > longest expected task duration
}
