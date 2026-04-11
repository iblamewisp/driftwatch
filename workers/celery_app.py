import asyncio
import functools

from celery import Celery

from app.config import settings

# ── Event loop ────────────────────────────────────────────────────────────────
# One loop per worker process, created on first use, reused for its lifetime.
# IMPORTANT: requires Celery prefork pool (the default, -P prefork).
# Breaks with -P threads / -P gevent / -P eventlet — do not switch pool type.
_worker_loop: asyncio.AbstractEventLoop | None = None


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop


# ── App ───────────────────────────────────────────────────────────────────────

celery_app = Celery(
    "driftwatch",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["workers.evaluator", "workers.detection"],
)

celery_app.conf.beat_schedule = {
    "drift-detection": {
        "task": "workers.detection.run_drift_detection",
        "schedule": settings.DRIFT_DETECTION_INTERVAL,
    },
    "cluster-split": {
        "task": "workers.detection.run_cluster_split",
        "schedule": settings.CLUSTER_SPLIT_INTERVAL,
    },
    "recover-unclustered": {
        "task": "workers.detection.recover_unclustered_responses",
        "schedule": settings.RECOVERY_INTERVAL,
    },
}

celery_app.conf.task_routes = {
    "workers.evaluator.*": {"queue": "evaluator"},
    "workers.detection.*": {"queue": "detection"},
}

celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]

# Acknowledge tasks only after completion — broker requeues on worker crash.
celery_app.conf.task_acks_late = True
celery_app.conf.broker_transport_options = {
    "visibility_timeout": settings.CELERY_VISIBILITY_TIMEOUT,
}


# ── Decorator ─────────────────────────────────────────────────────────────────

def async_task(**kwargs):
    """
    Define a Celery task as an async function.

    The worker's per-process event loop runs the coroutine synchronously via
    run_until_complete. All Celery task options (bind, max_retries, name, etc.)
    are forwarded unchanged.

    Usage:
        @async_task(name="workers.foo.do_thing")
        async def do_thing() -> None:
            await some_io()

        @async_task(name="workers.foo.do_bound", bind=True, max_retries=3)
        async def do_bound(self, arg: str) -> None:
            ...
            raise self.retry(countdown=5)
    """
    def decorator(async_fn):
        @celery_app.task(**kwargs)
        @functools.wraps(async_fn)
        def _sync(*args, **fn_kwargs):
            _get_worker_loop().run_until_complete(async_fn(*args, **fn_kwargs))
        return _sync
    return decorator
