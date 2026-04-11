import asyncio
import functools
import threading

from celery import Celery

from app.config import settings

# ── Event loop ────────────────────────────────────────────────────────────────
# One loop per thread, created on first use, reused for its lifetime.
# threading.local() makes this safe under -P threads (each thread owns its loop).
# With the default prefork pool there is one thread per worker process, so
# behaviour is identical to a plain global.
# Note: -P gevent and -P eventlet are still unsupported — run_until_complete is
# incompatible with cooperative multitasking schedulers regardless of this fix.
_thread_local = threading.local()


def _get_worker_loop() -> asyncio.AbstractEventLoop:
    loop: asyncio.AbstractEventLoop | None = getattr(_thread_local, "loop", None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _thread_local.loop = loop
    return loop


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
