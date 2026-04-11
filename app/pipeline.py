"""
Post-response pipeline: persist to DB, enqueue for clustering, schedule evaluation.

Called as a background task (asyncio.create_task) so it never blocks the response path.
"""

import redis.asyncio as aioredis

from app.config import settings
from app.schemas.internal import LoggedResponse, UniformResponse
from db.repositories import response_repo
from db.session import AsyncSessionLocal
from monitoring.logging import get_logger
from services.circuit_breaker import CircuitBreakerOpen, redis_breaker

logger = get_logger("pipeline")

_SAMPLING_COUNTER_KEY = "driftwatch:sampling_counter"


async def log_and_enqueue(
    *,
    uniform: UniformResponse,
    prompt_hash: str,
    latency_ms: int,
    request_id: str,
    request_text: str,
    redis: aioredis.Redis,
) -> None:
    logged = LoggedResponse(
        request_id=request_id,
        prompt_hash=prompt_hash,
        model=uniform.model,
        prompt_tokens=uniform.prompt_tokens,
        completion_tokens=uniform.completion_tokens,
        latency_ms=latency_ms,
        finish_reason=uniform.finish_reason,
        raw_content=uniform.content,
        request_text=request_text,
    )

    async with AsyncSessionLocal() as session:
        response_id = await response_repo.insert_response(session, logged)

    logger.info("response_logged", request_id=request_id, model=logged.model,
                provider=uniform.provider, latency_ms=latency_ms)

    xadd_ok = False
    try:
        await redis_breaker.call(redis.xadd(
            "driftwatch:clustering",
            {"response_id": str(response_id), "request_text": request_text, "response_text": uniform.content},
        ))
        xadd_ok = True
    except CircuitBreakerOpen:
        logger.warning("redis_circuit_open_skipping_clustering",
                       request_id=request_id, response_id=str(response_id))
    except Exception as exc:
        logger.error("redis_xadd_failed", request_id=request_id,
                     response_id=str(response_id), error=str(exc))

    if xadd_ok:
        async with AsyncSessionLocal() as session:
            await response_repo.mark_clustering_enqueued(session, response_id)

    try:
        count = await redis.incr(_SAMPLING_COUNTER_KEY)
        if count % settings.SAMPLING_RATE == 0:
            async with AsyncSessionLocal() as session:
                await response_repo.mark_needs_evaluation(session, response_id)
    except Exception as exc:
        logger.warning("sampling_counter_failed_skipping_evaluation",
                       request_id=request_id, error=str(exc))
