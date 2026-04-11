"""
HTTP forwarding logic: proxy a request to an upstream provider, stream or unary.

Both functions are pure request/response — no DB, no Redis.
Side effects (logging, enqueuing) are handed off to pipeline.log_and_enqueue
as a background task.
"""

import asyncio
import time

import redis.asyncio as aioredis
from fastapi.responses import JSONResponse, StreamingResponse

from app.pipeline import log_and_enqueue
from app.providers.base import AbstractProvider
from app.schemas.internal import UniformResponse
from monitoring.logging import get_logger
from monitoring.metrics import LATENCY_HISTOGRAM
from services.circuit_breaker import CircuitBreaker, CircuitBreakerOpen

logger = get_logger("forwarding")


async def forward_unary(
    *,
    provider: AbstractProvider,
    breaker: CircuitBreaker,
    body: dict,
    headers: dict,
    http_client,
    redis: aioredis.Redis,
    request_id: str,
    request_text: str,
    prompt_hash: str,
    model: str,
) -> JSONResponse:
    start = time.monotonic()
    try:
        resp = await breaker.call(
            http_client.post(provider.upstream_url(), json=body, headers=headers)
        )
    except CircuitBreakerOpen as exc:
        logger.error("provider_circuit_open", request_id=request_id)
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    latency_ms = int((time.monotonic() - start) * 1000)
    LATENCY_HISTOGRAM.observe(latency_ms)
    raw = resp.json()
    uniform = provider.extract_uniform(raw, model)
    asyncio.create_task(log_and_enqueue(
        uniform=uniform, prompt_hash=prompt_hash, latency_ms=latency_ms,
        request_id=request_id, request_text=request_text, redis=redis,
    ))
    return JSONResponse(content=raw, status_code=resp.status_code)


async def forward_streaming(
    *,
    provider: AbstractProvider,
    breaker: CircuitBreaker,
    body: dict,
    headers: dict,
    http_client,
    redis: aioredis.Redis,
    request_id: str,
    request_text: str,
    prompt_hash: str,
    model: str,
) -> StreamingResponse | JSONResponse:
    if breaker.state.value == "open":
        return JSONResponse(status_code=503, content={"detail": "Circuit breaker is OPEN"})

    async def _generate():
        start = time.monotonic()
        uniform: UniformResponse | None = None
        try:
            async for line, maybe_uniform in provider.stream(http_client, body, headers):
                breaker._on_success()
                yield line
                if maybe_uniform is not None:
                    uniform = maybe_uniform
        except CircuitBreakerOpen as exc:
            logger.error("provider_circuit_open_stream", request_id=request_id)
            yield f"data: {{'error': '{exc}'}}\n\n"
            return
        except Exception as exc:
            breaker._on_failure(exc)
            raise

        if uniform is not None:
            latency_ms = int((time.monotonic() - start) * 1000)
            LATENCY_HISTOGRAM.observe(latency_ms)
            asyncio.create_task(log_and_enqueue(
                uniform=uniform, prompt_hash=prompt_hash, latency_ms=latency_ms,
                request_id=request_id, request_text=request_text, redis=redis,
            ))

    return StreamingResponse(_generate(), media_type="text/event-stream")
