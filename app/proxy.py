import asyncio
import hashlib
import threading
import time

import redis.asyncio as aioredis
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import settings
from app.providers.registry import get_provider
from app.schemas.anthropic import AnthropicRequest
from app.schemas.internal import LoggedResponse, UniformResponse
from app.schemas.openai import ChatCompletionRequest
from db.repositories import response_repo
from db.session import AsyncSessionLocal
from monitoring.logging import get_logger
from monitoring.metrics import LATENCY_HISTOGRAM, REQUEST_COUNTER
from services.circuit_breaker import (
    CircuitBreakerOpen,
    get_provider_breaker,
    redis_breaker,
)

router = APIRouter()
logger = get_logger("proxy")

_counter_lock = threading.Lock()
_request_counter = 0


def _increment_counter() -> int:
    global _request_counter
    with _counter_lock:
        _request_counter += 1
        return _request_counter


def _extract_request_text(messages: list) -> str:
    """Extract the last user message as the query text for embedding."""
    for msg in reversed(messages):
        role = msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")
        content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role == "user":
            if isinstance(content, str):
                return content
            # Anthropic multi-part content blocks
            if isinstance(content, list):
                return "".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
    return ""


def _build_forwarded_headers(request: Request) -> dict[str, str]:
    skip = {"x-driftwatch-key", "host", "content-length"}
    return {k: v for k, v in request.headers.items() if k.lower() not in skip}


async def _xadd(redis: aioredis.Redis, response_id: str, request_text: str, response_text: str) -> None:
    await redis.xadd(
        "driftwatch:clustering",
        {
            "response_id": response_id,
            "request_text": request_text,
            "response_text": response_text,
        },
    )


async def _log_and_enqueue(
    uniform: UniformResponse,
    prompt_hash: str,
    latency_ms: int,
    request_id: str,
    redis: aioredis.Redis,
    request_text: str = "",
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
    )

    async with AsyncSessionLocal() as session:
        response_id = await response_repo.insert_response(session, logged)

    logger.info(
        "response_logged",
        request_id=request_id,
        model=logged.model,
        provider=uniform.provider,
        latency_ms=latency_ms,
    )

    try:
        await redis_breaker.call(_xadd(redis, str(response_id), request_text, uniform.content))
    except CircuitBreakerOpen:
        logger.warning("redis_circuit_open_skipping_clustering", request_id=request_id)
    except Exception as exc:
        logger.error("redis_xadd_failed", request_id=request_id, error=str(exc))

    counter = _increment_counter()
    if counter % settings.SAMPLING_RATE == 0:
        from workers.evaluator import evaluate_response
        evaluate_response.delay(str(response_id))


# ── OpenAI-compatible endpoint ────────────────────────────────────────────────

@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    request_id = request.state.request_id
    body = await request.json()
    parsed = ChatCompletionRequest.model_validate(body)

    provider = get_provider(parsed.model)
    provider_name = provider.__class__.__name__.replace("Provider", "").lower()
    breaker = get_provider_breaker(provider_name)

    request_text = _extract_request_text(parsed.messages)
    forwarded = _build_forwarded_headers(request)
    headers = provider.prepare_headers(forwarded)
    prompt_hash = hashlib.sha256(
        parsed.model_dump_json(include={"messages"}).encode()
    ).hexdigest()

    http_client = request.app.state.http_client
    redis = request.app.state.redis

    REQUEST_COUNTER.labels(model=parsed.model, stream=str(parsed.stream)).inc()

    if not parsed.stream:
        start = time.monotonic()
        try:
            resp = await breaker.call(
                http_client.post(provider.upstream_url(), json=body, headers=headers)
            )
        except CircuitBreakerOpen as exc:
            logger.error("provider_circuit_open", provider=provider_name, request_id=request_id)
            return JSONResponse(status_code=503, content={"detail": str(exc)})

        latency_ms = int((time.monotonic() - start) * 1000)
        LATENCY_HISTOGRAM.observe(latency_ms)

        raw = resp.json()
        uniform = provider.extract_uniform(raw, parsed.model)

        asyncio.create_task(
            _log_and_enqueue(uniform, prompt_hash, latency_ms, request_id, redis, request_text)
        )
        return JSONResponse(content=raw, status_code=resp.status_code)

    # ── Streaming ──
    if breaker.state.value == "open":
        return JSONResponse(status_code=503, content={"detail": f"Circuit breaker '{provider_name}' is OPEN"})

    async def _stream_generator():
        start = time.monotonic()
        uniform: UniformResponse | None = None

        try:
            async for line, maybe_uniform in provider.stream(http_client, body, headers):
                breaker._on_success()
                yield line
                if maybe_uniform is not None:
                    uniform = maybe_uniform

        except CircuitBreakerOpen as exc:
            logger.error("provider_circuit_open_stream", provider=provider_name, request_id=request_id)
            yield f"data: {{'error': '{exc}'}}\n\n"
            return
        except Exception as exc:
            breaker._on_failure(exc)
            raise

        if uniform is not None:
            latency_ms = int((time.monotonic() - start) * 1000)
            LATENCY_HISTOGRAM.observe(latency_ms)
            asyncio.create_task(
                _log_and_enqueue(uniform, prompt_hash, latency_ms, request_id, redis, request_text)
            )

    return StreamingResponse(_stream_generator(), media_type="text/event-stream")


# ── Anthropic-native endpoint ─────────────────────────────────────────────────

@router.post("/v1/messages")
async def messages(request: Request):
    request_id = request.state.request_id
    body = await request.json()
    parsed = AnthropicRequest.model_validate(body)

    from app.providers.anthropic import AnthropicProvider
    provider = AnthropicProvider()
    breaker = get_provider_breaker("anthropic")

    request_text = _extract_request_text(parsed.messages)
    forwarded = _build_forwarded_headers(request)
    headers = provider.prepare_headers(forwarded)
    prompt_hash = hashlib.sha256(
        parsed.model_dump_json(include={"messages"}).encode()
    ).hexdigest()

    http_client = request.app.state.http_client
    redis = request.app.state.redis

    REQUEST_COUNTER.labels(model=parsed.model, stream=str(parsed.stream)).inc()

    if not parsed.stream:
        start = time.monotonic()
        try:
            resp = await breaker.call(
                http_client.post(provider.upstream_url(), json=body, headers=headers)
            )
        except CircuitBreakerOpen as exc:
            logger.error("anthropic_circuit_open", request_id=request_id)
            return JSONResponse(status_code=503, content={"detail": str(exc)})

        latency_ms = int((time.monotonic() - start) * 1000)
        LATENCY_HISTOGRAM.observe(latency_ms)

        raw = resp.json()
        uniform = provider.extract_uniform(raw, parsed.model)

        asyncio.create_task(
            _log_and_enqueue(uniform, prompt_hash, latency_ms, request_id, redis, request_text)
        )
        return JSONResponse(content=raw, status_code=resp.status_code)

    if breaker.state.value == "open":
        return JSONResponse(status_code=503, content={"detail": "Circuit breaker 'anthropic' is OPEN"})

    async def _stream_generator():
        start = time.monotonic()
        uniform: UniformResponse | None = None

        try:
            async for line, maybe_uniform in provider.stream(http_client, body, headers):
                breaker._on_success()
                yield line
                if maybe_uniform is not None:
                    uniform = maybe_uniform

        except CircuitBreakerOpen as exc:
            logger.error("anthropic_circuit_open_stream", request_id=request_id)
            yield f"data: {{'error': '{exc}'}}\n\n"
            return
        except Exception as exc:
            breaker._on_failure(exc)
            raise

        if uniform is not None:
            latency_ms = int((time.monotonic() - start) * 1000)
            LATENCY_HISTOGRAM.observe(latency_ms)
            asyncio.create_task(
                _log_and_enqueue(uniform, prompt_hash, latency_ms, request_id, redis, request_text)
            )

    return StreamingResponse(_stream_generator(), media_type="text/event-stream")
