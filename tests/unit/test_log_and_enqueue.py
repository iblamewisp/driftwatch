"""
Unit tests for pipeline.log_and_enqueue — the fire-and-forget logging pipeline.

What we test:
  - LoggedResponse is constructed correctly from UniformResponse
  - response_repo.insert_response is called with the right payload
  - mark_clustering_enqueued is called after a successful XADD
  - mark_clustering_enqueued is NOT called when the circuit breaker is open
  - CircuitBreakerOpen and generic Redis errors are swallowed (best-effort)
  - mark_needs_evaluation is called when the Redis sampling counter hits the rate
  - mark_needs_evaluation is NOT called between samples
  - sampling counter failure is swallowed — evaluation is skipped silently
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.pipeline import log_and_enqueue
from app.schemas.internal import UniformResponse
from services.circuit_breaker import CircuitBreakerOpen

FIXED_RESPONSE_ID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_uniform(**overrides) -> UniformResponse:
    defaults = dict(
        content="The capital of France is Paris.",
        model="gpt-4o",
        prompt_tokens=20,
        completion_tokens=10,
        finish_reason="stop",
        provider="openai",
    )
    return UniformResponse(**(defaults | overrides))


def make_redis(*, incr_return: int = 999) -> AsyncMock:
    """Return a Redis mock whose incr() returns incr_return by default (no evaluation)."""
    redis = AsyncMock()
    redis.xadd = AsyncMock(return_value="1-0")
    redis.incr = AsyncMock(return_value=incr_return)
    return redis


@pytest.fixture(autouse=True)
def isolate_db_and_redis():
    """
    Patch DB session and repo methods for every test in this module.
    All patches target app.pipeline — the module that actually imports these names.
    """
    session = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("app.pipeline.AsyncSessionLocal", return_value=cm),
        patch("app.pipeline.response_repo.insert_response", new=AsyncMock(return_value=FIXED_RESPONSE_ID)),
        patch("app.pipeline.response_repo.mark_clustering_enqueued", new=AsyncMock()),
        patch("app.pipeline.response_repo.mark_needs_evaluation", new=AsyncMock()),
        patch("app.pipeline.redis_breaker") as mock_breaker,
    ):
        mock_breaker.call = AsyncMock(return_value=None)
        yield


# ── DB insert ─────────────────────────────────────────────────────────────────

async def test_inserts_response_to_db():
    await log_and_enqueue(
        uniform=make_uniform(),
        prompt_hash="hash-abc",
        latency_ms=120,
        request_id="req-001",
        request_text="What is the capital of France?",
        redis=make_redis(),
    )

    from app.pipeline import response_repo
    logged = response_repo.insert_response.call_args[0][1]
    assert logged.model == "gpt-4o"
    assert logged.raw_content == "The capital of France is Paris."
    assert logged.prompt_tokens == 20
    assert logged.completion_tokens == 10
    assert logged.finish_reason == "stop"
    assert logged.prompt_hash == "hash-abc"
    assert logged.latency_ms == 120
    assert logged.request_text == "What is the capital of France?"


async def test_anthropic_fields_mapped_correctly():
    uniform = make_uniform(
        model="claude-3-5-sonnet-20241022",
        prompt_tokens=30,
        completion_tokens=15,
        finish_reason="end_turn",
        provider="anthropic",
    )
    await log_and_enqueue(
        uniform=uniform,
        prompt_hash="hash-xyz",
        latency_ms=200,
        request_id="req-002",
        request_text="Translate to French.",
        redis=make_redis(),
    )

    from app.pipeline import response_repo
    logged = response_repo.insert_response.call_args[0][1]
    assert logged.model == "claude-3-5-sonnet-20241022"
    assert logged.prompt_tokens == 30
    assert logged.completion_tokens == 15
    assert logged.finish_reason == "end_turn"


# ── Redis stream ──────────────────────────────────────────────────────────────

async def test_mark_clustering_enqueued_after_successful_xadd():
    await log_and_enqueue(
        uniform=make_uniform(),
        prompt_hash="hash-abc",
        latency_ms=100,
        request_id="req-003",
        request_text="Hello",
        redis=make_redis(),
    )

    from app.pipeline import response_repo
    response_repo.mark_clustering_enqueued.assert_called_once()


async def test_mark_clustering_not_enqueued_when_circuit_open():
    """If XADD is blocked by the circuit breaker, mark_clustering_enqueued must not be called."""
    from app.pipeline import redis_breaker
    redis_breaker.call = AsyncMock(side_effect=CircuitBreakerOpen("redis"))

    await log_and_enqueue(
        uniform=make_uniform(),
        prompt_hash="hash-abc",
        latency_ms=100,
        request_id="req-004",
        request_text="Hello",
        redis=make_redis(),
    )

    from app.pipeline import response_repo
    response_repo.mark_clustering_enqueued.assert_not_called()


async def test_redis_circuit_open_does_not_raise():
    """CircuitBreakerOpen is best-effort — must not propagate to the caller."""
    from app.pipeline import redis_breaker
    redis_breaker.call = AsyncMock(side_effect=CircuitBreakerOpen("redis"))

    await log_and_enqueue(
        uniform=make_uniform(),
        prompt_hash="hash-abc",
        latency_ms=100,
        request_id="req-005",
        request_text="Hello",
        redis=make_redis(),
    )


async def test_redis_generic_error_does_not_raise():
    from app.pipeline import redis_breaker
    redis_breaker.call = AsyncMock(side_effect=ConnectionError("connection refused"))

    await log_and_enqueue(
        uniform=make_uniform(),
        prompt_hash="hash-abc",
        latency_ms=100,
        request_id="req-006",
        request_text="Hello",
        redis=make_redis(),
    )


# ── Sampling / evaluation scheduling ─────────────────────────────────────────

async def test_mark_needs_evaluation_called_when_counter_hits():
    """When redis.incr returns a multiple of SAMPLING_RATE, mark_needs_evaluation is called."""
    with patch("app.pipeline.settings.SAMPLING_RATE", 10):
        await log_and_enqueue(
            uniform=make_uniform(),
            prompt_hash="hash-abc",
            latency_ms=100,
            request_id="req-007",
            request_text="Hello",
            redis=make_redis(incr_return=10),  # 10 % 10 == 0
        )

    from app.pipeline import response_repo
    response_repo.mark_needs_evaluation.assert_called_once()
    assert response_repo.mark_needs_evaluation.call_args[0][1] == FIXED_RESPONSE_ID


async def test_mark_needs_evaluation_not_called_between_samples():
    with patch("app.pipeline.settings.SAMPLING_RATE", 10):
        await log_and_enqueue(
            uniform=make_uniform(),
            prompt_hash="hash-abc",
            latency_ms=100,
            request_id="req-008",
            request_text="Hello",
            redis=make_redis(incr_return=3),  # 3 % 10 != 0
        )

    from app.pipeline import response_repo
    response_repo.mark_needs_evaluation.assert_not_called()


async def test_sampling_counter_failure_does_not_raise():
    """If redis.incr fails, the function must not raise — evaluation is skipped silently."""
    redis = make_redis()
    redis.incr = AsyncMock(side_effect=ConnectionError("redis down"))

    await log_and_enqueue(
        uniform=make_uniform(),
        prompt_hash="hash-abc",
        latency_ms=100,
        request_id="req-009",
        request_text="Hello",
        redis=redis,
    )
