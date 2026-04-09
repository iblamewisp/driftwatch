"""
Unit tests for _log_and_enqueue — the fire-and-forget logging pipeline.

What we test:
  - LoggedResponse is constructed correctly from UniformResponse
  - response_repo.insert_response is called with that payload
  - Redis xadd is attempted via redis_breaker
  - Celery task is dispatched only when sampling rate hits
  - CircuitBreakerOpen on Redis is swallowed (best-effort logging)
"""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from app.proxy import _log_and_enqueue
from app.schemas.internal import UniformResponse
from services.circuit_breaker import CircuitBreakerOpen


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


FIXED_RESPONSE_ID = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")


@pytest.fixture(autouse=True)
def isolate_db_and_redis():
    """
    Patch DB session and Redis for every test in this module.
    autouse=True means tests don't have to declare these fixtures explicitly.
    """
    session = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)

    with (
        patch("app.proxy.AsyncSessionLocal", return_value=cm),
        patch(
            "app.proxy.response_repo.insert_response",
            new=AsyncMock(return_value=FIXED_RESPONSE_ID),
        ),
        patch("app.proxy.redis_breaker") as mock_redis_breaker,
    ):
        mock_redis_breaker.call = AsyncMock(return_value=None)
        yield


# ── DB logging ────────────────────────────────────────────────────────────────

async def test_inserts_response_to_db():
    uniform = make_uniform()
    with patch("app.proxy.settings.SAMPLING_RATE", 999):  # no Celery
        await _log_and_enqueue(uniform, "hash-abc", 120, "req-001")

    from app.proxy import response_repo
    call_args = response_repo.insert_response.call_args
    logged = call_args[0][1]  # insert_response(session, logged)

    assert logged.model == "gpt-4o"
    assert logged.raw_content == "The capital of France is Paris."
    assert logged.prompt_tokens == 20
    assert logged.completion_tokens == 10
    assert logged.finish_reason == "stop"
    assert logged.prompt_hash == "hash-abc"
    assert logged.latency_ms == 120


async def test_anthropic_uniform_fields_mapped_correctly():
    uniform = make_uniform(
        model="claude-3-5-sonnet-20241022",
        prompt_tokens=30,
        completion_tokens=15,
        finish_reason="end_turn",
        provider="anthropic",
    )
    with patch("app.proxy.settings.SAMPLING_RATE", 999):
        await _log_and_enqueue(uniform, "hash-xyz", 200, "req-002")

    from app.proxy import response_repo
    logged = response_repo.insert_response.call_args[0][1]
    assert logged.model == "claude-3-5-sonnet-20241022"
    assert logged.prompt_tokens == 30
    assert logged.completion_tokens == 15
    assert logged.finish_reason == "end_turn"


# ── Redis stream ──────────────────────────────────────────────────────────────

async def test_pushes_content_to_redis_stream():
    uniform = make_uniform()
    with patch("app.proxy.settings.SAMPLING_RATE", 999):
        await _log_and_enqueue(uniform, "hash-abc", 100, "req-003")

    from app.proxy import redis_breaker
    redis_breaker.call.assert_called_once()


async def test_redis_circuit_open_is_swallowed():
    """Redis is best-effort — CircuitBreakerOpen must not bubble up."""
    from app.proxy import redis_breaker
    redis_breaker.call = AsyncMock(side_effect=CircuitBreakerOpen("redis"))

    uniform = make_uniform()
    with patch("app.proxy.settings.SAMPLING_RATE", 999):
        # must not raise
        await _log_and_enqueue(uniform, "hash-abc", 100, "req-004")


async def test_redis_generic_error_is_swallowed():
    from app.proxy import redis_breaker
    redis_breaker.call = AsyncMock(side_effect=ConnectionError("refused"))

    uniform = make_uniform()
    with patch("app.proxy.settings.SAMPLING_RATE", 999):
        await _log_and_enqueue(uniform, "hash-abc", 100, "req-005")


# ── Celery sampling ───────────────────────────────────────────────────────────

async def test_celery_task_dispatched_on_sampling_hit():
    """
    We reset _request_counter to 0 and set SAMPLING_RATE=1 so every
    call triggers Celery dispatch.
    """
    mock_task = MagicMock()
    mock_task.delay = MagicMock()

    with (
        patch("app.proxy.settings.SAMPLING_RATE", 1),
        patch("app.proxy._request_counter", 0),
        patch("workers.evaluator.evaluate_response", mock_task),
    ):
        await _log_and_enqueue(make_uniform(), "hash-abc", 100, "req-006")

    mock_task.delay.assert_called_once_with(str(FIXED_RESPONSE_ID))


async def test_celery_not_dispatched_between_samples():
    """With SAMPLING_RATE=100 and counter not at a multiple, Celery stays quiet."""
    mock_task = MagicMock()
    mock_task.delay = MagicMock()

    with (
        patch("app.proxy.settings.SAMPLING_RATE", 100),
        patch("app.proxy._request_counter", 1),  # will become 2 after increment
        patch("workers.evaluator.evaluate_response", mock_task),
    ):
        await _log_and_enqueue(make_uniform(), "hash-abc", 100, "req-007")

    mock_task.delay.assert_not_called()
