"""
Global test configuration.

env vars MUST be set before any app module is imported — pydantic-settings
reads them at class-definition time, so there's no way to patch after the fact.
"""
import hashlib
import os

_TEST_KEY = "test-key-driftwatch"

os.environ.setdefault("DRIFTWATCH_KEY_HASH", hashlib.sha256(_TEST_KEY.encode()).hexdigest())
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test:test@localhost/testdb")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/15")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.openai.com")
os.environ.setdefault("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
os.environ.setdefault("ENABLE_METRICS", "false")

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch

from app.main import app

# ── Shared constants ──────────────────────────────────────────────────────────

VALID_KEY = _TEST_KEY
AUTH_HEADERS = {
    "X-DriftWatch-Key": VALID_KEY,
    "Authorization": "Bearer sk-test-openai-key",
}


# ── HTTP client fixture ───────────────────────────────────────────────────────

@pytest.fixture
async def async_client():
    """FastAPI test client that goes through the full ASGI stack (middleware included)."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client


# ── DB session mock ───────────────────────────────────────────────────────────

@pytest.fixture
def mock_db_session():
    """
    Replaces AsyncSessionLocal with a fake async context manager.
    Returns the inner session mock so tests can assert on it.
    """
    session = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)

    with patch("app.proxy.AsyncSessionLocal", return_value=cm):
        yield session


# ── response_repo mock ────────────────────────────────────────────────────────

@pytest.fixture
def mock_insert_response(mock_db_session):
    """Mocks insert_response and returns a fixed UUID."""
    import uuid
    fixed_id = uuid.UUID("aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")

    with patch("app.proxy.response_repo.insert_response", new=AsyncMock(return_value=fixed_id)) as mock:
        yield mock, fixed_id


# ── Redis breaker mock ────────────────────────────────────────────────────────

@pytest.fixture
def mock_redis_breaker():
    """Makes redis_breaker.call a no-op so tests don't need a real Redis."""
    with patch("app.proxy.redis_breaker") as mock_breaker:
        mock_breaker.call = AsyncMock(return_value=None)
        yield mock_breaker


# ── Celery mock ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_celery():
    """Prevents Celery from trying to connect to a broker during tests."""
    with patch("workers.evaluator.evaluate_response") as mock_task:
        mock_task.delay = MagicMock()
        yield mock_task
