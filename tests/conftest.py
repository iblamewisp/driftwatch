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
