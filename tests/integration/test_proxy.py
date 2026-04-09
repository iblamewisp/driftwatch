"""
Integration tests for proxy endpoints.

Strategy:
  - Run the full ASGI stack (middleware → router → provider)
  - Mock upstream HTTP with respx (intercepts httpx at transport level)
  - Mock _log_and_enqueue with AsyncMock — it's a background task and
    requires live DB/Redis which we don't have in CI
  - Auth middleware is exercised for real

What we test:
  - Auth: missing key, wrong key, valid key
  - /v1/chat/completions: OpenAI model → OpenAI upstream, response forwarded
  - /v1/chat/completions: Claude model → Anthropic upstream, headers transformed
  - /v1/messages: Anthropic-native endpoint
  - Circuit breaker OPEN → 503 before hitting upstream
  - Upstream 4xx is forwarded as-is
"""
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import respx

from tests.conftest import AUTH_HEADERS

# ── Fake upstream responses ───────────────────────────────────────────────────

OPENAI_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}

ANTHROPIC_RESPONSE = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-20241022",
    "content": [{"type": "text", "text": "Bonjour!"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 12, "output_tokens": 6},
}


# ── Auth middleware ───────────────────────────────────────────────────────────

class TestAuth:
    async def test_missing_key_returns_401(self, async_client):
        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 401
        assert "Missing" in resp.json()["detail"]

    async def test_wrong_key_returns_401(self, async_client):
        resp = await async_client.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-DriftWatch-Key": "wrong-key"},
        )
        assert resp.status_code == 401
        assert "Invalid" in resp.json()["detail"]

    async def test_healthz_does_not_require_auth(self, async_client):
        resp = await async_client.get("/healthz")
        assert resp.status_code == 200


# ── /v1/chat/completions — OpenAI model ──────────────────────────────────────

class TestChatCompletionsOpenAI:
    @respx.mock
    async def test_non_streaming_response_forwarded(self, async_client):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=OPENAI_RESPONSE)
        )

        with patch("app.proxy._log_and_enqueue", new=AsyncMock()), \
             patch("app.proxy.asyncio.create_task"):
            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
                headers=AUTH_HEADERS,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["choices"][0]["message"]["content"] == "Hello!"
        assert body["model"] == "gpt-4o"

    @respx.mock
    async def test_upstream_4xx_forwarded(self, async_client):
        respx.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=httpx.Response(400, json={"error": {"message": "bad request"}})
        )

        with patch("app.proxy.asyncio.create_task"):
            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
                headers=AUTH_HEADERS,
            )

        assert resp.status_code == 400

    async def test_circuit_open_returns_503(self, async_client):
        from services.circuit_breaker import _State, get_provider_breaker
        breaker = get_provider_breaker("openai")

        original_state = breaker._state
        original_fail = breaker._fail_count
        original_opened = breaker._opened_at

        try:
            import time
            breaker._state = _State.OPEN
            breaker._fail_count = 5
            breaker._opened_at = time.monotonic()  # just opened — won't transition to HALF_OPEN

            resp = await async_client.post(
                "/v1/chat/completions",
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
                headers=AUTH_HEADERS,
            )
            assert resp.status_code == 503
        finally:
            breaker._state = original_state
            breaker._fail_count = original_fail
            breaker._opened_at = original_opened


# ── /v1/chat/completions — Claude model → Anthropic upstream ─────────────────

class TestChatCompletionsClaude:
    @respx.mock
    async def test_claude_model_routes_to_anthropic(self, async_client):
        """gpt-style request with claude model goes to api.anthropic.com."""
        anthropic_route = respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=ANTHROPIC_RESPONSE)
        )

        with patch("app.proxy._log_and_enqueue", new=AsyncMock()), \
             patch("app.proxy.asyncio.create_task"):
            resp = await async_client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                headers=AUTH_HEADERS,
            )

        assert anthropic_route.called, "Expected request to Anthropic upstream"
        assert resp.status_code == 200

    @respx.mock
    async def test_anthropic_headers_transformed(self, async_client):
        """Authorization: Bearer → x-api-key + anthropic-version injected."""
        captured: dict = {}

        def capture(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, json=ANTHROPIC_RESPONSE)

        respx.post("https://api.anthropic.com/v1/messages").mock(side_effect=capture)

        with patch("app.proxy._log_and_enqueue", new=AsyncMock()), \
             patch("app.proxy.asyncio.create_task"):
            await async_client.post(
                "/v1/chat/completions",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
                headers={
                    "X-DriftWatch-Key": AUTH_HEADERS["X-DriftWatch-Key"],
                    "Authorization": "Bearer sk-ant-test-key",
                },
            )

        assert "authorization" not in captured["headers"], \
            "Raw Authorization header should be stripped for Anthropic"
        assert captured["headers"].get("x-api-key") == "sk-ant-test-key"
        assert captured["headers"].get("anthropic-version") == "2023-06-01"


# ── /v1/messages — Anthropic native endpoint ─────────────────────────────────

class TestMessagesEndpoint:
    @respx.mock
    async def test_non_streaming_response_forwarded(self, async_client):
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json=ANTHROPIC_RESPONSE)
        )

        with patch("app.proxy._log_and_enqueue", new=AsyncMock()), \
             patch("app.proxy.asyncio.create_task"):
            resp = await async_client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 1024,
                },
                headers=AUTH_HEADERS,
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["content"][0]["text"] == "Bonjour!"

    async def test_circuit_open_returns_503(self, async_client):
        from services.circuit_breaker import _State, get_provider_breaker
        breaker = get_provider_breaker("anthropic")

        original_state = breaker._state
        original_fail = breaker._fail_count
        original_opened = breaker._opened_at

        try:
            import time
            breaker._state = _State.OPEN
            breaker._fail_count = 5
            breaker._opened_at = time.monotonic()

            resp = await async_client.post(
                "/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 100,
                },
                headers=AUTH_HEADERS,
            )
            assert resp.status_code == 503
        finally:
            breaker._state = original_state
            breaker._fail_count = original_fail
            breaker._opened_at = original_opened
