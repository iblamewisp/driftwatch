"""
Unit tests for provider implementations.
These are pure-function tests — no I/O, no mocks needed.
"""
import pytest

from app.providers.openai import OpenAIProvider
from app.providers.anthropic import AnthropicProvider
from app.providers.registry import get_provider


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def openai():
    return OpenAIProvider()


@pytest.fixture
def anthropic():
    return AnthropicProvider()


# ── OpenAI: extract_uniform ───────────────────────────────────────────────────

class TestOpenAIExtractUniform:
    def test_happy_path(self, openai):
        raw = {
            "id": "chatcmpl-abc",
            "model": "gpt-4o",
            "choices": [{"message": {"content": "Hello!"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        u = openai.extract_uniform(raw, "gpt-4o")

        assert u.content == "Hello!"
        assert u.model == "gpt-4o"
        assert u.prompt_tokens == 10
        assert u.completion_tokens == 5
        assert u.finish_reason == "stop"
        assert u.provider == "openai"

    def test_model_from_response_wins_over_arg(self, openai):
        # The upstream may return a resolved model alias ("gpt-4o-2024-11-20")
        raw = {
            "model": "gpt-4o-2024-11-20",
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        u = openai.extract_uniform(raw, "gpt-4o")
        assert u.model == "gpt-4o-2024-11-20"

    def test_missing_usage_defaults_to_zero(self, openai):
        raw = {
            "model": "gpt-4",
            "choices": [{"message": {"content": "Hi"}, "finish_reason": "length"}],
        }
        u = openai.extract_uniform(raw, "gpt-4")
        assert u.prompt_tokens == 0
        assert u.completion_tokens == 0

    def test_empty_choices_gives_empty_content(self, openai):
        raw = {"model": "gpt-4", "choices": [], "usage": {}}
        u = openai.extract_uniform(raw, "gpt-4")
        assert u.content == ""


# ── OpenAI: prepare_headers ───────────────────────────────────────────────────

class TestOpenAIHeaders:
    def test_passes_through_authorization(self, openai):
        forwarded = {"authorization": "Bearer sk-abc", "content-type": "application/json"}
        headers = openai.prepare_headers(forwarded)
        assert headers["authorization"] == "Bearer sk-abc"

    def test_does_not_modify_headers(self, openai):
        forwarded = {"authorization": "Bearer sk-abc", "x-custom": "value"}
        headers = openai.prepare_headers(forwarded)
        assert headers == forwarded


# ── Anthropic: extract_uniform ────────────────────────────────────────────────

class TestAnthropicExtractUniform:
    def test_happy_path(self, anthropic):
        raw = {
            "id": "msg_abc",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 12, "output_tokens": 8},
        }
        u = anthropic.extract_uniform(raw, "claude-3-5-sonnet-20241022")

        assert u.content == "Hello from Claude!"
        assert u.model == "claude-3-5-sonnet-20241022"
        assert u.prompt_tokens == 12
        assert u.completion_tokens == 8
        assert u.finish_reason == "end_turn"
        assert u.provider == "anthropic"

    def test_multiple_text_blocks_joined(self, anthropic):
        raw = {
            "model": "claude-3-opus-20240229",
            "content": [
                {"type": "text", "text": "Part one."},
                {"type": "text", "text": "Part two."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 5},
        }
        u = anthropic.extract_uniform(raw, "claude-3-opus-20240229")
        assert u.content == "Part one.Part two."

    def test_tool_use_blocks_ignored(self, anthropic):
        # tool_use blocks have no "text" field — we only want text blocks
        raw = {
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {"type": "tool_use", "id": "tool_abc", "name": "calculator", "input": {}},
                {"type": "text", "text": "Result: 42"},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 10},
        }
        u = anthropic.extract_uniform(raw, "claude-3-5-sonnet-20241022")
        assert u.content == "Result: 42"
        assert u.finish_reason == "tool_use"

    def test_empty_content_array(self, anthropic):
        raw = {
            "model": "claude-3-haiku-20240307",
            "content": [],
            "stop_reason": "max_tokens",
            "usage": {"input_tokens": 5, "output_tokens": 0},
        }
        u = anthropic.extract_uniform(raw, "claude-3-haiku-20240307")
        assert u.content == ""


# ── Anthropic: prepare_headers ────────────────────────────────────────────────

class TestAnthropicHeaders:
    def test_converts_bearer_to_x_api_key(self, anthropic):
        forwarded = {"authorization": "Bearer sk-ant-abc123"}
        headers = anthropic.prepare_headers(forwarded)
        assert "authorization" not in headers
        assert headers["x-api-key"] == "sk-ant-abc123"

    def test_case_insensitive_bearer_strip(self, anthropic):
        forwarded = {"authorization": "bearer sk-ant-abc123"}
        headers = anthropic.prepare_headers(forwarded)
        assert headers["x-api-key"] == "sk-ant-abc123"

    def test_adds_anthropic_version(self, anthropic):
        headers = anthropic.prepare_headers({"authorization": "Bearer sk-ant-x"})
        assert headers["anthropic-version"] == "2023-06-01"

    def test_no_auth_header_no_x_api_key(self, anthropic):
        headers = anthropic.prepare_headers({"content-type": "application/json"})
        assert "x-api-key" not in headers

    def test_other_headers_preserved(self, anthropic):
        forwarded = {
            "authorization": "Bearer sk-ant-x",
            "x-request-id": "req-123",
        }
        headers = anthropic.prepare_headers(forwarded)
        assert headers["x-request-id"] == "req-123"


# ── Registry ──────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_gpt_routes_to_openai(self):
        assert type(get_provider("gpt-4o")).__name__ == "OpenAIProvider"
        assert type(get_provider("gpt-3.5-turbo")).__name__ == "OpenAIProvider"

    def test_o_series_routes_to_openai(self):
        assert type(get_provider("o1-preview")).__name__ == "OpenAIProvider"
        assert type(get_provider("o3-mini")).__name__ == "OpenAIProvider"
        assert type(get_provider("o4-mini")).__name__ == "OpenAIProvider"

    def test_claude_routes_to_anthropic(self):
        assert type(get_provider("claude-3-5-sonnet-20241022")).__name__ == "AnthropicProvider"
        assert type(get_provider("claude-3-opus-20240229")).__name__ == "AnthropicProvider"
        assert type(get_provider("claude-3-haiku-20240307")).__name__ == "AnthropicProvider"

    def test_unknown_model_falls_back_to_openai(self):
        assert type(get_provider("some-unknown-model-v99")).__name__ == "OpenAIProvider"
