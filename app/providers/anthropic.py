import json
from typing import AsyncGenerator

import httpx

from app.config import settings
from app.providers.base import AbstractProvider
from app.schemas.internal import UniformResponse

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(AbstractProvider):

    def upstream_url(self) -> str:
        return f"{settings.ANTHROPIC_BASE_URL}/v1/messages"

    def prepare_headers(self, forwarded: dict[str, str]) -> dict[str, str]:
        headers = {k: v for k, v in forwarded.items() if k.lower() != "authorization"}

        # Extract raw key from "Bearer sk-ant-..." or bare "sk-ant-..."
        auth = forwarded.get("authorization", "")
        api_key = auth.removeprefix("Bearer ").removeprefix("bearer ").strip()
        if api_key:
            headers["x-api-key"] = api_key

        headers["anthropic-version"] = _ANTHROPIC_VERSION
        headers.setdefault("content-type", "application/json")
        return headers

    def extract_uniform(self, raw: dict, model: str) -> UniformResponse:
        usage = raw.get("usage") or {}
        content_blocks = raw.get("content") or []
        # Multiple text blocks (rare) join without separator — they're contiguous prose
        text = "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")
        return UniformResponse(
            content=text,
            model=raw.get("model", model),
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            finish_reason=raw.get("stop_reason", ""),
            provider="anthropic",
        )

    async def stream(
        self,
        client: httpx.AsyncClient,
        body: dict,
        headers: dict[str, str],
    ) -> AsyncGenerator[tuple[str, UniformResponse | None], None]:
        """
        Anthropic SSE uses event/data pairs:
            event: message_start
            data: {...}

            event: content_block_delta
            data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"..."}}

            event: message_stop
            data: {"type":"message_stop"}

        We parse before yielding so we can attach UniformResponse to the
        message_stop line (the true terminal event) without a spurious [DONE].
        """
        chunks: list[str] = []
        model_name = body.get("model", "")
        finish_reason = ""
        input_tokens = 0
        output_tokens = 0

        async with client.stream("POST", self.upstream_url(), json=body, headers=headers, timeout=120.0) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    yield "\n", None
                    continue

                # Non-data lines (event:, id:, comment) pass through untouched
                if not line.startswith("data: "):
                    yield line + "\n", None
                    continue

                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    yield line + "\n", None
                    continue

                etype = event.get("type", "")

                if etype == "message_start":
                    msg = event.get("message", {})
                    model_name = msg.get("model", model_name)
                    usage = msg.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    yield line + "\n", None

                elif etype == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        chunks.append(delta.get("text", ""))
                    yield line + "\n", None

                elif etype == "message_delta":
                    finish_reason = event.get("delta", {}).get("stop_reason", "")
                    usage = event.get("usage", {})
                    output_tokens = usage.get("output_tokens", 0)
                    yield line + "\n", None

                elif etype == "message_stop":
                    # Terminal event — attach UniformResponse here, no synthetic [DONE]
                    uniform = UniformResponse(
                        content="".join(chunks),
                        model=model_name,
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        finish_reason=finish_reason,
                        provider="anthropic",
                    )
                    yield line + "\n", uniform
                    return

                else:
                    # content_block_start, content_block_stop, ping — forward as-is
                    yield line + "\n", None
