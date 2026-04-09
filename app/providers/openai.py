import json
from typing import AsyncGenerator

import httpx

from app.config import settings
from app.providers.base import AbstractProvider
from app.schemas.internal import UniformResponse


class OpenAIProvider(AbstractProvider):

    def upstream_url(self) -> str:
        return f"{settings.OPENAI_BASE_URL}/v1/chat/completions"

    def prepare_headers(self, forwarded: dict[str, str]) -> dict[str, str]:
        # Pass through as-is — client sends Authorization: Bearer sk-...
        return forwarded

    def extract_uniform(self, raw: dict, model: str) -> UniformResponse:
        usage = raw.get("usage") or {}
        choices = raw.get("choices") or [{}]
        return UniformResponse(
            content=choices[0].get("message", {}).get("content", ""),
            model=raw.get("model", model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choices[0].get("finish_reason", ""),
            provider="openai",
        )

    async def stream(
        self,
        client: httpx.AsyncClient,
        body: dict,
        headers: dict[str, str],
    ) -> AsyncGenerator[tuple[str, UniformResponse | None], None]:
        chunks: list[str] = []
        meta: dict = {}

        async with client.stream("POST", self.upstream_url(), json=body, headers=headers, timeout=120.0) as resp:
            async for line in resp.aiter_lines():
                if not line:
                    continue

                if not line.startswith("data: "):
                    yield line + "\n\n", None
                    continue

                data_str = line[6:]

                if data_str.strip() == "[DONE]":
                    uniform = UniformResponse(
                        content="".join(chunks),
                        model=meta.get("model", body.get("model", "")),
                        prompt_tokens=0,
                        completion_tokens=0,
                        finish_reason="stop",
                        provider="openai",
                    )
                    yield line + "\n\n", uniform
                    return

                yield line + "\n\n", None

                try:
                    chunk = json.loads(data_str)
                    if not meta:
                        meta = {"model": chunk.get("model", "")}
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content") or ""
                    if delta:
                        chunks.append(delta)
                except (json.JSONDecodeError, IndexError):
                    pass
