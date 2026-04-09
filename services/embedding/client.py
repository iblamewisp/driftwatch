import asyncio

import httpx

from app.config import settings
from services.circuit_breaker import CircuitBreakerOpen, litserve_breaker

_client: httpx.AsyncClient | None = None


async def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.EMBEDDING_SERVICE_URL,
            timeout=30.0,
        )
    return _client


async def _request_embed(text: str) -> list[float]:
    client = await _get_client()
    response = await client.post("/predict", json={"text": text})
    response.raise_for_status()
    return response.json()["embedding"]


async def embed(text: str) -> list[float]:
    """Embed a single text. Raises CircuitBreakerOpen if LitServe is unavailable."""
    return await litserve_breaker.call(_request_embed(text))


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Fire all texts as concurrent requests — LitServe's adaptive batching
    groups them into optimal batches server-side.
    Raises CircuitBreakerOpen if LitServe is unavailable.
    """
    client = await _get_client()

    async def _one(text: str) -> list[float]:
        response = await client.post("/predict", json={"text": text})
        response.raise_for_status()
        return response.json()["embedding"]

    # All requests share one breaker — first failure counts for all
    return await litserve_breaker.call(
        asyncio.gather(*[_one(t) for t in texts])
    )
