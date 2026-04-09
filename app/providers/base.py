from abc import ABC, abstractmethod
from typing import AsyncGenerator

import httpx

from app.schemas.internal import UniformResponse


class AbstractProvider(ABC):

    @abstractmethod
    def upstream_url(self) -> str:
        """Full URL for the chat/completions endpoint."""

    @abstractmethod
    def prepare_headers(self, forwarded: dict[str, str]) -> dict[str, str]:
        """Transform client headers into headers suitable for this provider."""

    @abstractmethod
    def extract_uniform(self, raw: dict, model: str) -> UniformResponse:
        """Parse a non-streaming JSON response into UniformResponse."""

    @abstractmethod
    async def stream(
        self,
        client: httpx.AsyncClient,
        body: dict,
        headers: dict[str, str],
    ) -> AsyncGenerator[tuple[str, UniformResponse | None], None]:
        """
        Yield (sse_line, uniform) tuples.
        uniform is None for every chunk except the final one.
        The final tuple carries the assembled UniformResponse.
        """
        # make mypy happy — subclasses must implement this as an async generator
        raise NotImplementedError
        yield  # type: ignore[misc]
