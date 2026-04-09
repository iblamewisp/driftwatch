from abc import ABC, abstractmethod
from uuid import UUID


class AbstractVectorRepository(ABC):

    @abstractmethod
    async def insert_embedding(
        self,
        response_id: UUID,
        embedding: list[float],
        quality_score: float,
    ) -> None:
        """Store embedding and quality score for a logged response."""
        ...

    @abstractmethod
    async def get_golden_embeddings(self, cluster_id: UUID | None = None) -> list[list[float]]:
        """
        Return golden set embeddings for cosine similarity computation.
        If cluster_id is provided, filter to that cluster only.
        """
        ...

    @abstractmethod
    async def insert_golden(
        self,
        prompt: str,
        embedding: list[float],
        description: str,
        cluster_id: UUID | None = None,
        request_embedding: list[float] | None = None,
    ) -> None:
        """Add a new golden set entry, optionally tagged to a cluster."""
        ...

    @abstractmethod
    async def count_golden(self, cluster_id: UUID | None = None) -> int:
        """Return number of golden set entries, optionally filtered by cluster."""
        ...
