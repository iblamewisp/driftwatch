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

    @abstractmethod
    async def insert_golden_if_under_cap(
        self,
        prompt: str,
        embedding: list[float],
        description: str,
        cluster_id: UUID,
        request_embedding: list[float] | None,
        cap: int,
    ) -> bool:
        """
        Atomically insert a golden entry only if the cluster still exists and its
        golden count is strictly less than cap. Returns True if inserted.

        Guards against two races:
          1. TOCTOU within a batch — multiple coroutines read the same count
             then all insert, overshooting the cap.
          2. Post-split stale cluster_id — cluster was deleted between assign_cluster
             and here; inserting into a deleted cluster creates an orphaned row.
        """
        ...
