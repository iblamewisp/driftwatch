from abc import ABC, abstractmethod
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession


class AbstractClusterRepository(ABC):
    """
    Contract for BIRCH cluster assignment and maintenance.

    Implementations are backend-specific (Postgres/pgvector, in-memory for tests, etc.).
    All methods receive an open AsyncSession — transaction management is the caller's
    responsibility.
    """

    @abstractmethod
    async def assign_cluster(
        self,
        session: AsyncSession,
        response_id: UUID,
        request_embedding: list[float],
        response_embedding: list[float],
    ) -> UUID:
        """
        Assign response_id to the best absorbing BIRCH cluster, or create a new one.
        Returns the cluster_id.
        """
        ...

    @abstractmethod
    async def split_oversized_clusters(self, session: AsyncSession) -> int:
        """
        Find clusters whose radius exceeds BIRCH_THRESHOLD and split each into two
        using k-means. Returns number of splits performed.
        """
        ...

    @abstractmethod
    async def get_all_cluster_ids(self, session: AsyncSession) -> list[UUID]:
        """Return ids of all existing clusters."""
        ...
