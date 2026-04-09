from uuid import UUID

from db.repositories.base import AbstractVectorRepository

_MSG = "Qdrant backend not yet implemented. Set VECTOR_BACKEND=pgvector."


class QdrantRepository(AbstractVectorRepository):

    async def insert_embedding(self, response_id: UUID, embedding: list[float], quality_score: float) -> None:
        raise NotImplementedError(_MSG)

    async def get_golden_embeddings(self, cluster_id: UUID | None = None) -> list[list[float]]:
        raise NotImplementedError(_MSG)

    async def insert_golden(self, prompt: str, embedding: list[float], description: str, cluster_id: UUID | None = None) -> None:
        raise NotImplementedError(_MSG)

    async def count_golden(self, cluster_id: UUID | None = None) -> int:
        raise NotImplementedError(_MSG)
