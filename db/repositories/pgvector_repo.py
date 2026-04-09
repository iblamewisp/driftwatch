from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import GoldenSet, LLMResponse
from db.repositories.base import AbstractVectorRepository
from db.session import AsyncSessionLocal


class PgvectorRepository(AbstractVectorRepository):

    async def insert_embedding(
        self,
        response_id: UUID,
        embedding: list[float],
        quality_score: float,
    ) -> None:
        # embedding here is the response_embedding (already stored by clustering service).
        # We write it again for idempotency; the meaningful update is quality_score.
        async with AsyncSessionLocal() as session:
            await session.execute(
                update(LLMResponse)
                .where(LLMResponse.id == response_id)
                .values(response_embedding=embedding, quality_score=quality_score)
            )
            await session.commit()

    async def get_golden_embeddings(self, cluster_id: UUID | None = None) -> list[list[float]]:
        async with AsyncSessionLocal() as session:
            query = select(GoldenSet.expected_embedding)
            if cluster_id is not None:
                query = query.where(GoldenSet.cluster_id == cluster_id)
            result = await session.execute(query)
            return [list(row) for row in result.scalars().all()]

    async def insert_golden(
        self,
        prompt: str,
        embedding: list[float],
        description: str,
        cluster_id: UUID | None = None,
        request_embedding: list[float] | None = None,
    ) -> None:
        async with AsyncSessionLocal() as session:
            entry = GoldenSet(
                id=uuid4(),
                prompt=prompt,
                expected_embedding=embedding,
                request_embedding=request_embedding,
                description=description,
                cluster_id=cluster_id,
            )
            session.add(entry)
            await session.commit()

    async def count_golden(self, cluster_id: UUID | None = None) -> int:
        async with AsyncSessionLocal() as session:
            query = select(func.count()).select_from(GoldenSet)
            if cluster_id is not None:
                query = query.where(GoldenSet.cluster_id == cluster_id)
            result = await session.execute(query)
            return result.scalar_one()
