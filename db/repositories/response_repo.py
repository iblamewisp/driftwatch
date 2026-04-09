from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.internal import LoggedResponse
from db.models import LLMResponse


async def insert_response(session: AsyncSession, data: LoggedResponse) -> UUID:
    response_id = uuid4()
    row = LLMResponse(
        id=response_id,
        request_id=data.request_id,
        prompt_hash=data.prompt_hash,
        model=data.model,
        prompt_tokens=data.prompt_tokens,
        completion_tokens=data.completion_tokens,
        latency_ms=data.latency_ms,
        finish_reason=data.finish_reason,
        raw_content=data.raw_content,
    )
    session.add(row)
    await session.commit()
    return response_id


async def get_response_by_id(session: AsyncSession, response_id: UUID) -> LLMResponse:
    result = await session.execute(
        select(LLMResponse).where(LLMResponse.id == response_id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise ValueError(f"LLMResponse {response_id} not found")
    return row


async def get_recent_quality_scores(
    session: AsyncSession,
    limit: int = 50,
    cluster_id: UUID | None = None,
) -> list[float]:
    query = (
        select(LLMResponse.quality_score)
        .where(LLMResponse.quality_score.is_not(None))
    )
    if cluster_id is not None:
        query = query.where(LLMResponse.cluster_id == cluster_id)

    result = await session.execute(
        query.order_by(LLMResponse.created_at.desc()).limit(limit)
    )
    # Return in ascending time order so caller can split oldest/newest correctly
    scores = list(result.scalars().all())
    scores.reverse()
    return scores
