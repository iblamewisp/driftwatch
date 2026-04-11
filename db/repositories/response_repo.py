from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

from sqlalchemy import select, update
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
        request_text=data.request_text or None,
        needs_clustering=True,
    )
    session.add(row)
    await session.commit()
    return response_id


async def mark_clustering_enqueued(session: AsyncSession, response_id: UUID) -> None:
    """Flip needs_clustering=False after a successful XADD to the clustering stream."""
    await session.execute(
        update(LLMResponse)
        .where(LLMResponse.id == response_id)
        .values(needs_clustering=False)
    )
    await session.commit()


async def get_unclustered_responses(
    session: AsyncSession,
    older_than_seconds: int = 300,
    limit: int = 100,
) -> list[LLMResponse]:
    """
    Return rows that never made it into the clustering stream.
    The age guard prevents racing with in-flight XADD calls.
    """
    cutoff = datetime.now(tz=timezone.utc) - timedelta(seconds=older_than_seconds)
    result = await session.execute(
        select(LLMResponse)
        .where(LLMResponse.needs_clustering == True)  # noqa: E712 — SQLAlchemy requires ==
        .where(LLMResponse.created_at < cutoff)
        .limit(limit)
    )
    return list(result.scalars().all())


async def mark_needs_evaluation(session: AsyncSession, response_id: UUID) -> None:
    """Flag a response for evaluation. Clustering service will enqueue the task after assign_cluster."""
    await session.execute(
        update(LLMResponse)
        .where(LLMResponse.id == response_id)
        .values(needs_evaluation=True)
    )
    await session.commit()


async def get_flagged_for_evaluation(
    session: AsyncSession, response_ids: list[UUID]
) -> list[UUID]:
    """Return subset of response_ids where needs_evaluation=True."""
    result = await session.execute(
        select(LLMResponse.id)
        .where(LLMResponse.id.in_(response_ids))
        .where(LLMResponse.needs_evaluation == True)  # noqa: E712
    )
    return list(result.scalars().all())


async def clear_needs_evaluation(session: AsyncSession, response_ids: list[UUID]) -> None:
    """Flip needs_evaluation=False after enqueuing the evaluator task."""
    await session.execute(
        update(LLMResponse)
        .where(LLMResponse.id.in_(response_ids))
        .values(needs_evaluation=False)
    )
    await session.commit()


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
