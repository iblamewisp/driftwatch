from uuid import UUID

from monitoring.logging import get_logger
from monitoring.metrics import QUALITY_SCORE_GAUGE
from services.embedding.similarity import mean_cosine_similarity
from workers.celery_app import celery_app, get_worker_loop

logger = get_logger("evaluator")


async def _evaluate_response(task, response_id_str: str) -> None:
    from services.embedding.client import embed
    from db.repositories import response_repo
    from db.session import AsyncSessionLocal
    from db.vector_store import get_vector_store
    from app.schemas.internal import EvaluationResult

    response_id = UUID(response_id_str)
    vector_store = get_vector_store()

    async with AsyncSessionLocal() as session:
        response = await response_repo.get_response_by_id(session, response_id)
        cluster_id = response.cluster_id
        # Clustering service stores response_embedding directly — no need to re-embed.
        # Fall back to embedding raw_content for rows written before this change.
        stored_embedding = list(response.response_embedding) if response.response_embedding is not None else None
        raw_content = response.raw_content

    if cluster_id is None:
        raise task.retry(countdown=10)

    if stored_embedding is not None:
        embedding = stored_embedding
    else:
        # Legacy fallback — rows from before request/response embedding split
        embedding = await embed(raw_content)

    golden_embeddings = await vector_store.get_golden_embeddings(cluster_id=cluster_id)
    if not golden_embeddings:
        logger.warning(
            "golden_set_empty_skipping_evaluation",
            response_id=response_id_str,
            cluster_id=str(cluster_id),
        )
        return

    quality_score = mean_cosine_similarity(embedding, golden_embeddings)

    result = EvaluationResult(
        response_id=response_id,
        embedding=embedding,
        quality_score=quality_score,
    )

    await vector_store.insert_embedding(result.response_id, result.embedding, result.quality_score)
    QUALITY_SCORE_GAUGE.set(quality_score)

    logger.info(
        "response_evaluated",
        response_id=response_id_str,
        quality_score=quality_score,
        cluster_id=str(cluster_id),
    )


@celery_app.task(name="workers.evaluator.evaluate_response", bind=True, max_retries=5)
def evaluate_response(self, response_id: str) -> None:
    get_worker_loop().run_until_complete(_evaluate_response(self, response_id))
