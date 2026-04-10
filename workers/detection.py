from datetime import datetime, timezone
from uuid import UUID, uuid4

from monitoring.logging import get_logger
from monitoring.metrics import DRIFT_ALERT_COUNTER
from workers.celery_app import celery_app, get_worker_loop

logger = get_logger("detection")


async def _detect_for_cluster(cluster_id: UUID) -> None:
    from app.config import settings
    from app.schemas.internal import DriftDetectionResult
    from app.schemas.notifications import DriftAlertPayload
    from db.models import DriftEvent
    from db.repositories import response_repo
    from db.session import AsyncSessionLocal
    from services.notifications.base import get_notification_service
    from sqlalchemy import update

    async with AsyncSessionLocal() as session:
        scores = await response_repo.get_recent_quality_scores(
            session, limit=50, cluster_id=cluster_id
        )

    if len(scores) < 10:
        logger.info(
            "insufficient_data_for_drift_detection",
            cluster_id=str(cluster_id),
            score_count=len(scores),
        )
        return

    midpoint = len(scores) // 2
    baseline_score = float(sum(scores[:midpoint]) / midpoint)
    current_score = float(sum(scores[midpoint:]) / len(scores[midpoint:]))
    delta = (baseline_score - current_score) / baseline_score if baseline_score > 0 else 0.0

    result = DriftDetectionResult(
        baseline_score=baseline_score,
        current_score=current_score,
        delta=delta,
        threshold=settings.DRIFT_THRESHOLD,
        alert_triggered=delta > settings.DRIFT_THRESHOLD,
    )

    logger.info(
        "drift_detection_run",
        cluster_id=str(cluster_id),
        baseline_score=baseline_score,
        current_score=current_score,
        delta=delta,
        alert_triggered=result.alert_triggered,
    )

    if not result.alert_triggered:
        return

    now = datetime.now(tz=timezone.utc)
    channel = settings.NOTIFICATION_CHANNEL

    async with AsyncSessionLocal() as session:
        event = DriftEvent(
            id=uuid4(),
            detected_at=now,
            cluster_id=cluster_id,
            similarity_score=current_score,
            baseline_score=baseline_score,
            delta=delta,
            alert_sent=False,
            alert_channel=channel,
        )
        session.add(event)
        await session.commit()
        event_id = event.id

    payload = DriftAlertPayload(
        detected_at=now,
        cluster_id=cluster_id,
        baseline_score=baseline_score,
        current_score=current_score,
        delta_percent=round(delta * 100, 1),
        threshold_percent=round(settings.DRIFT_THRESHOLD * 100, 1),
        alert_channel=channel,
    )

    service = get_notification_service(channel)
    alert_sent = False
    if service:
        try:
            await service.send_alert(payload)
            alert_sent = True
            DRIFT_ALERT_COUNTER.inc()
        except Exception as exc:
            logger.error("notification_failed", error=str(exc), channel=channel)

    async with AsyncSessionLocal() as session:
        await session.execute(
            update(DriftEvent)
            .where(DriftEvent.id == event_id)
            .values(alert_sent=alert_sent, alert_channel=channel)
        )
        await session.commit()


async def _run_drift_detection() -> None:
    from db.repositories.cluster_repo import get_all_cluster_ids
    from db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        cluster_ids = await get_all_cluster_ids(session)

    if not cluster_ids:
        logger.info("no_clusters_yet_skipping_drift_detection")
        return

    for cluster_id in cluster_ids:
        await _detect_for_cluster(cluster_id)


@celery_app.task(name="workers.detection.run_drift_detection")
def run_drift_detection() -> None:
    get_worker_loop().run_until_complete(_run_drift_detection())


async def _run_cluster_split() -> None:
    from db.repositories.cluster_repo import split_oversized_clusters
    from db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        splits = await split_oversized_clusters(session)

    logger.info("cluster_split_check_complete", splits_performed=splits)


@celery_app.task(name="workers.detection.run_cluster_split")
def run_cluster_split() -> None:
    get_worker_loop().run_until_complete(_run_cluster_split())


async def _recover_unclustered_responses() -> None:
    """
    Re-enqueue llm_responses rows that never made it into the clustering stream
    (XADD failed while Redis was down). Targets needs_clustering=True rows older
    than 5 minutes to avoid racing with in-flight XADD calls from the proxy.
    """
    import redis.asyncio as aioredis

    from app.config import settings
    from db.repositories import response_repo
    from db.session import AsyncSessionLocal

    async with AsyncSessionLocal() as session:
        rows = await response_repo.get_unclustered_responses(session, older_than_seconds=300)

    if not rows:
        return

    redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    recovered = 0

    for row in rows:
        if not row.request_text:
            # Pre-migration row — no request_text stored, cannot reconstruct stream entry.
            # Mark done to stop the recovery job from retrying it forever.
            async with AsyncSessionLocal() as session:
                await response_repo.mark_clustering_enqueued(session, row.id)
            logger.warning("recovery_skipped_no_request_text", response_id=str(row.id))
            continue

        try:
            await redis.xadd(
                "driftwatch:clustering",
                {
                    "response_id": str(row.id),
                    "request_text": row.request_text,
                    "response_text": row.raw_content,
                },
            )
            async with AsyncSessionLocal() as session:
                await response_repo.mark_clustering_enqueued(session, row.id)
            recovered += 1
        except Exception as exc:
            logger.error("recovery_xadd_failed", response_id=str(row.id), error=str(exc))

    if recovered:
        logger.info("unclustered_responses_recovered", count=recovered)


@celery_app.task(name="workers.detection.recover_unclustered_responses")
def recover_unclustered_responses() -> None:
    get_worker_loop().run_until_complete(_recover_unclustered_responses())
