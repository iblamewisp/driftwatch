"""
BIRCH cluster assignment and maintenance — repository layer.

Only DB I/O lives here. All CF math is in services/clustering/birch.py.
"""

import numpy as np
from uuid import UUID, uuid4

from sqlalchemy import case, delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from db.locks import AdvisoryLock, acquire_xact_lock
from db.models import Cluster, GoldenSet, LLMResponse
from monitoring.logging import get_logger
from services.clustering.birch import ClusteringFeature, cosine_similarity, kmeans_split

logger = get_logger("cluster_repo")


# ── Assignment ────────────────────────────────────────────────────────────────

async def assign_cluster(
    session: AsyncSession,
    response_id: UUID,
    request_embedding: list[float],
    response_embedding: list[float],
) -> UUID:
    """
    BIRCH absorption using request_embedding to define topic clusters.

    Clustering on the *request* (not the response) means each cluster represents
    a semantic topic — e.g. "data analysis questions" or "code review requests".
    quality_score then measures whether the *response* matches golden responses
    for that topic, which is a meaningful signal of model degradation.

    Candidate selection uses an HNSW ANN query (centroid <=> req_emb, top-10) so
    we never pull the full cluster table into Python. The absorption check then
    runs over only those candidates.

    NOTE: K=10 is a tuning knob. If BIRCH_THRESHOLD is very tight and clusters
    are dense, a true absorbing cluster might fall outside the top-10 — see TODO.md.
    """
    req_emb = np.array(request_embedding, dtype=np.float32)

    best_cluster, best_similarity = await _find_best_candidate(session, req_emb, request_embedding)

    if best_cluster is not None:
        cluster_id = await _absorb_into(session, best_cluster, req_emb, best_similarity, event="cluster_absorbed")
    else:
        # No absorbing cluster found. Acquire a transaction-scoped advisory lock so
        # concurrent workers don't both create a cluster for the same topic.
        # After acquiring, re-query — if another worker just created one, absorb instead.
        await acquire_xact_lock(session, AdvisoryLock.CLUSTER_CREATION)
        logger.debug("cluster_creation_lock_acquired")

        best_cluster, best_similarity = await _find_best_candidate(session, req_emb, request_embedding)

        if best_cluster is not None:
            cluster_id = await _absorb_into(session, best_cluster, req_emb, best_similarity, event="cluster_absorbed_after_recheck")
        else:
            cluster_id = uuid4()
            cf = ClusteringFeature.from_point(req_emb)
            session.add(Cluster(
                id=cluster_id,
                n=cf.n,
                ls=cf.ls.tolist(),
                ss=cf.ss,
                centroid=cf.centroid.tolist(),
            ))
            logger.info("cluster_created", cluster_id=str(cluster_id))

    await session.execute(
        update(LLMResponse)
        .where(LLMResponse.id == response_id)
        .values(
            cluster_id=cluster_id,
            request_embedding=request_embedding,
            response_embedding=response_embedding,
        )
    )
    await session.commit()
    return cluster_id


async def _find_best_candidate(
    session: AsyncSession,
    req_emb: np.ndarray,
    request_embedding: list[float],
) -> tuple[Cluster | None, float]:
    """ANN query + absorption check. Returns (best cluster, similarity) or (None, -1)."""
    result = await session.execute(
        select(Cluster)
        .where(Cluster.centroid.is_not(None))
        .order_by(Cluster.centroid.op("<=>")(request_embedding))
        .limit(10)
    )
    candidates = result.scalars().all()
    logger.debug("ann_candidates_fetched", candidate_count=len(candidates))

    best_cluster: Cluster | None = None
    best_similarity = -1.0

    for cluster in candidates:
        if cluster.ls is None:
            continue
        cf = ClusteringFeature(
            n=cluster.n,
            ls=np.array(list(cluster.ls), dtype=np.float32),
            ss=cluster.ss,
        )
        sim = cosine_similarity(req_emb, cf.centroid)
        if sim > best_similarity and cf.would_absorb(req_emb, settings.BIRCH_THRESHOLD):
            best_similarity = sim
            best_cluster = cluster

    return best_cluster, best_similarity


async def _absorb_into(
    session: AsyncSession,
    cluster: Cluster,
    req_emb: np.ndarray,
    similarity: float,
    event: str,
) -> UUID:
    """Update the cluster CF to include req_emb, return its id."""
    cf = ClusteringFeature(
        n=cluster.n,
        ls=np.array(list(cluster.ls), dtype=np.float32),
        ss=cluster.ss,
    )
    new_cf = cf.absorb(req_emb)

    await session.execute(
        update(Cluster)
        .where(Cluster.id == cluster.id)
        .values(
            ls=new_cf.ls.tolist(),
            n=new_cf.n,
            ss=new_cf.ss,
            centroid=new_cf.centroid.tolist(),
        )
    )
    logger.info(event, cluster_id=str(cluster.id), new_n=new_cf.n, similarity=round(similarity, 4))
    return cluster.id


# ── Splitting ─────────────────────────────────────────────────────────────────

async def split_oversized_clusters(session: AsyncSession) -> int:
    """
    Find clusters whose radius exceeds BIRCH_THRESHOLD, split each into two
    using k-means, reassign all responses and golden set entries.
    Returns number of splits performed.
    """
    result = await session.execute(select(Cluster))
    clusters = result.scalars().all()
    splits = 0

    for cluster in clusters:
        if cluster.ls is None or cluster.n < 4:
            continue

        cf = ClusteringFeature(
            n=cluster.n,
            ls=np.array(list(cluster.ls), dtype=np.float32),
            ss=cluster.ss,
        )
        if cf.radius <= settings.BIRCH_THRESHOLD:
            continue

        logger.info(
            "cluster_oversized_splitting",
            cluster_id=str(cluster.id),
            radius=round(cf.radius, 4),
            n=cluster.n,
            threshold=settings.BIRCH_THRESHOLD,
        )

        emb_result = await session.execute(
            select(LLMResponse.id, LLMResponse.request_embedding)
            .where(LLMResponse.cluster_id == cluster.id)
            .where(LLMResponse.request_embedding.is_not(None))
        )
        rows = emb_result.all()
        if len(rows) < 4:
            continue

        ids = [r.id for r in rows]
        embeddings = np.array([list(r.request_embedding) for r in rows], dtype=np.float32)
        labels = kmeans_split(embeddings)

        new_clusters: list[Cluster] = []
        for label in (0, 1):
            mask = labels == label
            if not mask.any():
                continue
            sub = embeddings[mask]
            sub_cf = ClusteringFeature(
                n=int(mask.sum()),
                ls=sub.sum(axis=0),
                ss=float(np.sum(np.sum(sub ** 2, axis=1))),
            )
            new_clusters.append(Cluster(
                id=uuid4(),
                n=sub_cf.n,
                ls=sub_cf.ls.tolist(),
                ss=sub_cf.ss,
                centroid=sub_cf.centroid.tolist(),
            ))

        for nc in new_clusters:
            session.add(nc)

        if len(new_clusters) < 2:
            continue

        c0 = np.array(list(new_clusters[0].centroid), dtype=np.float32)
        c1 = np.array(list(new_clusters[1].centroid), dtype=np.float32)

        id_to_cluster = {rid: new_clusters[int(lbl)].id for rid, lbl in zip(ids, labels)}
        await session.execute(
            update(LLMResponse)
            .where(LLMResponse.id.in_(id_to_cluster))
            .values(
                cluster_id=case(
                    {rid: cid for rid, cid in id_to_cluster.items()},
                    value=LLMResponse.id,
                )
            )
        )

        # Reassign golden set entries — N+1, acceptable (small set). See TODO.md.
        gs_result = await session.execute(
            select(GoldenSet.id, GoldenSet.expected_embedding, GoldenSet.request_embedding)
            .where(GoldenSet.cluster_id == cluster.id)
        )
        for gs_id, gs_resp_emb, gs_req_emb in gs_result.all():
            pivot = gs_req_emb if gs_req_emb is not None else gs_resp_emb
            emb = np.array(list(pivot), dtype=np.float32)
            new_cid = new_clusters[0].id if cosine_similarity(emb, c0) >= cosine_similarity(emb, c1) else new_clusters[1].id
            await session.execute(
                update(GoldenSet).where(GoldenSet.id == gs_id).values(cluster_id=new_cid)
            )

        await session.execute(delete(Cluster).where(Cluster.id == cluster.id))
        splits += 1
        logger.info(
            "cluster_split_complete",
            original_cluster_id=str(cluster.id),
            new_cluster_ids=[str(nc.id) for nc in new_clusters],
        )

    await session.commit()
    return splits


# ── Queries ───────────────────────────────────────────────────────────────────

async def get_all_cluster_ids(session: AsyncSession) -> list[UUID]:
    result = await session.execute(select(Cluster.id))
    return list(result.scalars().all())
