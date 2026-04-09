"""
BIRCH cluster assignment and maintenance.

Clustering Feature: CF = (N, LS, SS)
  N  = number of points in cluster
  LS = linear sum of all embedding vectors
  SS = sum of squared norms (= N for unit-norm sentence-transformer embeddings)

Derived quantities (computed, never stored):
  centroid = LS / N
  radius   = sqrt(max(0, SS/N - ||LS/N||²))
           = sqrt(max(0, 1 - ||LS||²/N²))   [with unit-norm simplification]

Absorption criterion:
  A new point p is absorbed into cluster C if:
      new_radius(C + p) <= BIRCH_THRESHOLD
  This is geometrically meaningful — unlike cosine threshold, it accounts
  for how diverse the cluster already is.
"""

import numpy as np
from uuid import UUID, uuid4

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from db.models import Cluster, GoldenSet, LLMResponse
from monitoring.logging import get_logger

logger = get_logger("cluster_repo")


# ── CF math ───────────────────────────────────────────────────────────────────

def _centroid(ls: np.ndarray, n: int) -> np.ndarray:
    return ls / n


def _radius(ls: np.ndarray, n: int, ss: float) -> float:
    """
    R = sqrt(SS/N - ||LS/N||²)
    With unit-norm vectors: SS = N, simplifies to sqrt(1 - ||LS||²/N²)
    """
    ls_norm_sq = float(np.dot(ls, ls))
    r_sq = max(0.0, ss / n - ls_norm_sq / (n * n))
    return float(np.sqrt(r_sq))


def _would_absorb(ls: np.ndarray, n: int, ss: float, new_embedding: np.ndarray) -> bool:
    """True if absorbing new_embedding keeps cluster radius <= BIRCH_THRESHOLD."""
    new_ls = ls + new_embedding
    new_n = n + 1
    new_ss = ss + float(np.dot(new_embedding, new_embedding))
    return _radius(new_ls, new_n, new_ss) <= settings.BIRCH_THRESHOLD


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


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

    CF is built from request_embeddings. response_embedding is stored alongside
    for the evaluator to use without re-computing it.
    """
    req_emb = np.array(request_embedding, dtype=np.float32)

    result = await session.execute(select(Cluster))
    clusters = result.scalars().all()

    best_cluster: Cluster | None = None
    best_similarity = -1.0

    for cluster in clusters:
        if cluster.ls is None:
            continue
        ls = np.array(list(cluster.ls), dtype=np.float32)
        centroid = _centroid(ls, cluster.n)
        sim = _cosine_similarity(req_emb, centroid)

        if sim > best_similarity and _would_absorb(ls, cluster.n, cluster.ss, req_emb):
            best_similarity = sim
            best_cluster = cluster

    if best_cluster is not None:
        ls = np.array(list(best_cluster.ls), dtype=np.float32)
        new_ls = (ls + req_emb).tolist()
        new_n = best_cluster.n + 1
        new_ss = best_cluster.ss + float(np.dot(req_emb, req_emb))

        await session.execute(
            update(Cluster)
            .where(Cluster.id == best_cluster.id)
            .values(ls=new_ls, n=new_n, ss=new_ss)
        )
        cluster_id = best_cluster.id
    else:
        cluster_id = uuid4()
        session.add(Cluster(
            id=cluster_id,
            n=1,
            ls=request_embedding,
            ss=float(np.dot(req_emb, req_emb)),
        ))

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


# ── Splitting ─────────────────────────────────────────────────────────────────

def _kmeans_split(embeddings: np.ndarray, max_iter: int = 50) -> np.ndarray:
    """
    Split embeddings into 2 groups using k-means.
    Initialises with the two most distant points (deterministic, good for BIRCH).
    Returns label array (0 or 1) for each embedding.
    """
    # Pick two most distant points as initial centroids
    dists = np.sum((embeddings - embeddings[0]) ** 2, axis=1)
    i1 = int(np.argmax(dists))
    dists = np.sum((embeddings - embeddings[i1]) ** 2, axis=1)
    i0 = int(np.argmax(dists))

    c0, c1 = embeddings[i0].copy(), embeddings[i1].copy()
    labels = np.zeros(len(embeddings), dtype=int)

    for _ in range(max_iter):
        d0 = np.sum((embeddings - c0) ** 2, axis=1)
        d1 = np.sum((embeddings - c1) ** 2, axis=1)
        new_labels = (d1 < d0).astype(int)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        mask0 = labels == 0
        mask1 = labels == 1
        if mask0.any():
            c0 = embeddings[mask0].mean(axis=0)
        if mask1.any():
            c1 = embeddings[mask1].mean(axis=0)

    return labels


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
            # Need at least 4 points to split meaningfully
            continue

        ls = np.array(list(cluster.ls), dtype=np.float32)
        r = _radius(ls, cluster.n, cluster.ss)

        if r <= settings.BIRCH_THRESHOLD:
            continue

        logger.info(
            "cluster_oversized_splitting",
            cluster_id=str(cluster.id),
            radius=round(r, 4),
            n=cluster.n,
            threshold=settings.BIRCH_THRESHOLD,
        )

        # Fetch all request_embeddings for this cluster — k-means splits on request space
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

        labels = _kmeans_split(embeddings)

        # Build CFs for each new cluster
        new_clusters: list[Cluster] = []
        for label in (0, 1):
            mask = labels == label
            if not mask.any():
                continue
            sub = embeddings[mask]
            n = int(mask.sum())
            ls_vec = sub.sum(axis=0)
            ss = float(np.sum(np.sum(sub ** 2, axis=1)))

            new_clusters.append(Cluster(
                id=uuid4(),
                n=n,
                ls=ls_vec.tolist(),
                ss=ss,
            ))

        for nc in new_clusters:
            session.add(nc)

        if len(new_clusters) < 2:                                                                                                 
            continue

        # Reassign responses
        ls0 = np.array(list(new_clusters[0].ls), dtype=np.float32)
        c0 = _centroid(ls0, new_clusters[0].n)
        ls1 = np.array(list(new_clusters[1].ls), dtype=np.float32)
        c1 = _centroid(ls1, new_clusters[1].n)

        for idx, (response_id, label) in enumerate(zip(ids, labels)):
            new_cid = new_clusters[int(label)].id
            await session.execute(
                update(LLMResponse)
                .where(LLMResponse.id == response_id)
                .values(cluster_id=new_cid)
            )

        # Reassign golden set entries using request_embedding vs new cluster centroids.
        # Centroids are in request-embedding space, so we compare on request_embedding.
        # Falls back to expected_embedding (response) for legacy entries without request_embedding.
        gs_result = await session.execute(
            select(GoldenSet.id, GoldenSet.expected_embedding, GoldenSet.request_embedding)
            .where(GoldenSet.cluster_id == cluster.id)
        )
        for gs_id, gs_resp_emb, gs_req_emb in gs_result.all():
            pivot = gs_req_emb if gs_req_emb is not None else gs_resp_emb
            emb = np.array(list(pivot), dtype=np.float32)
            new_cid = (
                new_clusters[0].id
                if _cosine_similarity(emb, c0) >= _cosine_similarity(emb, c1)
                else new_clusters[1].id
            )
            await session.execute(
                update(GoldenSet)
                .where(GoldenSet.id == gs_id)
                .values(cluster_id=new_cid)
            )

        # Delete original cluster
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
