"""
Postgres/pgvector implementation of AbstractClusterRepository.

All pgvector-specific constructs live here:
  - HNSW ANN query          (centroid <=> embedding)
  - Radius SQL expression   (ls <#> ls inner product)
  - Advisory lock           (pg_advisory_xact_lock)
  - Bulk CASE WHEN UPDATE

Nothing in this file is backend-agnostic. If a second backend is needed,
implement AbstractClusterRepository in a new file — do not modify this one.
"""

import numpy as np
from uuid import UUID, uuid4

from sqlalchemy import Float, case as sa_case, delete, func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from db.locks import AdvisoryLock, acquire_xact_lock
from db.models import Cluster, GoldenSet, LLMResponse
from db.repositories.cluster_base import AbstractClusterRepository
from monitoring.logging import get_logger
from services.clustering.birch import ClusteringFeature, cosine_similarity

logger = get_logger("cluster_repo.postgres")



class PostgresClusterRepository(AbstractClusterRepository):

    # ── Public interface ──────────────────────────────────────────────────────

    async def assign_cluster(
        self,
        session: AsyncSession,
        response_id: UUID,
        request_embedding: list[float],
        response_embedding: list[float],
    ) -> UUID:
        """
        BIRCH absorption using request_embedding to define topic clusters.

        Candidate selection uses an HNSW ANN query (centroid <=> req_emb) so
        the full cluster table is never loaded into Python.

        Concurrent create-on-miss is guarded by a Postgres advisory lock:
        workers that find no absorbing cluster re-query after acquiring the lock,
        absorbing into any cluster created while waiting.
        """
        req_emb = np.array(request_embedding, dtype=np.float32)
        best_cluster, best_sim = await self.find_best_candidate(session, req_emb, request_embedding)

        if best_cluster is not None:
            cluster_id = await self.absorb_into(session, best_cluster, req_emb, best_sim, "cluster_absorbed")
        else:
            await acquire_xact_lock(session, AdvisoryLock.CLUSTER_CREATION)
            logger.debug("cluster_creation_lock_acquired")

            best_cluster, best_sim = await self.find_best_candidate(session, req_emb, request_embedding)
            if best_cluster is not None:
                cluster_id = await self.absorb_into(session, best_cluster, req_emb, best_sim, "cluster_absorbed_after_recheck")
            else:
                cluster_id = await self.create_cluster(session, req_emb)

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

    async def split_oversized_clusters(self, session: AsyncSession) -> int:
        """
        Find clusters whose radius exceeds BIRCH_THRESHOLD and split each into two
        via k-means. Candidates are pre-filtered in SQL using the pgvector inner
        product operator — only clusters that actually need splitting are fetched.
        Returns number of splits performed.
        """
        result = await session.execute(
            select(Cluster)
            .where(Cluster.ls.is_not(None))
            .where(Cluster.n >= 4)
            .where(self.radius_sql_expr() > settings.BIRCH_THRESHOLD)
        )
        clusters = result.scalars().all()
        splits = 0

        for cluster in clusters:
            split_happened = await self.split_one(session, cluster)
            if split_happened:
                splits += 1

        await session.commit()
        return splits

    async def get_all_cluster_ids(self, session: AsyncSession) -> list[UUID]:
        result = await session.execute(select(Cluster.id))
        return list(result.scalars().all())

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def find_best_candidate(
        self,
        session: AsyncSession,
        req_emb: np.ndarray,
        request_embedding: list[float],
    ) -> tuple[Cluster | None, float]:
        """ANN query + BIRCH absorption check. Returns (best cluster, similarity)."""
        result = await session.execute(
            select(Cluster)
            .where(Cluster.centroid.is_not(None))
            .order_by(Cluster.centroid.op("<=>")(request_embedding))
            .limit(settings.CLUSTERING_ANN_CANDIDATES)
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

    async def absorb_into(
        self,
        session: AsyncSession,
        cluster: Cluster,
        req_emb: np.ndarray,
        similarity: float,
        event: str,
    ) -> UUID:
        """Absorb req_emb into cluster, update CF and centroid."""
        cf = ClusteringFeature(
            n=cluster.n,
            ls=np.array(list(cluster.ls), dtype=np.float32),
            ss=cluster.ss,
        )
        new_cf = cf.absorb(req_emb)
        await session.execute(
            update(Cluster)
            .where(Cluster.id == cluster.id)
            .values(ls=new_cf.ls.tolist(), n=new_cf.n, ss=new_cf.ss, centroid=new_cf.centroid.tolist())
        )
        logger.info(event, cluster_id=str(cluster.id), new_n=new_cf.n, similarity=round(similarity, 4))
        return cluster.id

    async def create_cluster(self, session: AsyncSession, req_emb: np.ndarray) -> UUID:
        """Create a new single-point cluster from req_emb."""
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
        return cluster_id

    async def split_one(self, session: AsyncSession, cluster: Cluster) -> bool:
        """
        Split a single oversized cluster into two via kmeans_split_cluster().

        The SQL function runs k-means entirely inside Postgres — no embedding
        vectors cross the network. Python only inserts two new Cluster rows
        and reassigns FKs.

        Returns True if the split was performed.
        """
        cf = ClusteringFeature(
            n=cluster.n,
            ls=np.array(list(cluster.ls), dtype=np.float32),
            ss=cluster.ss,
        )
        logger.info(
            "cluster_oversized_splitting",
            cluster_id=str(cluster.id),
            radius=round(cf.radius, 4),
            n=cluster.n,
            threshold=settings.BIRCH_THRESHOLD,
        )

        result = await session.execute(
            text("SELECT * FROM kmeans_split_cluster(:cid)").bindparams(cid=cluster.id)
        )
        rows = result.all()

        # Group by label — CF columns are the same for every row within a label.
        label_cfs: dict[int, tuple] = {}
        label_response_ids: dict[int, list[UUID]] = {}
        for row in rows:
            lbl = int(row.label)
            if lbl not in label_cfs:
                label_cfs[lbl] = (row.cluster_n, row.cluster_ls, row.cluster_ss, row.cluster_centroid)
                label_response_ids[lbl] = []
            label_response_ids[lbl].append(row.response_id)

        if len(label_cfs) < 2:
            return False

        new_clusters: dict[int, Cluster] = {}
        for lbl, (n, ls, ss, centroid) in label_cfs.items():
            nc = Cluster(id=uuid4(), n=n, ls=list(ls), ss=ss, centroid=list(centroid))
            session.add(nc)
            new_clusters[lbl] = nc

        # Single UPDATE — CASE WHEN maps each response_id to its new cluster_id.
        id_to_cluster_id = {
            rid: new_clusters[lbl].id
            for lbl, response_ids in label_response_ids.items()
            for rid in response_ids
        }
        await session.execute(
            update(LLMResponse)
            .where(LLMResponse.id.in_(id_to_cluster_id))
            .values(cluster_id=sa_case(id_to_cluster_id, value=LLMResponse.id))
        )

        # Reassign golden set entries — single CASE WHEN UPDATE.
        c0 = np.array(list(new_clusters[0].centroid), dtype=np.float32)
        c1 = np.array(list(new_clusters[1].centroid), dtype=np.float32)
        gs_result = await session.execute(
            select(GoldenSet.id, GoldenSet.expected_embedding, GoldenSet.request_embedding)
            .where(GoldenSet.cluster_id == cluster.id)
        )
        gs_id_to_cluster_id = {}
        for gs_id, gs_resp_emb, gs_req_emb in gs_result.all():
            pivot = gs_req_emb if gs_req_emb is not None else gs_resp_emb
            emb = np.array(list(pivot), dtype=np.float32)
            lbl = 0 if cosine_similarity(emb, c0) >= cosine_similarity(emb, c1) else 1
            gs_id_to_cluster_id[gs_id] = new_clusters[lbl].id
        if gs_id_to_cluster_id:
            await session.execute(
                update(GoldenSet)
                .where(GoldenSet.id.in_(gs_id_to_cluster_id))
                .values(cluster_id=sa_case(gs_id_to_cluster_id, value=GoldenSet.id))
            )

        await session.execute(delete(Cluster).where(Cluster.id == cluster.id))
        logger.info(
            "cluster_split_complete",
            original_cluster_id=str(cluster.id),
            new_cluster_ids=[str(nc.id) for nc in new_clusters.values()],
        )
        return True

    @staticmethod
    def radius_sql_expr():
        """
        SQL equivalent of ClusteringFeature.radius:
            R = sqrt(max(0, SS/N - ||LS||² / N²))

        pgvector: (ls <#> ls) = negative inner product, so * -1 gives ||LS||².
        """
        n = Cluster.n.cast(Float)
        ls_norm_sq = Cluster.ls.op("<#>")(Cluster.ls) * -1
        return func.sqrt(func.greatest(0, Cluster.ss / n - ls_norm_sq / (n * n)))
