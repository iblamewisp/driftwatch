"""
Clustering service — reads from Redis Stream, assigns cluster_id to each response.

Stream entry written by proxy:
    XADD driftwatch:clustering * response_id <uuid> request_text <str> response_text <str>

This service:
    1. Reads entries in adaptive batches (drain up to MAX_BATCH pairs or wait MAX_WAIT_MS)
    2. Embeds all texts in one LitServe call using interleaved batching:
           [req1, resp1, req2, resp2, ...]  →  [req1_emb, resp1_emb, req2_emb, resp2_emb, ...]
       Split by stride: request_embeddings = result[::2], response_embeddings = result[1::2]
    3. Assigns each pair to a cluster via BIRCH on request_embedding
    4. Stores both embeddings on the llm_responses row
    5. If auto golden set mode: inserts response_embedding into golden_set for that cluster

MAX_BATCH is 32 *pairs* = 64 texts — stays within LitServe's max_batch_size=64.
"""

import asyncio
import time
from uuid import UUID

import redis.asyncio as aioredis

from app.config import settings
from db.repositories.cluster_repo import assign_cluster
from db.session import AsyncSessionLocal
from db.vector_store import get_vector_store
from monitoring.logging import get_logger
from services.circuit_breaker import CircuitBreakerOpen
from services.embedding.client import embed_batch

logger = get_logger("clustering")

STREAM_KEY = "driftwatch:clustering"
CONSUMER_GROUP = "clustering-group"
CONSUMER_NAME = "clustering-worker-1"
MAX_BATCH = 32   # pairs — 64 texts total per LitServe call
MAX_WAIT_MS = 200
MAX_DELIVERY_ATTEMPTS = 3  # messages delivered this many times without ACK are dropped


async def _ensure_stream_group(redis: aioredis.Redis) -> None:
    try:
        await redis.xgroup_create(STREAM_KEY, CONSUMER_GROUP, id="0", mkstream=True)
    except Exception:
        pass  # group already exists


async def _process_batch(entries: list[dict]) -> None:
    if not entries:
        return

    request_texts = [e["request_text"] for e in entries]
    response_texts = [e["response_text"] for e in entries]
    response_ids = [UUID(e["response_id"]) for e in entries]

    # Interleaved batch: [req1, resp1, req2, resp2, ...]
    # One HTTP round-trip to LitServe for both embeddings per pair.
    interleaved = [text for pair in zip(request_texts, response_texts) for text in pair]
    all_embeddings = await embed_batch(interleaved)

    # Split by stride
    request_embeddings = all_embeddings[::2]
    response_embeddings = all_embeddings[1::2]

    # Assign clusters in parallel — absorption path is fully independent per item.
    # Advisory lock serializes only the rare "create new cluster" branch on the DB side.
    async def _assign(response_id: UUID, req_emb: list[float], resp_emb: list[float]) -> UUID:
        async with AsyncSessionLocal() as session:
            return await assign_cluster(session, response_id, req_emb, resp_emb)

    cluster_ids: list[UUID] = await asyncio.gather(*[
        _assign(rid, req, resp)
        for rid, req, resp in zip(response_ids, request_embeddings, response_embeddings)
    ])

    logger.info("batch_clustered", count=len(entries))

    if settings.GOLDEN_SET_MODE != "auto":
        return

    vector_store = get_vector_store()

    # Count golden set entries for all clusters in parallel
    counts: list[int] = await asyncio.gather(*[
        vector_store.count_golden(cluster_id=cid) for cid in cluster_ids
    ])

    for cluster_id, count, resp_emb, req_emb, response_text in zip(
        cluster_ids, counts, response_embeddings, request_embeddings, response_texts
    ):
        if count < settings.GOLDEN_SET_WARMUP:
            await vector_store.insert_golden(
                prompt=response_text,
                embedding=resp_emb,
                description="auto",
                cluster_id=cluster_id,
                request_embedding=req_emb,
            )
            logger.info("golden_set_entry_added", cluster_id=str(cluster_id), cluster_size=count + 1)


async def _recover_pending(redis: aioredis.Redis) -> None:
    """
    On startup, process any messages left in the PEL from a previous crash.

    Steps:
    1. Check delivery counts — ACK and drop messages that have failed too many
       times (poison pills) so they don't block recovery forever.
    2. Fetch remaining pending messages with xreadgroup("0") and process them.
    """
    pending_meta = await redis.xpending_range(
        STREAM_KEY,
        CONSUMER_GROUP,
        min="-",
        max="+",
        count=MAX_BATCH * 2,
        consumername=CONSUMER_NAME,
    )
    if not pending_meta:
        return

    skip_ids = [
        p["message_id"]
        for p in pending_meta
        if p["times_delivered"] >= MAX_DELIVERY_ATTEMPTS
    ]
    if skip_ids:
        logger.warning(
            "dropping_unprocessable_pending_messages",
            count=len(skip_ids),
            ids=skip_ids,
        )
        await redis.xack(STREAM_KEY, CONSUMER_GROUP, *skip_ids)

    # Fetch content of remaining pending messages (skip_ids already removed from PEL)
    results = await redis.xreadgroup(
        groupname=CONSUMER_GROUP,
        consumername=CONSUMER_NAME,
        streams={STREAM_KEY: "0"},
        count=MAX_BATCH,
    )
    if not results:
        return

    entries = []
    message_ids = []
    for _, messages in results:
        for msg_id, fields in messages:
            entries.append(fields)
            message_ids.append(msg_id)

    if not entries:
        return

    logger.info("recovering_pending_messages", count=len(entries))
    await _process_batch(entries)
    await redis.xack(STREAM_KEY, CONSUMER_GROUP, *message_ids)


async def run() -> None:
    redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    await _ensure_stream_group(redis)
    await _recover_pending(redis)

    logger.info("clustering_service_started", stream=STREAM_KEY)

    buffer: list[dict] = []
    message_ids: list[str] = []
    last_flush = time.monotonic()

    while True:
        results = await redis.xreadgroup(
            groupname=CONSUMER_GROUP,
            consumername=CONSUMER_NAME,
            streams={STREAM_KEY: ">"},
            count=MAX_BATCH,
            block=MAX_WAIT_MS,
        )

        if results:
            for _, messages in results:
                for msg_id, fields in messages:
                    buffer.append(fields)
                    message_ids.append(msg_id)

        elapsed_ms = (time.monotonic() - last_flush) * 1000
        should_flush = len(buffer) >= MAX_BATCH or elapsed_ms >= MAX_WAIT_MS

        if should_flush and buffer:
            try:
                await _process_batch(buffer)
                await redis.xack(STREAM_KEY, CONSUMER_GROUP, *message_ids)
            except CircuitBreakerOpen as exc:
                logger.error("litserve_circuit_open_batch_deferred", error=str(exc))
                await asyncio.sleep(5)
                continue
            except Exception as exc:
                logger.error("batch_processing_failed", error=str(exc))
            finally:
                buffer.clear()
                message_ids.clear()
                last_flush = time.monotonic()


if __name__ == "__main__":
    asyncio.run(run())
