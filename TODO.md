# Known Issues

Architectural issues that are understood, accepted for now, and should be addressed before scaling.
Not bugs — the system works — but each item is a landmine at higher load or cluster count.

---

## I/O Blocking

### [RESOLVED] `send_alert()` blocked the async event loop
`send_alert` made async. `requests` replaced with `httpx.AsyncClient` in both
`telegram.py` and `webhook.py`. `detection.py` now `await`s the call.

### [BLOCKING] Non-atomic DriftEvent write
**File:** `workers/detection.py:59-103`
**Issue:** A `DriftEvent` is written in one transaction (`alert_sent=False`), the alert is
sent, then a second transaction updates `alert_sent=True`. If the worker dies between the
two commits, the event is stuck as `alert_sent=False` permanently with no retry path.
**Fix:** Outbox pattern — write `alert_sent=False`, have a reliable sender pick it up and
flip the flag. Or wrap send + update in a single transaction with a savepoint.

---

## Thread Safety

### [RESOLVED] `get_worker_loop()` uses unprotected global state
Replaced module-level global with `threading.local()`. Each thread now owns its
event loop — safe under `-P threads`. Prefork behaviour is unchanged (one thread
per process). `-P gevent` / `-P eventlet` remain unsupported.

### [RESOLVED] `assign_cluster` create-on-miss race
Guarded by `pg_advisory_xact_lock(AdvisoryLock.CLUSTER_CREATION)` via `db/locks.py`.
Workers contending on the "no absorbing cluster" branch serialize, re-query ANN under
the lock, and absorb into any cluster created while waiting.

### [RESOLVED] Auto golden set: TOCTOU overshoot and post-split orphaned entries
Two races in `_process_batch`:

1. **TOCTOU within a batch** — all coroutines called `count_golden` in parallel, then all
   inserted in parallel. N items for the same near-threshold cluster all read the same
   count and all passed the cap check, overshooting `GOLDEN_SET_WARMUP` by up to N-1.

2. **Post-split stale cluster_id** — `assign_cluster` returned OLD_ID, then
   `run_cluster_split` committed (OLD deleted, goldens redistributed to A and B).
   `count_golden(OLD_ID)` returned 0, `0 < GOLDEN_SET_WARMUP` passed, and
   `insert_golden(cluster_id=OLD_ID)` wrote an orphaned row (no FK constraint on
   `golden_set.cluster_id`) that no evaluator would ever find.

**Fix:** Replaced the `count → filter → insert` pattern with `insert_golden_if_under_cap`
(`db/repositories/pgvector_repo.py`). Opens a transaction, does `SELECT ... FOR UPDATE` on
the cluster row (serializes concurrent callers for the same cluster; returns `None` if the
cluster was deleted), re-counts under the lock, inserts only if under cap. Clustering
service now calls this directly with no pre-count step.

### [RESOLVED] Sampling counter was process-local — blocked horizontal scaling
Replaced in-memory global with `redis.incr("driftwatch:sampling_counter")`. Atomic,
shared across all proxy replicas. If Redis is unreachable the request is skipped for
evaluation with a warning — not a hard failure.

---

## Performance / Scalability

### [RESOLVED] `assign_cluster` full table scan on every request
Replaced with HNSW ANN query (`centroid <=> req_emb LIMIT 10`) via migration `0005`.
`centroid` column added to `clusters`, kept in sync on every absorption and split.

### [RESOLVED] `_process_batch` N+1 — sequential DB writes per batch item
`assign_cluster` calls now run via `asyncio.gather` — parallel across all batch items.
Advisory lock serializes only the rare "create cluster" branch on the DB side.
`count_golden` calls also gathered. Was up to 96 sequential round-trips at `MAX_BATCH=32`.

### [RESOLVED] LLMResponse N+1 in `split_oversized_clusters`
Replaced per-row UPDATEs with a single bulk `UPDATE ... CASE WHEN` statement.

### [PERF] ANN candidate limit K=10 may miss the true best absorbing cluster
**File:** `db/repositories/cluster_postgres.py — assign_cluster`
**Issue:** HNSW returns top-10 by centroid cosine distance. If `BIRCH_THRESHOLD` is tight
and clusters are dense, the correct absorbing cluster could rank 11th. System would then
create a new cluster unnecessarily.
**Fix:** Tune K upward (20–50) or monitor `cluster_created` log rate — a spike means K is
too small or `BIRCH_THRESHOLD` too tight.

### [RESOLVED] GoldenSet reassignment in `split_oversized_clusters` was N+1
Replaced per-row UPDATE loop with a single bulk `UPDATE golden_set ... CASE WHEN` statement.
Both response and golden set reassignment now complete in one round-trip each.

### [PERF] `updated_at` on Cluster is never updated on writes
**File:** `db/models.py`
**Issue:** `server_default=func.now()` only fires on INSERT. Every absorption and split
leaves `updated_at` at creation time — useless for debugging.
**Fix:** Add `onupdate=func.now()` or include it explicitly in UPDATE statements.

---

## Data Integrity

### [RESOLVED] Orphaned `llm_responses` rows on Redis stream failure
When `XADD` to `driftwatch:clustering` failed (Redis down), the DB row existed but
never entered the clustering pipeline — `cluster_id`, embeddings, and `quality_score`
stayed NULL permanently with no recovery path.

**Fix applied:**
- `request_text` stored on `llm_responses` (needed to reconstruct the stream entry)
- `needs_clustering` flag: `True` on insert, `False` after successful XADD
- Celery beat task `recover_unclustered_responses` (every 5 min) finds `needs_clustering=True`
  rows older than 5 min and re-XADDs them. Migration `0006` adds a partial index on
  `(needs_clustering, created_at) WHERE needs_clustering = true` for the recovery query.

---

## Evaluation ordering

### [RESOLVED] Evaluator could start before clustering finished
Evaluator was enqueued by the proxy at the same time as XADD — before `cluster_id`
and `response_embedding` were written. With default settings (5 retries × 10s) any
clustering delay over 50 seconds would silently drop the evaluation permanently.

**Fix applied:**
- Proxy sets `needs_evaluation=True` flag instead of enqueuing directly
- Clustering service reads the flag after `assign_cluster` completes, enqueues
  `evaluate_response`, then clears the flag — guaranteeing cluster_id is present
- Migration `0008` adds `needs_evaluation` column with partial index
- Evaluator no longer retries on `cluster_id is None` — logs error and returns instead

---

## Observability

### [OBS] Detection worker silently skips clusters on insufficient data
**File:** `workers/detection.py`
**Issue:** Clusters with fewer than 10 quality scores are skipped — correct behaviour,
but no metric is emitted. A new cluster is silently unmonitored until it warms up.
**Fix:** Emit a gauge for `clusters_below_detection_threshold` count.
