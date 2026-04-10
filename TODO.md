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

### [THREAD-UNSAFE] `get_worker_loop()` uses unprotected global state
**File:** `workers/celery_app.py:8-17`
**Issue:** `_worker_loop` is a module-level global mutated without a lock. Safe with Celery's
default `prefork` pool (one OS process per worker, one thread per process). Breaks silently
with `-P threads`, `-P gevent`, or `-P eventlet`.
**Fix:** Add a `threading.Lock` guard or use `threading.local()`. Or enforce prefork in
deployment config and document the constraint.

### [RESOLVED] `assign_cluster` create-on-miss race
Guarded by `pg_advisory_xact_lock(AdvisoryLock.CLUSTER_CREATION)` via `db/locks.py`.
Workers contending on the "no absorbing cluster" branch serialize, re-query ANN under
the lock, and absorb into any cluster created while waiting.

### [RESOLVED] `threading.Lock` in async proxy counter
Removed. `asyncio` is single-threaded — the lock was unnecessary. Counter increment
is now a plain global mutation.

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
**File:** `db/repositories/cluster_repo.py — assign_cluster`
**Issue:** HNSW returns top-10 by centroid cosine distance. If `BIRCH_THRESHOLD` is tight
and clusters are dense, the correct absorbing cluster could rank 11th. System would then
create a new cluster unnecessarily.
**Fix:** Tune K upward (20–50) or monitor `cluster_created` log rate — a spike means K is
too small or `BIRCH_THRESHOLD` too tight.

### [PERF] GoldenSet reassignment in `split_oversized_clusters` is still N+1
**File:** `db/repositories/cluster_repo.py`
**Issue:** Golden set entries reassigned one UPDATE per row. Low priority — golden set is
small and user-managed. Documented here for completeness.
**Fix:** Bulk `UPDATE ... CASE WHEN` same as LLMResponse fix.

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

## Observability

### [OBS] Detection worker silently skips clusters on insufficient data
**File:** `workers/detection.py`
**Issue:** Clusters with fewer than 10 quality scores are skipped — correct behaviour,
but no metric is emitted. A new cluster is silently unmonitored until it warms up.
**Fix:** Emit a gauge for `clusters_below_detection_threshold` count.
