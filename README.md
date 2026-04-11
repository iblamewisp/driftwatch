# Driftwatch

> Open-source, self-hosted transparent proxy between your application and OpenAI / Anthropic that detects when LLM response quality drifts from a baseline — zero code changes required, just change `base_url`.

---

## How it works

Driftwatch sits between your app and the upstream LLM API. Every request passes through unchanged. In the background it embeds responses, clusters them by topic using BIRCH, and periodically checks whether recent response quality has dropped relative to a per-cluster baseline. When it has, you get notified.

```
Your App  →  Driftwatch Proxy  →  OpenAI / Anthropic
                    │
                    ├─ INSERT llm_responses (needs_clustering=True, request_text)
                    ├─ XADD driftwatch:clustering
                    └─ every Nth: SET needs_evaluation=True  (flag only, no task yet)
                              │
                              ▼
                    Clustering Service (Redis Stream)
                      interleaved batch embed → LitServe
                      BIRCH assign_cluster (HNSW ANN + advisory lock)
                      if needs_evaluation=True → enqueue Evaluator
                              │
                    ┌─────────┴──────────────┐
                    ▼                        ▼
           PostgreSQL + pgvector      Evaluator (Celery)
           llm_responses              cluster_id guaranteed set
           clusters / golden_set      cosine sim vs golden set
           drift_events               → quality_score
                    │
                    ▼
          Drift Detection (Celery beat, every 10 min)
          per cluster: baseline vs current quality scores
          delta > DRIFT_THRESHOLD → DriftEvent → Notification
```

---

## Quick Start

**Prerequisites:** Docker, Docker Compose

```bash
# 1. Clone
git clone https://github.com/iblamewisp/driftwatch && cd driftwatch

# 2. Configure
cp .env.example .env

# 3. Generate an API key — paste the hash into .env as DRIFTWATCH_KEY_HASH
docker-compose -f docker/docker-compose.yml run --rm proxy python -m cli generate-key

# 4. Start
docker-compose -f docker/docker-compose.yml up -d

# 5. Point your client at the proxy
#    Replace:  base_url="https://api.openai.com"
#    With:     base_url="http://localhost:8000"
#    Add header: X-DriftWatch-Key: dwk_<your key>
```

Your OpenAI API key travels in `Authorization: Bearer sk-...` as normal — Driftwatch forwards it and never touches it.

### Anthropic

The proxy also accepts Anthropic-native requests at `/v1/messages`. Point `base_url` at `http://localhost:8000` with your Anthropic client and add the same `X-DriftWatch-Key` header.

---

## Code Flow

### 1. Proxy (`app/`)

```
app/proxy.py          — FastAPI router, two endpoints (/v1/chat/completions, /v1/messages)
app/forwarding.py     — forward_unary() / forward_streaming() — httpx to upstream
app/pipeline.py       — log_and_enqueue(): INSERT + XADD + needs_evaluation flag
app/utils.py          — extract_request_text(), forwarded_headers()
```

Request path:
1. `AuthMiddleware` validates `X-DriftWatch-Key` via `hmac.compare_digest`
2. Request forwarded to upstream (OpenAI or Anthropic) via `httpx.AsyncClient`
3. Response returned to client immediately
4. `asyncio.create_task(log_and_enqueue(...))` fires in the background — never on the hot path
5. Inside `log_and_enqueue`: INSERT `llm_responses` with `needs_clustering=True` and `request_text`; then XADD to Redis stream. If XADD fails (Redis down), the row stays with `needs_clustering=True` and the recovery task picks it up later.
6. Every `SAMPLING_RATE`-th request sets `needs_evaluation=True` on the row — the evaluator is **not** enqueued here

### 2. Clustering service (`services/clustering/service.py`)

Runs as a standalone asyncio process (not Celery). Reads from `driftwatch:clustering` Redis Stream in adaptive batches:

1. On startup: recover any pending messages left in PEL from a previous crash, dropping poison pills (delivery count ≥ `CLUSTERING_MAX_DELIVERY_ATTEMPTS`)
2. Main loop: `XREADGROUP` blocks up to `CLUSTERING_MAX_WAIT_MS`, flushes when buffer reaches `CLUSTERING_MAX_BATCH` pairs or timeout fires
3. Per batch: interleaved embed `[req1, resp1, req2, resp2, ...]` → one LitServe call → split by stride → `asyncio.gather` over all `assign_cluster` calls in parallel
4. After clustering: check which response_ids in the batch have `needs_evaluation=True` — enqueue `evaluate_response` Celery task for each, then flip the flag to False. Evaluator is guaranteed to start only after `cluster_id` and `response_embedding` are written.
5. Auto golden set: if `GOLDEN_SET_MODE=auto` and cluster size < `GOLDEN_SET_WARMUP`, insert response embedding into `golden_set`

### 3. Clustering logic (`db/repositories/cluster_postgres.py`, `services/clustering/birch.py`)

`assign_cluster` — BIRCH absorption:
1. ANN query: `SELECT ... ORDER BY centroid <=> req_emb LIMIT CLUSTERING_ANN_CANDIDATES` (HNSW index)
2. For each candidate: compute `ClusteringFeature.would_absorb()` — checks if adding the point keeps radius ≤ `BIRCH_THRESHOLD`
3. If a candidate absorbs: UPDATE cluster CF (N, LS, SS, centroid) in one statement
4. If no candidate absorbs: acquire `pg_advisory_xact_lock(CLUSTER_CREATION)`, re-run ANN (another worker may have created a cluster while we waited), create new cluster if still nothing

`split_oversized_clusters` — runs via Celery beat every `CLUSTER_SPLIT_INTERVAL`:
1. SQL pre-filter: find clusters where `sqrt(SS/N - ||LS||²/N²) > BIRCH_THRESHOLD` — radius computed entirely in Postgres via `<#>` operator, no Python round-trip
2. Per oversized cluster: call `kmeans_split_cluster(cluster_id)` PL/pgSQL function
   - Initialises with two farthest points (deterministic)
   - Iterates: reassign via `<=>`, recompute centroids via `avg(vector)` aggregate — entirely inside Postgres
   - Returns `(response_id, label, cluster_n, cluster_ls, cluster_ss, cluster_centroid)` — no embedding vectors cross the network
3. Python inserts two new `Cluster` rows from the returned CF components
4. Single `UPDATE llm_responses ... CASE WHEN` reassigns all response FKs
5. Single `UPDATE golden_set ... CASE WHEN` reassigns golden set entries by cosine similarity to new centroids
6. DELETE old cluster

### 4. Evaluator (`workers/evaluator.py`)

Celery task, retries up to `EVALUATOR_MAX_RETRIES` with `EVALUATOR_RETRY_COUNTDOWN` seconds between attempts:

1. Load response from DB — use stored `response_embedding` if present (set by clustering service), otherwise fall back to embedding `raw_content` (legacy rows)
2. If `cluster_id` is NULL: retry — clustering may not have finished yet
3. Fetch golden embeddings for the cluster
4. `mean_cosine_similarity(response_embedding, golden_embeddings)` — single numpy matrix multiply, no loop
5. Store `quality_score` and emit `driftwatch_quality_score_latest` Prometheus gauge

### 5. Drift detection (`workers/detection.py`)

Three Celery beat tasks:

| Task | Schedule | What it does |
|---|---|---|
| `run_drift_detection` | every `DRIFT_DETECTION_INTERVAL` | Per cluster: compare oldest 50% vs newest 50% of quality scores; if delta > `DRIFT_THRESHOLD` write `DriftEvent` and send alert |
| `run_cluster_split` | every `CLUSTER_SPLIT_INTERVAL` | Find and split oversized clusters via `kmeans_split_cluster()` |
| `recover_unclustered_responses` | every `RECOVERY_INTERVAL` | Re-XADD rows with `needs_clustering=True` older than `RECOVERY_MIN_AGE_SECONDS` |

---

## Golden Set

The golden set is the quality baseline. Responses are compared against it per semantic cluster to detect drift.

### Auto mode (default)

The first `GOLDEN_SET_WARMUP` responses per cluster (default: 100) are automatically added to the golden set. After that, auto-insertion stops and quality scoring begins.

`quality_score` will be `NULL` until the golden set is seeded — this is expected.

**Caveat:** if your traffic is unusual or the model behaves anomalously at launch, those early responses become your baseline. If you care about baseline quality, use manual mode.

### Manual mode

Set `GOLDEN_SET_MODE=manual`. Populate the golden set explicitly via CLI:

```bash
python -m cli golden-set add \
  --prompt "Summarize the theory of relativity in one paragraph." \
  --description "Physics summary baseline"
```

Requires `OPENAI_API_KEY` in your environment (only used at this call site — never stored or forwarded by the proxy).

---

## CLI Reference

```
python -m cli generate-key
    Generate a new API key and its SHA-256 hash for .env

python -m cli golden-set add --prompt "..." --description "..."
    Call OpenAI with the prompt, embed the response, add to golden set

python -m cli golden-set list
    Print all golden set entries as a table

python -m cli golden-set clear
    Delete all golden set entries (prompts confirmation)
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DRIFTWATCH_KEY_HASH` | — | SHA-256 of your API key. **Required.** |
| `DATABASE_URL` | — | `postgresql+asyncpg://...`. **Required.** |
| `REDIS_URL` | `redis://redis:6379/0` | Celery broker + Redis Stream |
| `OPENAI_BASE_URL` | `https://api.openai.com` | Upstream OpenAI endpoint |
| `OPENAI_API_KEY` | — | Only used by CLI `golden-set add`. Never logged. |
| `ANTHROPIC_BASE_URL` | `https://api.anthropic.com` | Upstream Anthropic endpoint |
| `EMBEDDING_SERVICE_URL` | `http://litserve:8001` | LitServe embedding service |
| `SAMPLING_RATE` | `10` | Every Nth response is evaluated |
| `GOLDEN_SET_MODE` | `auto` | `auto` or `manual` |
| `GOLDEN_SET_WARMUP` | `100` | Entries per cluster before evaluation starts (auto mode) |
| `DRIFT_THRESHOLD` | `0.15` | Relative quality drop that triggers an alert (15%) |
| `BIRCH_THRESHOLD` | `0.25` | Max cluster radius — lower = tighter clusters, higher = fewer |
| `CLUSTERING_ANN_CANDIDATES` | `10` | HNSW candidates fetched before absorption check — raise if cluster_created rate spikes |
| `CLUSTERING_MAX_BATCH` | `32` | Max pairs per clustering batch (× 2 texts to LitServe) |
| `CLUSTERING_MAX_WAIT_MS` | `200` | Max wait before flushing a partial batch |
| `CLUSTERING_MAX_DELIVERY_ATTEMPTS` | `3` | Drop Redis Stream messages delivered this many times without ACK |
| `DRIFT_DETECTION_INTERVAL` | `600` | Seconds between drift detection runs |
| `CLUSTER_SPLIT_INTERVAL` | `3600` | Seconds between cluster split checks |
| `RECOVERY_INTERVAL` | `300` | Seconds between unclustered-response recovery runs |
| `RECOVERY_MIN_AGE_SECONDS` | `300` | Only recover rows older than this — avoids racing with live proxy |
| `CELERY_VISIBILITY_TIMEOUT` | `3600` | Must be ≥ `CLUSTER_SPLIT_INTERVAL` or split tasks may requeue mid-run |
| `EVALUATOR_MAX_RETRIES` | `5` | Max Celery retries for evaluate_response |
| `EVALUATOR_RETRY_COUNTDOWN` | `10` | Seconds between evaluator retries |
| `NOTIFICATION_CHANNEL` | `none` | `none`, `telegram`, or `webhook` |
| `TELEGRAM_BOT_TOKEN` | — | Telegram bot token |
| `TELEGRAM_CHAT_ID` | — | Telegram chat ID to send alerts to |
| `WEBHOOK_URL` | — | Webhook endpoint for drift alerts |
| `WEBHOOK_SECRET` | — | HMAC-SHA256 signing secret for webhook payload |
| `ENABLE_METRICS` | `true` | Expose Prometheus metrics at `/metrics` |
| `VECTOR_BACKEND` | `pgvector` | `pgvector` or `qdrant` (qdrant is a stub, not yet implemented) |
| `QDRANT_URL` | — | Qdrant endpoint, only if `VECTOR_BACKEND=qdrant` |
| `QDRANT_COLLECTION` | `driftwatch` | Qdrant collection name |

All variables are validated at startup via Pydantic `BaseSettings`. The process will refuse to start if required fields are missing or cross-field constraints are violated (e.g. `CELERY_VISIBILITY_TIMEOUT < CLUSTER_SPLIT_INTERVAL`).

---

## Known Risks and Limitations

### High risk

**Non-atomic DriftEvent write** (`workers/detection.py`)
A `DriftEvent` is written with `alert_sent=False`, the notification is sent, then a second transaction flips `alert_sent=True`. If the worker dies between the two commits the event is permanently stuck as unsent with no retry path. Fix: outbox pattern — write event, have a reliable sender flip the flag transactionally.

**`get_worker_loop()` global state** (`workers/celery_app.py`)
The per-process event loop is stored in a module-level global without a lock. Safe with Celery's default `prefork` pool (one OS process per worker). Silently breaks with `-P threads`, `-P gevent`, or `-P eventlet`. Do not change the pool type.

### Performance caveats

**ANN candidate limit** (`CLUSTERING_ANN_CANDIDATES=10`)
HNSW returns the top-K nearest clusters by centroid. If `BIRCH_THRESHOLD` is tight and clusters are dense, the correct absorbing cluster could rank outside K — a new cluster is created unnecessarily. Monitor the `cluster_created` log rate; a spike means K is too small or `BIRCH_THRESHOLD` too tight.

**`updated_at` on Cluster is never updated**
`server_default=func.now()` fires only on INSERT. Every absorption and split leaves `updated_at` at creation time — useless for debugging cluster activity.

**Sampling counter is shared via Redis — multiple proxy replicas are safe**
`SAMPLING_RATE` uses `INCR driftwatch:sampling_counter` — atomic, shared across all instances. If Redis is unreachable the counter call fails gracefully and that request is skipped for evaluation (logged as a warning, not an error).

**Beat does not catch up**
If `celery beat` is down, scheduled detection runs are lost. Detection resumes from the next interval after restart.

### Observability gap

Clusters with fewer than 10 quality scores are silently skipped by drift detection. No metric is emitted for this — a freshly created cluster is unmonitored until it warms up with no visibility into how many clusters are in this state.

---

## Optional: Monitoring Stack

```bash
docker-compose -f docker/docker-compose.yml -f docker/docker-compose.monitoring.yml up -d
```

- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin / admin)

Metrics exposed at `/metrics` (Prometheus format):

| Metric | Description |
|---|---|
| `driftwatch_requests_total` | Total proxied requests, labeled by model and stream |
| `driftwatch_latency_ms` | Request latency histogram |
| `driftwatch_drift_alerts_total` | Total drift alerts fired |
| `driftwatch_quality_score_latest` | Latest quality score from evaluator |
| `driftwatch_golden_set_size` | Current golden set entry count |

---

## Architecture

```
  Your App
     │  Authorization: Bearer sk-...   (forwarded as-is, never stored)
     │  X-DriftWatch-Key: dwk_...      (validated, then stripped)
     ▼
┌──────────────────────────────────────────────────┐
│                 Driftwatch Proxy                  │
│                  (FastAPI)                        │
│                                                   │
│  AuthMiddleware → /v1/chat/completions            │
│                   /v1/messages                    │
│                        │                          │
│                  forward request                  │
│                        ▼                          │
│           OpenAI API / Anthropic API              │
│                        │                          │
│             return response to client             │
│                        │                          │
│  fire-and-forget (asyncio.create_task)            │
│    INSERT llm_responses                           │
│      needs_clustering=True                        │
│      request_text stored                          │
│    XADD driftwatch:clustering                     │
│    every Nth: SET needs_evaluation=True           │
└──────────────────────────────────────────────────┘
                    │
                    ▼
          ┌─────────────────────────┐
          │    Clustering Service   │
          │    (Redis Stream loop)  │
          │                         │
          │  interleaved batch      │
          │  embed → LitServe       │
          │                         │
          │  BIRCH assign_cluster   │
          │  HNSW ANN top-K         │
          │  advisory lock on       │
          │  cluster create         │
          │                         │
          │  needs_evaluation=True? │
          │  → enqueue Evaluator    │
          │  → flip flag False      │
          │                         │
          │  auto golden set insert │
          └─────────┬───────────────┘
                    │                      ┌──────────────────────┐
                    │        ┌────────────►│   Evaluator (Celery)  │
                    │        │             │                       │
                    ▼        │             │  cluster_id is set    │
          ┌──────────────────┴───┐         │  use stored embedding │
          │     PostgreSQL        │         │  (re-embed legacy    │
          │     + pgvector        │         │   rows as fallback)  │
          │                       │         │                       │
          │  llm_responses        │         │  cosine sim vs        │
          │  golden_set           │◄────────│  golden set           │
          │  clusters             │         │  → quality_score      │
          │  drift_events         │         └──────────────────────┘
          └──────────┬────────────┘
                     │
           ┌─────────┴────────────┐
           │                      │
           ▼                      ▼
 ┌──────────────────┐   ┌──────────────────────┐
 │ Detection        │   │ Split                │
 │ (beat, 10 min)   │   │ (beat, 1 hour)       │
 │                  │   │                      │
 │ per cluster:     │   │ kmeans_split_cluster │
 │ baseline vs      │   │ PL/pgSQL — runs      │
 │ current scores   │   │ inside Postgres      │
 │ → DriftEvent     │   └──────────────────────┘
 │ → Notification   │
 └──────────────────┘
```

---

## Security Notes

- `Authorization: Bearer sk-...` is forwarded to the upstream API and **never** logged, stored, or read by Driftwatch.
- API key validation uses `hmac.compare_digest` (timing-safe).
- Webhook payloads are signed with HMAC-SHA256.
- `OPENAI_API_KEY` is only read by the CLI at call time and is never persisted.

---

## Contributing

Driftwatch is open-source. Issues and pull requests are welcome.

Things that would be good to have:
- Qdrant backend implementation
- Dashboard for cluster inspection and BIRCH threshold tuning
- Distributed sampling counter (Redis-based)
- Judge-model scoring as an alternative to cosine similarity
- Outbox pattern for reliable alert delivery
