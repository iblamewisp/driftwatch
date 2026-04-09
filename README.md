# Driftwatch

> Open-source, self-hosted transparent proxy between your application and OpenAI / Anthropic that detects when LLM response quality drifts from a baseline — zero code changes required, just change `base_url`.

---

## How it works

Driftwatch sits between your app and the upstream LLM API. Every request passes through unchanged. In the background it embeds responses, clusters them by topic using BIRCH, and periodically checks whether recent response quality has dropped relative to a per-cluster baseline. When it has, you get notified.

```
Your App  →  Driftwatch Proxy  →  OpenAI / Anthropic
                    ↓
             Redis Stream
                    ↓
         Clustering Service (BIRCH)
                    ↓
         Evaluator (cosine similarity vs golden set)
                    ↓
         Drift Detection (every 10 min, per cluster)
                    ↓
         Alert (Telegram / Webhook)
```

---

## Quick Start

**Prerequisites:** Docker, Docker Compose

```bash
# 1. Clone
git clone https://github.com/yourname/driftwatch && cd driftwatch

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
| `NOTIFICATION_CHANNEL` | `none` | `none`, `telegram`, or `webhook` |
| `TELEGRAM_BOT_TOKEN` | — | Telegram bot token |
| `TELEGRAM_CHAT_ID` | — | Telegram chat ID to send alerts to |
| `WEBHOOK_URL` | — | Webhook endpoint for drift alerts |
| `WEBHOOK_SECRET` | — | HMAC-SHA256 signing secret for webhook payload |
| `ENABLE_METRICS` | `true` | Expose Prometheus metrics at `/metrics` |
| `VECTOR_BACKEND` | `pgvector` | `pgvector` or `qdrant` (qdrant is a stub, not yet implemented) |
| `QDRANT_URL` | — | Qdrant endpoint, only if `VECTOR_BACKEND=qdrant` |
| `QDRANT_COLLECTION` | `driftwatch` | Qdrant collection name |

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
│           fire-and-forget (asyncio.create_task)   │
│             INSERT llm_responses                  │
│             XADD driftwatch:clustering            │
│             enqueue evaluator (every Nth)         │
└──────────────────────────────────────────────────┘
          │                          │
          ▼                          ▼
 ┌─────────────────┐       ┌──────────────────────┐
 │ Clustering Svc  │       │   Evaluator (Celery)  │
 │ (Redis Stream)  │       │                       │
 │                 │       │  embed response       │
 │ BIRCH on        │       │  cosine sim vs        │
 │ request embeds  │       │  golden set           │
 │ → cluster_id    │       │  → quality_score      │
 │ → golden set    │       └──────────────────────┘
 │   (auto mode)   │                 │
 └─────────────────┘                 ▼
          │                 ┌──────────────────────┐
          └────────────────►│     PostgreSQL        │
                            │     + pgvector        │
                            │                       │
                            │  llm_responses        │
                            │  golden_set           │
                            │  clusters             │
                            │  drift_events         │
                            └──────────────────────┘
                                      │
                            ┌─────────▼────────────┐
                            │  Detection (Celery    │
                            │  beat, every 10 min)  │
                            │                       │
                            │  per cluster:         │
                            │  baseline = oldest 50%│
                            │  current  = newest 50%│
                            │  delta > threshold    │
                            │    → DriftEvent       │
                            │    → Notification     │
                            └──────────────────────┘
```

---

## Limitations

- **Single instance only.** The request sampling counter is in-memory and process-local. Running multiple proxy replicas will multiply your sampling rate by the number of replicas. There is no distributed counter.
- **Beat does not catch up.** If `celery beat` is down, scheduled drift detection runs are lost — there is no catch-up mechanism. Detection resumes from the next scheduled interval after restart.
- **Cosine similarity is a proxy for quality.** Driftwatch measures whether responses are semantically similar to your golden set, not whether they are factually correct. A confident wrong answer can score high if it sounds like a good answer.
- **Cold-start in auto mode.** The first `GOLDEN_SET_WARMUP` responses per cluster become your baseline. If early traffic is atypical, the baseline is polluted. Use manual mode if baseline integrity matters.
- **Qdrant backend is not implemented.** `VECTOR_BACKEND=qdrant` raises `NotImplementedError`. Only `pgvector` is supported.
- **GPU is optional but not orchestrated.** The LitServe embedding service auto-detects CUDA if available. No configuration needed, but `nvidia-container-toolkit` must be installed on the host for GPU passthrough to work.
- **No multi-tenancy.** One `DRIFTWATCH_KEY_HASH`, one deployment, one golden set namespace.

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
