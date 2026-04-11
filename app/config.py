from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    # ── Auth ──────────────────────────────────────────────────────────────────
    DRIFTWATCH_KEY_HASH: str

    # ── Database ──────────────────────────────────────────────────────────────
    DATABASE_URL: str
    DB_COMMAND_TIMEOUT: int = 20  # seconds — asyncpg kills hung queries after this

    # ── Redis ─────────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://redis:6379/0"

    # ── LLM Providers ────────────────────────────────────────────────────────
    OPENAI_BASE_URL: str = "https://api.openai.com"
    OPENAI_API_KEY: str = ""
    ANTHROPIC_BASE_URL: str = "https://api.anthropic.com"

    # ── Embedding service ─────────────────────────────────────────────────────
    EMBEDDING_SERVICE_URL: str = "http://litserve:8001"

    # ── Vector backend ────────────────────────────────────────────────────────
    VECTOR_BACKEND: Literal["pgvector", "qdrant"] = "pgvector"
    QDRANT_URL: str = ""
    QDRANT_COLLECTION: str = "driftwatch"

    # ── Sampling ──────────────────────────────────────────────────────────────
    # 1-in-N responses are sent to the evaluator worker.
    SAMPLING_RATE: int = Field(default=10, ge=1)

    # ── Golden set ───────────────────────────────────────────────────────────
    GOLDEN_SET_MODE: Literal["auto", "manual"] = "auto"
    # Auto mode: stop inserting goldens once a cluster has this many entries.
    GOLDEN_SET_WARMUP: int = Field(default=100, ge=1)

    # ── Drift detection ───────────────────────────────────────────────────────
    # Alert if quality drops by more than this fraction relative to baseline.
    DRIFT_THRESHOLD: float = Field(default=0.15, gt=0.0, lt=1.0)

    # ── BIRCH clustering ──────────────────────────────────────────────────────
    # A point is absorbed into a cluster if the resulting radius stays <= this.
    # Lower = tighter, more clusters. Higher = looser, fewer clusters.
    BIRCH_THRESHOLD: float = Field(default=0.25, gt=0.0, lt=1.0)
    # ANN candidates fetched from HNSW index before running absorption check.
    # Raise if cluster_created log rate spikes — see TODO.md.
    CLUSTERING_ANN_CANDIDATES: int = Field(default=10, ge=1, le=100)

    # ── Redis clustering stream ───────────────────────────────────────────────
    CLUSTERING_STREAM_KEY: str = "driftwatch:clustering"
    CLUSTERING_CONSUMER_GROUP: str = "clustering-group"
    CLUSTERING_CONSUMER_NAME: str = "clustering-worker-1"
    # Max pairs per batch (× 2 texts sent to LitServe; must not exceed LitServe max_batch_size / 2).
    CLUSTERING_MAX_BATCH: int = Field(default=32, ge=1, le=32)
    # Wait up to this many ms for a full batch before flushing a partial one.
    CLUSTERING_MAX_WAIT_MS: int = Field(default=200, ge=10)
    # Drop stream messages that have been delivered this many times without ACK (poison pills).
    CLUSTERING_MAX_DELIVERY_ATTEMPTS: int = Field(default=3, ge=1)

    # ── Celery schedules (seconds) ────────────────────────────────────────────
    DRIFT_DETECTION_INTERVAL: int = Field(default=600, ge=60)
    CLUSTER_SPLIT_INTERVAL: int = Field(default=3600, ge=60)
    RECOVERY_INTERVAL: int = Field(default=300, ge=60)
    # Must be greater than the longest expected task duration.
    CELERY_VISIBILITY_TIMEOUT: int = Field(default=3600, ge=60)

    # ── Recovery ──────────────────────────────────────────────────────────────
    # Only re-enqueue rows older than this (seconds) — avoids racing with live proxy.
    RECOVERY_MIN_AGE_SECONDS: int = Field(default=300, ge=30)

    # ── Evaluator ─────────────────────────────────────────────────────────────
    EVALUATOR_MAX_RETRIES: int = Field(default=5, ge=1)
    EVALUATOR_RETRY_COUNTDOWN: int = Field(default=10, ge=1)

    # ── Notifications ─────────────────────────────────────────────────────────
    NOTIFICATION_CHANNEL: Literal["telegram", "webhook", "none"] = "none"
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    WEBHOOK_URL: str = ""
    WEBHOOK_SECRET: str = ""

    # ── Observability ─────────────────────────────────────────────────────────
    ENABLE_METRICS: bool = True

    @model_validator(mode="after")
    def validate_notification_config(self) -> "Settings":
        if self.NOTIFICATION_CHANNEL == "telegram":
            if not self.TELEGRAM_BOT_TOKEN or not self.TELEGRAM_CHAT_ID:
                raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required when NOTIFICATION_CHANNEL=telegram")
        if self.NOTIFICATION_CHANNEL == "webhook":
            if not self.WEBHOOK_URL:
                raise ValueError("WEBHOOK_URL is required when NOTIFICATION_CHANNEL=webhook")
        return self

    @model_validator(mode="after")
    def validate_celery_visibility_timeout(self) -> "Settings":
        if self.CELERY_VISIBILITY_TIMEOUT < self.CLUSTER_SPLIT_INTERVAL:
            raise ValueError(
                "CELERY_VISIBILITY_TIMEOUT must be >= CLUSTER_SPLIT_INTERVAL "
                "or the split task may be requeued mid-run"
            )
        return self

    model_config = SettingsConfigDict(env_file=".env")
        env_file_encoding = "utf-8"


settings = Settings()
