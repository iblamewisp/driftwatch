from typing import Literal
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Auth
    DRIFTWATCH_KEY_HASH: str

    # Database
    DATABASE_URL: str
    # SYNC_DATABASE_URL removed — workers use asyncio.run() with asyncpg

    # Redis / Celery
    REDIS_URL: str = "redis://redis:6379/0"

    # Providers
    OPENAI_BASE_URL: str = "https://api.openai.com"
    OPENAI_API_KEY: str = ""  # only used by CLI golden-set add, never logged or stored
    ANTHROPIC_BASE_URL: str = "https://api.anthropic.com"

    # Sampling
    SAMPLING_RATE: int = 10

    # Golden set
    GOLDEN_SET_MODE: Literal["auto", "manual"] = "auto"
    GOLDEN_SET_WARMUP: int = 100

    # Drift detection
    DRIFT_THRESHOLD: float = 0.15

    # Clustering — BIRCH radius threshold.
    # A point is absorbed into a cluster if adding it keeps radius <= this value.
    # Lower = tighter clusters (more of them). Higher = looser (fewer).
    # Splitting occurs when an existing cluster's radius exceeds this threshold.
    BIRCH_THRESHOLD: float = 0.25

    # Notifications
    NOTIFICATION_CHANNEL: Literal["telegram", "webhook", "none"] = "none"
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    WEBHOOK_URL: str = ""
    WEBHOOK_SECRET: str = ""

    # Monitoring
    ENABLE_METRICS: bool = True

    # Database
    DB_COMMAND_TIMEOUT: int = 20  # seconds — asyncpg kills hung queries after this

    # Embedding service
    EMBEDDING_SERVICE_URL: str = "http://litserve:8001"

    # Vector backend
    VECTOR_BACKEND: Literal["pgvector", "qdrant"] = "pgvector"
    QDRANT_URL: str = ""
    QDRANT_COLLECTION: str = "driftwatch"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
