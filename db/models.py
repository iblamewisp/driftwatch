from datetime import datetime
from uuid import UUID, uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Cluster(Base):
    __tablename__ = "clusters"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    # BIRCH Clustering Feature: CF = (N, LS, SS)
    # centroid = LS / N  (derived, not stored)
    # radius   = sqrt(1 - ||LS||² / N²)  (derived, not stored)
    n: Mapped[int] = mapped_column(default=0)
    ls: Mapped[Vector(384)] = mapped_column(nullable=True)       # linear sum of all embeddings
    ss: Mapped[float] = mapped_column(default=0.0)               # sum of squared norms; equals N for unit-norm vectors
    centroid: Mapped[Vector(384)] = mapped_column(nullable=True) # cached LS/N — kept in sync on every write; drives ANN search
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, server_default=func.now())


class LLMResponse(Base):
    __tablename__ = "llm_responses"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    request_id: Mapped[str]
    prompt_hash: Mapped[str]
    model: Mapped[str]
    prompt_tokens: Mapped[int]
    completion_tokens: Mapped[int]
    latency_ms: Mapped[int]
    finish_reason: Mapped[str]
    raw_content: Mapped[str]
    request_text: Mapped[str] = mapped_column(nullable=True)  # original user message; needed for re-enqueue on Redis recovery
    needs_clustering: Mapped[bool] = mapped_column(default=True)  # False after successful XADD; recovery job targets True rows
    # request_embedding — drives BIRCH clustering (what topic is this question about?)
    request_embedding: Mapped[Vector(384)] = mapped_column(nullable=True)
    # response_embedding — used for quality scoring vs golden set
    response_embedding: Mapped[Vector(384)] = mapped_column(nullable=True)
    quality_score: Mapped[float] = mapped_column(nullable=True)
    cluster_id: Mapped[UUID] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, server_default=func.now())


class GoldenSet(Base):
    __tablename__ = "golden_set"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    prompt: Mapped[str]
    expected_embedding: Mapped[Vector(384)]        # response embedding — what a good answer looks like
    request_embedding: Mapped[Vector(384)] = mapped_column(nullable=True)  # for cluster reassignment on split
    description: Mapped[str]
    cluster_id: Mapped[UUID] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, server_default=func.now())


class DriftEvent(Base):
    __tablename__ = "drift_events"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    detected_at: Mapped[datetime] = mapped_column(default=datetime.utcnow, server_default=func.now())
    cluster_id: Mapped[UUID] = mapped_column(nullable=True)
    similarity_score: Mapped[float]
    baseline_score: Mapped[float]
    delta: Mapped[float]
    alert_sent: Mapped[bool] = mapped_column(default=False)
    alert_channel: Mapped[str]
