from typing import Literal
from uuid import UUID
from pydantic import BaseModel, Field


class UniformResponse(BaseModel):
    """Provider-agnostic response DTO used for logging and drift detection."""
    content: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    provider: Literal["openai", "anthropic"]


class LoggedResponse(BaseModel):
    request_id: str
    prompt_hash: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    finish_reason: str
    raw_content: str


class EvaluationResult(BaseModel):
    response_id: UUID
    embedding: list[float]
    quality_score: float = Field(ge=0.0, le=1.0)


class DriftDetectionResult(BaseModel):
    baseline_score: float = Field(ge=0.0, le=1.0)
    current_score: float = Field(ge=0.0, le=1.0)
    delta: float
    threshold: float
    alert_triggered: bool
