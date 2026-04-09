"""Initial schema: enable pgvector, create llm_responses, golden_set, drift_events

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "llm_responses",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("request_id", sa.String(), nullable=False),
        sa.Column("prompt_hash", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column("finish_reason", sa.String(), nullable=False),
        sa.Column("raw_content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=True),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_llm_responses_created_at", "llm_responses", ["created_at"])
    op.create_index("ix_llm_responses_prompt_hash", "llm_responses", ["prompt_hash"])

    op.create_table(
        "golden_set",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("expected_embedding", Vector(384), nullable=False),
        sa.Column("description", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "drift_events",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("detected_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("similarity_score", sa.Float(), nullable=False),
        sa.Column("baseline_score", sa.Float(), nullable=False),
        sa.Column("delta", sa.Float(), nullable=False),
        sa.Column("alert_sent", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("alert_channel", sa.String(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_drift_events_detected_at", "drift_events", ["detected_at"])


def downgrade() -> None:
    op.drop_table("drift_events")
    op.drop_table("golden_set")
    op.drop_table("llm_responses")
