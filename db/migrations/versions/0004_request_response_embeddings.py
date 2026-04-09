"""Split embedding into request_embedding + response_embedding.

BIRCH clustering now operates on request_embedding (the query topic),
while quality scoring compares response_embedding against golden set.

Revision ID: 0004
Revises: 0003
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename existing embedding column (response content) → response_embedding
    op.alter_column("llm_responses", "embedding", new_column_name="response_embedding")

    # Add request_embedding column (nullable — backfill not feasible for old rows)
    op.add_column(
        "llm_responses",
        sa.Column("request_embedding", Vector(384), nullable=True),
    )

    # Add request_embedding to golden_set for cluster reassignment on split
    op.add_column(
        "golden_set",
        sa.Column("request_embedding", Vector(384), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("golden_set", "request_embedding")
    op.drop_column("llm_responses", "request_embedding")
    op.alter_column("llm_responses", "response_embedding", new_column_name="embedding")
