"""Add clusters table and cluster_id to llm_responses and golden_set

Revision ID: 0002
Revises: 0001
Create Date: 2024-01-01 00:00:01.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "clusters",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("centroid", Vector(384), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    op.add_column("llm_responses", sa.Column("cluster_id", sa.UUID(), nullable=True))
    op.add_column("golden_set", sa.Column("cluster_id", sa.UUID(), nullable=True))

    op.create_index("ix_llm_responses_cluster_id", "llm_responses", ["cluster_id"])
    op.create_index("ix_golden_set_cluster_id", "golden_set", ["cluster_id"])


def downgrade() -> None:
    op.drop_index("ix_golden_set_cluster_id", "golden_set")
    op.drop_index("ix_llm_responses_cluster_id", "llm_responses")
    op.drop_column("golden_set", "cluster_id")
    op.drop_column("llm_responses", "cluster_id")
    op.drop_table("clusters")
