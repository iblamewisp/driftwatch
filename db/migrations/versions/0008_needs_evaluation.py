"""Add needs_evaluation flag to llm_responses.

Revision ID: 0008
Revises: 0007

needs_evaluation — set True by the proxy when the sampling counter fires.
                   The clustering service reads it after assign_cluster and
                   enqueues evaluate_response only then — guaranteeing that
                   cluster_id and response_embedding are already written before
                   the evaluator task starts. Eliminates the retry-on-null-cluster
                   pattern that could silently drop evaluations if clustering was slow.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "llm_responses",
        sa.Column("needs_evaluation", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.create_index(
        "ix_llm_responses_needs_evaluation",
        "llm_responses",
        ["needs_evaluation"],
        postgresql_where=sa.text("needs_evaluation = true"),
    )


def downgrade() -> None:
    op.drop_index("ix_llm_responses_needs_evaluation", "llm_responses")
    op.drop_column("llm_responses", "needs_evaluation")
