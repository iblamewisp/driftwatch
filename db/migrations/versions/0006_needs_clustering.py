"""Add request_text and needs_clustering to llm_responses for Redis recovery.

Revision ID: 0006
Revises: 0005

request_text — original user message, needed to re-enqueue rows whose XADD
               to the clustering stream failed (Redis down). Without it we
               cannot reconstruct the stream entry from the DB row alone.

needs_clustering — True on insert, flipped to False after successful XADD.
                   A Celery beat task (recover_unclustered_responses, every 5min)
                   finds True rows older than 5min and re-enqueues them.
                   Existing rows default to False — they predate this mechanism
                   and their stream fate is unknown; re-enqueueing would cause
                   duplicates in the clustering pipeline.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("llm_responses", sa.Column("request_text", sa.Text(), nullable=True))
    # Existing rows: False — unknown stream fate, do not re-enqueue.
    # New rows: True (Python default), flipped to False after XADD.
    op.add_column(
        "llm_responses",
        sa.Column("needs_clustering", sa.Boolean(), nullable=False, server_default="false"),
    )
    op.create_index(
        "ix_llm_responses_needs_clustering",
        "llm_responses",
        ["needs_clustering", "created_at"],
        postgresql_where=sa.text("needs_clustering = true"),
    )


def downgrade() -> None:
    op.drop_index("ix_llm_responses_needs_clustering", "llm_responses")
    op.drop_column("llm_responses", "needs_clustering")
    op.drop_column("llm_responses", "request_text")
