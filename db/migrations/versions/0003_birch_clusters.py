"""Replace centroid+size with BIRCH Clustering Feature (n, ls, ss)

Revision ID: 0003
Revises: 0002
Create Date: 2024-01-01 00:00:02.000000

CF = (N, LS, SS) where:
  N  = number of points
  LS = linear sum of all embeddings (vector)
  SS = sum of squared norms (scalar) — equals N for unit-norm vectors

Centroid is derived: LS / N  (computed, not stored)
Radius is derived:   sqrt(1 - ||LS||² / N²)  (computed, not stored)
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop old columns
    op.drop_column("clusters", "centroid")
    op.drop_column("clusters", "size")

    # Add BIRCH CF columns
    op.add_column("clusters", sa.Column("n", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("clusters", sa.Column("ls", Vector(384), nullable=True))
    op.add_column("clusters", sa.Column("ss", sa.Float(), nullable=False, server_default="0"))


def downgrade() -> None:
    op.drop_column("clusters", "ss")
    op.drop_column("clusters", "ls")
    op.drop_column("clusters", "n")

    op.add_column("clusters", sa.Column("centroid", Vector(384), nullable=True))
    op.add_column("clusters", sa.Column("size", sa.Integer(), nullable=False, server_default="0"))
