"""Add centroid column to clusters and HNSW index for ANN search.

Revision ID: 0005
Revises: 0004

Centroid = LS / N — derived from the BIRCH CF, stored redundantly so Postgres
can run an approximate nearest-neighbour search (HNSW, cosine distance) without
pulling the full cluster table into Python on every incoming request.

Invariant: centroid must be kept in sync with ls/n on every write.
See db/repositories/cluster_postgres.py — PostgresClusterRepository.

The HNSW index (m=16, ef_construction=64) is appropriate for 384-dim embeddings
at moderate cluster counts (<100k). Tune m/ef_construction upward if recall
degrades at scale.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("clusters", sa.Column("centroid", Vector(384), nullable=True))

    # Backfill: centroid = ls / n for all existing clusters.
    # pgvector supports vector / scalar, so this is a single pass with no Python round-trip.
    op.execute("UPDATE clusters SET centroid = ls / n WHERE ls IS NOT NULL AND n > 0")

    # HNSW index on centroid using cosine distance operator class.
    # Concurrent build avoids a full table lock — safe to run on a live database.
    op.execute(
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS clusters_centroid_hnsw "
        "ON clusters USING hnsw (centroid vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS clusters_centroid_hnsw")
    op.drop_column("clusters", "centroid")
