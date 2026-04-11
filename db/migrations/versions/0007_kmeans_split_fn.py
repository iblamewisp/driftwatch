"""Add kmeans_split_cluster SQL function.

Revision ID: 0007
Revises: 0006

Runs k-means (k=2) entirely inside Postgres so the caller never has to
pull embedding vectors into Python.

Initialisation: pick the farthest pair of points from the cluster —
same deterministic strategy as the Python kmeans_split() it replaces.

Each iteration:
  1. Re-assign every point to the nearer centroid (cosine distance via <=>).
  2. Recompute each centroid with avg(vector) — pgvector aggregate.

Returns one row per response_id with:
  label         — 0 or 1
  cluster_n     — CF.N  for this label's group (same value repeated per group)
  cluster_ls    — CF.LS (sum of embeddings)   requires pgvector >= 0.7.0
  cluster_ss    — CF.SS (sum of squared norms)
  cluster_centroid — LS / N

Python uses label+CF rows to create two new Cluster rows and reassign FKs.
No embedding data crosses the network.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_CREATE = """
CREATE OR REPLACE FUNCTION kmeans_split_cluster(
    p_cluster_id uuid,
    p_iterations  int DEFAULT 10
)
RETURNS TABLE(
    response_id      uuid,
    label            smallint,
    cluster_n        bigint,
    cluster_ls       vector,
    cluster_ss       double precision,
    cluster_centroid vector
)
LANGUAGE plpgsql AS $$
DECLARE
    c0 vector;
    c1 vector;
    tmp vector;
BEGIN
    -- Init: pick the two most distant points.
    -- Step 1 — pick any point as seed.
    SELECT request_embedding INTO c0
    FROM   llm_responses
    WHERE  cluster_id = p_cluster_id
      AND  request_embedding IS NOT NULL
    ORDER BY random()
    LIMIT 1;

    IF c0 IS NULL THEN RETURN; END IF;

    -- Step 2 — point farthest from seed (largest cosine distance).
    SELECT request_embedding INTO c1
    FROM   llm_responses
    WHERE  cluster_id = p_cluster_id
      AND  request_embedding IS NOT NULL
    ORDER BY c0 <=> request_embedding DESC
    LIMIT 1;

    IF c1 IS NULL OR c0 = c1 THEN RETURN; END IF;

    -- Iterate: reassign → recompute centroids.
    FOR i IN 1..p_iterations LOOP
        SELECT avg(request_embedding) INTO tmp
        FROM   llm_responses
        WHERE  cluster_id = p_cluster_id
          AND  request_embedding IS NOT NULL
          AND  (c0 <=> request_embedding) <= (c1 <=> request_embedding);
        IF tmp IS NOT NULL THEN c0 := tmp; END IF;

        SELECT avg(request_embedding) INTO tmp
        FROM   llm_responses
        WHERE  cluster_id = p_cluster_id
          AND  request_embedding IS NOT NULL
          AND  (c1 <=> request_embedding) < (c0 <=> request_embedding);
        IF tmp IS NOT NULL THEN c1 := tmp; END IF;
    END LOOP;

    -- Final assignments + CF aggregates in one pass.
    RETURN QUERY
    WITH assignments AS (
        SELECT
            r.id                                                    AS response_id,
            CASE
                WHEN (c0 <=> r.request_embedding) <= (c1 <=> r.request_embedding)
                THEN 0::smallint
                ELSE 1::smallint
            END                                                     AS label,
            r.request_embedding
        FROM llm_responses r
        WHERE r.cluster_id = p_cluster_id
          AND r.request_embedding IS NOT NULL
    ),
    cfs AS (
        SELECT
            a.label,
            count(*)                                                AS n,
            sum(a.request_embedding)                                AS ls,
            sum(0.0 - (a.request_embedding <#> a.request_embedding)) AS ss
        FROM assignments a
        GROUP BY a.label
    )
    SELECT
        a.response_id,
        a.label,
        f.n,
        f.ls,
        f.ss,
        (f.ls / f.n::float)                                        AS centroid
    FROM assignments a
    JOIN cfs f USING (label);
END;
$$;
"""

_DROP = "DROP FUNCTION IF EXISTS kmeans_split_cluster(uuid, int);"


def upgrade() -> None:
    op.execute(_CREATE)


def downgrade() -> None:
    op.execute(_DROP)
