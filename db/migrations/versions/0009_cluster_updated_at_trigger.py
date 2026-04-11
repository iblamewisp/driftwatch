"""Add trigger to auto-update clusters.updated_at on every row change.

Revision ID: 0009
Revises: 0008

updated_at was set via server_default=func.now() (INSERT only) — it never
changed after creation. All cluster writes in the codebase use SQLAlchemy Core
UPDATE statements, so onupdate= on the column has no effect. A BEFORE UPDATE
trigger is the only reliable way to keep updated_at current.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE OR REPLACE FUNCTION set_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER clusters_set_updated_at
        BEFORE UPDATE ON clusters
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS clusters_set_updated_at ON clusters;")
    op.execute("DROP FUNCTION IF EXISTS set_updated_at;")
