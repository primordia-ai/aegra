"""Add expression index on thread.metadata_json->>'graph_id'.

Speeds up any query that filters threads by graph_id (e.g. TTL cleanup jobs,
reporting, admin operations).

Revision ID: 20260304120000
Revises: 20260304101500
Create Date: 2026-03-04 12:00:00.000000

"""

from alembic import op

revision = "20260304120000"
down_revision = "20260304101500"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_thread_graph_id "
        "ON thread ((metadata_json->>'graph_id'))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_thread_graph_id")
