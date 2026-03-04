"""Add ON DELETE CASCADE foreign keys to LangGraph checkpoint tables.

LangGraph's PostgresSaver creates checkpoint tables at runtime. This migration
takes ownership of those tables by adding them (if not already present) and
ensuring their thread_id foreign keys cascade on thread deletion.

Revision ID: 20260304101500
Revises: d042a0ca1cb5
Create Date: 2026-03-04 10:15:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "20260304101500"
down_revision = "d042a0ca1cb5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create checkpoint tables (if absent) and add ON DELETE CASCADE FKs."""

    conn = op.get_bind()

    def table_exists(name: str) -> bool:
        return conn.execute(
            sa.text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables"
                " WHERE table_schema = 'public' AND table_name = :t)"
            ),
            {"t": name},
        ).scalar()

    def fk_exists(table: str, fk_name: str) -> bool:
        return conn.execute(
            sa.text(
                "SELECT EXISTS (SELECT 1 FROM information_schema.table_constraints"
                " WHERE table_schema = 'public' AND table_name = :t"
                " AND constraint_name = :c AND constraint_type = 'FOREIGN KEY')"
            ),
            {"t": table, "c": fk_name},
        ).scalar()

    # Create checkpoint tables if LangGraph hasn't created them yet.
    if not table_exists("checkpoint_migrations"):
        op.create_table(
            "checkpoint_migrations",
            sa.Column("v", sa.Integer(), nullable=False),
            sa.PrimaryKeyConstraint("v"),
        )

    if not table_exists("checkpoints"):
        op.create_table(
            "checkpoints",
            sa.Column("thread_id", sa.Text(), nullable=False),
            sa.Column("checkpoint_ns", sa.Text(), server_default=sa.text("''::text"), nullable=False),
            sa.Column("checkpoint_id", sa.Text(), nullable=False),
            sa.Column("parent_checkpoint_id", sa.Text(), nullable=True),
            sa.Column("type", sa.Text(), nullable=True),
            sa.Column("checkpoint", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
            sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), server_default=sa.text("'{}'::jsonb"), nullable=False),
            sa.ForeignKeyConstraint(["thread_id"], ["thread.thread_id"], name="checkpoints_thread_id_fkey", ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("thread_id", "checkpoint_ns", "checkpoint_id"),
        )
        op.create_index("checkpoints_thread_id_idx", "checkpoints", ["thread_id"])
    elif not fk_exists("checkpoints", "checkpoints_thread_id_fkey"):
        op.execute(sa.text(
            "DELETE FROM checkpoints WHERE thread_id NOT IN (SELECT thread_id FROM thread)"
        ))
        op.create_foreign_key(
            "checkpoints_thread_id_fkey", "checkpoints", "thread", ["thread_id"], ["thread_id"], ondelete="CASCADE"
        )

    if not table_exists("checkpoint_blobs"):
        op.create_table(
            "checkpoint_blobs",
            sa.Column("thread_id", sa.Text(), nullable=False),
            sa.Column("checkpoint_ns", sa.Text(), server_default=sa.text("''::text"), nullable=False),
            sa.Column("channel", sa.Text(), nullable=False),
            sa.Column("version", sa.Text(), nullable=False),
            sa.Column("type", sa.Text(), nullable=False),
            sa.Column("blob", sa.LargeBinary(), nullable=True),
            sa.ForeignKeyConstraint(["thread_id"], ["thread.thread_id"], name="checkpoint_blobs_thread_id_fkey", ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("thread_id", "checkpoint_ns", "channel", "version"),
        )
        op.create_index("checkpoint_blobs_thread_id_idx", "checkpoint_blobs", ["thread_id"])
    elif not fk_exists("checkpoint_blobs", "checkpoint_blobs_thread_id_fkey"):
        op.execute(sa.text(
            "DELETE FROM checkpoint_blobs WHERE thread_id NOT IN (SELECT thread_id FROM thread)"
        ))
        op.create_foreign_key(
            "checkpoint_blobs_thread_id_fkey", "checkpoint_blobs", "thread", ["thread_id"], ["thread_id"], ondelete="CASCADE"
        )

    if not table_exists("checkpoint_writes"):
        op.create_table(
            "checkpoint_writes",
            sa.Column("thread_id", sa.Text(), nullable=False),
            sa.Column("checkpoint_ns", sa.Text(), server_default=sa.text("''::text"), nullable=False),
            sa.Column("checkpoint_id", sa.Text(), nullable=False),
            sa.Column("task_id", sa.Text(), nullable=False),
            sa.Column("idx", sa.Integer(), nullable=False),
            sa.Column("channel", sa.Text(), nullable=False),
            sa.Column("type", sa.Text(), nullable=True),
            sa.Column("blob", sa.LargeBinary(), nullable=False),
            sa.Column("task_path", sa.Text(), server_default=sa.text("''::text"), nullable=False),
            sa.ForeignKeyConstraint(["thread_id"], ["thread.thread_id"], name="checkpoint_writes_thread_id_fkey", ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"),
        )
        op.create_index("checkpoint_writes_thread_id_idx", "checkpoint_writes", ["thread_id"])
    elif not fk_exists("checkpoint_writes", "checkpoint_writes_thread_id_fkey"):
        op.execute(sa.text(
            "DELETE FROM checkpoint_writes WHERE thread_id NOT IN (SELECT thread_id FROM thread)"
        ))
        op.create_foreign_key(
            "checkpoint_writes_thread_id_fkey", "checkpoint_writes", "thread", ["thread_id"], ["thread_id"], ondelete="CASCADE"
        )

    if not table_exists("store"):
        op.create_table(
            "store",
            sa.Column("prefix", sa.Text(), nullable=False),
            sa.Column("key", sa.Text(), nullable=False),
            sa.Column("value", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
            sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.Column("updated_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
            sa.PrimaryKeyConstraint("prefix", "key"),
        )

    if not table_exists("store_migrations"):
        op.create_table(
            "store_migrations",
            sa.Column("v", sa.Integer(), nullable=False),
            sa.PrimaryKeyConstraint("v"),
        )


def downgrade() -> None:
    op.drop_constraint("checkpoint_writes_thread_id_fkey", "checkpoint_writes", type_="foreignkey")
    op.drop_constraint("checkpoint_blobs_thread_id_fkey", "checkpoint_blobs", type_="foreignkey")
    op.drop_constraint("checkpoints_thread_id_fkey", "checkpoints", type_="foreignkey")
