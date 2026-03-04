"""Integration tests for Alembic migrations against a real Postgres DB.

Requires a running Postgres instance. Set AEGRA_TEST_DATABASE_URL to point at it,
e.g.:
    AEGRA_TEST_DATABASE_URL="postgresql+asyncpg://user:password@localhost:5432/aegra"

These tests are skipped automatically when AEGRA_TEST_DATABASE_URL is not set.
"""

import os
from unittest.mock import patch

import pytest
import sqlalchemy as sa
from alembic import command

TEST_DB_URL = os.environ.get("AEGRA_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DB_URL,
    reason="AEGRA_TEST_DATABASE_URL not set — skipping migration integration tests",
)


def _run_alembic(direction: str) -> None:
    """Run alembic upgrade/downgrade with the test DB URL.

    env.py calls `config.set_main_option("sqlalchemy.url", settings.db.database_url)`
    unconditionally at module load, overriding anything we set on the Config
    object. We patch the DatabaseSettings instance's computed property via
    patch.object with a PropertyMock so env.py picks up the test URL.
    """
    from unittest.mock import PropertyMock
    from aegra_api.core.migrations import get_alembic_config
    from aegra_api.settings import settings
    cfg = get_alembic_config()
    with patch.object(type(settings.db), "database_url", new_callable=PropertyMock, return_value=TEST_DB_URL):
        if direction == "head":
            command.upgrade(cfg, "head")
        else:
            command.downgrade(cfg, "base")


@pytest.fixture(scope="module", autouse=True)
def apply_migrations():
    """Ensure DB is at head before running tests.

    Note: downgrade base is intentionally omitted — an existing upstream migration
    (aee821a02fc8) has a broken downgrade path with an unnamed FK constraint.
    upgrade head is idempotent so running it against an already-migrated DB is safe.
    """
    _run_alembic("head")
    yield


@pytest.fixture(scope="module")
def sync_engine():
    # psycopg (v3) sync driver for simple inspection queries
    sync_url = TEST_DB_URL.replace("postgresql+asyncpg://", "postgresql+psycopg://")
    engine = sa.create_engine(sync_url)
    yield engine
    engine.dispose()


class TestMigrationSchema:
    """Verify the migrated schema is correct."""

    def test_checkpoint_tables_exist(self, sync_engine):
        with sync_engine.connect() as conn:
            result = conn.execute(sa.text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
                  AND table_name IN ('checkpoints', 'checkpoint_blobs', 'checkpoint_writes',
                                     'checkpoint_migrations', 'store', 'store_migrations')
                ORDER BY table_name
            """))
            tables = {row[0] for row in result}
        assert tables == {
            "checkpoints",
            "checkpoint_blobs",
            "checkpoint_writes",
            "checkpoint_migrations",
            "store",
            "store_migrations",
        }

    def test_checkpoint_fk_constraints_exist(self, sync_engine):
        with sync_engine.connect() as conn:
            result = conn.execute(sa.text("""
                SELECT conname
                FROM pg_constraint
                WHERE contype = 'f'
                  AND conrelid::regclass::text IN ('checkpoints', 'checkpoint_blobs', 'checkpoint_writes')
                ORDER BY conname
            """))
            constraints = {row[0] for row in result}
        assert "checkpoints_thread_id_fkey" in constraints
        assert "checkpoint_blobs_thread_id_fkey" in constraints
        assert "checkpoint_writes_thread_id_fkey" in constraints

    def test_alembic_version_at_head(self, sync_engine):
        from alembic.script import ScriptDirectory
        from aegra_api.core.migrations import get_alembic_config
        script = ScriptDirectory.from_config(get_alembic_config())
        head = script.get_current_head()

        with sync_engine.connect() as conn:
            result = conn.execute(sa.text("SELECT version_num FROM alembic_version"))
            version = result.scalar()
        assert version == head


class TestCheckpointCascadeDelete:
    """Verify that deleting a thread cascades to checkpoint tables."""

    @pytest.fixture(autouse=True)
    def cleanup(self, sync_engine):
        yield
        with sync_engine.begin() as conn:
            conn.execute(sa.text("DELETE FROM thread WHERE thread_id = 'test-cascade-thread'"))

    def test_delete_thread_cascades_to_checkpoints(self, sync_engine):
        with sync_engine.begin() as conn:
            # Insert a thread
            conn.execute(sa.text("""
                INSERT INTO thread (thread_id, status, metadata_json, user_id)
                VALUES ('test-cascade-thread', 'idle', '{}', 'test-user')
            """))
            # Insert a checkpoint row tied to that thread
            conn.execute(sa.text("""
                INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, checkpoint, metadata)
                VALUES ('test-cascade-thread', '', 'ckpt-1', '{}', '{}')
            """))
            # Insert a checkpoint_blob row
            conn.execute(sa.text("""
                INSERT INTO checkpoint_blobs (thread_id, checkpoint_ns, channel, version, type, blob)
                VALUES ('test-cascade-thread', '', 'channel-1', 'v1', 'bytes', NULL)
            """))
            # Insert a checkpoint_write row
            conn.execute(sa.text("""
                INSERT INTO checkpoint_writes
                    (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, blob, task_path)
                VALUES ('test-cascade-thread', '', 'ckpt-1', 'task-1', 0, 'channel-1', 'bytes', '\\x00', '')
            """))

        # Now delete the thread — cascade should clean up all checkpoint rows
        with sync_engine.begin() as conn:
            conn.execute(sa.text("DELETE FROM thread WHERE thread_id = 'test-cascade-thread'"))

        with sync_engine.connect() as conn:
            cp = conn.execute(sa.text(
                "SELECT count(*) FROM checkpoints WHERE thread_id = 'test-cascade-thread'"
            )).scalar()
            blobs = conn.execute(sa.text(
                "SELECT count(*) FROM checkpoint_blobs WHERE thread_id = 'test-cascade-thread'"
            )).scalar()
            writes = conn.execute(sa.text(
                "SELECT count(*) FROM checkpoint_writes WHERE thread_id = 'test-cascade-thread'"
            )).scalar()

        assert cp == 0, f"Expected 0 checkpoints after thread delete, got {cp}"
        assert blobs == 0, f"Expected 0 checkpoint_blobs after thread delete, got {blobs}"
        assert writes == 0, f"Expected 0 checkpoint_writes after thread delete, got {writes}"
