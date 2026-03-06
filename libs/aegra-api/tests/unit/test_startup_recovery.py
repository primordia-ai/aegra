"""Unit tests for startup_recovery.recover_stale_runs()."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aegra_api.core.orm import Assistant as AssistantORM
from aegra_api.core.orm import Run as RunORM  # noqa: F401
from aegra_api.core.orm import Thread as ThreadORM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_orm(
    run_id: str = "run-1",
    thread_id: str = "thread-1",
    assistant_id: str = "asst-1",
    status: str = "pending",
    user_id: str = "user-1",
    input_data: dict | None = None,
    config: dict | None = None,
    context: dict | None = None,
) -> RunORM:
    run = RunORM()
    run.run_id = run_id
    run.thread_id = thread_id
    run.assistant_id = assistant_id
    run.status = status
    run.user_id = user_id
    run.input = input_data or {}
    run.config = config or {}
    run.context = context or {}
    run.created_at = datetime.now(UTC)
    run.updated_at = datetime.now(UTC)
    return run


def _make_assistant_orm(
    assistant_id: str = "asst-1",
    graph_id: str = "vibe",
) -> AssistantORM:
    asst = AssistantORM()
    asst.assistant_id = assistant_id
    asst.graph_id = graph_id
    asst.name = "test"
    asst.user_id = "user-1"
    asst.config = {}
    asst.context = {}
    return asst


def _make_thread_orm(
    thread_id: str = "thread-1",
    status: str = "busy",
) -> ThreadORM:
    thread = ThreadORM()
    thread.thread_id = thread_id
    thread.status = status
    thread.user_id = "user-1"
    thread.metadata_json = {}
    thread.created_at = datetime.now(UTC)
    thread.updated_at = datetime.now(UTC)
    return thread


class _FakeScalars:
    """Mimics the object returned by session.scalars(...)."""

    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


def _build_session(
    stale_runs: list[RunORM],
    assistant: AssistantORM | None,
    stuck_threads: list[ThreadORM],
    pending_runs_after: list[RunORM] | None = None,
) -> AsyncMock:
    """Build a mock AsyncSession whose .scalars() and .scalar() calls return
    test data in the expected call order."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()

    # Track call count to distinguish which query is which.
    call_order: list[int] = [0]

    async def _scalars(_stmt):
        call_order[0] += 1
        if call_order[0] == 1:
            # First call: stale runs
            return _FakeScalars(stale_runs)
        elif call_order[0] == 2:
            # Second call: stuck threads
            return _FakeScalars(stuck_threads)
        return _FakeScalars([])

    async def _scalar(_stmt):
        call_order[0] += 1
        # Called once per pending run (assistant lookup), then once per stuck
        # thread (active run count check).
        # For simplicity we return the assistant for all early calls, then
        # None for thread-active-run checks (meaning no active runs remain).
        if assistant is not None and call_order[0] <= 1 + len(stale_runs):
            return assistant
        return None  # no active runs → thread should be reset

    session.scalars = _scalars
    session.scalar = _scalar

    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_stale_runs_returns_empty_stats():
    """When there are no stale runs, stats should all be zero and nothing is committed."""
    session = AsyncMock()
    session.scalars = AsyncMock(return_value=_FakeScalars([]))
    session.commit = AsyncMock()

    mock_maker = MagicMock()
    mock_maker.return_value.__aenter__ = AsyncMock(return_value=session)
    mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("aegra_api.startup_recovery._get_session_maker", return_value=mock_maker):
        from aegra_api.startup_recovery import recover_stale_runs

        stats = await recover_stale_runs()

    assert stats == {"pending_restarted": 0, "running_errored": 0, "threads_reset": 0}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_running_run_is_marked_error():
    """A run stuck in 'running' state should be marked 'error' with an explanation."""
    run = _make_run_orm(run_id="run-crash", status="running")

    executed_updates: list[dict] = []

    async def _fake_execute(stmt):
        # Capture the values dict from the UPDATE statement via compiled params.
        try:
            compiled = stmt.compile()
            executed_updates.append(dict(compiled.params))
        except Exception:
            pass

    call_order = [0]

    async def _scalars(_stmt):
        call_order[0] += 1
        if call_order[0] == 1:
            return _FakeScalars([run])
        return _FakeScalars([])  # no stuck threads

    async def _scalar(_stmt):
        return None  # thread active-run check → no active runs

    session = AsyncMock()
    session.scalars = _scalars
    session.scalar = _scalar
    session.execute = _fake_execute
    session.commit = AsyncMock()

    mock_maker = MagicMock()
    mock_maker.return_value.__aenter__ = AsyncMock(return_value=session)
    mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("aegra_api.startup_recovery._get_session_maker", return_value=mock_maker):
        from aegra_api.startup_recovery import recover_stale_runs

        stats = await recover_stale_runs()

    assert stats["running_errored"] == 1
    assert stats["pending_restarted"] == 0
    # The UPDATE should include status='error'
    assert any("error" in str(u) for u in executed_updates)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pending_run_is_restarted():
    """A run stuck in 'pending' state should be restarted as a new asyncio task."""
    run = _make_run_orm(run_id="run-pending", status="pending", input_data={"step": "RESEARCH"})
    assistant = _make_assistant_orm(graph_id="vibe")

    call_order = [0]

    async def _scalars(_stmt):
        call_order[0] += 1
        if call_order[0] == 1:
            return _FakeScalars([run])
        return _FakeScalars([])  # no stuck threads

    async def _scalar(_stmt):
        call_order[0] += 1
        if call_order[0] <= 3:
            return assistant  # assistant lookup
        return None  # thread active-run check

    session = AsyncMock()
    session.scalars = _scalars
    session.scalar = _scalar
    session.execute = AsyncMock()
    session.commit = AsyncMock()

    mock_maker = MagicMock()
    mock_maker.return_value.__aenter__ = AsyncMock(return_value=session)
    mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

    created_tasks: list = []

    def _fake_create_task(coro):
        # Cancel the coroutine immediately to avoid dangling tasks in tests.
        task = MagicMock(spec=asyncio.Task)
        created_tasks.append(task)
        # Close the coroutine to avoid ResourceWarning.
        coro.close()
        return task

    fake_active_runs: dict = {}
    fake_execute_run_async = AsyncMock()

    with (
        patch("aegra_api.startup_recovery._get_session_maker", return_value=mock_maker),
        patch("aegra_api.startup_recovery.asyncio.create_task", side_effect=_fake_create_task),
        patch("aegra_api.api.runs.active_runs", fake_active_runs),
        patch("aegra_api.api.runs.execute_run_async", fake_execute_run_async),
    ):
        from aegra_api.startup_recovery import recover_stale_runs

        stats = await recover_stale_runs()

    assert stats["pending_restarted"] == 1
    assert stats["running_errored"] == 0
    assert len(created_tasks) == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pending_run_without_assistant_marked_error():
    """A pending run whose assistant is missing should be marked error, not restarted."""
    run = _make_run_orm(run_id="run-orphan", status="pending", assistant_id="gone")

    call_order = [0]

    async def _scalars(_stmt):
        call_order[0] += 1
        if call_order[0] == 1:
            return _FakeScalars([run])
        return _FakeScalars([])

    async def _scalar(_stmt):
        return None  # assistant not found

    session = AsyncMock()
    session.scalars = _scalars
    session.scalar = _scalar
    session.execute = AsyncMock()
    session.commit = AsyncMock()

    mock_maker = MagicMock()
    mock_maker.return_value.__aenter__ = AsyncMock(return_value=session)
    mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("aegra_api.startup_recovery._get_session_maker", return_value=mock_maker):
        from aegra_api.startup_recovery import recover_stale_runs

        stats = await recover_stale_runs()

    assert stats["running_errored"] == 1
    assert stats["pending_restarted"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mixed_runs_handled_independently():
    """One pending and one running run are each handled according to their status."""
    run_pending = _make_run_orm(run_id="run-p", status="pending")
    run_running = _make_run_orm(run_id="run-r", status="running")
    assistant = _make_assistant_orm()

    call_order = [0]

    async def _scalars(_stmt):
        call_order[0] += 1
        if call_order[0] == 1:
            return _FakeScalars([run_running, run_pending])
        return _FakeScalars([])

    async def _scalar(_stmt):
        call_order[0] += 1
        if call_order[0] <= 4:
            return assistant
        return None

    session = AsyncMock()
    session.scalars = _scalars
    session.scalar = _scalar
    session.execute = AsyncMock()
    session.commit = AsyncMock()

    mock_maker = MagicMock()
    mock_maker.return_value.__aenter__ = AsyncMock(return_value=session)
    mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

    def _fake_create_task(coro):
        task = MagicMock(spec=asyncio.Task)
        coro.close()
        return task

    fake_active_runs: dict = {}
    fake_execute_run_async = AsyncMock()

    with (
        patch("aegra_api.startup_recovery._get_session_maker", return_value=mock_maker),
        patch("aegra_api.startup_recovery.asyncio.create_task", side_effect=_fake_create_task),
        patch("aegra_api.api.runs.active_runs", fake_active_runs),
        patch("aegra_api.api.runs.execute_run_async", fake_execute_run_async),
    ):
        from aegra_api.startup_recovery import recover_stale_runs

        stats = await recover_stale_runs()

    assert stats["pending_restarted"] == 1
    assert stats["running_errored"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stuck_busy_thread_is_reset():
    """A thread stuck in 'busy' with no remaining active runs should be reset to idle."""
    thread = _make_thread_orm(thread_id="thread-stuck", status="busy")

    call_order = [0]

    async def _scalars(_stmt):
        call_order[0] += 1
        if call_order[0] == 1:
            return _FakeScalars([])  # no stale runs
        elif call_order[0] == 2:
            return _FakeScalars([thread])  # one stuck thread
        return _FakeScalars([])

    async def _scalar(_stmt):
        return None  # no active runs on stuck thread

    session = AsyncMock()
    session.scalars = _scalars
    session.scalar = _scalar
    session.execute = AsyncMock()
    session.commit = AsyncMock()

    mock_maker = MagicMock()
    mock_maker.return_value.__aenter__ = AsyncMock(return_value=session)
    mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("aegra_api.startup_recovery._get_session_maker", return_value=mock_maker):
        from aegra_api.startup_recovery import recover_stale_runs

        stats = await recover_stale_runs()

    assert stats["threads_reset"] == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_busy_thread_with_active_run_not_reset():
    """A 'busy' thread that still has an active pending run should NOT be reset."""
    thread = _make_thread_orm(thread_id="thread-active", status="busy")
    # This run is newly restarted, so it's still pending on the thread.
    active_run = _make_run_orm(run_id="run-restarted", thread_id="thread-active", status="pending")

    call_order = [0]

    async def _scalars(_stmt):
        call_order[0] += 1
        if call_order[0] == 1:
            return _FakeScalars([])  # no stale runs to process
        elif call_order[0] == 2:
            return _FakeScalars([thread])
        return _FakeScalars([])

    async def _scalar(_stmt):
        # Return a non-None value → thread has active runs → should not be reset.
        return active_run

    session = AsyncMock()
    session.scalars = _scalars
    session.scalar = _scalar
    session.execute = AsyncMock()
    session.commit = AsyncMock()

    mock_maker = MagicMock()
    mock_maker.return_value.__aenter__ = AsyncMock(return_value=session)
    mock_maker.return_value.__aexit__ = AsyncMock(return_value=False)

    with patch("aegra_api.startup_recovery._get_session_maker", return_value=mock_maker):
        from aegra_api.startup_recovery import recover_stale_runs

        stats = await recover_stale_runs()

    assert stats["threads_reset"] == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_lifespan_calls_recover_stale_runs():
    """The lifespan function should call recover_stale_runs() during startup."""
    import importlib

    import aegra_api.main as main_module

    importlib.reload(main_module)
    from aegra_api.main import lifespan

    with (
        patch("aegra_api.main.run_migrations_async", new_callable=AsyncMock),
        patch("aegra_api.main.db_manager") as mock_db_manager,
        patch("aegra_api.main.get_langgraph_service") as mock_get_langgraph_service,
        patch("aegra_api.main.event_store") as mock_event_store,
        patch("aegra_api.main.setup_observability"),
        patch("aegra_api.main.recover_stale_runs", new_callable=AsyncMock) as mock_recover,
    ):
        mock_db_manager.initialize = AsyncMock()
        mock_db_manager.close = AsyncMock()
        mock_langgraph_service = MagicMock()
        mock_langgraph_service.initialize = AsyncMock()
        mock_get_langgraph_service.return_value = mock_langgraph_service
        mock_event_store.start_cleanup_task = AsyncMock()
        mock_event_store.stop_cleanup_task = AsyncMock()
        mock_recover.return_value = {"pending_restarted": 0, "running_errored": 0, "threads_reset": 0}

        async with lifespan(MagicMock()):
            pass

        mock_recover.assert_called_once()
