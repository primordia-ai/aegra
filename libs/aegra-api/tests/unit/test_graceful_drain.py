"""Unit tests for graceful-drain behaviour (Option 2 pod-restart recovery)."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# _handle_sigterm / _draining flag
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_handle_sigterm_sets_draining_flag():
    """_handle_sigterm() must set _draining to True."""
    import aegra_api.main as main_module

    original = main_module._draining
    try:
        main_module._draining = False
        main_module._handle_sigterm()
        assert main_module._draining is True
    finally:
        main_module._draining = original


@pytest.mark.unit
def test_draining_resets_to_false_by_default():
    """The module-level default must be False (no drain until SIGTERM)."""
    import aegra_api.main as main_module

    # If previous test left it True, reset it so isolation is verified.
    main_module._draining = False
    assert main_module._draining is False


# ---------------------------------------------------------------------------
# create_run — 503 when draining
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_run_returns_503_when_draining():
    """create_run must raise HTTP 503 while _draining is True."""
    from fastapi import HTTPException

    import aegra_api.main as main_module
    from aegra_api.api.runs import create_run

    original = main_module._draining
    try:
        main_module._draining = True

        mock_request = MagicMock()
        mock_request.assistant_id = "test-assistant"
        mock_request.input = {}
        mock_request.config = None
        mock_request.context = None
        mock_request.command = None
        mock_request.model_dump = MagicMock(return_value={})

        mock_user = MagicMock()
        mock_session = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await create_run(
                thread_id="thread-1",
                request=mock_request,
                user=mock_user,
                session=mock_session,
            )

        assert exc_info.value.status_code == 503
        assert "shutting down" in exc_info.value.detail
    finally:
        main_module._draining = original


@pytest.mark.unit
@pytest.mark.asyncio
async def test_create_run_does_not_reject_when_not_draining():
    """create_run must NOT raise 503 when _draining is False."""
    import aegra_api.main as main_module
    from aegra_api.api.runs import create_run

    original = main_module._draining
    try:
        main_module._draining = False

        mock_request = MagicMock()
        mock_request.assistant_id = "test-assistant"
        mock_request.input = {}
        mock_request.config = None
        mock_request.context = None
        mock_request.command = None
        mock_request.model_dump = MagicMock(return_value={})
        mock_user = MagicMock()
        mock_session = AsyncMock()

        # We only want to verify it gets past the drain check.
        # Patch the next thing that runs (auth context) to stop early.
        with patch(
            "aegra_api.api.runs.build_auth_context",
            side_effect=RuntimeError("stop here"),
        ), pytest.raises(RuntimeError, match="stop here"):
            await create_run(
                thread_id="thread-1",
                request=mock_request,
                user=mock_user,
                session=mock_session,
            )
    finally:
        main_module._draining = original


# ---------------------------------------------------------------------------
# execute_run_async CancelledError — draining vs non-draining
# ---------------------------------------------------------------------------


def _make_execute_run_patches(draining: bool):
    """Return a context manager that patches all execute_run_async dependencies."""
    return [
        patch("aegra_api.main._draining", draining),
        patch("aegra_api.api.runs.update_run_status", new_callable=AsyncMock),
        patch("aegra_api.api.runs.set_thread_status", new_callable=AsyncMock),
        patch("aegra_api.api.runs.streaming_service"),
        patch("aegra_api.api.runs.get_langgraph_service"),
        patch("aegra_api.api.runs.create_run_config", return_value={}),
        patch("aegra_api.api.runs.stream_graph_events"),
        patch("aegra_api.api.runs.with_auth_ctx"),
        patch("aegra_api.api.runs._get_session_maker"),
    ]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_error_during_drain_sets_pending():
    """When _draining=True and CancelledError is raised, run → pending, thread → busy."""
    from aegra_api.api.runs import execute_run_async

    async def _raising_stream(**kwargs):
        raise asyncio.CancelledError()
        yield  # make it an async generator

    mock_streaming = MagicMock()
    mock_streaming.signal_run_cancelled = AsyncMock()
    mock_streaming.put_to_broker = AsyncMock()
    mock_streaming.store_event_from_raw = AsyncMock()
    mock_streaming.cleanup_run = AsyncMock()

    mock_graph_ctx = MagicMock()
    mock_graph_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
    mock_graph_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_lg_service = MagicMock()
    mock_lg_service.get_graph = MagicMock(return_value=mock_graph_ctx)

    mock_auth_ctx = MagicMock()
    mock_auth_ctx.__aenter__ = AsyncMock(return_value=None)
    mock_auth_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()

    with (
        patch("aegra_api.api.runs.update_run_status", new_callable=AsyncMock) as mock_update,
        patch("aegra_api.api.runs.set_thread_status", new_callable=AsyncMock) as mock_thread,
        patch("aegra_api.api.runs.streaming_service", mock_streaming),
        patch("aegra_api.api.runs.get_langgraph_service", return_value=mock_lg_service),
        patch("aegra_api.api.runs.create_run_config", return_value={}),
        patch("aegra_api.api.runs.stream_graph_events", side_effect=_raising_stream),
        patch("aegra_api.api.runs.with_auth_ctx", return_value=mock_auth_ctx),
        patch("aegra_api.main._draining", True),pytest.raises(asyncio.CancelledError)
    ):
        await execute_run_async(
            run_id="run-1",
            thread_id="thread-1",
            graph_id="vibe",
            input_data={"step": "RESEARCH"},
            user=MagicMock(),
            session=mock_session,
        )

    # run → pending (retriable), thread → busy (startup sweep will restart it)
    status_calls = [call.args[1] for call in mock_update.call_args_list]
    assert "pending" in status_calls, f"Expected 'pending' in status calls, got {status_calls}"

    thread_calls = [call.args[2] for call in mock_thread.call_args_list]
    assert "busy" in thread_calls, f"Expected 'busy' in thread calls, got {thread_calls}"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancelled_error_not_draining_sets_interrupted():
    """When _draining=False and CancelledError is raised, run → interrupted, thread → idle."""
    from aegra_api.api.runs import execute_run_async

    async def _raising_stream(**kwargs):
        raise asyncio.CancelledError()
        yield

    mock_streaming = MagicMock()
    mock_streaming.signal_run_cancelled = AsyncMock()
    mock_streaming.put_to_broker = AsyncMock()
    mock_streaming.store_event_from_raw = AsyncMock()
    mock_streaming.cleanup_run = AsyncMock()

    mock_graph_ctx = MagicMock()
    mock_graph_ctx.__aenter__ = AsyncMock(return_value=MagicMock())
    mock_graph_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_lg_service = MagicMock()
    mock_lg_service.get_graph = MagicMock(return_value=mock_graph_ctx)

    mock_auth_ctx = MagicMock()
    mock_auth_ctx.__aenter__ = AsyncMock(return_value=None)
    mock_auth_ctx.__aexit__ = AsyncMock(return_value=False)

    mock_session = AsyncMock()

    with (
        patch("aegra_api.api.runs.update_run_status", new_callable=AsyncMock) as mock_update,
        patch("aegra_api.api.runs.set_thread_status", new_callable=AsyncMock) as mock_thread,
        patch("aegra_api.api.runs.streaming_service", mock_streaming),
        patch("aegra_api.api.runs.get_langgraph_service", return_value=mock_lg_service),
        patch("aegra_api.api.runs.create_run_config", return_value={}),
        patch("aegra_api.api.runs.stream_graph_events", side_effect=_raising_stream),
        patch("aegra_api.api.runs.with_auth_ctx", return_value=mock_auth_ctx),
        patch("aegra_api.main._draining", False),pytest.raises(asyncio.CancelledError)
    ):
        await execute_run_async(
            run_id="run-2",
            thread_id="thread-2",
            graph_id="vibe",
            input_data={"step": "MODEL"},
            user=MagicMock(),
            session=mock_session,
        )

    status_calls = [call.args[1] for call in mock_update.call_args_list]
    assert "interrupted" in status_calls, f"Expected 'interrupted' in status calls, got {status_calls}"

    thread_calls = [call.args[2] for call in mock_thread.call_args_list]
    assert "idle" in thread_calls, f"Expected 'idle' in thread calls, got {thread_calls}"
