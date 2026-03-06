"""Startup recovery: restart or clean up stale runs left by a previous pod crash.

On pod restart, any runs that were ``pending`` or ``running`` at crash time are
orphaned.  This module queries those runs at startup and either:

- ``pending``  → restart via ``execute_run_async`` (run never started; graph
  checkpoint may carry prior state for the thread, which LangGraph handles).
- ``running``  → mark as ``error`` (mid-execution; cannot safely resume without
  knowing which node was active).

After processing runs, any threads still in ``busy`` or ``error`` state that have
no remaining active (pending/running) runs are reset to ``idle``.
"""

import asyncio
from datetime import UTC, datetime

import structlog
from sqlalchemy import select, update

from aegra_api.core.orm import Run as RunORM
from aegra_api.core.orm import Thread as ThreadORM
from aegra_api.core.orm import _get_session_maker

logger = structlog.getLogger(__name__)

_STALE_STATUSES = ("pending", "running")
_STUCK_THREAD_STATUSES = ("busy", "error")


async def recover_stale_runs() -> dict[str, int]:
    """Query and recover runs left in a non-terminal state from a previous crash.

    Returns a summary dict::

        {
            "pending_restarted": <int>,
            "running_errored": <int>,
            "threads_reset": <int>,
        }
    """
    from aegra_api.api.runs import active_runs, execute_run_async
    from aegra_api.models import User

    maker = _get_session_maker()
    stats = {"pending_restarted": 0, "running_errored": 0, "threads_reset": 0}

    async with maker() as session:
        # ------------------------------------------------------------------ #
        # 1. Find all stale runs                                              #
        # ------------------------------------------------------------------ #
        result = await session.scalars(
            select(RunORM)
            .where(RunORM.status.in_(_STALE_STATUSES))
            .order_by(RunORM.created_at.asc())
        )
        stale_runs: list[RunORM] = list(result.all())

        if not stale_runs:
            logger.info("[startup_recovery] No stale runs found.")
        else:
            logger.warning(
                f"[startup_recovery] Found {len(stale_runs)} stale run(s) from previous crash.",
                statuses={s: sum(1 for r in stale_runs if r.status == s) for s in _STALE_STATUSES},
            )

        # ------------------------------------------------------------------ #
        # 2. Handle each stale run                                            #
        # ------------------------------------------------------------------ #
        for run in stale_runs:
            if run.status == "running":
                # Cannot safely resume mid-execution — mark as error.
                logger.warning(
                    f"[startup_recovery] Marking run {run.run_id} as error "
                    f"(was 'running' at crash time)."
                )
                await session.execute(
                    update(RunORM)
                    .where(RunORM.run_id == run.run_id)
                    .values(
                        status="error",
                        error_message="Pod restarted while run was executing. Run was abandoned.",
                        updated_at=datetime.now(UTC),
                    )
                )
                stats["running_errored"] += 1

            elif run.status == "pending":
                # Run was created but never picked up — restart it.
                logger.info(
                    f"[startup_recovery] Restarting pending run {run.run_id} "
                    f"on thread {run.thread_id}."
                )
                # Reconstruct a minimal User from the persisted user_id.
                user = User(identity=run.user_id, scopes=[])

                # Retrieve the assistant's graph_id from the assistant table.
                from aegra_api.core.orm import Assistant as AssistantORM

                assistant = await session.scalar(
                    select(AssistantORM).where(
                        AssistantORM.assistant_id == run.assistant_id
                    )
                )
                if assistant is None:
                    logger.error(
                        f"[startup_recovery] Cannot restart run {run.run_id}: "
                        f"assistant {run.assistant_id} not found. Marking as error."
                    )
                    await session.execute(
                        update(RunORM)
                        .where(RunORM.run_id == run.run_id)
                        .values(
                            status="error",
                            error_message="Startup recovery: assistant not found.",
                            updated_at=datetime.now(UTC),
                        )
                    )
                    stats["running_errored"] += 1
                    continue

                task = asyncio.create_task(
                    execute_run_async(
                        run_id=run.run_id,
                        thread_id=run.thread_id,
                        graph_id=assistant.graph_id,
                        input_data=run.input or {},
                        user=user,
                        config=run.config or {},
                        context=run.context or {},
                    )
                )
                active_runs[run.run_id] = task
                stats["pending_restarted"] += 1

        await session.commit()

        # ------------------------------------------------------------------ #
        # 3. Reset threads that are stuck in busy/error with no active runs   #
        # ------------------------------------------------------------------ #
        # Re-query to find threads that are still busy/error after the above.
        stuck_threads_result = await session.scalars(
            select(ThreadORM).where(ThreadORM.status.in_(_STUCK_THREAD_STATUSES))
        )
        stuck_threads: list[ThreadORM] = list(stuck_threads_result.all())

        for thread in stuck_threads:
            # Check if there are any still-active runs on this thread.
            active_count = await session.scalar(
                select(RunORM)
                .where(
                    RunORM.thread_id == thread.thread_id,
                    RunORM.status.in_(_STALE_STATUSES),
                )
            )
            if active_count is None:
                logger.info(
                    f"[startup_recovery] Resetting stuck thread {thread.thread_id} "
                    f"from '{thread.status}' to 'idle'."
                )
                await session.execute(
                    update(ThreadORM)
                    .where(ThreadORM.thread_id == thread.thread_id)
                    .values(status="idle", updated_at=datetime.now(UTC))
                )
                stats["threads_reset"] += 1

        await session.commit()

    logger.info(f"[startup_recovery] Recovery complete: {stats}")
    return stats
