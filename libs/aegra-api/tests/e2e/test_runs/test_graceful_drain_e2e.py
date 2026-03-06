"""End-to-end tests for graceful-drain (SIGTERM) behaviour.

These tests spawn the Aegra API server as a subprocess, issue real HTTP
requests, send SIGTERM, and then verify the database state — exactly what
happens in production when k8s terminates a pod.

Postgres must already be running (started by ``aegra dev`` or
``docker compose up postgres -d``).  The test binds to a dedicated port
(18765) so it does not conflict with a concurrently running dev server.

Run with::

    uv run python -m pytest tests/e2e/test_runs/test_graceful_drain_e2e.py -v -s
"""

import asyncio
import os
import signal
import subprocess
import sys
import time

import httpx
import pytest
import sqlalchemy as sa
from langgraph_sdk import get_client

from tests.e2e._utils import elog, get_sync_db_engine

# Dedicated port for this test so it never clashes with the dev server
_TEST_PORT = 18765
_TEST_URL = f"http://127.0.0.1:{_TEST_PORT}"
_STARTUP_TIMEOUT = 30  # seconds to wait for the server to become ready
_SIGTERM_DRAIN_TIMEOUT = 30  # seconds to wait for clean exit after SIGTERM


def _load_dotenv_file(path: str, env: dict) -> None:
    """Parse a .env file and set missing keys into env (does not override existing vars)."""
    if not os.path.isfile(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            env.setdefault(key, value)


def _repo_root() -> str:
    """Absolute path to the aegra repo root (5 levels up from dirname of this file)."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))


def _server_env() -> dict:
    """Build subprocess env: inherit os.environ, overlay .env then .env.local, set port."""
    env = os.environ.copy()
    root = _repo_root()
    _load_dotenv_file(os.path.join(root, ".env"), env)
    _load_dotenv_file(os.path.join(root, ".env.local"), env)
    env["SERVER_URL"] = _TEST_URL
    env["AEGRA_CONFIG"] = os.path.join(_repo_root(), "aegra.test.json")
    return env


def _load_env_into_process() -> None:
    """Load .env/.env.local and set AEGRA_DATABASE_URL so get_sync_db_engine bypasses the frozen settings singleton."""
    env: dict = {}
    root = _repo_root()
    _load_dotenv_file(os.path.join(root, ".env"), env)
    _load_dotenv_file(os.path.join(root, ".env.local"), env)
    for k, v in env.items():
        os.environ.setdefault(k, v)
    # Build and export the DB URL so get_sync_db_engine picks it up via os.environ
    user = env.get("POSTGRES_USER", "postgres")
    password = env.get("POSTGRES_PASSWORD", "postgres")
    host = env.get("POSTGRES_HOST", "localhost")
    port = env.get("POSTGRES_PORT", "5432")
    db = env.get("POSTGRES_DB", "aegra")
    os.environ["AEGRA_DATABASE_URL"] = f"postgresql+psycopg://{user}:{password}@{host}:{port}/{db}"


_load_env_into_process()


def _start_server() -> subprocess.Popen:
    """Spawn a uvicorn process on _TEST_PORT and return the Popen handle."""
    cmd = [
        sys.executable, "-m", "uvicorn",
        "aegra_api.main:app",
        "--host", "127.0.0.1",
        "--port", str(_TEST_PORT),
        "--no-access-log",
    ]
    return subprocess.Popen(
        cmd,
        cwd=_repo_root(),
        env=_server_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _wait_for_ready(timeout: int = _STARTUP_TIMEOUT) -> None:
    """Poll /health until the server responds or timeout is reached."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{_TEST_URL}/health", timeout=1.0)
            if r.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Server on {_TEST_URL} did not become ready within {timeout}s")


def _drain_server_output(proc: subprocess.Popen) -> str:
    """Read all remaining stdout/stderr from a finished process."""
    try:
        stdout, _ = proc.communicate(timeout=5)
        return stdout.decode(errors="replace") if stdout else ""
    except Exception:
        return ""


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sigterm_rejects_new_runs_with_503():
    """POST /threads/.../runs must return 503 immediately after SIGTERM.

    Flow:
    1. Start a fresh server subprocess.
    2. Send SIGTERM.
    3. In the brief window before the process exits, POST a new run.
    4. Assert 503 with the drain detail message.
    """
    proc = _start_server()
    try:
        _wait_for_ready()

        client = get_client(url=_TEST_URL)
        assistant = await client.assistants.create(
            graph_id="sleep_agent",
            config={"tags": ["drain-503"]},
            if_exists="do_nothing",
        )
        thread = await client.threads.create()
        thread_id = thread["thread_id"]

        # SIGTERM — server starts draining
        proc.send_signal(signal.SIGTERM)

        # Retry for up to 3 s until we get a 503 (the signal propagates async)
        response = None
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                async with httpx.AsyncClient(timeout=2.0) as http:
                    response = await http.post(
                        f"{_TEST_URL}/threads/{thread_id}/runs",
                        json={
                            "assistant_id": assistant["assistant_id"],
                            "input": {"messages": [{"role": "user", "content": "Hello"}]},
                        },
                    )
                if response.status_code == 503:
                    break
            except Exception:
                pass
            await asyncio.sleep(0.1)

        elog("503 response", {"status": getattr(response, "status_code", None)})

        assert response is not None, "Never got a response after SIGTERM"
        assert response.status_code == 503, (
            f"Expected 503, got {response.status_code}. "
            "Server may have already shut down before the request arrived."
        )
        body = response.json()
        detail = (body.get("detail") or body.get("message") or "").lower()
        assert "shutting down" in detail, f"Expected 'shutting down' in detail, got: {body}"
    finally:
        if proc.returncode is None:
            proc.kill()
        _drain_server_output(proc)


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sigterm_requeues_active_run_as_pending():
    """In-flight run is re-queued as 'pending' and thread stays 'busy' on graceful drain.

    Flow:
    1. Start a fresh server subprocess.
    2. Create an assistant + thread, start a long-running background run.
    3. Wait for the run to reach 'running' status.
    4. Send SIGTERM to the server process.
    5. Wait for the process to exit cleanly (graceful drain).
    6. Query the database directly — run must be 'pending', thread must be 'busy'.
    7. Start the server again and verify startup recovery logs + run restarts.
    """
    proc = _start_server()
    run_id = None
    thread_id = None

    try:
        _wait_for_ready()
        elog("Server started", {"pid": proc.pid, "url": _TEST_URL})

        client = get_client(url=_TEST_URL)
        assistant = await client.assistants.create(
            graph_id="sleep_agent",
            config={"tags": ["drain-requeue"]},
            if_exists="do_nothing",
        )
        assistant_id = assistant["assistant_id"]

        thread = await client.threads.create()
        thread_id = thread["thread_id"]
        elog("Thread", {"thread_id": thread_id})

        run = await client.runs.create(
            thread_id=thread_id,
            assistant_id=assistant_id,
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Write a very detailed 800-word essay about the history of databases, "
                            "covering relational, NoSQL, and NewSQL systems with specific dates."
                        ),
                    }
                ]
            },
        )
        run_id = run["run_id"]
        elog("Run created", {"run_id": run_id})

        # Wait for run to be 'running'
        for _ in range(40):
            await asyncio.sleep(0.5)
            current = await client.runs.get(thread_id, run_id)
            if current["status"] == "running":
                break
        else:
            pytest.skip("Run did not reach 'running' within 20s — skipping")

        elog("Run is running — sending SIGTERM", {"run_id": run_id})

        # SIGTERM — triggers _handle_sigterm() in the server
        proc.send_signal(signal.SIGTERM)

        # Wait for the process to exit (graceful drain)
        try:
            proc.wait(timeout=_SIGTERM_DRAIN_TIMEOUT)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            pytest.fail(f"Server did not exit within {_SIGTERM_DRAIN_TIMEOUT}s after SIGTERM")

        elog("Server exited", {"returncode": proc.returncode})
        server_output = _drain_server_output(proc)
        elog("Server output (last 2000 chars)", server_output[-2000:])

        # Verify drain log lines appeared
        assert "SIGTERM received" in server_output, (
            "Expected SIGTERM drain log in server output. "
            f"Got: {server_output[-500:]}"
        )
        assert "re-queued as pending" in server_output, (
            "Expected 're-queued as pending' log in server output. "
            f"Got: {server_output[-500:]}"
        )

    finally:
        if proc.returncode is None:
            proc.kill()
            _drain_server_output(proc)

    # ------------------------------------------------------------------ #
    # Phase 2: verify DB state directly (server is down)                  #
    # ------------------------------------------------------------------ #
    assert run_id is not None and thread_id is not None

    with get_sync_db_engine() as engine, engine.connect() as conn:
        run_row = conn.execute(
            sa.text("SELECT status FROM runs WHERE run_id = :run_id"),
            {"run_id": run_id},
        ).fetchone()
        thread_row = conn.execute(
            sa.text("SELECT status FROM thread WHERE thread_id = :thread_id"),
            {"thread_id": thread_id},
        ).fetchone()

    elog("DB run status", {"run_id": run_id, "status": run_row[0] if run_row else None})
    elog("DB thread status", {"thread_id": thread_id, "status": thread_row[0] if thread_row else None})

    assert run_row is not None, f"Run {run_id} not found in DB"
    assert run_row[0] == "pending", (
        f"Expected run status 'pending' after SIGTERM drain, got '{run_row[0]}'"
    )
    assert thread_row is not None, f"Thread {thread_id} not found in DB"
    assert thread_row[0] == "busy", (
        f"Expected thread status 'busy' after SIGTERM drain, got '{thread_row[0]}'"
    )

    # ------------------------------------------------------------------ #
    # Phase 3: restart server — startup recovery must pick up the run     #
    # ------------------------------------------------------------------ #
    proc2 = _start_server()
    try:
        _wait_for_ready()
        elog("Server restarted", {"pid": proc2.pid})

        # Give startup recovery time to run and restart the run
        await asyncio.sleep(5.0)

        client2 = get_client(url=_TEST_URL)

        # Poll for the run to be restarted (running or success)
        for _ in range(30):
            await asyncio.sleep(1.0)
            restarted = await client2.runs.get(thread_id, run_id)
            elog("Run status after restart", {"status": restarted["status"]})
            if restarted["status"] in ("running", "success", "interrupted"):
                break
        else:
            pytest.fail(
                f"Run {run_id} was not restarted by startup recovery within 30s. "
                f"Final status: {restarted['status']}"
            )

        assert restarted["status"] in ("running", "success", "interrupted"), (
            f"Expected run to be restarted, got status '{restarted['status']}'"
        )
        elog("✅ Run successfully restarted by startup recovery", restarted)

    finally:
        if proc2.returncode is None:
            proc2.send_signal(signal.SIGTERM)
            try:
                proc2.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc2.kill()
        _drain_server_output(proc2)
