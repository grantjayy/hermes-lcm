"""Startup and session-end lock contract (grantjayy/hermes-lcm#1).

Hermes gives a plugin 120 s to load. Each SQLite write that LCM issues while
another process holds the database writer can wait the full 30 s busy timeout,
so a healthy reopen must not request the writer at all. The background FTS
integrity scan must claim without waiting and publish its result inside the
transaction that ran the check. Session-end must not shrink the busy timeout
that sibling engine clones share.

Writer contention comes from a real child process holding ``BEGIN IMMEDIATE``.
Readiness and release travel over pipes; the only time bounds are the
contract limits each test asserts.
"""

from __future__ import annotations

import importlib.util
import logging
import sqlite3
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest

from hermes_lcm import db_bootstrap
from hermes_lcm.config import LCMConfig
from hermes_lcm.dag import SummaryDAG
from hermes_lcm.engine import LCMEngine
from hermes_lcm.lifecycle_state import LifecycleStateStore
from hermes_lcm.sqlite_util import _temporary_sqlite_busy_timeout
from hermes_lcm.store import MessageStore, build_message_fts_spec

INTERVAL_ENV = "LCM_FTS_INTEGRITY_CHECK_INTERVAL_HOURS"
PLUGIN_DIR = Path(__file__).resolve().parent.parent

# Contract limits from the approved plan: a reopen or register that waited on
# the writer would exceed these, while the write-free path finishes far below.
REOPEN_LIMIT_S = 2.0
CLAIM_LIMIT_S = 0.5
SESSION_END_SKIP_LIMIT_S = 0.5
# Hang guard only, matching the join timeouts used by test_db_bootstrap_fts.py.
JOIN_GUARD_S = 30.0

WRITE_PREFIXES = ("INSERT", "UPDATE", "DELETE", "REPLACE", "DROP", "ALTER", "BEGIN IMMEDIATE")
SKIP_WARNING = "session-end flush skipped because the store is busy"

# Holds the writer until stdin closes, or for argv[2] seconds when given.
_HOLDER = r"""
import sqlite3, sys, time
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("BEGIN IMMEDIATE")
sys.stdout.write("LOCKED\n")
sys.stdout.flush()
if len(sys.argv) > 2:
    time.sleep(float(sys.argv[2]))
else:
    sys.stdin.readline()
conn.execute("ROLLBACK")
conn.close()
"""


def _start_holder(db_path: Path, hold_seconds: float | None = None) -> subprocess.Popen[str]:
    args = [sys.executable, "-c", _HOLDER, str(db_path)]
    if hold_seconds is not None:
        args.append(str(hold_seconds))
    holder = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    assert holder.stdout is not None
    ready = holder.stdout.readline().strip()
    if ready != "LOCKED":
        holder.kill()
        holder.wait()
        pytest.fail(f"writer holder did not acquire BEGIN IMMEDIATE (got {ready!r})")
    return holder


def _stop_holder(holder: subprocess.Popen[str]) -> None:
    assert holder.stdin is not None and holder.stdout is not None
    holder.stdin.close()
    holder.wait(timeout=JOIN_GUARD_S)
    holder.stdout.close()


@contextmanager
def _held_writer(db_path: Path) -> Iterator[None]:
    holder = _start_holder(db_path)
    try:
        yield
    finally:
        _stop_holder(holder)


def _call_while_writer_held(
    db_path: Path, fn: Callable[[], Any], limit_s: float
) -> dict[str, Any]:
    """Run ``fn`` on a worker thread while another process holds the writer.

    The writer is released once ``fn`` returns or ``limit_s`` passes, so a call
    that waits on the writer finishes (and can be inspected) instead of
    running into SQLite's 30 s busy timeout.
    """
    outcome: dict[str, Any] = {}

    def run() -> None:
        outcome["start"] = time.monotonic()
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # surfaced to the test below
            outcome["error"] = exc
        outcome["end"] = time.monotonic()

    worker = threading.Thread(target=run, name="lcm-startup-contract-call")
    with _held_writer(db_path):
        worker.start()
        worker.join(limit_s)
        outcome["returned_within_limit"] = not worker.is_alive()
    worker.join(JOIN_GUARD_S)
    assert not worker.is_alive(), "call still running after the writer was released"
    if "error" in outcome:
        raise outcome["error"]
    outcome["elapsed"] = outcome["end"] - outcome["start"]
    return outcome


def _bootstrap(db_path: Path) -> None:
    """First open, without contention: creates schema and one-time markers."""
    store = MessageStore(db_path)
    store.append("bootstrap-session", {"role": "user", "content": "bootstrap row"})
    store.close()
    SummaryDAG(db_path).close()
    LifecycleStateStore(db_path).close()


@pytest.mark.parametrize(
    "store_cls",
    [MessageStore, SummaryDAG, LifecycleStateStore],
    ids=["MessageStore", "SummaryDAG", "LifecycleStateStore"],
)
def test_healthy_reopen_issues_no_writes_under_held_writer(tmp_path, monkeypatch, store_cls):
    monkeypatch.setenv(INTERVAL_ENV, "-1")
    db_path = tmp_path / "lcm.db"
    _bootstrap(db_path)

    statements: list[str] = []
    real_connect = sqlite3.connect

    def tracing_connect(*args: Any, **kwargs: Any) -> sqlite3.Connection:
        conn = real_connect(*args, **kwargs)
        conn.set_trace_callback(statements.append)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracing_connect)
    outcome = _call_while_writer_held(db_path, lambda: store_cls(db_path), REOPEN_LIMIT_S)
    outcome["value"].close()

    writes = [
        " ".join(sql.split())
        for sql in statements
        if " ".join(sql.split()).upper().startswith(WRITE_PREFIXES)
    ]
    assert writes == [], f"healthy {store_cls.__name__} reopen still requests the writer: {writes}"
    assert outcome["returned_within_limit"], (
        f"{store_cls.__name__} reopen waited on the held writer for more than {REOPEN_LIMIT_S} s"
    )
    assert outcome["elapsed"] < REOPEN_LIMIT_S


class _StubCtx:
    """Minimal plugin context, after repro_register_lock.py ``StubCtx``."""

    def __init__(self) -> None:
        self.engine: Any = None
        self.engine_registered_at: float | None = None

    def register_context_engine(self, engine: Any) -> None:
        self.engine = engine
        self.engine_registered_at = time.monotonic()

    def __getattr__(self, name: str) -> Callable[..., None]:
        return lambda *args, **kwargs: None


@pytest.mark.parametrize("interval", ["-1", "0"], ids=["scan-disabled", "scan-due"])
def test_register_returns_fast_under_held_writer(tmp_path, monkeypatch, interval):
    home = tmp_path / "hermes_home"
    home.mkdir()
    db_path = home / "lcm.db"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    monkeypatch.setenv(INTERVAL_ENV, "-1")
    _bootstrap(db_path)
    monkeypatch.setenv(INTERVAL_ENV, interval)

    module_name = f"hermes_lcm_startup_lock_register_{'due' if interval == '0' else 'off'}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(PLUGIN_DIR / "__init__.py"),
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    ctx = _StubCtx()
    try:
        outcome = _call_while_writer_held(db_path, lambda: module.register(ctx), REOPEN_LIMIT_S)
    finally:
        if ctx.engine is not None:
            ctx.engine.shutdown()
        sys.modules[f"{module_name}.db_bootstrap"].join_background_integrity_scans(JOIN_GUARD_S)

    assert ctx.engine is not None
    assert ctx.engine_registered_at is not None
    registered_after = ctx.engine_registered_at - outcome["start"]
    assert registered_after < REOPEN_LIMIT_S, (
        f"register() waited {registered_after:.2f} s on the held writer before registering the engine"
    )


def test_dispatch_claim_skips_when_writer_busy(tmp_path, monkeypatch):
    monkeypatch.setenv(INTERVAL_ENV, "-1")
    db_path = tmp_path / "lcm.db"
    _bootstrap(db_path)
    spec = build_message_fts_spec()
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    key = (db_bootstrap._database_path_for_connection(conn), spec.table_name)

    def dispatch() -> tuple[bool, bool]:
        dispatched = db_bootstrap._dispatch_background_integrity_scan(conn, spec)
        return dispatched, key in db_bootstrap._integrity_scan_threads

    try:
        outcome = _call_while_writer_held(db_path, dispatch, CLAIM_LIMIT_S)
    finally:
        db_bootstrap.join_background_integrity_scans(JOIN_GUARD_S)
        conn.close()

    assert outcome["returned_within_limit"], (
        f"scan claim waited on the held writer for more than {CLAIM_LIMIT_S} s"
    )
    assert outcome["elapsed"] < CLAIM_LIMIT_S
    dispatched, thread_registered = outcome["value"]
    assert dispatched is True
    assert thread_registered is False


def _metadata_value(db_path: Path, key: str) -> str | None:
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def _independent_writer_result(db_path: Path) -> str:
    probe = sqlite3.connect(str(db_path), timeout=0, isolation_level=None)
    try:
        probe.execute("BEGIN IMMEDIATE")
    except sqlite3.OperationalError as exc:
        return str(getattr(exc, "sqlite_errorname", exc))
    else:
        probe.execute("ROLLBACK")
        return "acquired"
    finally:
        probe.close()


def test_scan_result_is_published_in_the_scan_transaction(tmp_path, monkeypatch):
    monkeypatch.setenv(INTERVAL_ENV, "-1")
    monkeypatch.delenv("LCM_FTS_INTEGRITY_BACKGROUND", raising=False)
    db_path = tmp_path / "lcm.db"
    _bootstrap(db_path)
    spec = build_message_fts_spec()
    marker_key = f"fts_integrity_checked_at:{spec.table_name}"
    started_key = f"fts_integrity_scan_started_at:{spec.table_name}"
    aged = str(time.time() - 100 * 3600)
    conn = sqlite3.connect(str(db_path))
    conn.execute("UPDATE metadata SET value = ? WHERE key = ?", (aged, marker_key))
    conn.commit()

    scan_connections: list[sqlite3.Connection] = []
    publications: list[dict[str, Any]] = []
    real_check = db_bootstrap.check_external_content_fts_integrity
    real_record = db_bootstrap._record_integrity_checked

    def capturing_check(conn_: sqlite3.Connection, spec_: Any) -> dict[str, str]:
        scan_connections.append(conn_)
        return real_check(conn_, spec_)

    def observing_record(conn_: sqlite3.Connection, spec_: Any, *, now: float | None = None) -> None:
        # Asserting here would be swallowed by the scan thread's exception
        # handler, so record what publication saw and assert after the join.
        publications.append(
            {
                "same_connection": bool(scan_connections) and conn_ is scan_connections[-1],
                "in_transaction": conn_.in_transaction,
                "independent_writer": _independent_writer_result(db_path),
            }
        )
        real_record(conn_, spec_, now=now)

    monkeypatch.setattr(db_bootstrap, "check_external_content_fts_integrity", capturing_check)
    monkeypatch.setattr(db_bootstrap, "_record_integrity_checked", observing_record)
    monkeypatch.setenv(INTERVAL_ENV, "0")
    try:
        db_bootstrap.ensure_external_content_fts(conn, spec)
        db_bootstrap.join_background_integrity_scans(JOIN_GUARD_S)
    finally:
        conn.close()

    assert publications == [
        {"same_connection": True, "in_transaction": True, "independent_writer": "SQLITE_BUSY"}
    ]
    marker = _metadata_value(db_path, marker_key)
    assert marker is not None and float(marker) > float(aged)
    assert _metadata_value(db_path, started_key) is None
    assert _independent_writer_result(db_path) == "acquired"


def _engine_family(tmp_path: Path) -> tuple[LCMEngine, LCMEngine, LCMEngine]:
    root = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "lcm.db")))
    clone_a = root.clone_for_agent()
    clone_b = root.clone_for_agent()
    clone_a.on_session_start("session-a", platform="cli")
    clone_b.on_session_start("session-b", platform="cli")
    return root, clone_a, clone_b


def _shutdown(*engines: LCMEngine) -> None:
    for engine in engines:
        engine.shutdown()


def _busy_timeouts(engine: LCMEngine) -> tuple[int, int]:
    return (
        engine._store._conn.execute("PRAGMA busy_timeout").fetchone()[0],
        engine._lifecycle._conn.execute("PRAGMA busy_timeout").fetchone()[0],
    )


def _pause_session_end_flush(monkeypatch, engine: LCMEngine) -> tuple[threading.Event, threading.Event]:
    """Make ``engine``'s session-end flush wait inside its timeout scope."""
    entered = threading.Event()
    release = threading.Event()

    def paused_ingest(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entered.set()
        release.wait(JOIN_GUARD_S)
        return messages

    monkeypatch.setattr(engine, "_ingest_messages", paused_ingest)
    return entered, release


MESSAGES = [
    {"role": "user", "content": "first"},
    {"role": "assistant", "content": "second"},
    {"role": "user", "content": "third"},
]


def test_session_end_does_not_shorten_sibling_ingest(tmp_path, monkeypatch):
    monkeypatch.setenv(INTERVAL_ENV, "-1")
    root, clone_a, clone_b = _engine_family(tmp_path)
    entered, release = _pause_session_end_flush(monkeypatch, clone_a)
    session_end = threading.Thread(target=clone_a.on_session_end, args=("session-a", MESSAGES))
    sibling_errors: list[BaseException] = []

    def sibling_ingest() -> None:
        try:
            clone_b._ingest_messages(MESSAGES)
        except BaseException as exc:  # asserted below
            sibling_errors.append(exc)

    sibling = threading.Thread(target=sibling_ingest)
    try:
        session_end.start()
        assert entered.wait(JOIN_GUARD_S)
        # The plan's one-second external hold: the sibling's write lands
        # inside it while clone A sits in its session-end timeout scope.
        holder = _start_holder(tmp_path / "lcm.db", hold_seconds=1.0)
        sibling.start()
        holder.wait(timeout=JOIN_GUARD_S)
        assert holder.stdout is not None and holder.stdin is not None
        holder.stdout.close()
        holder.stdin.close()
        release.set()
        session_end.join(JOIN_GUARD_S)
        sibling.join(JOIN_GUARD_S)
        assert not sibling.is_alive()
        assert sibling_errors == []
        assert clone_b._store.get_session_count("session-b") == len(MESSAGES)
    finally:
        release.set()
        session_end.join(JOIN_GUARD_S)
        sibling.join(JOIN_GUARD_S)
        _shutdown(clone_a, clone_b, root)


def test_session_end_skips_when_store_busy(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(INTERVAL_ENV, "-1")
    root, clone_a, clone_b = _engine_family(tmp_path)
    store_lock_held = threading.Event()
    release_store = threading.Event()

    def hold_store_lock() -> None:
        with clone_a._store._write_lock:
            store_lock_held.set()
            release_store.wait(JOIN_GUARD_S)

    lock_holder = threading.Thread(target=hold_store_lock)
    session_end = threading.Thread(target=clone_a.on_session_end, args=("session-a", MESSAGES))
    try:
        lock_holder.start()
        assert store_lock_held.wait(JOIN_GUARD_S)
        with caplog.at_level(logging.WARNING):
            started = time.monotonic()
            session_end.start()
            session_end.join(SESSION_END_SKIP_LIMIT_S)
            returned = not session_end.is_alive()
            elapsed = time.monotonic() - started
            release_store.set()
            lock_holder.join(JOIN_GUARD_S)
            session_end.join(JOIN_GUARD_S)

        assert returned, f"on_session_end waited {elapsed:.2f} s for the busy store"
        assert clone_a._store.get_session_count("session-a") == 0
        skips = [record for record in caplog.records if SKIP_WARNING in record.getMessage()]
        assert len(skips) == 1
        assert skips[0].levelno == logging.WARNING
    finally:
        release_store.set()
        lock_holder.join(JOIN_GUARD_S)
        session_end.join(JOIN_GUARD_S)
        _shutdown(clone_a, clone_b, root)


def test_overlapping_session_end_hooks_restore_timeouts(tmp_path, monkeypatch):
    monkeypatch.setenv(INTERVAL_ENV, "-1")
    root, clone_a, clone_b = _engine_family(tmp_path)
    # Distinct originals prove each connection gets its own value back.
    clone_a._store._conn.execute("PRAGMA busy_timeout=31000")
    clone_a._lifecycle._conn.execute("PRAGMA busy_timeout=32000")
    entered, release = _pause_session_end_flush(monkeypatch, clone_a)
    b_flushes: list[int] = []

    def recording_ingest(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        b_flushes.append(len(messages))
        return messages

    monkeypatch.setattr(clone_b, "_ingest_messages", recording_ingest)
    session_end_a = threading.Thread(target=clone_a.on_session_end, args=("session-a", MESSAGES))
    session_end_b = threading.Thread(target=clone_b.on_session_end, args=("session-b", MESSAGES))
    try:
        session_end_a.start()
        assert entered.wait(JOIN_GUARD_S)
        during_a = _busy_timeouts(clone_a)
        session_end_b.start()
        session_end_b.join(JOIN_GUARD_S)
        assert not session_end_b.is_alive()
        after_b = _busy_timeouts(clone_a)
        release.set()
        session_end_a.join(JOIN_GUARD_S)
        assert not session_end_a.is_alive()

        assert b_flushes == [], "clone B flushed instead of skipping while clone A held the session-end scope"
        assert after_b == during_a
        assert _busy_timeouts(clone_a) == (31000, 32000)
        assert _busy_timeouts(clone_b) == (31000, 32000)
    finally:
        release.set()
        session_end_a.join(JOIN_GUARD_S)
        session_end_b.join(JOIN_GUARD_S)
        _shutdown(clone_a, clone_b, root)


class _RefusesBusyTimeoutChange(sqlite3.Connection):
    def execute(self, sql: str, *args: Any) -> sqlite3.Cursor:
        if sql.startswith("PRAGMA busy_timeout="):
            raise sqlite3.OperationalError("injected busy_timeout failure")
        return super().execute(sql, *args)


def test_timeout_scope_restores_first_connection_when_second_fails(tmp_path):
    first = sqlite3.connect(str(tmp_path / "first.db"))
    second = sqlite3.connect(str(tmp_path / "second.db"), factory=_RefusesBusyTimeoutChange)
    try:
        first.execute("PRAGMA busy_timeout=31000")
        with pytest.raises(sqlite3.OperationalError, match="injected busy_timeout failure"):
            with _temporary_sqlite_busy_timeout([first, second], 50):
                pytest.fail("the scope body must not run when a timeout cannot be set")
        assert first.execute("PRAGMA busy_timeout").fetchone()[0] == 31000
    finally:
        first.close()
        second.close()
