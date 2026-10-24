"""Tests for B2 compaction telemetry (per-conversation snapshot in metadata)."""

import json
import sqlite3
import threading
import time

import pytest

from hermes_lcm import command as lcm_command
from hermes_lcm import db_bootstrap, sqlite_util
from hermes_lcm import engine as engine_module
from hermes_lcm import tools as lcm_tools
from hermes_lcm.command import handle_lcm_command
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.store import MessageStore


@pytest.fixture
def engine(tmp_path):
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_test.db")
    e = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    e._session_id = "test-session"
    e._conversation_id = "conv-1"
    e.update_model("gpt-test", 200000, provider="openai-codex", api_mode="responses")
    return e


def _hot_usage(prompt=1000, read=400, write=50):
    return {
        "prompt_tokens": prompt,
        "completion_tokens": 100,
        "total_tokens": prompt + 100,
        "input_tokens": prompt - read - write,
        "cache_read_tokens": read,
        "cache_write_tokens": write,
    }


def _cold_usage(prompt=1000):
    # Cache keys present (so cache_metrics_available) but both zero -> cold.
    return {"prompt_tokens": prompt, "cache_read_tokens": 0, "cache_write_tokens": 0}


def _telemetry(engine):
    return engine.get_status().get("compaction_telemetry")


def _slash_status_fields(engine):
    return {
        key.strip(): value.strip()
        for line in handle_lcm_command("status", engine).splitlines()
        if ":" in line
        for key, value in [line.split(":", 1)]
    }


def _assert_total_compaction_surfaces(engine, expected):
    status = engine.get_status()
    tool_status = json.loads(lcm_tools.lcm_status({}, engine=engine))
    slash_status = _slash_status_fields(engine)

    assert status["total_compactions"] == expected
    assert status["total_compactions_scope"] == "current_conversation"
    assert tool_status["total_compactions"] == expected
    assert tool_status["total_compactions_scope"] == "current_conversation"
    assert slash_status["total_compactions"] == str(expected)
    assert slash_status["total_compactions_scope"] == "current_conversation"


def test_records_per_turn_snapshot(engine):
    engine.update_from_response(_hot_usage(prompt=1050, read=400, write=50))
    t = _telemetry(engine)
    assert t is not None
    assert t["cache_state"] == "hot"
    assert t["consecutive_cold_observations"] == 0
    assert t["turns_since_leaf_compaction"] == 1
    assert t["last_observed_prompt_tokens"] == 1050
    assert t["last_observed_cache_read"] == 400
    assert t["last_observed_cache_write"] == 50
    assert t["peak_prompt_tokens_since_leaf_compaction"] == 1050
    assert t["provider"] == "openai-codex"
    assert t["model"] == "gpt-test"
    assert t["activity_band"] == "low"


def test_turns_since_increments_and_peak_is_max(engine):
    engine.update_from_response(_hot_usage(prompt=1000))
    engine.update_from_response(_hot_usage(prompt=600))
    t = _telemetry(engine)
    assert t["turns_since_leaf_compaction"] == 2
    assert t["peak_prompt_tokens_since_leaf_compaction"] == 1000  # max across turns


def test_cold_streak_counts_and_hot_resets(engine):
    engine.update_from_response(_cold_usage())
    engine.update_from_response(_cold_usage())
    assert _telemetry(engine)["cache_state"] == "cold"
    assert _telemetry(engine)["consecutive_cold_observations"] == 2
    engine.update_from_response(_hot_usage())
    assert _telemetry(engine)["cache_state"] == "hot"
    assert _telemetry(engine)["consecutive_cold_observations"] == 0


def test_idle_turn_is_skipped(engine):
    engine.update_from_response(_hot_usage())
    before = _telemetry(engine)["turns_since_leaf_compaction"]
    # No prompt tokens and no cache keys at all -> no signal -> no write.
    engine.update_from_response({"completion_tokens": 5})
    assert _telemetry(engine)["turns_since_leaf_compaction"] == before


def test_resets_on_compaction(engine):
    engine.update_from_response(_hot_usage(prompt=1000))
    engine.update_from_response(_hot_usage(prompt=1000))
    assert _telemetry(engine)["turns_since_leaf_compaction"] == 2

    # Simulate a leaf compaction happening between turns.
    engine.compression_count += 1
    engine._last_compaction_duration_ms = 12.5
    engine.update_from_response(_hot_usage(prompt=400))

    t = _telemetry(engine)
    assert t["turns_since_leaf_compaction"] == 0
    assert t["total_compactions"] == 1
    assert t["last_leaf_compaction_at"] is not None
    assert t["last_compaction_duration_ms"] == 12.5
    assert t["peak_prompt_tokens_since_leaf_compaction"] == 400  # reset to current


def test_total_compactions_status_includes_pending_compaction(engine):
    engine.compression_count = 3
    engine.update_from_response(_hot_usage())
    _assert_total_compaction_surfaces(engine, 3)

    # Status can be requested after compaction increments the live counter but
    # before the response hook persists the next telemetry snapshot.
    engine.compression_count += 1

    _assert_total_compaction_surfaces(engine, 4)


def test_total_compactions_status_survives_session_rollover_for_conversation(engine):
    engine.compression_count = 3
    engine.update_from_response(_hot_usage())
    _assert_total_compaction_surfaces(engine, 3)

    engine.on_session_start(
        "test-session-next",
        platform="discord",
        conversation_id="conv-1",
    )

    assert engine.compression_count == 0
    _assert_total_compaction_surfaces(engine, 3)


@pytest.mark.parametrize("previous_session_count", [1, 3])
def test_total_compactions_includes_first_compaction_after_session_rollover(
    engine,
    previous_session_count,
):
    engine.compression_count = previous_session_count
    engine.update_from_response(_hot_usage())

    engine.on_session_start(
        "test-session-next",
        platform="discord",
        conversation_id="conv-1",
    )
    assert engine.compression_count == 0

    # The first response telemetry can arrive after the new session has already
    # compacted. Count it even when the new session's counter is equal to or
    # below the persisted baseline from the previous session.
    engine.compression_count = 1
    engine.update_from_response(_hot_usage())

    _assert_total_compaction_surfaces(engine, previous_session_count + 1)


def test_total_compactions_includes_first_compaction_after_engine_restart(tmp_path):
    config = LCMConfig()
    config.database_path = str(tmp_path / "lcm_test.db")

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    first.on_session_start(
        "test-session",
        platform="discord",
        conversation_id="conv-1",
    )
    first.compression_count = 3
    first.update_from_response(_hot_usage())
    first.shutdown()

    restarted = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    try:
        restarted.on_session_start(
            "test-session-next",
            platform="discord",
            conversation_id="conv-1",
        )
        restarted.compression_count = 1
        restarted.update_from_response(_hot_usage())

        _assert_total_compaction_surfaces(restarted, 4)
    finally:
        restarted.shutdown()


def test_total_compactions_rebaseline_when_session_id_binds_new_conversation(engine):
    engine.compression_count = 3
    engine.update_from_response(_hot_usage())
    _assert_total_compaction_surfaces(engine, 3)

    engine.on_session_start(
        "test-session",
        platform="discord",
        conversation_id="conv-2",
    )

    assert engine.compression_count == 0
    _assert_total_compaction_surfaces(engine, 0)
    assert engine._store.read_compaction_telemetry("conv-1")["total_compactions"] == 3

    engine.compression_count = 1
    engine.update_from_response(_hot_usage())
    _assert_total_compaction_surfaces(engine, 1)


@pytest.mark.parametrize("record_mode", ["successful_compaction", "response_hook"])
def test_compactions_increment_total_atomically_across_engines(tmp_path, record_mode):
    config = LCMConfig(database_path=str(tmp_path / "lcm_test.db"))
    engines = [
        LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home_a")),
        LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home_b")),
    ]
    barrier = threading.Barrier(2, timeout=5)
    errors = []

    try:
        for index, current in enumerate(engines):
            current.on_session_start(
                f"test-session-{index}",
                platform="cli",
                conversation_id="conv-1",
            )
            current.compression_count = 1
            original_increment = current._store.increment_compaction_telemetry

            def paused_increment(
                conversation_id,
                increment,
                updates,
                *,
                _increment=original_increment,
            ):
                barrier.wait()
                return _increment(conversation_id, increment, updates)

            current._store.increment_compaction_telemetry = paused_increment

        def record(current):
            try:
                if record_mode == "successful_compaction":
                    current._record_successful_compaction_telemetry()
                else:
                    current.update_from_response(_hot_usage())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=record, args=(current,)) for current in engines]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not [thread for thread in threads if thread.is_alive()]
        assert errors == []
        telemetry = engines[0]._store.read_compaction_telemetry("conv-1")
        assert telemetry["total_compactions"] == 2
    finally:
        for current in engines:
            current.shutdown()


def test_zero_delta_snapshot_cannot_overwrite_concurrent_increment(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm_test.db"))
    engines = [
        LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home_a")),
        LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home_b")),
    ]
    barrier = threading.Barrier(2, timeout=5)
    errors = []

    try:
        for index, current in enumerate(engines):
            current.on_session_start(
                f"test-session-{index}",
                platform="cli",
                conversation_id="conv-1",
            )
            original_increment = current._store.increment_compaction_telemetry

            def paused_increment(
                conversation_id,
                increment,
                updates,
                *,
                _increment=original_increment,
            ):
                barrier.wait()
                return _increment(conversation_id, increment, updates)

            current._store.increment_compaction_telemetry = paused_increment

        engines[1].compression_count = 1

        def record_snapshot():
            try:
                engines[0].update_from_response(_hot_usage())
            except Exception as exc:
                errors.append(exc)

        def record_increment():
            try:
                engines[1]._record_successful_compaction_telemetry()
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=record_snapshot),
            threading.Thread(target=record_increment),
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        assert not [thread for thread in threads if thread.is_alive()]
        assert errors == []
        telemetry = engines[0]._store.read_compaction_telemetry("conv-1")
        assert telemetry["total_compactions"] == 1
    finally:
        for current in engines:
            current.shutdown()


def test_zero_delta_snapshot_preserves_concurrent_compaction_metadata(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm_test.db"))
    snapshot_engine = LCMEngine(
        config=config,
        hermes_home=str(tmp_path / "hermes_home_snapshot"),
    )
    compaction_engine = LCMEngine(
        config=config,
        hermes_home=str(tmp_path / "hermes_home_compaction"),
    )
    snapshot_ready = threading.Event()
    compaction_done = threading.Event()
    errors = []

    try:
        snapshot_engine.on_session_start(
            "snapshot-session",
            platform="cli",
            conversation_id="conv-1",
        )
        compaction_engine.on_session_start(
            "compaction-session",
            platform="cli",
            conversation_id="conv-1",
        )
        original_increment = snapshot_engine._store.increment_compaction_telemetry

        def paused_snapshot(conversation_id, increment, updates):
            snapshot_ready.set()
            assert compaction_done.wait(timeout=5)
            return original_increment(conversation_id, increment, updates)

        snapshot_engine._store.increment_compaction_telemetry = paused_snapshot

        def record_snapshot():
            try:
                snapshot_engine.update_from_response(_hot_usage(prompt=1000))
            except Exception as exc:
                errors.append(exc)

        snapshot_thread = threading.Thread(target=record_snapshot)
        snapshot_thread.start()
        assert snapshot_ready.wait(timeout=5)

        compaction_engine.compression_count = 1
        compaction_engine._last_compaction_duration_ms = 12.5
        compaction_engine._record_successful_compaction_telemetry()
        compaction_done.set()
        snapshot_thread.join(timeout=10)

        assert not snapshot_thread.is_alive()
        assert errors == []
        telemetry = snapshot_engine._store.read_compaction_telemetry("conv-1")
        assert telemetry["total_compactions"] == 1
        assert telemetry["turns_since_leaf_compaction"] == 0
        assert telemetry["peak_prompt_tokens_since_leaf_compaction"] == 0
        assert telemetry["last_leaf_compaction_at"] is not None
        assert telemetry["last_compaction_duration_ms"] == 12.5
    finally:
        compaction_done.set()
        snapshot_engine.shutdown()
        compaction_engine.shutdown()


def test_compaction_telemetry_lock_contention_is_nonblocking(tmp_path):
    db_path = tmp_path / "lcm_test.db"
    lock_owner = MessageStore(db_path)
    telemetry_store = MessageStore(db_path)
    lock_ready = threading.Event()
    release_lock = threading.Event()
    write_done = threading.Event()
    write_errors = []

    def hold_write_lock():
        lock_owner._conn.execute("BEGIN IMMEDIATE")
        lock_ready.set()
        release_lock.wait(timeout=5)
        lock_owner._conn.rollback()

    def write_telemetry():
        try:
            telemetry_store.increment_compaction_telemetry(
                "conv-1",
                1,
                {"compression_count_at_record": 1},
            )
        except Exception as exc:
            write_errors.append(exc)
        finally:
            write_done.set()

    holder = threading.Thread(target=hold_write_lock)
    writer = threading.Thread(target=write_telemetry)
    try:
        holder.start()
        assert lock_ready.wait(timeout=5)
        assert telemetry_store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000

        started = time.monotonic()
        writer.start()
        assert write_done.wait(timeout=0.25)
        elapsed = time.monotonic() - started

        assert elapsed < 0.25
        assert len(write_errors) == 1
        assert isinstance(write_errors[0], sqlite3.OperationalError)
        assert "locked" in str(write_errors[0]).lower()
        assert telemetry_store._conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        assert telemetry_store.read_compaction_telemetry("conv-1") is None
    finally:
        release_lock.set()
        writer.join(timeout=5)
        holder.join(timeout=5)
        lock_owner.close()
        telemetry_store.close()


def test_successful_compaction_is_durable_before_response_hook(tmp_path, monkeypatch):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm_test.db"),
        fresh_tail_count=2,
        leaf_chunk_tokens=1,
    )
    conversation_id = "conv-1"
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "old question one"},
        {"role": "assistant", "content": "old answer one"},
        {"role": "user", "content": "old question two"},
        {"role": "assistant", "content": "old answer two"},
        {"role": "user", "content": "fresh question"},
        {"role": "assistant", "content": "fresh answer"},
    ]

    first = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    first.on_session_start(
        "test-session",
        platform="cli",
        conversation_id=conversation_id,
        context_length=200_000,
    )

    def summarize(initial_chunk, **_kwargs):
        return (
            list(initial_chunk),
            100,
            "Durable summary.\nExpand for details about: old turns",
            1,
            1,
        )

    monkeypatch.setattr(first, "_summarize_leaf_chunk_with_rescue", summarize)
    first.compress(messages)
    assert first.compression_count == 1
    persisted = first._store.read_compaction_telemetry(conversation_id)
    assert persisted is not None
    assert persisted["total_compactions"] == 1
    assert persisted["compression_count_at_record"] == 1

    # Simulate process interruption before update_from_response() can persist
    # the next per-turn telemetry snapshot.
    first.shutdown()

    restarted = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    try:
        restarted.on_session_start(
            "test-session-next",
            platform="cli",
            conversation_id=conversation_id,
            context_length=200_000,
        )

        _assert_total_compaction_surfaces(restarted, 1)
    finally:
        restarted.shutdown()


def test_response_hook_does_not_recount_durably_recorded_compaction(engine):
    engine.compression_count = 1
    engine._last_compaction_duration_ms = 12.5
    engine._record_successful_compaction_telemetry()

    engine.update_from_response(_hot_usage(prompt=400))

    telemetry = _telemetry(engine)
    assert telemetry["total_compactions"] == 1
    assert telemetry["turns_since_leaf_compaction"] == 0
    assert telemetry["peak_prompt_tokens_since_leaf_compaction"] == 400
    assert telemetry["last_compaction_duration_ms"] == 12.5


def test_response_hook_does_not_recount_ambiguous_committed_increment(engine, monkeypatch):
    real_increment = engine._store.increment_compaction_telemetry

    def commit_then_raise(conversation_id, increment, updates):
        real_increment(conversation_id, increment, updates)
        raise RuntimeError("simulated ambiguous commit outcome")

    engine.compression_count = 1
    engine._last_compaction_duration_ms = 12.5
    monkeypatch.setattr(engine._store, "increment_compaction_telemetry", commit_then_raise)
    engine._record_successful_compaction_telemetry()

    committed = engine._store.read_compaction_telemetry("conv-1")
    assert committed["total_compactions"] == 1
    assert engine._compaction_telemetry_counter_rebaseline_pending is True

    monkeypatch.setattr(engine._store, "increment_compaction_telemetry", real_increment)
    engine.update_from_response(_hot_usage(prompt=400))

    telemetry = engine._store.read_compaction_telemetry("conv-1")
    assert telemetry["total_compactions"] == 1
    assert telemetry["turns_since_leaf_compaction"] == 0
    assert telemetry["peak_prompt_tokens_since_leaf_compaction"] == 400

    engine.compression_count = 2
    engine._record_successful_compaction_telemetry()
    assert engine._store.read_compaction_telemetry("conv-1")["total_compactions"] == 2


def test_total_compactions_status_defaults_to_zero_without_telemetry(engine):
    _assert_total_compaction_surfaces(engine, 0)


def test_total_compactions_status_is_scoped_to_current_conversation(engine):
    engine._store.write_compaction_telemetry(
        "conv-1",
        {"conversation_id": "conv-1", "total_compactions": 2},
    )
    engine._store.write_compaction_telemetry(
        "other-conversation",
        {"conversation_id": "other-conversation", "total_compactions": 99},
    )

    _assert_total_compaction_surfaces(engine, 2)


@pytest.mark.parametrize("malformed", [-1, "7", 1.5, True, None, [], {}])
def test_total_compactions_status_fails_closed_for_malformed_telemetry(engine, malformed):
    engine._store.write_compaction_telemetry(
        "conv-1",
        {"conversation_id": "conv-1", "total_compactions": malformed},
    )

    _assert_total_compaction_surfaces(engine, 0)
    assert _telemetry(engine)["total_compactions"] == 0


def test_no_conversation_records_nothing(engine):
    engine._conversation_id = ""
    engine.update_from_response(_hot_usage())
    assert _telemetry(engine) is None


def test_unknown_cache_state_when_no_cache_signal(engine):
    # Prompt tokens but no cache keys -> still recorded, state unknown.
    engine.update_from_response({"prompt_tokens": 800})
    t = _telemetry(engine)
    assert t is not None
    assert t["cache_state"] == "unknown"


def test_store_roundtrip_and_skip_unchanged(tmp_path):
    store = MessageStore(tmp_path / "t.db")
    assert store.read_compaction_telemetry("c1") is None
    record = {"conversation_id": "c1", "cache_state": "hot", "turns_since_leaf_compaction": 3}
    store.write_compaction_telemetry("c1", record)
    assert store.read_compaction_telemetry("c1") == record

    # Unchanged payload must not rewrite the row.
    key = store._compaction_telemetry_key("c1")
    store.write_compaction_telemetry("c1", dict(record))
    row = store._conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    assert row is not None

    # Empty conversation id is a no-op.
    store.write_compaction_telemetry("", {"x": 1})
    assert store.read_compaction_telemetry("") is None
    store.close()


def _busy_timeout_ms(conn):
    return conn.execute("PRAGMA busy_timeout").fetchone()[0]


def _fts_match_count(db_path, term):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT count(*) FROM messages_fts WHERE messages_fts MATCH ?",
            (term,),
        ).fetchone()[0]
    finally:
        conn.close()


def _shared_storage_pair(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm_test.db"))
    first = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    sibling = first.clone_for_agent()
    assert sibling._store is first._store
    return config, first, sibling


def test_sibling_telemetry_does_not_roll_back_foreground_fts_repair(tmp_path, monkeypatch):
    config, foreground, sibling = _shared_storage_pair(tmp_path)
    term = "zanzibarrepairterm"
    paused_after_drop = threading.Event()
    telemetry_errors = []

    def record_sibling_telemetry():
        try:
            sibling._record_successful_compaction_telemetry()
        except Exception as exc:
            telemetry_errors.append(exc)

    real_drop = db_bootstrap._drop_fts_table

    def drop_then_record_sibling_telemetry(conn, table_name):
        real_drop(conn, table_name)
        if table_name != "messages_fts" or paused_after_drop.is_set():
            return
        paused_after_drop.set()
        assert conn.in_transaction
        telemetry_thread = threading.Thread(target=record_sibling_telemetry)
        telemetry_thread.start()
        telemetry_thread.join(timeout=10)
        assert not telemetry_thread.is_alive()

    try:
        foreground.on_session_start(
            "foreground-session",
            platform="cli",
            conversation_id="conv-foreground",
        )
        foreground._ingest_messages([{"role": "user", "content": f"remember {term}"}])
        sibling.on_session_start("sibling-session", platform="cli", conversation_id="conv-1")
        sibling.compression_count = 1

        # Clear the index behind the content table so explicit repair must
        # drop and rebuild messages_fts inside its own write transaction.
        shared_conn = foreground._store.connection
        shared_conn.execute("INSERT INTO messages_fts(messages_fts) VALUES('delete-all')")
        shared_conn.commit()
        assert _fts_match_count(config.database_path, term) == 0

        monkeypatch.setattr(db_bootstrap, "_drop_fts_table", drop_then_record_sibling_telemetry)
        result = lcm_command._doctor_repair_apply_text(foreground)

        assert paused_after_drop.is_set()
        assert telemetry_errors == []
        assert "status: ok" in result, result
        assert _fts_match_count(config.database_path, term) == 1

        # The repair held SQLite's write reservation, so the sibling's count
        # stayed pending and the next call records it.
        sibling._record_successful_compaction_telemetry()
        assert sibling._store.read_compaction_telemetry("conv-1")["total_compactions"] == 1
    finally:
        sibling.shutdown()
        foreground.shutdown()


def test_session_end_overlapping_sibling_telemetry_restores_shared_busy_timeout(
    tmp_path,
    monkeypatch,
):
    _config, ending, sibling = _shared_storage_pair(tmp_path)
    shared_conn = ending._store.connection
    original_timeout = _busy_timeout_ms(shared_conn)
    user_message = {"role": "user", "content": "already ingested before session end"}
    telemetry_thread_name = "sibling-telemetry"
    telemetry_progressed = threading.Event()
    session_end_in_bounded_flush = threading.Event()
    session_end_finished = threading.Event()
    telemetry_errors = []

    real_busy_timeout_ms = sqlite_util._sqlite_busy_timeout_ms

    def observe_busy_timeout(conn):
        value = real_busy_timeout_ms(conn)
        if threading.current_thread().name == telemetry_thread_name:
            # Hold telemetry after it saved the shared timeout so session-end
            # restores first, the interleaving that leaks the 50 ms value.
            telemetry_progressed.set()
            session_end_finished.wait(timeout=5)
        return value

    def record_sibling_telemetry():
        try:
            sibling._record_successful_compaction_telemetry()
        except Exception as exc:
            telemetry_errors.append(exc)
        finally:
            telemetry_progressed.set()

    telemetry_thread = threading.Thread(
        target=record_sibling_telemetry,
        name=telemetry_thread_name,
    )
    try:
        ending.on_session_start("ending-session", platform="cli", conversation_id="conv-ending")
        ending._ingest_messages([user_message])
        sibling.on_session_start("sibling-session", platform="cli", conversation_id="conv-1")
        sibling.compression_count = 1

        real_ingest = ending._ingest_messages

        def ingest_inside_bounded_flush(messages):
            if (
                not session_end_in_bounded_flush.is_set()
                and _busy_timeout_ms(shared_conn) == engine_module._SESSION_END_BUSY_TIMEOUT_MS
            ):
                session_end_in_bounded_flush.set()
                telemetry_thread.start()
                assert telemetry_progressed.wait(timeout=5)
            return real_ingest(messages)

        monkeypatch.setattr(sqlite_util, "_sqlite_busy_timeout_ms", observe_busy_timeout)
        monkeypatch.setattr(ending, "_ingest_messages", ingest_inside_bounded_flush)
        ending.on_session_end("ending-session", [user_message])
        session_end_finished.set()
        telemetry_thread.join(timeout=10)

        assert session_end_in_bounded_flush.is_set()
        assert not telemetry_thread.is_alive()
        assert telemetry_errors == []
        assert _busy_timeout_ms(shared_conn) == original_timeout
        assert sibling._store.read_compaction_telemetry("conv-1")["total_compactions"] == 1
    finally:
        session_end_finished.set()
        if telemetry_thread.is_alive():
            telemetry_thread.join(timeout=10)
        sibling.shutdown()
        ending.shutdown()


def test_sibling_ingest_contention_does_not_block_telemetry(tmp_path):
    config, ingesting, sibling = _shared_storage_pair(tmp_path)
    shared_conn = ingesting._store.connection
    ingest_thread_name = "sibling-ingest"
    insert_started = threading.Event()
    telemetry_done = threading.Event()
    errors = []

    def observe_statement(statement):
        if (
            threading.current_thread().name == ingest_thread_name
            and statement.lstrip().upper().startswith("INSERT INTO MESSAGES")
        ):
            insert_started.set()

    def ingest():
        try:
            ingesting._ingest_messages([{"role": "user", "content": "contended ingest"}])
        except Exception as exc:
            errors.append(exc)

    telemetry_elapsed = []

    def record_sibling_telemetry():
        # Time the call inside its own thread: a telemetry thread blocked on
        # the shared connection can also stall the observing thread.
        started = time.monotonic()
        try:
            sibling._record_successful_compaction_telemetry()
        except Exception as exc:
            errors.append(exc)
        finally:
            telemetry_elapsed.append(time.monotonic() - started)
            telemetry_done.set()

    ingesting.on_session_start("ingest-session", platform="cli", conversation_id="conv-ingest")
    sibling.on_session_start("sibling-session", platform="cli", conversation_id="conv-1")
    sibling.compression_count = 1

    external_writer = sqlite3.connect(config.database_path, check_same_thread=False)
    ingest_thread = threading.Thread(target=ingest, name=ingest_thread_name)
    telemetry_thread = threading.Thread(target=record_sibling_telemetry)
    try:
        external_writer.execute("BEGIN IMMEDIATE")
        shared_conn.set_trace_callback(observe_statement)
        ingest_thread.start()
        # The sibling's INSERT now waits on SQLite's write lock while it holds
        # the shared connection and the store write lock.
        assert insert_started.wait(timeout=5)

        telemetry_thread.start()
        finished_while_contended = telemetry_done.wait(timeout=1.0)
        persisted_while_contended = sibling._store.read_compaction_telemetry("conv-1")
    finally:
        external_writer.rollback()
        external_writer.close()
        ingest_thread.join(timeout=35)
        telemetry_thread.join(timeout=35)
        shared_conn.set_trace_callback(None)

    try:
        assert finished_while_contended
        assert telemetry_elapsed[0] < 1.0
        assert persisted_while_contended is None
        assert not ingest_thread.is_alive()
        assert not telemetry_thread.is_alive()
        assert errors == []

        sibling._record_successful_compaction_telemetry()
        assert sibling._store.read_compaction_telemetry("conv-1")["total_compactions"] == 1
    finally:
        sibling.shutdown()
        ingesting.shutdown()


def test_busy_telemetry_connection_skips_and_keeps_pending_counts(tmp_path):
    config = LCMConfig(database_path=str(tmp_path / "lcm_test.db"))
    first = LCMEngine(config=config, hermes_home=str(tmp_path / "hermes_home"))
    first.on_session_start("first-session", platform="cli", conversation_id="conv-1")
    first.compression_count = 3
    first._record_successful_compaction_telemetry()
    first.on_session_start("next-session", platform="cli", conversation_id="conv-1")
    first.compression_count = 1
    assert first._compaction_telemetry_counter_rebaseline_pending is True

    telemetry_lock = first._store._telemetry_lock
    errors = []
    elapsed = []

    def record_while_busy():
        started = time.monotonic()
        try:
            first._record_successful_compaction_telemetry()
        except Exception as exc:
            errors.append(exc)
        finally:
            elapsed.append(time.monotonic() - started)

    try:
        assert telemetry_lock.acquire(timeout=5)
        try:
            recorder = threading.Thread(target=record_while_busy)
            recorder.start()
            recorder.join(timeout=10)
        finally:
            telemetry_lock.release()

        assert not recorder.is_alive()
        assert errors == []
        assert elapsed[0] < 1.0
        assert first._store.read_compaction_telemetry("conv-1")["total_compactions"] == 3
        # A skipped write must not consume the post-rollover rebaseline, or the
        # next write would treat the new session's first compaction as old.
        assert first._compaction_telemetry_counter_rebaseline_pending is True

        first._record_successful_compaction_telemetry()
        assert first._store.read_compaction_telemetry("conv-1")["total_compactions"] == 4
    finally:
        first.shutdown()
