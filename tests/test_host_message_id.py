"""Rows that carry the Hermes message id (message_uid) are decided by that id.

Rows without an id keep the older rule. Every list here is synthetic. Each test
names the rule of the plan that it proves.
"""
import json
import sqlite3
import threading
import time
from copy import deepcopy

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
from hermes_lcm.message_content import normalize_content_value
from hermes_lcm.store import MessageStore

S = "uid-session"
CHILD = "uid-session-child"
C = "uid-chat"
T = 1_795_000_000


@pytest.fixture(autouse=True)
def _stub_summarizer(monkeypatch):
    monkeypatch.setattr(engine_module, "summarize_with_escalation",
                        lambda **_kwargs: ("synthetic summary.\nExpand for details about: older work", 1))


def bind(tmp_path, session=S, platform="cli", conversation_id=C):
    config = LCMConfig(database_path=str(tmp_path / "lcm.db"),
                       large_output_externalization_path=str(tmp_path / "ext"),
                       fresh_tail_count=10, leaf_chunk_tokens=150)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(session, platform=platform, conversation_id=conversation_id, context_length=200_000)
    engine.threshold_tokens = 170_000
    return engine


def msg(role, content, at, uid=None, **extra):
    row = dict(role=role, content=content, **extra)
    if at is not None:
        row["timestamp"] = float(at)
    if uid:
        row["message_uid"] = uid
    return row


def call(call_id, content, at, uid, args='{"path":"build.txt"}'):
    spec = dict(id=call_id, call_id=call_id, type="function", function=dict(name="read_file", arguments=args))
    return msg("assistant", content, at, uid, tool_calls=[spec])


def result(call_id, content, at, uid):
    return msg("tool", content, at, uid, tool_call_id=call_id, name="read_file", tool_name="read_file")


def rows(engine, session=S):
    return [(r["role"], r["content"], r["observed_at"])
            for r in engine._store.get_session_messages(session, limit=10_000)]


def stored_uids(engine, session=S):
    return [row[0] for row in engine._store._conn.execute(
        "SELECT message_uid FROM messages WHERE session_id = ? ORDER BY store_id", (session,))]


def view(engine, session=S, conversation_id=C):
    """What a scoped read of the session returns: (content, message_uid) in store order."""
    return [(row[0], row[1]) for row in engine._store._conn.execute(
        "SELECT content, message_uid FROM messages WHERE session_id = ? AND conversation_id = ? ORDER BY store_id",
        (session, conversation_id))]


def followups(prefix, start):
    return [row for n in range(1, 19) for row in (
        msg("user", f"Follow-up {n}", start + n * 2, f"{prefix}-fu-{n}"),
        msg("assistant", f"Follow-up response {n}: " + "detail " * 320, start + 1 + n * 2, f"{prefix}-fr-{n}"),
    )]


def handoff(engine, before, *, session=S, child=None):
    kept = engine.compress(deepcopy(before), current_tokens=160_000, force=True)
    assert 1 < len(kept) < len(before)
    engine.on_session_end(session, deepcopy(before))
    engine.on_session_start(child or session, boundary_reason="compression", old_session_id=session,
                            platform="cli", conversation_id=C)
    # Hermes stamps a new id on the summary row when it writes the compacted list.
    kept[0] = dict(kept[0], message_uid=f"summary-{len(before)}")
    return kept


def legacy_store(tmp_path, stored, conversation=C):
    """A store written before this change: no message_uid column, rows of session S in *conversation*."""
    conn = sqlite3.connect(tmp_path / "lcm.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        store_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, source TEXT DEFAULT '',
        conversation_id TEXT DEFAULT '', role TEXT NOT NULL, content TEXT, tool_call_id TEXT,
        tool_calls TEXT, tool_name TEXT, timestamp REAL NOT NULL, token_estimate INTEGER DEFAULT 0,
        pinned INTEGER DEFAULT 0, ingested_at REAL, observed_at REAL, observed_at_source TEXT)""")
    for row in stored:
        conn.execute(
            "INSERT INTO messages (session_id, source, conversation_id, role, content, timestamp, ingested_at, "
            "observed_at, observed_at_source) VALUES (?, 'cli', ?, ?, ?, ?, ?, ?, 'host_message_timestamp')",
            (S, conversation, row["role"], row["content"], 1.0, 1.0, row["timestamp"]))
    conn.commit()
    conn.close()


# R1: the store has one nullable column for the id, its index, and the first-open mark.
def test_store_adds_message_uid_column_to_an_existing_store(tmp_path):
    legacy_store(tmp_path, [msg("user", "kept text", T)])
    store = MessageStore(tmp_path / "lcm.db")
    try:
        columns = {row[1] for row in store._conn.execute("PRAGMA table_info(messages)")}
        indexes = {row[1] for row in store._conn.execute("PRAGMA index_list(messages)")}
        old = store._conn.execute("SELECT content, message_uid FROM messages").fetchall()
        mark = store.message_uid_first_open_store_id()
    finally:
        store.close()
    assert "message_uid" in columns
    assert "idx_msg_message_uid" in indexes
    assert old == [("kept text", None)]
    assert mark == 1


# R1: the first-open mark is written once; rows stored later are above it.
def test_first_open_mark_is_written_once(tmp_path):
    legacy_store(tmp_path, [msg("user", "one", T), msg("assistant", "two", T + 1)])
    MessageStore(tmp_path / "lcm.db").close()
    engine = bind(tmp_path)
    try:
        engine.ingest([msg("user", "one", T, "u-1"), msg("assistant", "two", T + 1, "a-1"),
                       msg("user", "three", T + 2, "u-2")])
        assert engine._store.message_uid_first_open_store_id() == 2
    finally:
        engine.shutdown()
    reopened = MessageStore(tmp_path / "lcm.db")
    try:
        assert reopened.message_uid_first_open_store_id() == 2
    finally:
        reopened.close()


# R1: a stored row keeps the id it arrived with; a row without one stores NULL.
def test_new_rows_store_their_message_uid(tmp_path):
    engine = bind(tmp_path)
    try:
        engine.ingest([msg("user", "Check the build", T, "u-1"), msg("assistant", "Started.", T + 1)])
        assert stored_uids(engine) == ["u-1", None]
    finally:
        engine.shutdown()


# R4: the edit-revert case in its host form, every row with an id.
def test_reverted_edit_and_repeated_tool_run_are_stored_after_restart(tmp_path):
    calls = "read_file:0"
    output = json.dumps({"content": "1|build: green", "total_lines": 1})
    before = [msg("user", "Check the build", T, "send"),
              msg("assistant", "Build check started.", T + 5, "a-0"),
              msg("user", "Check the test build", T, "edit-1"),
              call(calls, "Checking the test build.", T + 61, "c-1"),
              result(calls, output, T + 62, "r-1"),
              msg("assistant", "Test build check complete.", T + 65, "a-1")] + followups("old", T + 100)
    engine = bind(tmp_path, platform="telegram")
    try:
        for n in range(2, len(before) + 1, 2):
            engine.ingest(deepcopy(before[:n]))
        kept = handoff(engine, before)
    finally:
        engine.shutdown()
    fresh = bind(tmp_path, platform="telegram")
    try:
        old = rows(fresh)
        tail = [msg("user", "Check the build", T, "edit-2"),
                call(calls, "Checking the build again.", T + 4501, "c-2"),
                result(calls, output, T + 4502, "r-2"),
                msg("assistant", "The build is green.", T + 4505, "a-2")]
        fresh.ingest(deepcopy(kept + tail))
        added = rows(fresh)[len(old):]
        assert added == [(r["role"], r["content"], r["timestamp"]) for r in tail]
        assert stored_uids(fresh)[len(old):] == ["edit-2", "c-2", "r-2", "a-2"]
    finally:
        fresh.shutdown()


# R4: a new request without a host timestamp, then the same tool run as before (same provider
# id, arguments and result). Every row has a new id, so every row is new.
def test_unstamped_request_with_a_repeated_tool_run_is_stored(tmp_path):
    output = "build: green"
    old = [msg("user", "Check the build", T, "u-1"), call("read_file:0", "Checking.", T + 1, "c-1"),
           result("read_file:0", output, T + 2, "r-1"), msg("assistant", "Green.", T + 3, "a-1")]
    engine = bind(tmp_path)
    try:
        engine.ingest(deepcopy(old))
    finally:
        engine.shutdown()
    fresh = bind(tmp_path)
    try:
        new = [msg("user", "And now?", None, "u-2"), msg("assistant", "Checking again.", None, "a-2"),
               call("read_file:0", "", None, "c-2"), result("read_file:0", output, None, "r-2")]
        fresh.ingest(deepcopy(new))
        assert rows(fresh)[4:] == [(r["role"], r["content"], None) for r in new]
    finally:
        fresh.shutdown()


# R4: a short delta after a rebind that repeats an old text is new by its id.
@pytest.mark.parametrize("rebind", [True, False])
def test_delta_with_a_new_id_keeps_the_repeated_text(tmp_path, rebind):
    engine = bind(tmp_path)
    try:
        engine.ingest([msg("user", "Approved", 100, "u-1"), msg("assistant", "ack", None, "a-1")])
        if rebind:
            engine.on_session_start(S, platform="cli", conversation_id=C, context_length=200_000)
        engine.ingest([msg("user", "Approved", 200, "u-2")])
        assert ("user", "Approved", 200.0) in rows(engine)
    finally:
        engine.shutdown()


# R4: after an undo the host list is shorter; the fresh turn has new ids.
def test_undo_with_ids_keeps_the_fresh_turn(tmp_path):
    engine = bind(tmp_path)
    try:
        old = [msg("user", "Retained request", 100, "u-1"), msg("assistant", "Retained reply", 101, "a-1"),
               msg("user", "Request later undone", 102, "u-2"), msg("assistant", "Reply later undone", 103, "a-2")]
        engine.ingest(deepcopy(old))
        engine.ingest(deepcopy(old[:2]) + [msg("user", "Fresh question after undo", 200, "u-3"),
                                           msg("assistant", "Fresh response after undo", 201, "a-3")])
        assert rows(engine)[4:] == [("user", "Fresh question after undo", 200.0),
                                    ("assistant", "Fresh response after undo", 201.0)]
    finally:
        engine.shutdown()


# R5 proof 1: rows stored before this code get their id once and are not stored again.
def test_rows_stored_before_this_code_get_their_id(tmp_path):
    history = [row for n in range(4) for row in (
        msg("user", f"Question {n}", T + 2 * n, f"u-{n}"), msg("assistant", f"Answer {n}", T + 2 * n + 1, f"a-{n}"))]
    legacy_store(tmp_path, history)
    fresh = bind(tmp_path)
    try:
        turn = [msg("user", "Question 4", T + 8, "u-4"), msg("assistant", "Answer 4", T + 9, "a-4")]
        fresh.ingest(deepcopy(history + turn))
        assert len(rows(fresh)) == len(history) + 2
        assert stored_uids(fresh) == [row["message_uid"] for row in history + turn]
    finally:
        fresh.shutdown()


# R5 proof 1 reads the view's own tail: a newer row with the same text and timestamp in another
# lane of the session does not take the pairing, so the view's rows get their ids and no repeat.
def test_rows_stored_before_this_code_pair_inside_the_view(tmp_path):
    history = [msg("user", "Alpha", T), msg("assistant", "Beta", T + 1),
               msg("user", "Gamma", T + 2), msg("assistant", "Delta", T + 3)]
    legacy_store(tmp_path, history)
    legacy_store(tmp_path, history[3:], conversation="lane-2")
    engine = bind(tmp_path)
    try:
        engine.ingest([dict(row, message_uid=f"{n}" * 32) for n, row in enumerate(history, 1)])
        own, other = view(engine), view(engine, S, "lane-2")
    finally:
        engine.shutdown()
    assert own == [(row["content"], f"{n}" * 32) for n, row in enumerate(history, 1)]
    assert other == [("Delta", None)]


# R5: a row stored by this code without an id is not adopted by text, even with the
# same text and host timestamp; it is stored again (a repeat, never a loss).
def test_row_stored_by_this_code_without_id_is_not_adopted_by_text(tmp_path):
    engine = bind(tmp_path)
    try:
        engine.ingest([msg("user", "Same ask", T), msg("assistant", "Answer", T + 1)])
        engine.ingest([msg("user", "Same ask", T, "u-1"), msg("assistant", "Answer", T + 1, "a-1")])
        assert [content for _role, content, _at in rows(engine)] == ["Same ask", "Answer", "Same ask", "Answer"]
        assert stored_uids(engine) == [None, None, "u-1", "a-1"]
    finally:
        engine.shutdown()


# R5 proof 2: the current-turn row reaches LCM before the host flush gives it an id; the host
# stamps the id onto the same dict (preflight only) or onto the dict compress() returned.
@pytest.mark.parametrize("compress", [False, True])
def test_current_turn_row_gets_its_id_on_the_same_dict(tmp_path, compress):
    history = followups("cur", T)
    engine = bind(tmp_path)
    try:
        engine.ingest(deepcopy(history))
        pending = msg("user", "Next question", T + 900)
        host_list = deepcopy(history) + [pending]
        engine.should_compress_preflight(host_list)
        if compress:
            host_list = engine.compress(host_list, current_tokens=160_000, force=True)
        count = len(rows(engine))
        assert rows(engine)[-1] == ("user", "Next question", T + 900.0)
        host_list[-1]["message_uid"] = "u-next"
        engine.ingest(host_list + [msg("assistant", "Next answer", T + 901, "a-next")])
        assert rows(engine)[count:] == [("assistant", "Next answer", T + 901.0)]
        assert stored_uids(engine)[-2:] == ["u-next", "a-next"]
    finally:
        engine.shutdown()


# R5: a replaced dict with the same text and timestamp is a new row (the first request can
# have been rewound before LCM saw its id).
def test_replaced_current_turn_row_is_a_new_row(tmp_path):
    history = followups("rep", T)
    engine = bind(tmp_path)
    try:
        engine.ingest(deepcopy(history))
        pending = msg("user", "Same ask", T + 900)
        host_list = deepcopy(history) + [pending]
        engine.should_compress_preflight(host_list)
        replacement = dict(pending, message_uid="u-new")
        engine.ingest(deepcopy(history) + [replacement, msg("assistant", "Answered.", T + 1000, "reply")])
        assert [content for _role, content, _at in rows(engine)].count("Same ask") == 2
        assert stored_uids(engine)[-3:] == [None, "u-new", "reply"]
    finally:
        engine.shutdown()


# R5: an adoption whose UPDATE changes no row is not a success: the row is stored with its id.
def test_lost_adoption_race_stores_the_row(tmp_path):
    engine = bind(tmp_path)
    other = MessageStore(tmp_path / "lcm.db")
    try:
        first = msg("user", "same", T)
        host_list = [first]
        engine.ingest(host_list)
        original = engine._store._append_protected_batch

        def competing_writer_first(*args, **kwargs):
            with other._write_lock, other._conn:
                other._conn.execute("UPDATE messages SET message_uid = 'winner' WHERE message_uid IS NULL")
            return original(*args, **kwargs)

        engine._store._append_protected_batch = competing_writer_first
        first["message_uid"] = "loser"
        engine.ingest(host_list)
        assert stored_uids(engine) == ["winner", "loser"]
    finally:
        other.close()
        engine.shutdown()


# R12: the write transaction checks each new id again: a competing writer that stored the same id
# in the same view between this engine's read and its write leaves one row, not two.
def test_competing_writer_of_the_same_new_id_adds_no_second_row(tmp_path):
    engine = bind(tmp_path)
    other = MessageStore(tmp_path / "lcm.db")
    try:
        original = engine._store._append_protected_batch

        def competing_writer_first(*args, **kwargs):
            other.append(S, msg("user", "shared", T, "shared"), source="cli", conversation_id=C)
            return original(*args, **kwargs)

        engine._store._append_protected_batch = competing_writer_first
        engine.ingest([msg("user", "shared", T, "shared")])
        assert stored_uids(engine) == ["shared"]
    finally:
        other.close()
        engine.shutdown()

# R5, R7a, R12: the host rewrites the dict that LCM stored without an id and stamps it. The
# write adopts the stored row, then decides the rewritten text against the rows now held: the
# finalized list keeps both texts (a replay adds nothing); an intermediate surface keeps the first.
@pytest.mark.parametrize("surface", ["finalized", "intermediate"])
def test_row_rewritten_in_place_after_storage_keeps_both_texts(tmp_path, surface):
    engine = bind(tmp_path)
    pending = msg("user", "[Voice input] Read the label", T)
    try:
        engine.should_compress_preflight([pending])
        pending.update(message_uid="1" * 32, content="Read the label")
        host_list = [pending, msg("assistant", "Done.", T + 1, "2" * 32)]
        for _ in range(2):
            if surface == "finalized":
                engine.ingest(host_list)
            else:
                engine.handle_tool_call("lcm_status", {}, messages=host_list)
        got = view(engine)
    finally:
        engine.shutdown()
    clean = [("Read the label", "1" * 32)] if surface == "finalized" else []
    assert got == [("[Voice input] Read the label", "1" * 32)] + clean + [("Done.", "2" * 32)]


# R7a, R12: another engine stores the working text under the id that this finalized write
# planned as new; the write decides again against the rows now held and keeps the final text.
def test_writer_between_read_and_write_does_not_drop_the_final_text(tmp_path):
    engine, sibling = bind(tmp_path), bind(tmp_path)
    clean = msg("user", "Read the label", T, "1" * 32)
    working = dict(clean, content="[Voice input] Read the label")
    original = engine._store._append_protected_batch

    def sibling_first(*args, **kwargs):
        sibling.handle_tool_call("lcm_status", {}, messages=[deepcopy(working)])
        return original(*args, **kwargs)

    try:
        engine._store._append_protected_batch = sibling_first
        engine.ingest([deepcopy(clean), msg("assistant", "Done.", T + 1, "2" * 32)])
        engine._store._append_protected_batch = original
        engine.ingest([deepcopy(clean), msg("assistant", "Done.", T + 1, "2" * 32)])
        got = view(engine)
    finally:
        sibling.shutdown()
        engine.shutdown()
    assert got == [("[Voice input] Read the label", "1" * 32), ("Read the label", "1" * 32), ("Done.", "2" * 32)]



# R2: a rotation child holds its parent's carried rows in this process; after a restart the
# child stores them again (an accepted repeat). No id of the list is missing from the views.
def test_rotation_child_and_restart(tmp_path):
    history = followups("rot", T)
    engine = bind(tmp_path)
    try:
        engine.ingest(deepcopy(history))
        kept = handoff(engine, history, child=CHILD)
        turn = [msg("user", "After rotation", T + 900, "u-rot"), msg("assistant", "Rotated answer", T + 901, "a-rot")]
        engine.ingest(deepcopy(kept + turn))
        assert rows(engine, CHILD) == [(r["role"], r["content"], r["timestamp"]) for r in turn]
    finally:
        engine.shutdown()
    fresh = bind(tmp_path, session=CHILD)
    try:
        later = [msg("user", "After restart", T + 950, "u-late"), msg("assistant", "Late answer", T + 951, "a-late")]
        fresh.ingest(deepcopy(kept + turn + later))
        child_rows = rows(fresh, CHILD)
        assert child_rows[:2] == [(r["role"], r["content"], r["timestamp"]) for r in turn]
        assert child_rows[2:] == [(r["role"], r["content"], r["timestamp"]) for r in kept[1:] + later]
        every_id = {r["message_uid"] for r in kept[1:] + turn + later}
        assert every_id <= set(stored_uids(fresh, CHILD)) | set(stored_uids(fresh, S))
    finally:
        fresh.shutdown()


# R2: the same session on a new conversation lane stores the replayed rows in the new view.
def test_same_session_on_a_new_lane_stores_its_view(tmp_path):
    engine = bind(tmp_path)
    old = [msg("user", "alpha", T, "a"), msg("assistant", "beta", T + 1, "b")]
    try:
        engine.ingest(deepcopy(old))
        engine.on_session_start(S, platform="telegram", conversation_id="new-lane")
        engine.ingest(deepcopy(old) + [msg("user", "gamma", T + 2, "c")])
        assert [content for content, _uid in view(engine, S, "new-lane")] == ["alpha", "beta", "gamma"]
        assert [content for content, _uid in view(engine, S, C)] == ["alpha", "beta"]
    finally:
        engine.shutdown()


# R2: a new session on the same lane with copied ids keeps its own rows.
def test_branch_on_the_same_lane_stores_its_own_rows(tmp_path):
    parent = bind(tmp_path)
    old = [msg("user", "Alpha", T, "1" * 32), msg("assistant", "Beta", T + 1, "2" * 32)]
    try:
        parent.ingest(deepcopy(old))
    finally:
        parent.shutdown()
    branch = bind(tmp_path, session="branch-owner")
    try:
        branch.ingest(deepcopy(old) + [msg("user", "Gamma", T + 2, "3" * 32)])
        assert [content for _role, content, _at in rows(branch, "branch-owner")] == ["Alpha", "Beta", "Gamma"]
    finally:
        branch.shutdown()

# R5, R2: both adoption proofs act only inside the archive view. After the same session moves to
# another lane, its rows there are stored with their ids; the old lane's rows keep none.
@pytest.mark.parametrize("proof", ["same_dict", "pre_upgrade"])
def test_adoption_stays_in_the_archive_it_reads(tmp_path, proof):
    first = msg("user", "Alpha", T)
    second = msg("assistant", "Beta", T + 1)
    if proof == "pre_upgrade":
        legacy_store(tmp_path, [first, second])
    engine = bind(tmp_path)
    try:
        if proof == "same_dict":
            engine.should_compress_preflight([first])
        engine.on_session_start(S, platform="telegram", conversation_id="lane-2")
        first["message_uid"] = "1" * 32
        second["message_uid"] = "2" * 32
        engine.ingest([first, second, msg("user", "Gamma", T + 2, "3" * 32)])
        old, new = view(engine), view(engine, S, "lane-2")
    finally:
        engine.shutdown()
    assert new == [("Alpha", "1" * 32), ("Beta", "2" * 32), ("Gamma", "3" * 32)]
    assert old and all(uid is None for _content, uid in old)


# R2: a rotation link belongs to one store. After the engine moves to another home, a normal
# start of the child there settles against the child alone.
def test_home_rebind_drops_rotation_links(tmp_path):
    one, two = str(tmp_path / "one"), str(tmp_path / "two")
    earlier = msg("user", "Earlier", T, "a" * 32)
    engine = LCMEngine(config=LCMConfig(database_path=""), hermes_home=one)
    try:
        engine.on_session_start("parent", platform="cli", conversation_id=C, hermes_home=one)
        engine.ingest([earlier])
        engine.on_session_start("child", old_session_id="parent", boundary_reason="compression",
                                platform="cli", conversation_id=C)
        engine.on_session_start("parent", platform="cli", conversation_id=C, hermes_home=two)
        engine.ingest([deepcopy(earlier)])
        engine.on_session_start("child", platform="cli", conversation_id=C)
        engine.ingest([deepcopy(earlier), msg("assistant", "New branch", T + 1, "b" * 32)])
        got = view(engine, "child", C)
    finally:
        engine.shutdown()
    assert got == [("Earlier", "a" * 32), ("New branch", "b" * 32)]


# R5: a same-dict proof belongs to one store. A row stored without an id in one home does not
# adopt the row of another home's store that has the same store_id.
def test_home_rebind_drops_same_dict_proofs(tmp_path):
    one, two = str(tmp_path / "one"), str(tmp_path / "two")
    pending = msg("user", "Pending ask", T + 5)
    other = msg("user", "Unrelated row", T)
    engine = LCMEngine(config=LCMConfig(database_path=""), hermes_home=one)
    try:
        engine.on_session_start(S, platform="cli", conversation_id=C, hermes_home=one)
        engine.should_compress_preflight([pending])
        engine.on_session_start(S, platform="cli", conversation_id=C, hermes_home=two)
        engine.ingest([other])
        pending["message_uid"] = "p" * 32
        engine.ingest([other, pending])
        got = view(engine)
    finally:
        engine.shutdown()
    assert got == [("Unrelated row", None), ("Pending ask", "p" * 32)]



# R2: a late session end decides its whole list by id against its own session; a row with
# another id and the same text as a stored row is stored; a second close adds nothing.
def test_late_close_stores_a_different_message_with_the_same_text(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "late.db"), ignore_session_patterns=["cron"]),
                       hermes_home=str(tmp_path / "home"))
    ended = "reused-id"
    a = {"role": "user", "content": "Again", "message_uid": "a" * 32}
    b = dict(a, message_uid="b" * 32)
    c = {"role": "assistant", "content": "Done", "message_uid": "c" * 32}
    try:
        engine.on_session_start(ended, platform="cron", context_length=1000)
        engine.ingest([{"role": "user", "content": "Other", "message_uid": "d" * 32}])
        engine.on_session_start(ended, platform="cli", context_length=1000)
        engine.ingest([a])
        engine.on_session_start("foreground", platform="cli", context_length=1000)
        engine.on_session_end(ended, [b, c])
        engine.on_session_end(ended, [b, c])
        uids = [row[0] for row in engine._store._conn.execute(
            "SELECT message_uid FROM messages WHERE session_id = ? ORDER BY store_id", (ended,))]
        assert uids == ["a" * 32, "b" * 32, "c" * 32]
        assert engine._store.get_session_count("foreground") == 0
    finally:
        engine.shutdown()

# R11: a late close of a reused (bypass, then normal) session settles there by id after its
# finalized list stored a version; the version's extra row does not hide the new message, a
# second close adds nothing, and the foreground stays empty.
@pytest.mark.parametrize("rewrite", [False, True])
def test_late_close_after_a_stored_version_keeps_the_new_message(tmp_path, rewrite):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "late.db"), ignore_session_patterns=["cron"]),
                       hermes_home=str(tmp_path / "home"))
    clean = msg("user", "Read label.", T, "a" * 32)
    seen_by_tool = dict(clean, content="[Voice input] Read label.") if rewrite else deepcopy(clean)
    answer = msg("assistant", "Ready.", T + 1, "b" * 32)
    late = msg("user", "Last message.", T + 2, "c" * 32)
    try:
        engine.on_session_start(S, platform="cron", conversation_id="ignored")
        engine.ingest([msg("user", "Cron task", T, "f" * 32)])
        engine.on_session_start(S, platform="cli", conversation_id=C)
        engine.handle_tool_call("lcm_status", {}, messages=[seen_by_tool])
        engine.ingest([clean, answer])
        engine.on_session_start("foreground", platform="cli", conversation_id="other")
        for _ in range(2):
            engine.on_session_end(S, deepcopy([clean, answer, late]))
        ended, foreground = view(engine), view(engine, "foreground", "other")
    finally:
        engine.shutdown()
    working = [("[Voice input] Read label.", "a" * 32)] if rewrite else []
    assert ended == working + [("Read label.", "a" * 32), ("Ready.", "b" * 32), ("Last message.", "c" * 32)]
    assert foreground == []


# R11: a late close of a normal session after the engine moved to another normal session settles
# the ended session's own archive, twice without a repeat; the foreground keeps its own rows and
# its binding.
def test_late_close_of_a_normal_session_settles_its_own_archive(tmp_path):
    engine = bind(tmp_path)
    opener = msg("user", "Normal A opener", T, "1" * 32)
    answer = msg("assistant", "A final answer", T + 1, "2" * 32)
    try:
        engine.ingest([opener])
        engine.on_session_start("B", platform="cli", conversation_id="B-lane")
        engine.ingest([msg("user", "Normal B opener", T + 2, "3" * 32)])
        for _ in range(2):
            engine.on_session_end(S, deepcopy([opener, answer]))
        ended, foreground = view(engine), view(engine, "B", "B-lane")
        foreground_owner = engine._lifecycle.get_by_conversation("B-lane").current_session_id
    finally:
        engine.shutdown()
    assert ended == [("Normal A opener", "1" * 32), ("A final answer", "2" * 32)]
    assert foreground == [("Normal B opener", "3" * 32)]
    assert foreground_owner == "B"


# R11: that settlement keeps the bounded session-end flush: under another writer's lock it gives
# up after the short wait instead of the store's busy timeout; the next close stores the rows.
def test_late_close_in_its_own_archive_keeps_the_bounded_flush(tmp_path):
    engine = bind(tmp_path)
    opener = msg("user", "Normal A opener", T, "1" * 32)
    answer = msg("assistant", "A final answer", T + 1, "2" * 32)
    keeper = sqlite3.connect(tmp_path / "lcm.db")
    try:
        engine.ingest([opener])
        engine.on_session_start("B", platform="cli", conversation_id="B-lane")
        keeper.execute("BEGIN IMMEDIATE")
        began = time.monotonic()
        engine.on_session_end(S, deepcopy([opener, answer]))
        waited = time.monotonic() - began
        keeper.rollback()
        during = view(engine)
        engine.on_session_end(S, deepcopy([opener, answer]))
        after, foreground = view(engine), view(engine, "B", "B-lane")
    finally:
        keeper.close()
        engine.shutdown()
    assert waited < 5
    assert during == [("Normal A opener", "1" * 32)]
    assert after == [("Normal A opener", "1" * 32), ("A final answer", "2" * 32)]
    assert foreground == []


# R11: a late close of a session that this engine never bound (bound before a restart) settles
# that session in the conversation its lifecycle record names, never the foreground.
def test_late_close_of_a_session_bound_before_a_restart(tmp_path):
    opener = msg("user", "Before restart", T, "1" * 32)
    answer = msg("assistant", "Closing answer", T + 1, "2" * 32)
    first = bind(tmp_path)
    try:
        first.ingest([opener])
    finally:
        first.shutdown()
    engine = bind(tmp_path, session="foreground", conversation_id="other")
    try:
        for _ in range(2):
            engine.on_session_end(S, deepcopy([opener, answer]))
        ended, foreground = view(engine), view(engine, "foreground", "other")
    finally:
        engine.shutdown()
    assert ended == [("Before restart", "1" * 32), ("Closing answer", "2" * 32)]
    assert foreground == []


# R11: the bypass checks keep precedence over the id rule: a late close of an ignored session
# stores nothing, in its own archive or the foreground's.
def test_late_close_of_an_ignored_session_stores_nothing(tmp_path):
    engine = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "late.db"), ignore_session_patterns=["cron"]),
                       hermes_home=str(tmp_path / "home"))
    task = msg("user", "Cron task", T, "f" * 32)
    try:
        engine.on_session_start("job", platform="cron", conversation_id="cron-lane")
        engine.ingest([task])
        engine.on_session_start("foreground", platform="cli", conversation_id="other")
        engine.on_session_end("job", [task, msg("assistant", "Cron result", T + 1, "e" * 32)])
        count = engine._store._conn.execute("SELECT count(*) FROM messages").fetchone()[0]
    finally:
        engine.shutdown()
    assert count == 0



# R7: a merged row is stored once for each constituent id that is not held, one id per row;
# repeats add nothing, in the same batch and after a restart.
def test_merged_rows_are_stored_once_per_missing_id(tmp_path):
    engine = bind(tmp_path)
    try:
        engine.ingest([msg("user", "First part", T, "u-1")])
        merged = msg("user", "First part\n\nSecond part", T, "u-1", _absorbed_message_uids=["u-2"])
        engine.ingest([deepcopy(merged), msg("assistant", "Both parts read.", T + 5, "a-1")])
        engine.ingest([deepcopy(merged), msg("assistant", "Both parts read.", T + 5, "a-1")])
        assert [content for _role, content, _at in rows(engine)] == [
            "First part", "First part\n\nSecond part", "Both parts read."]
        assert stored_uids(engine) == ["u-1", "u-2", "a-1"]
        three = msg("user", "A\n\nB\n\nC", T + 10, "x-a", _absorbed_message_uids=["x-b", "x-c"])
        counts = []
        for _ in range(3):
            engine.ingest([deepcopy(three), deepcopy(three)])
            counts.append(len(rows(engine)))
        assert counts == [6, 6, 6]
        assert stored_uids(engine)[3:] == ["x-a", "x-b", "x-c"]
    finally:
        engine.shutdown()
    fresh = bind(tmp_path)
    try:
        fresh.ingest([deepcopy(three)])
        assert len(rows(fresh)) == 6
    finally:
        fresh.shutdown()


# R7: two list slots with one new id give one stored row.
def test_two_slots_with_one_new_id_store_one_row(tmp_path):
    engine = bind(tmp_path)
    try:
        echo = msg("user", "Echo", T, "e" * 32)
        engine.ingest([echo, deepcopy(echo)])
        assert stored_uids(engine) == ["e" * 32]
    finally:
        engine.shutdown()


# R7: changed text under a held id on an intermediate surface keeps the first text.
def test_changed_text_on_an_intermediate_surface_keeps_the_first_text(tmp_path):
    engine = bind(tmp_path)
    try:
        engine.ingest([msg("user", "Check", T, "u-1"), msg("assistant", "First form", T + 1, "a-1")])
        engine.on_session_start(S, platform="cli", conversation_id=C, context_length=200_000)
        engine.handle_tool_call("lcm_status", {}, messages=[msg("user", "Check", T, "u-1"),
                                                            msg("assistant", "Second form", T + 1, "a-1")])
        assert rows(engine) == [("user", "Check", float(T)), ("assistant", "First form", T + 1.0)]
    finally:
        engine.shutdown()


# R7a: the finalized transcript (ingest(), the post_llm_call entry) stores each new text of a
# held user or assistant id once; the earlier text stays. Here the user row that a context tool
# saw carried a persist override; the finalized list carries the clean text.
def test_finalized_version_after_a_context_tool_is_stored_once(tmp_path):
    engine = bind(tmp_path)
    try:
        user = msg("user", "[Voice input] Read the label", T, "u")
        working = call("read", "Checking.", T + 1, "a")
        engine.handle_tool_call("lcm_status", {}, messages=deepcopy([user, working]))
        final = [dict(user, content="Read the label"), working, result("read", "ok", T + 2, "t"),
                 msg("assistant", "Done.", T + 3, "f")]
        engine.ingest(deepcopy(final))
        engine.ingest(deepcopy(final))
        assert [content for _role, content, _at in rows(engine)] == [
            "[Voice input] Read the label", "Checking.", "Read the label", "ok", "Done."]
        assert stored_uids(engine) == ["u", "a", "u", "t", "f"]
    finally:
        engine.shutdown()


# R7b: a row stored blank that arrives with text is stored once, on any surface: after a
# blank list form, through another clone, and after a reopen.
@pytest.mark.parametrize("blank", ["", [], [{"type": "text", "text": ""}]])
@pytest.mark.parametrize("where", ["same", "clone", "reopen"])
def test_text_filled_into_a_row_stored_blank_is_stored_once(tmp_path, blank, where):
    engine = bind(tmp_path)
    user = msg("user", "Check the build", T, "u-1")
    pending = call("read_file:0", blank, T + 1, "c-1")
    filled = dict(deepcopy(pending), content="The build is green.")
    other = None
    try:
        engine.handle_tool_call("lcm_status", {}, messages=[deepcopy(user), deepcopy(pending)])
        if where == "clone":
            other = engine.clone_for_agent()
            other.on_session_start(S, platform="cli", conversation_id=C)
            target = other
        elif where == "reopen":
            engine.shutdown()
            target = engine = bind(tmp_path)
        else:
            target = engine
        for _ in range(2):
            target.handle_tool_call("lcm_status", {}, messages=[deepcopy(user), deepcopy(filled)])
        assert [content for _role, content, _at in rows(target)][-1] == "The build is green."
        assert stored_uids(target) == ["u-1", "c-1", "c-1"]
    finally:
        if other is not None:
            other.shutdown()
        engine.shutdown()

# D5: the blank test is the host's: string parts and the text, content, input_text, output_text
# and summary_text fields give visible text; image and audio parts give none; any other single
# value counts by its own text. A stored list is decoded first; a literal string is text.
def test_host_blank_test_follows_the_host_text_rules():
    cases = [
        ("", True), ("  ", True), (None, True), ([], True), ([{"type": "text", "text": ""}], True),
        ([{"type": "image_url", "image_url": {"url": "x"}}], True),
        ([{"type": "input_audio", "input_audio": {"data": "x"}}], True),
        ([{"type": "image", "text": "caption"}], True),
        ([{"type": "output_text", "output_text": "Final text."}], False),
        ([{"type": "output_text", "text": "normal key"}], False),
        ([{"type": "summary_text", "summary_text": "Summary."}], False),
        ([{"type": "input_text", "input_text": "Typed."}], False),
        (["plain part"], False),
        ({"type": "text", "text": ""}, False),
        ("[]", False),
    ]
    assert [(content, LCMEngine._host_blank_text(content)) for content, _blank in cases] == cases
    stored = [normalize_content_value([]), normalize_content_value([{"type": "text", "text": ""}]),
              normalize_content_value([{"type": "output_text", "output_text": "Final text."}])]
    assert [LCMEngine._host_blank_text(value, stored=True) for value in stored] == [True, True, False]


# R7a, D5: a finalized assistant row whose text sits in an output_text field has visible text;
# it is stored once as a version beside the working text, and a replay adds nothing.
def test_finalized_output_text_part_is_stored_as_a_version(tmp_path):
    engine = bind(tmp_path)
    working = msg("assistant", "Working.", T, "1" * 32)
    final = dict(working, content=[{"type": "output_text", "output_text": "Final text."}])
    try:
        engine.handle_tool_call("lcm_status", {}, messages=[deepcopy(working)])
        engine.ingest([deepcopy(final)])
        engine.ingest([deepcopy(final)])
        got = view(engine)
    finally:
        engine.shutdown()
    assert [uid for _content, uid in got] == ["1" * 32, "1" * 32]
    assert got[0][0] == "Working." and "Final text." in got[1][0]



# R8: the older rule does not reuse a row matched by id.
def test_older_rule_does_not_reuse_a_row_matched_by_id(tmp_path):
    engine = bind(tmp_path)
    try:
        engine.ingest([msg("user", "Check the build", T, "u-1"), msg("assistant", "Started.", T + 1, "a-1")])
    finally:
        engine.shutdown()
    fresh = bind(tmp_path)
    try:
        # Four rows, so the older rule reads the store (its multiset path).
        incoming = [msg("user", "Check the build, please", T, "u-1"), msg("assistant", "Started.", T + 1, "a-1"),
                    msg("user", "Check the build", T), msg("assistant", "Started again.", T + 3)]
        fresh.handle_tool_call("lcm_status", {}, messages=deepcopy(incoming))
        assert "multiset" in fresh._last_ingest_reconciliation["reason"], fresh._last_ingest_reconciliation
        assert rows(fresh)[2:] == [("user", "Check the build", float(T)), ("assistant", "Started again.", T + 3.0)]
    finally:
        fresh.shutdown()


# R8: a row decided by id keeps its place for the older rule and is no neighbor, so a row
# without an id is not proven by a neighbor that only a removed row made adjacent.
def test_row_decided_by_id_keeps_its_place_for_the_older_rule(tmp_path):
    engine = bind(tmp_path)
    try:
        start = msg("user", "start", T)
        barrier = msg("assistant", "intervening", T + 1, "x")
        echo = msg("assistant", "echo", None)
        end = msg("user", "end", T + 2)
        engine.ingest([start, barrier, echo, end])
        engine._ingest_held_message_uids = frozenset()
        control = engine._multiset_persisted_indices([start, echo, barrier, end])[0]
        engine._ingest_held_message_uids = frozenset(["x"])
        decided = engine._multiset_persisted_indices([start, echo, barrier, end])[0]
        assert control == [1]
        assert decided == [1]
    finally:
        engine.shutdown()


# R9: scaffold rows stay out of the store even with a host id.
def test_summary_row_with_a_host_id_is_not_stored(tmp_path):
    history = followups("sum", T)
    engine = bind(tmp_path)
    try:
        engine.ingest(deepcopy(history))
        kept = handoff(engine, history)
        count = len(rows(engine))
        engine.ingest(deepcopy(kept) + [msg("user", "Next", T + 900, "u-n")])
        assert rows(engine)[count:] == [("user", "Next", T + 900.0)]
    finally:
        engine.shutdown()


class _DropPattern:
    """An ignore pattern without the optional regex dependency: drops text that starts with DROP:."""

    pattern = "^DROP:"

    def __init__(self):
        self.seen = []

    def search(self, text, timeout=None):
        self.seen.append(str(text))
        return object() if str(text).startswith("DROP:") else None


# R9: the ignore filter judges a row with a new id below the cursor in its own form.
def test_ignored_row_with_a_new_id_below_the_cursor_is_not_stored(tmp_path):
    engine = bind(tmp_path)
    pattern = _DropPattern()
    engine._compiled_ignore_message_patterns = [pattern]
    try:
        engine.ingest([msg("user", "ordinary", T, "a"), msg("assistant", "ok", T + 1, "b")])
        pattern.seen.clear()
        engine.ingest([msg("user", "DROP: private", T + 2, "c")])
        assert "DROP: private" in pattern.seen
        engine.ingest([msg("user", "ordinary", T, "a"), msg("assistant", "kept", T + 3, "d")])
        assert [content for _role, content, _at in rows(engine)] == ["ordinary", "ok", "kept"]
    finally:
        engine.shutdown()


# R6: a row returned to the host keeps the host identity of its incoming row, never the id of a
# cached row with the same text.
def test_returned_rows_keep_their_incoming_ids(tmp_path):
    engine = bind(tmp_path)
    try:
        first = msg("user", "Same", T, "1" * 32)
        second = dict(first, message_uid="2" * 32)
        engine.ingest([first, second])
        out = engine.compress([deepcopy(second)], current_tokens=3, force=True)
        assert out[-1].get("message_uid") == "2" * 32
        pending = msg("user", "q", T + 10)
        engine.should_compress_preflight([deepcopy(pending)])
        out = engine.compress([dict(pending, message_uid="u")], current_tokens=2, force=True)
        assert out[-1].get("message_uid") == "u"
    finally:
        engine.shutdown()


# R10: the status record counts rows decided by id and by the older rule.
def test_status_counts_rows_decided_by_id_and_by_rule(tmp_path):
    engine = bind(tmp_path)
    try:
        engine.ingest([msg("user", "Check", T, "u-1"), msg("assistant", "Started.", T + 1)])
        record = engine.get_status()["ingest_reconciliation"]
        assert (record["rows_decided_by_id"], record["rows_decided_by_rule"]) == (1, 1)
    finally:
        engine.shutdown()

# R10: a row with an id below the scan start is counted only once its ignore decision is known.
# A held row changed below the cursor into an ignored form is not counted, also when the next list
# repeats it unchanged; a change into another kept form is counted. The archive keeps the first text.
def test_counters_skip_an_ignored_observation_below_the_cursor(tmp_path):
    engine = bind(tmp_path)
    engine._compiled_ignore_message_patterns = [_DropPattern()]
    row = msg("user", "Safe text", T, "1" * 32)
    counts = []
    try:
        for content in ("Safe text", "DROP: excluded", "DROP: excluded", "Other kept text"):
            engine.handle_tool_call("lcm_status", {}, messages=[dict(row, content=content)])
            counts.append(engine.get_status()["ingest_reconciliation"]["rows_decided_by_id"])
        got = view(engine)
    finally:
        engine.shutdown()
    assert counts == [1, 1, 1, 2]
    assert got == [("Safe text", "1" * 32)]


# R10, R11: a late close counts its rows with ids after the ignore filters judged them; an
# ignored row is neither counted nor stored.
def test_counters_of_a_late_close_skip_ignored_rows(tmp_path):
    engine = bind(tmp_path)
    engine._compiled_ignore_message_patterns = [_DropPattern()]
    opener = msg("user", "Opener", T, "1" * 32)
    try:
        engine.ingest([opener])
        engine.on_session_start("B", platform="cli", conversation_id="B-lane")
        before = engine.get_status()["ingest_reconciliation"]["rows_decided_by_id"]
        engine.on_session_end(S, [deepcopy(opener), msg("assistant", "Kept answer", T + 1, "2" * 32),
                                  msg("user", "DROP: excluded", T + 2, "3" * 32)])
        after = engine.get_status()["ingest_reconciliation"]["rows_decided_by_id"]
        got = view(engine)
    finally:
        engine.shutdown()
    assert after - before == 2
    assert got == [("Opener", "1" * 32), ("Kept answer", "2" * 32)]


# R10: the id rule keeps the scan cost of the older rule: the rows of an unchanged prefix with
# ids are not matched again; only the new row is.
def test_unchanged_prefix_with_ids_is_not_matched_again(tmp_path):
    engine = bind(tmp_path)
    pattern = _DropPattern()
    engine._compiled_ignore_message_patterns = [pattern]
    history = [msg("user", f"old message {n}", T + n, f"u-{n}") for n in range(50)]
    try:
        engine.ingest(deepcopy(history))
        pattern.seen.clear()
        engine.ingest(deepcopy(history) + [msg("user", "new message", T + 50, "u-50")])
        seen, count = list(pattern.seen), len(rows(engine))
    finally:
        engine.shutdown()
    assert seen == ["new message"]
    assert count == 51



# Gate: two first openers of one store end with one column, one index and one mark; a first
# open under another writer's lock fails with "locked" and the next open succeeds.
def test_first_open_concurrency_and_lock(tmp_path):
    path = tmp_path / "openers.db"
    barrier = threading.Barrier(2)
    found = []

    def opener():
        barrier.wait()
        store = MessageStore(path)
        try:
            found.append((
                store._conn.execute("SELECT count(*) FROM sqlite_master WHERE name = 'idx_msg_message_uid'").fetchone()[0],
                store.message_uid_first_open_store_id(),
            ))
        finally:
            store.close()

    threads = [threading.Thread(target=opener) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert found == [(1, 0), (1, 0)]
    locked = tmp_path / "locked.db"
    keeper = sqlite3.connect(locked)
    keeper.execute("BEGIN EXCLUSIVE")
    began = time.monotonic()
    try:
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            MessageStore(locked)
        assert time.monotonic() - began < 30
    finally:
        keeper.rollback()
        keeper.close()
    store = MessageStore(locked)
    try:
        assert "message_uid" in {row[1] for row in store._conn.execute("PRAGMA table_info(messages)")}
    finally:
        store.close()
