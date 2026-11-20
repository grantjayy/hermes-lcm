"""Loss cases that two plan reviews found in earlier designs of the repeated-row fix.

Each case drives a real engine through public host callbacks (with a stubbed
summarizer) and asserts that a new message reaches the store.
"""
from contextlib import closing
from copy import deepcopy
import sqlite3

import pytest

import hermes_lcm.engine as engine_module
from tests.test_repeat_rows_rule import (
    S,
    SUMMARY,
    _answer,
    _call,
    _engine,
    _long_conversation,
    _replay,
    _result,
    _user,
)

CONVERSATION = "rule-case-conversation"


def _stub_summarizer(monkeypatch):
    monkeypatch.setattr(
        engine_module, "summarize_with_escalation",
        lambda **_kwargs: ("synthetic summary.\nExpand for details about: older work", 1),
    )


def _compacting_engine(tmp_path):
    engine = _engine(tmp_path, fresh_tail_count=8, leaf_chunk_tokens=150)
    engine.context_length = 200_000
    engine.threshold_tokens = 170_000
    return engine


def _stored_rows(tmp_path):
    """(session_id, role, content, observed_at) of every stored row, in store order."""
    with closing(sqlite3.connect(f"file:{tmp_path / 'lcm.db'}?mode=ro", uri=True)) as conn:
        return list(conn.execute("SELECT session_id, role, content, observed_at FROM messages ORDER BY store_id"))


def _q(n, at):
    return _user(f"q{n}", at)


def _a(n):
    return _answer(f"a{n}")


def _tool_step(n):
    return [_call(f"c{n}"), _result(f"c{n}", f"t{n}")]


@pytest.fixture
def compact(tmp_path, monkeypatch):
    _stub_summarizer(monkeypatch)
    engine = _compacting_engine(tmp_path)
    before = _long_conversation()
    try:
        engine.ingest(deepcopy(before))
        kept = engine.compress(deepcopy(before), current_tokens=160_000, force=True)
        assert 1 < len(kept) < len(before)
        assert engine._ingest_cursor == len(kept)
        assert len(_stored_rows(tmp_path)) == len(before)
        yield engine, before, kept
    finally:
        engine.shutdown()


@pytest.mark.parametrize("continuation", ["short", "long", "rotated", "end"])
def test_end_continuations(tmp_path, compact, continuation):
    engine, before, kept = compact
    position = (engine._ingest_cursor, engine._ingest_cursor_needs_reconcile)
    late = _user("late before flush", 200)
    new = _user("never previously seen", 210)
    engine.on_session_end(S, deepcopy(before + [late]))
    assert (engine._ingest_cursor, engine._ingest_cursor_needs_reconcile) == position
    assert sum(row[3] == 200 for row in _stored_rows(tmp_path)) == 1
    if continuation == "end":
        return
    expected_session = S
    if continuation == "rotated":
        expected_session = S + "-child"
        engine.on_session_start(expected_session, boundary_reason="compression", old_session_id=S, platform="cli", conversation_id=CONVERSATION)
        assert engine._persisted_prefix_len(before) is None
    sequence = (before + [late]) if continuation == "long" else kept
    engine.ingest(deepcopy(sequence + [new]))
    assert (expected_session, "user", new["content"], 210.0) in _stored_rows(tmp_path)
    assert len(_stored_rows(tmp_path)) == len(before) + 2


def test_pending_reconcile_survives_final_flush(tmp_path):
    before = _long_conversation()
    engine = _compacting_engine(tmp_path)
    try:
        engine.ingest(deepcopy(before))
    finally:
        engine.shutdown()
    fresh = _compacting_engine(tmp_path)
    try:
        position = (fresh._ingest_cursor, fresh._ingest_cursor_needs_reconcile)
        assert position == (0, True)
        late = _user("late boundary input", 600)
        next_row = _answer("following real response")
        fresh.on_session_end(S, deepcopy(before + [late]))
        assert (fresh._ingest_cursor, fresh._ingest_cursor_needs_reconcile) == position
        fresh.ingest(deepcopy(before + [late, next_row]))
        added = _stored_rows(tmp_path)[len(before):]
        assert [row[2] for row in added] == [late["content"], next_row["content"]]
    finally:
        fresh.shutdown()


def test_restart_after_real_text_only_compaction(tmp_path, compact):
    engine, before, kept = compact
    assert all(message.get("content") != before[0]["content"] for message in kept)
    engine.on_session_end(S, deepcopy(before))
    engine.on_session_start(S, boundary_reason="compression", old_session_id=S, platform="cli", conversation_id=CONVERSATION)
    fresh = _compacting_engine(tmp_path)
    try:
        new = dict(before[0], timestamp=300.0)
        reply = _answer("new text-only response")
        count = len(_stored_rows(tmp_path))
        fresh.ingest(deepcopy(kept + [new, reply]))
        added = _stored_rows(tmp_path)[count:]
        assert any(row[2] == reply["content"] for row in added)
        assert any(row[3] == 300 for row in added)
    finally:
        fresh.shutdown()


@pytest.mark.parametrize("continuation", ["notice", "no_notice", "rotation"])
def test_shorter_committed_list_after_host_drops_completed_todo(tmp_path, monkeypatch, continuation):
    # The host removes a finished todo snapshot from the list that compress() returned before it
    # continues with that list. Only the host's list changes; no private cursor field is touched.
    _stub_summarizer(monkeypatch)
    engine = _compacting_engine(tmp_path)
    todo = {
        "role": "user",
        "content": "[Your active task list was preserved across context compression]\n- [ ] test item",
        "_todo_snapshot_synthetic": True,
    }
    before = _long_conversation() + [todo]
    try:
        kept = engine.compress(deepcopy(before), current_tokens=160_000, force=True)
        assert len(kept) < len(before)
        assert kept[-1].get("_todo_snapshot_synthetic") and kept[-1]["content"] == todo["content"]
        assert any(row.get("role") == "user" and row.get("timestamp") for row in kept[:-1])
        trimmed = kept[:-1]
        engine.on_session_end(S, deepcopy(before))
        assert engine._ingest_cursor == len(kept)
        if continuation != "no_notice":
            engine.on_session_start(
                S if continuation == "notice" else S + "-child",
                boundary_reason="compression", old_session_id=S, platform="cli", conversation_id=CONVERSATION,
            )
        new = _user("brand new after host trim", 400)
        engine.ingest(deepcopy(trimmed + [new]))
        assert any(row[3] == 400 for row in _stored_rows(tmp_path))
    finally:
        engine.shutdown()


def test_no_anchor_after_text_only_compaction(tmp_path):
    # Text-only turns: a summary, the kept recent turns, and one new turn after a restart.
    old = [message for n in range(6) for message in (_q(n, 100 + n), _a(n))]
    incoming = [SUMMARY] + old[-6:] + [_q(0, 200), _a(9)]
    assert ("user", "q0", 200.0) in _replay(tmp_path, old, incoming)


def test_new_notification_before_a_stored_call(tmp_path):
    # The incoming list puts a new stamped notification in front of tool rows that the store holds.
    old = [_q(0, 100), _a(0), _q(1, 101), *_tool_step(1), _a(1), _q(2, 102), *_tool_step(2), _a(2)]
    incoming = [SUMMARY] + old[2:6] + [_q(0, 200)] + old[6:]
    assert ("user", "q0", 200.0) in _replay(tmp_path, old, incoming)


def test_idless_answer_after_a_new_call_is_not_lost(tmp_path):
    # The store holds an old repeat of the last answer that the host list does not hold. A new
    # answer with that text follows a tool step that the store has never seen.
    old = [_q(0, 100), *_tool_step(0), _a(0), _q(1, 101), *_tool_step(1), _a(1), _a(1)]
    incoming = [SUMMARY] + old[4:8] + [_q(2, 200), *_tool_step(2), _a(1)]
    added = _replay(tmp_path, old, incoming)
    assert ("user", "q2", 200.0) in added
    assert ("tool", "t2", None) in added
    assert ("assistant", "a1", None) in added


def test_pair_multiset_and_collision_control(tmp_path):
    old = [_q(0, 100), *_tool_step(0), _a(0), _q(1, 101)]
    engine = _engine(tmp_path)
    try:
        engine.ingest(deepcopy(old))
        assert engine._store.get_session_count(S) == len(old)
    finally:
        engine.shutdown()
    engine = _engine(tmp_path)
    try:
        incoming = old[:4] + [_q(2, 101), _q(1, 101), _q(1, 101)]
        new, _, _ = engine._multiset_persisted_indices(incoming)
        assert new == [4, 6]
    finally:
        engine.shutdown()


def _invocation(kind, second):
    arguments = '{"station":"west"}' if second and kind != "same_signature" else '{"station":"east"}'
    if kind == "result_alias":
        # A Responses-style call carries two ids, and its result answers the call_id.
        call = _call("fc_second" if second else "fc_first", arguments)
        call["tool_calls"][0]["call_id"] = "call_shared"
        result_id = "call_shared"
    else:
        call = _call("call_constant", arguments)
        result_id = "call_constant"
    call["tool_calls"][0]["function"]["name"] = "read_status"
    return [call, _result(result_id, "sample two" if second else "sample one")]


@pytest.mark.parametrize("handoff", ["same_id_notice", "fresh_bind"])
@pytest.mark.parametrize("kind", ["same_signature", "changed_arguments_control", "result_alias"])
def test_second_poll_result_is_archived(tmp_path, monkeypatch, kind, handoff):
    # A tool invocation is an event, not a function of its id and arguments.
    _stub_summarizer(monkeypatch)
    engine = _compacting_engine(tmp_path)
    try:
        old = [_user("Check the station", 51), *_invocation(kind, False), _answer("first reading")] + _long_conversation()
        engine.ingest(deepcopy(old))
        # The host compacts the list before the turn, including a new stamped user request.
        before = old + [_user("Check the station again", 500)]
        kept = engine.compress(deepcopy(before), current_tokens=160_000, force=True)
        assert 4 <= len(kept) < len(before)
        assert any(message.get("timestamp") == 500 for message in kept)
        assert not any(message.get("tool_calls") or message.get("role") == "tool" for message in kept)
        engine.on_session_end(S, deepcopy(before))
        engine.on_session_start(S, boundary_reason="compression", old_session_id=S, platform="cli", conversation_id=CONVERSATION)
        if handoff == "fresh_bind":
            engine.shutdown()
            engine = _compacting_engine(tmp_path)
        before_count = engine._store.get_session_count(S)
        # The host continues with the compacted list and runs the same tool a second time.
        completed = kept + _invocation(kind, True) + [_answer("second reading")]
        engine.ingest(deepcopy(completed))
        rows = engine._store.get_session_messages(S, limit=10_000)
        assert sum(row["role"] == "tool" and row["content"] == "sample one" for row in rows) == 1
        added = rows[before_count:]
        decision = engine.get_status()["ingest_reconciliation"]
        assert "multiset" in decision["reason"]
        assert any(row["role"] == "assistant" and row["content"] == "second reading" for row in added)
        assert any(row["role"] == "tool" and row["content"] == "sample two" for row in rows)
        assert sum(bool(row["tool_calls"]) for row in added) == 1
    finally:
        engine.shutdown()
