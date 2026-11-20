"""A row is a copy only when the store shows it: one case for each part of the rule.

When the position search fails, ``_multiset_persisted_indices`` compares the
incoming list with the stored rows of the session. A row that nothing in the
store shows as a copy is stored, so a doubt ends in a stored row and never in a
lost message.

The answers of THREE_TURNS carry no host timestamp. A list that ends with such
an answer gets that answer stored again at a position search, because nothing
shows that it is not a new answer with the same text. LAST_ANSWER names that
row where a case expects it.
"""
import base64
import json

import pytest

from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine
import hermes_lcm.engine as engine_module

S = "rule-case-session"
SUMMARY = {
    "role": "user",
    "content": "[Recent Summary (d0, node 7)]\n# Session summary: synthetic\n[Expand for details: lcm_expand(node_id=7)]",
}


def _engine(tmp_path, session_id=S, **config_overrides):
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        large_output_externalization_path=str(tmp_path / "ext"),
    )
    for key, value in config_overrides.items():
        setattr(config, key, value)
    engine = LCMEngine(config=config, hermes_home=str(tmp_path / "home"))
    engine.on_session_start(session_id, platform="cli", conversation_id="rule-case-conversation", context_length=200_000)
    return engine


def _copy(messages):
    return json.loads(json.dumps(messages))


def _user(text, at=None):
    message = {"role": "user", "content": text}
    if at is not None:
        message["timestamp"] = float(at)
    return message


def _answer(text):
    return {"role": "assistant", "content": text}


def _call(call_id, arguments="{}"):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": "probe", "arguments": arguments}}],
    }


def _result(call_id, text):
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def _turn(n, user_text, at=None, answer=None):
    return [_user(user_text, at), _call(f"call_{n}", json.dumps({"n": n})), _result(f"call_{n}", f"result {n}"), _answer(answer or f"answer {n}")]


def _stored(tmp_path):
    engine = _engine(tmp_path)
    try:
        return [
            (row["role"], row["content"], row.get("observed_at"))
            for row in engine._store.get_session_messages(S, limit=10_000)
        ]
    finally:
        engine.shutdown()


def _replay(tmp_path, stored, incoming, **config_overrides):
    """Store ``stored``, bind a new engine, ingest ``incoming``. Return the rows that the new engine added."""
    engine = _engine(tmp_path, **config_overrides)
    try:
        engine.ingest(_copy(stored))
        before = engine._store.get_session_count(S)
        assert before == len(stored)
    finally:
        engine.shutdown()
    engine = _engine(tmp_path, **config_overrides)
    try:
        engine.ingest(_copy(incoming))
        rows = engine._store.get_session_messages(S, limit=10_000)[before:]
        return [(row["role"], row["content"], row.get("observed_at")) for row in rows]
    finally:
        engine.shutdown()


THREE_TURNS = _turn(1, "first request", 100) + _turn(2, "second request", 200) + _turn(3, "third request", 300)
LAST_ANSWER = ("assistant", "answer 3", None)
THANKS = _user("thanks", 900)


def _stamped(message, at):
    return dict(message, timestamp=float(at))


def test_full_replay_stores_nothing(tmp_path):
    assert _replay(tmp_path, THREE_TURNS, THREE_TURNS) == []


def test_replay_behind_a_summary_stores_nothing(tmp_path):
    stored = THREE_TURNS + [THANKS]
    assert _replay(tmp_path, stored, [SUMMARY] + stored[4:]) == []


def test_last_answer_with_a_stored_host_timestamp_is_a_copy(tmp_path):
    # A host that loads its history from its session database puts a timestamp on every message.
    stored = THREE_TURNS[:-1] + [_stamped(_answer("answer 3"), 305)]
    assert _replay(tmp_path, stored, [SUMMARY] + stored[4:]) == []


def test_user_copy_with_stored_time_and_text_is_a_copy(tmp_path):
    # The last stored row is a user message, so no stored neighbor follows it. Its timestamp and text show the copy.
    stored = THREE_TURNS + [_user("one more thing", 400)]
    assert _replay(tmp_path, stored, [SUMMARY] + stored[4:]) == []


def test_text_after_a_proven_copy_is_a_copy(tmp_path):
    # The store holds "answer 3" before a note. The list holds it before another row, so its only stored
    # neighbor pair is the one with the row before it, the result of call_3.
    stored = THREE_TURNS + [_user("a note without a timestamp"), THANKS]
    assert _replay(tmp_path, stored, [SUMMARY] + THREE_TURNS[4:] + [THANKS]) == []


def test_text_before_a_proven_copy_is_a_copy(tmp_path):
    # The list starts with "answer 2". Its only identified neighbor is the next row, a stamped user message.
    stored = THREE_TURNS + [THANKS]
    assert _replay(tmp_path, stored, [SUMMARY] + stored[7:]) == []


def test_old_result_with_other_content_is_a_copy(tmp_path):
    # A host can change the text of an old tool result. The result sits before a stored user row, so its id
    # and its call show the copy. The store keeps the first form.
    stored = THREE_TURNS + [THANKS]
    incoming = [SUMMARY] + stored[4:6] + [_result("call_2", "result 2 in another form")] + stored[7:]
    assert _replay(tmp_path, stored, incoming) == []


def test_last_result_with_a_stored_host_timestamp_and_other_content_is_a_copy(tmp_path):
    # The result sits after the last stored user row. Its id and its host timestamp are stored.
    stored = THREE_TURNS + [_user("fourth request", 400), _call("call_4"), _stamped(_result("call_4", "first form"), 450)]
    incoming = [SUMMARY] + stored[4:14] + [_stamped(_result("call_4", "another form"), 450)]
    assert _replay(tmp_path, stored, incoming) == []


def test_last_result_in_a_redacted_form_is_a_copy(tmp_path):
    secret = 'loaded\napi_key = "invented-key-0123456789abcdef"\npassword=hunter2hunter2hunter2'
    stored = THREE_TURNS + [_user("show the configuration", 400), _call("call_s"), _result("call_s", secret)]
    added = _replay(
        tmp_path, stored, [SUMMARY] + stored[4:],
        sensitive_patterns_enabled=True, sensitive_patterns=["api_key", "bearer_token", "password_assignment"],
    )
    assert added == []


def test_last_result_with_a_large_payload_is_a_copy(tmp_path):
    payload = "data:image/png;base64," + base64.b64encode(b"rule case payload " * 900).decode("ascii")
    content = json.dumps([{"type": "text", "text": "image loaded"}, {"type": "image_url", "image_url": {"url": payload}}])
    stored = THREE_TURNS + [_user("look at this", 400), _call("call_img"), _result("call_img", content)]
    assert _replay(tmp_path, stored, [SUMMARY] + stored[4:]) == []


def test_same_call_with_a_redacted_argument_is_a_copy(tmp_path):
    secret_call = _call("call_s", json.dumps({"api_key": "invented-key-0123456789abcdef"}))
    stored = THREE_TURNS + [_user("deploy it", 400), secret_call, _result("call_s", "deployed"), _answer("done deploying"), THANKS]
    # The summary in front forces the comparison with the stored rows.
    added = _replay(
        tmp_path, stored, [SUMMARY] + stored[4:],
        sensitive_patterns_enabled=True, sensitive_patterns=["api_key", "bearer_token", "password_assignment"],
    )
    assert added == []


def test_same_call_with_a_large_payload_argument_is_a_copy(tmp_path):
    payload = "data:image/png;base64," + base64.b64encode(b"rule case payload " * 900).decode("ascii")
    big_call = _call("call_big", json.dumps({"image": payload}))
    stored = THREE_TURNS + [_user("look at this", 400), big_call, _result("call_big", "seen"), _answer("looked"), THANKS]
    assert _replay(tmp_path, stored, [SUMMARY] + stored[4:]) == []


def test_second_message_with_the_same_text_is_new(tmp_path):
    stored = _turn(1, "Approved", 100) + _turn(2, "second request", 200) + _turn(3, "third request", 300)
    incoming = [SUMMARY] + stored[4:] + [_user("Approved", 400)]
    assert _replay(tmp_path, stored, incoming) == [LAST_ANSWER, ("user", "Approved", 400.0)]


def test_tail_answer_with_the_text_of_an_older_answer_is_new(tmp_path):
    stored = _turn(1, "first request", 100, answer="Done.") + _turn(2, "second request", 200) + _turn(3, "third request", 300)
    incoming = [SUMMARY] + stored[4:] + [_user("fourth request", 400), _call("call_4"), _result("call_4", "result 4"), _answer("Done.")]
    added = _replay(tmp_path, stored, incoming)
    assert added[0] == LAST_ANSWER
    assert [row[0] for row in added[1:]] == ["user", "assistant", "tool", "assistant"]
    assert added[-1][:2] == ("assistant", "Done.")


def test_same_timestamp_with_another_text_is_new(tmp_path):
    stored = THREE_TURNS + [_user("note two", 400)]
    incoming = THREE_TURNS + [_user("note one", 400), _user("note two", 400)]
    assert _replay(tmp_path, stored, incoming) == [("user", "note one", 400.0)]


def test_text_only_conversation_after_a_summary_keeps_the_new_turn(tmp_path):
    stored = [message for n in range(6) for message in (_user(f"question {n}", 100 + n), _answer(f"answer {n}"))]
    incoming = [SUMMARY] + stored[-6:] + [_user("question 0", 200), _answer("a new answer")]
    assert _replay(tmp_path, stored, incoming) == [
        ("assistant", "answer 5", None), ("user", "question 0", 200.0), ("assistant", "a new answer", None),
    ]


def test_new_message_before_a_stored_call_is_new(tmp_path):
    incoming = [SUMMARY] + THREE_TURNS[4:8] + [_user("first request", 500)] + THREE_TURNS[8:]
    assert _replay(tmp_path, THREE_TURNS, incoming) == [("user", "first request", 500.0), LAST_ANSWER]


def test_new_call_that_reuses_a_provider_id_is_new(tmp_path):
    # A provider can number the tool calls of each turn from zero.
    new_turn = [_user("first request", 400), _call("call_1", json.dumps({"n": 99})), _result("call_1", "another result"), _answer("answer 4")]
    added = _replay(tmp_path, THREE_TURNS, [SUMMARY] + THREE_TURNS[4:] + new_turn)
    assert added[0] == LAST_ANSWER
    assert [row[0] for row in added[1:]] == ["user", "assistant", "tool", "assistant"]
    assert added[3][:2] == ("tool", "another result")


def test_provider_with_one_id_for_every_call_loses_no_turn(tmp_path):
    # A provider can give every tool call the same id. The new call repeats the call of the first turn,
    # which left the list. The store holds the id with three different calls, so the id shows nothing.
    def fixed_turn(n, text, at, arguments_n=None, result=None):
        arguments = json.dumps({"n": n if arguments_n is None else arguments_n})
        return [_user(text, at), _call("call_fixed", arguments), _result("call_fixed", result or f"result {n}"), _answer(f"answer {n}")]

    stored = fixed_turn(1, "first request", 100) + fixed_turn(2, "second request", 200) + fixed_turn(3, "third request", 300)
    new_turn = fixed_turn(4, "fourth request", 400, arguments_n=1, result="another result")
    added = _replay(tmp_path, stored, [SUMMARY] + stored[4:] + new_turn)
    assert [row[:2] for row in added[-4:]] == [
        ("user", "fourth request"), ("assistant", ""), ("tool", "another result"), ("assistant", "answer 4"),
    ]


def test_new_call_with_other_arguments_and_the_same_result_is_new(tmp_path):
    # No user row carries a timestamp and the result has the content of the old result.
    # Only the arguments tell the new call from the old call with the same id.
    stored = _turn(1, "first request") + _turn(2, "second request") + _turn(3, "third request")
    new_turn = [_user("first request again"), _call("call_1", json.dumps({"n": 99})), _result("call_1", "result 1"), _answer("answer 4")]
    added = _replay(tmp_path, stored, [SUMMARY] + stored[4:] + new_turn)
    assert [row[:2] for row in added[-4:]] == [
        ("user", "first request again"), ("assistant", ""), ("tool", "result 1"), ("assistant", "answer 4"),
    ]


def test_provider_with_one_id_for_every_call_keeps_a_run_that_repeats_an_old_one(tmp_path):
    # No user row carries a timestamp. The new run has the id, the arguments and the result of the first run.
    # The store holds the id with three different calls, so the id shows nothing.
    def fixed_turn(n, text):
        return [_user(text), _call("call_fixed", json.dumps({"n": n})), _result("call_fixed", f"result {n}"), _answer(f"answer {n}")]

    stored = fixed_turn(1, "first request") + fixed_turn(2, "second request") + fixed_turn(3, "third request")
    new_turn = [_user("first request again"), _call("call_fixed", json.dumps({"n": 1})), _result("call_fixed", "result 1"), _answer("answer 4")]
    added = _replay(tmp_path, stored, [SUMMARY] + stored[4:] + new_turn)
    assert [row[:2] for row in added[-4:]] == [
        ("user", "first request again"), ("assistant", ""), ("tool", "result 1"), ("assistant", "answer 4"),
    ]


def _poll(call_id, reading, alias=None):
    """One run of a status tool. A provider can give every run the same id, name and arguments."""
    call = _call(call_id, json.dumps({"station": "east"}))
    call["tool_calls"][0]["function"]["name"] = "read_status"
    if alias:
        call["tool_calls"][0]["call_id"] = alias
    return [call, _result(alias or call_id, reading)]


def test_second_run_of_the_same_call_keeps_its_new_result(tmp_path):
    # No user row carries a timestamp, so only the content of the result tells the second run from the first.
    stored = [_user("check the station"), *_poll("call_constant", "sample one"), _answer("first reading")] + _turn(2, "second request") + _turn(3, "third request")
    second_run = [_user("check it again"), *_poll("call_constant", "sample two"), _answer("second reading")]
    added = _replay(tmp_path, stored, [SUMMARY] + stored[4:] + second_run)
    assert [row[:2] for row in added[-4:]] == [
        ("user", "check it again"), ("assistant", ""), ("tool", "sample two"), ("assistant", "second reading"),
    ]


def test_second_run_in_the_open_turn_keeps_its_new_result(tmp_path):
    # Both runs belong to one turn. The first run left the list, and the store holds it after the request.
    # Only the content of the result tells the second run from a copy of the first.
    stored = THREE_TURNS + [_user("watch the station", 400), *_poll("call_constant", "sample one"), _answer("still waiting")]
    second_run = [*_poll("call_constant", "sample two"), _answer("second reading")]
    added = _replay(tmp_path, stored, [SUMMARY] + stored[8:13] + second_run)
    assert [row[:2] for row in added] == [("assistant", ""), ("tool", "sample two"), ("assistant", "second reading")]


def test_open_turn_that_is_stored_is_a_copy(tmp_path):
    # The same turn with nothing new: the call and its result are stored after the request, with the same content.
    stored = THREE_TURNS + [_user("watch the station", 400), *_poll("call_constant", "sample one")]
    assert _replay(tmp_path, stored, [SUMMARY] + stored[8:]) == []


def test_second_run_with_the_same_result_after_a_stored_request_is_new(tmp_path):
    # The request for the second run is stored. The store holds the call and its result only before that request.
    stored = (
        [_user("check the station", 50), *_poll("call_constant", "sample one"), _answer("first reading")]
        + THREE_TURNS + [_user("check it again", 400)]
    )
    second_run = [*_poll("call_constant", "sample one"), _answer("the same reading")]
    added = _replay(tmp_path, stored, [SUMMARY] + stored[8:] + second_run)
    assert [row[:2] for row in added] == [("assistant", ""), ("tool", "sample one"), ("assistant", "the same reading")]


def test_second_run_with_the_same_result_after_a_new_request_is_new(tmp_path):
    # The store holds the first run after the last stored request. A new request follows, then a second run
    # with the same id, arguments and result. The first run left the list. Rows after a new request are new.
    stored = THREE_TURNS[:8] + [_user("watch the station", 300), *_poll("call_constant", "sample one"), _answer("first reading")]
    second_run = [_user("check it again", 400), *_poll("call_constant", "sample one"), _answer("the same reading")]
    added = _replay(tmp_path, stored, [SUMMARY] + stored[4:9] + second_run)
    assert [row[:2] for row in added] == [
        ("user", "check it again"), ("assistant", ""), ("tool", "sample one"), ("assistant", "the same reading"),
    ]


def test_result_that_answers_another_id_of_a_new_call_is_new(tmp_path):
    # A call can carry two ids. The result answers the second one, which an older call also carried.
    stored = [_user("check the station"), *_poll("fc_first", "sample one", alias="call_shared"), _answer("first reading")] + _turn(2, "second request") + _turn(3, "third request")
    new_call = _poll("fc_second", "sample one", alias="call_shared")
    new_call[0]["tool_calls"][0]["function"]["arguments"] = json.dumps({"station": "west"})
    added = _replay(tmp_path, stored, [SUMMARY] + stored[4:] + [_user("check the other one"), *new_call, _answer("second reading")])
    assert [row[:2] for row in added[-4:]] == [
        ("user", "check the other one"), ("assistant", ""), ("tool", "sample one"), ("assistant", "second reading"),
    ]


def test_second_equal_text_beside_one_stored_text_is_new(tmp_path):
    # One stored "answer 2" sits between two proven rows. It shows one incoming row, not two.
    stored = THREE_TURNS + [THANKS]
    incoming = [SUMMARY] + stored[4:8] + [_answer("answer 2")] + stored[8:]
    assert _replay(tmp_path, stored, incoming) == [("assistant", "answer 2", None)]


def _ingest_with_new_engine(tmp_path, messages):
    engine = _engine(tmp_path)
    try:
        engine.ingest(_copy(messages))
    finally:
        engine.shutdown()


def test_new_answer_with_the_text_of_an_old_answer_that_was_stored_again(tmp_path):
    # A position search with nothing new stores an old text again, directly after the last stored row.
    # The answer of the open turn then arrives with that text. It is new.
    history = [
        _user("first request", 100), _call("call_1"), _result("call_1", "result 1"), _answer("answer 1"),
        _answer("Done."), _user("a note without a timestamp"), _answer("fine"),
        _user("go", 300), _call("call_3"), _result("call_3", "result 3"),
    ]
    _ingest_with_new_engine(tmp_path, history)
    _ingest_with_new_engine(tmp_path, [SUMMARY] + history[1:])
    _ingest_with_new_engine(tmp_path, [SUMMARY] + history[1:] + [_answer("Done.")])
    assert _stored(tmp_path)[-1] == ("assistant", "Done.", None)


def test_new_answer_after_an_inserted_row_left_the_list(tmp_path):
    # A new text row in the middle of the list is stored directly after the last stored row. It then leaves
    # the list, and the answer of the open turn arrives with the same text. The store holds both.
    history = _turn(1, "first request", 100) + _turn(2, "second request", 200) + [_user("go", 300), _call("call_3"), _result("call_3", "result 3")]
    _ingest_with_new_engine(tmp_path, history)
    _ingest_with_new_engine(tmp_path, [SUMMARY] + history[:4] + [_answer("Done.")] + history[4:])
    _ingest_with_new_engine(tmp_path, [SUMMARY] + history[4:] + [_answer("Done.")])
    assert sum(1 for row in _stored(tmp_path) if row[:2] == ("assistant", "Done.")) == 2


def test_last_answer_without_proof_is_stored_again(tmp_path):
    # The price of the rule. The list ends with "answer 3". The store holds it after the result of call_3,
    # but the store holds the same rows when a new answer with that text arrives, so nothing shows the copy.
    assert _replay(tmp_path, THREE_TURNS, [SUMMARY] + THREE_TURNS[4:]) == [LAST_ANSWER]


def test_last_answer_is_stored_again_once_when_the_host_adds_its_timestamp_later(tmp_path):
    # LCM stored the answer during the turn, without a host timestamp. The host then loads its history from
    # its session database, where the answer carries a timestamp. The first position search stores the answer
    # again, with the timestamp. Each later search finds that timestamp in the store.
    reloaded = [SUMMARY] + THREE_TURNS[4:-1] + [_stamped(_answer("answer 3"), 305)]
    _ingest_with_new_engine(tmp_path, THREE_TURNS)
    _ingest_with_new_engine(tmp_path, reloaded)
    assert _stored(tmp_path)[len(THREE_TURNS):] == [("assistant", "answer 3", 305.0)]
    _ingest_with_new_engine(tmp_path, reloaded)
    assert len(_stored(tmp_path)) == len(THREE_TURNS) + 1


def test_user_row_with_a_host_note_is_stored_and_its_turn_is_not(tmp_path):
    # A host can show one user message with a note added or removed. It keeps the timestamp.
    # The other form of the row is stored. The tool rows of its turn are copies.
    stored = THREE_TURNS + [_user("fourth request", 400), _call("call_4"), _result("call_4", "result 4")]
    with_note = _user("fourth request\n\n[a note that the host added]", 400)
    assert _replay(tmp_path, stored, [SUMMARY] + stored[4:12] + [with_note] + stored[13:]) == [
        ("user", "fourth request\n\n[a note that the host added]", 400.0),
    ]


def test_user_copy_with_a_new_timestamp_is_stored_again(tmp_path):
    # The host stamped a copy again. It cannot be told from a second message with that text.
    stored = THREE_TURNS + [_user("one more thing", 400)]
    incoming = [SUMMARY] + THREE_TURNS[4:] + [_user("one more thing", 999)]
    assert _replay(tmp_path, stored, incoming) == [LAST_ANSWER, ("user", "one more thing", 999.0)]


def test_text_without_a_proven_neighbor_is_stored_again(tmp_path):
    # A host that does not stamp user messages. A turn without a tool call then has no identified
    # neighbor: its question and its answer are stored again. Nothing is lost.
    stored = (
        _turn(1, "first request") + _turn(2, "second request")
        + [_user("plain question"), _answer("plain answer")] + _turn(3, "third request")
    )
    added = _replay(tmp_path, stored, [SUMMARY] + stored[4:])
    assert added == [("user", "plain question", None), ("assistant", "plain answer", None), LAST_ANSWER]


def _long_conversation():
    return [
        message
        for n in range(30)
        for message in (_user(f"question {n} " + "w " * 150, 100 + n), _answer(f"answer {n} " + "r " * 150))
    ]


@pytest.fixture
def compacted(tmp_path, monkeypatch):
    monkeypatch.setattr(
        engine_module, "summarize_with_escalation",
        lambda **_kwargs: ("synthetic summary.\nExpand for details about: older work", 1),
    )
    engine = _engine(tmp_path, fresh_tail_count=8, leaf_chunk_tokens=150)
    engine.context_length = 200_000
    engine.threshold_tokens = 170_000
    before = _long_conversation()
    try:
        engine.ingest(_copy(before))
        kept = engine.compress(_copy(before), current_tokens=160_000, force=True)
        assert 1 < len(kept) < len(before)
        yield engine, before, kept
    finally:
        engine.shutdown()


def _count(engine):
    return engine._store._conn.execute("SELECT count(*) FROM messages").fetchone()[0]


def test_flush_keeps_the_position_and_stores_what_is_missing(compacted):
    engine, before, _kept = compacted
    position = (engine._ingest_cursor, engine._ingest_cursor_needs_reconcile)
    rows = _count(engine)
    engine.on_session_end(S, _copy(before) + [_user("late row", 900)])
    assert (engine._ingest_cursor, engine._ingest_cursor_needs_reconcile) == position
    assert _count(engine) == rows + 1


@pytest.mark.parametrize("boundary", ["no_notice", "same_id_notice", "new_session_id"])
@pytest.mark.parametrize("host_list", ["as_returned", "last_row_removed", "row_inserted"])
def test_first_new_message_after_a_compaction_is_stored(compacted, boundary, host_list):
    engine, before, kept = compacted
    rows = _count(engine)
    engine.on_session_end(S, _copy(before))
    assert _count(engine) == rows
    session_id = S
    if boundary == "same_id_notice":
        engine.on_session_start(S, boundary_reason="compression", old_session_id=S, platform="cli", conversation_id="rule-case-conversation")
    elif boundary == "new_session_id":
        session_id = S + "-next"
        engine.on_session_start(session_id, boundary_reason="compression", old_session_id=S, platform="cli", conversation_id="rule-case-conversation")
    committed = _copy(kept)
    if host_list == "last_row_removed":
        committed = committed[:-1]
    elif host_list == "row_inserted":
        committed.insert(len(committed) - 2, _user("host note", 950))
    new_message = _user("brand new after the compaction", 1000)
    engine.ingest(committed + [new_message])
    stored = engine._store._conn.execute(
        "SELECT session_id, content, observed_at FROM messages WHERE observed_at = 1000.0"
    ).fetchall()
    assert stored == [(session_id, "brand new after the compaction", 1000.0)]
    # After a compaction notice, and after a list that the host changed, the stored rows decide what is new.
    # The last kept answer is then stored again, because it has no host timestamp.
    if host_list == "as_returned":
        assert _count(engine) == rows + (2 if boundary == "same_id_notice" else 1)
    elif host_list == "row_inserted" and boundary != "new_session_id":
        # The inserted host row, the new message, and the last kept answer. No other kept row is stored again.
        assert _count(engine) == rows + 3
