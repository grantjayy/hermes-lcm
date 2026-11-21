"""Host call sequences of one long session keep every message and store few or no rows twice.

A real engine goes through the calls that the Hermes host makes for one long
session, with invented messages and a stubbed summarizer (no model call):

- agent start:       on_session_start(session, platform, conversation_id, ...)
- end of each turn:  ingest(messages), through the post_llm_call hook
- compaction:        compress(messages)
- after compaction:  on_session_end(session, the list from BEFORE the compaction)
- after the commit:  on_session_start(session, boundary_reason="compression",
                                      old_session_id=session)

The sizes (9 turns, 12 tool steps per turn, 3,200 characters per tool result,
compaction at 70 messages) and the config values decide where the compactions
fall, so the bounds below hold for these sizes.

A repeat row has the session, role, tool_call_id, tool_calls and content of an
earlier row of its session.
"""
from contextlib import closing
import json
import sqlite3

import pytest

import hermes_lcm.engine as engine_module
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

TURNS = 9
TOOL_STEPS_PER_TURN = 12
CHARS_PER_TOOL_RESULT = 3200
COMPACT_AT_MESSAGES = 70
CONVERSATION = "conversation:probe"


@pytest.fixture(autouse=True)
def _stub_summarizer(monkeypatch):
    monkeypatch.setattr(
        engine_module, "summarize_with_escalation",
        lambda **_kwargs: ("Summary of earlier probe work.\nExpand for details about: probe work", 1),
    )


def _config(tmp_path):
    return LCMConfig(
        database_path=str(tmp_path / "lcm.db"),
        fresh_tail_count=24,
        leaf_chunk_tokens=8000,
        dynamic_leaf_chunk_enabled=True,
        dynamic_leaf_chunk_max=40000,
    )


def _new_engine(tmp_path, config):
    engine = LCMEngine(config=config, hermes_home=str(tmp_path))
    engine.context_length = 450_000
    engine.threshold_tokens = int(450_000 * 0.88)
    return engine


def _bind_stored_session(engine, tmp_path, session_id):
    engine.on_session_start(
        session_id, hermes_home=str(tmp_path), platform="discord", model="probe-model",
        context_length=450_000, conversation_id=CONVERSATION,
    )


def _stored_rows(tmp_path):
    """(session_id, role, tool_call_id, tool_calls, content) of every stored row, in store order."""
    with closing(sqlite3.connect(f"file:{tmp_path / 'lcm.db'}?mode=ro", uri=True)) as conn:
        return list(conn.execute(
            "SELECT session_id, role, tool_call_id, tool_calls, content FROM messages ORDER BY store_id"
        ))


def _repeat_rows(rows):
    seen: set[tuple] = set()
    repeats = 0
    for row in rows:
        if row in seen:
            repeats += 1
        seen.add(row)
    return repeats


class _Messages:
    """Invented messages. A running number makes every message distinct."""

    def __init__(self):
        self.serial = 0

    def text(self, kind, chars):
        self.serial += 1
        seed = f"{kind} {self.serial:05d} "
        return (seed * (chars // len(seed) + 1))[:chars]

    def tool_step(self):
        call_id = f"call_{self.serial + 1:05d}"
        call = {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": call_id, "type": "function",
                            "function": {"name": "probe_tool", "arguments": json.dumps({"n": self.serial})}}],
        }
        self.serial += 1
        return [call, {"role": "tool", "tool_call_id": call_id, "content": self.text("tool output", CHARS_PER_TOOL_RESULT)}]


def _run_host_sequence(tmp_path, variant, user_time):
    """Drive one session through ``variant``: 9 turns of 26 messages, 234 messages in all."""
    config = _config(tmp_path)
    compact_at = 10 ** 9 if variant == "new_agent_no_compaction" else COMPACT_AT_MESSAGES
    session = {"id": "probe-session-1", "generation": 1}
    source = _Messages()
    messages: list = []
    engine = _new_engine(tmp_path, config)
    try:
        _bind_stored_session(engine, tmp_path, session["id"])
        for turn in range(1, TURNS + 1):
            if variant.startswith("new_agent") and turn > 1:
                # The gateway builds a new agent for the stored session (a restart or a cache eviction).
                engine.shutdown()
                engine = None
                engine = _new_engine(tmp_path, config)
                _bind_stored_session(engine, tmp_path, session["id"])
            user_message: dict[str, object] = {"role": "user", "content": source.text("user request", 300)}
            if user_time:
                user_message["timestamp"] = 1_795_000_000.0 + 60.0 * turn
            messages.append(user_message)
            for _step in range(TOOL_STEPS_PER_TURN):
                messages.extend(source.tool_step())
                if len(messages) < compact_at:
                    continue
                before = list(messages)
                compressed = engine.compress(list(messages), current_tokens=400_000)
                assert len(compressed) < len(before)
                old_id = session["id"]
                if variant == "rotate":
                    # The older host behavior: every compaction starts a new session id.
                    session["generation"] += 1
                    session["id"] = f"probe-session-{session['generation']}"
                if variant in ("host", "no_start", "rotate"):
                    engine.on_session_end(old_id, before)
                if variant in ("host", "no_end", "rotate"):
                    engine.on_session_start(
                        session["id"], boundary_reason="compression", old_session_id=old_id,
                        platform="discord", conversation_id=CONVERSATION,
                    )
                messages = [dict(message) for message in compressed]
            messages.append({"role": "assistant", "content": source.text("final answer", 500)})
            # The post_llm_call hook binds the current session id before it ingests.
            if str(getattr(engine, "_session_id", "") or "") != session["id"]:
                engine.on_session_start(session["id"], platform="discord", conversation_id=CONVERSATION)
            engine.ingest(list(messages))
    finally:
        if engine is not None:
            engine.shutdown()


# Measured on the change: with no host timestamp on an assistant answer, each position search
# stores the last answer of the list again (one row per new agent, and one in "host" and
# "no_end" when user messages carry no timestamp either).
REPEAT_LIMITS = {
    True: {"host": 0, "no_end": 0, "no_start": 0, "neither": 0, "rotate": 0, "new_agent": 6, "new_agent_no_compaction": 0},
    False: {"host": 1, "no_end": 1, "no_start": 0, "neither": 0, "rotate": 0, "new_agent": 7, "new_agent_no_compaction": 0},
}


@pytest.mark.parametrize(
    ("variant", "user_time"),
    [
        pytest.param(variant, user_time, id=f"{variant}-{'user_time' if user_time else 'no_user_time'}")
        for user_time in (True, False)
        for variant in REPEAT_LIMITS[user_time]
    ],
)
def test_host_sequence_keeps_every_message(tmp_path, variant, user_time):
    _run_host_sequence(tmp_path, variant, user_time)
    rows = _stored_rows(tmp_path)
    repeats = _repeat_rows(rows)
    # Every generated message is distinct, so 234 distinct rows means that none was lost.
    assert len(rows) - repeats == 234
    assert repeats <= REPEAT_LIMITS[user_time][variant]


REPEATED_TEXT = "Approved"


@pytest.mark.parametrize("host_time", [True, False], ids=["user_time", "no_user_time"])
@pytest.mark.parametrize("case", ["clean", "inflated"])
def test_new_message_with_repeated_text_reaches_the_store_of_a_new_engine(tmp_path, case, host_time):
    """A user sends "Approved" twice. The first leaves the list in a compaction, then a new agent binds.

    In the "inflated" case the store also holds three old repeat rows of the first
    message, written before the compaction, as live stores hold today.
    """
    config = LCMConfig(
        database_path=str(tmp_path / "lcm.db"), fresh_tail_count=24, leaf_chunk_tokens=8000,
        dynamic_leaf_chunk_enabled=True, dynamic_leaf_chunk_max=40000,
    )
    source = _Messages()
    clock = [1_795_000_000.0]

    def user(content):
        clock[0] += 60.0
        message: dict[str, object] = {"role": "user", "content": content}
        if host_time:
            message["timestamp"] = clock[0]
        return message

    def tool_steps(steps):
        return [message for _ in range(steps) for message in source.tool_step()]

    def answer():
        return {"role": "assistant", "content": source.text("final answer", 400)}

    def rows_with_the_repeated_text():
        with closing(sqlite3.connect(f"file:{tmp_path / 'lcm.db'}?mode=ro", uri=True)) as conn:
            return conn.execute(
                "SELECT count(*) FROM messages WHERE role = 'user' AND content = ?", (REPEATED_TEXT,)
            ).fetchone()[0]

    engine = _new_engine(tmp_path, config)
    try:
        _bind_stored_session(engine, tmp_path, "probe-session")
        first_message = user(REPEATED_TEXT)
        messages: list = [first_message, *tool_steps(12), answer()]
        engine.ingest(list(messages))
        if case == "inflated":
            engine._store.append_batch("probe-session", [dict(first_message) for _ in range(3)])
        for _turn in range(2):
            messages += [user(source.text("user request", 300)), *tool_steps(12), answer()]
            engine.ingest(list(messages))
        messages = [dict(message) for message in engine.compress(list(messages), current_tokens=400_000)]
        assert not any(message.get("content") == REPEATED_TEXT for message in messages)
        engine.on_session_start(
            "probe-session", boundary_reason="compression", old_session_id="probe-session",
            platform="discord", conversation_id=CONVERSATION,
        )
        messages.append(answer())
        engine.ingest(list(messages))
    finally:
        engine.shutdown()

    with_text_before = rows_with_the_repeated_text()
    with closing(sqlite3.connect(f"file:{tmp_path / 'lcm.db'}?mode=ro", uri=True)) as conn:
        last_store_id = conn.execute("SELECT max(store_id) FROM messages").fetchone()[0]

    new_messages = [user(REPEATED_TEXT), *tool_steps(3), answer()]
    engine = _new_engine(tmp_path, config)
    try:
        _bind_stored_session(engine, tmp_path, "probe-session")
        engine.ingest(list(messages + new_messages))
    finally:
        engine.shutdown()

    with closing(sqlite3.connect(f"file:{tmp_path / 'lcm.db'}?mode=ro", uri=True)) as conn:
        added = list(conn.execute(
            "SELECT role, content, tool_call_id, tool_calls FROM messages WHERE store_id > ? ORDER BY store_id",
            (last_store_id,),
        ))
    assert rows_with_the_repeated_text() == with_text_before + 1
    for message in new_messages:
        if message.get("tool_calls"):
            call_id = message["tool_calls"][0]["id"]
            assert any(
                role == "assistant" and tool_calls and call_id in [call["id"] for call in json.loads(tool_calls)]
                for role, _content, _tool_call_id, tool_calls in added
            ), call_id
        else:
            assert any(
                role == message["role"] and content == message["content"]
                and (tool_call_id or "") == message.get("tool_call_id", "")
                for role, content, tool_call_id, _tool_calls in added
            ), message["content"][:40]
    assert len(added) <= 10
