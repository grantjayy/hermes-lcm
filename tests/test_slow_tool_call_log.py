"""LCM tool calls that pass the slow-call threshold log where they spent their time."""

import logging
import re
import threading

import pytest

import hermes_lcm.engine as lcm_engine
import hermes_lcm.tools as lcm_tools
from hermes_lcm.config import LCMConfig
from hermes_lcm.engine import LCMEngine

SLOW_PREFIX = "LCM slow tool call"
TEST_THRESHOLD_SECONDS = 0.05
# Only a broken test waits this long; it is far above the test threshold.
WAIT_TIMEOUT_SECONDS = 10.0


class _SlowCallRecords(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []
        self.running_logged = threading.Event()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        if not message.startswith(SLOW_PREFIX):
            return
        self.messages.append(message)
        if message.startswith(f"{SLOW_PREFIX} running:"):
            self.running_logged.set()

    def only(self, kind: str) -> str:
        matching = [m for m in self.messages if m.startswith(f"{SLOW_PREFIX} {kind}:")]
        assert len(matching) == 1, self.messages
        return matching[0]


@pytest.fixture
def records(caplog):
    caplog.set_level(logging.INFO, logger="hermes_lcm.engine")
    handler = _SlowCallRecords()
    engine_logger = logging.getLogger("hermes_lcm.engine")
    engine_logger.addHandler(handler)
    try:
        yield handler
    finally:
        engine_logger.removeHandler(handler)


@pytest.fixture
def engine(tmp_path):
    lcm = LCMEngine(config=LCMConfig(database_path=str(tmp_path / "slow-call.db")))
    try:
        yield lcm
    finally:
        lcm.shutdown()


def _sampler_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name.startswith("lcm-slow-tool-call")]


def _field(message: str, name: str) -> str:
    match = re.search(rf"\b{name}=(\S+)", message)
    assert match, message
    return match.group(1)


def test_slow_handler_logs_running_stack_and_done_split(engine, records, monkeypatch):
    monkeypatch.setattr(lcm_engine, "_SLOW_TOOL_CALL_SECONDS", TEST_THRESHOLD_SECONDS)

    def waiting_status(args, *, engine):
        assert records.running_logged.wait(WAIT_TIMEOUT_SECONDS)
        return "status-ok"

    monkeypatch.setattr(lcm_tools, "lcm_status", waiting_status)

    assert engine.handle_tool_call("lcm_status", {}) == "status-ok"

    running = records.only("running")
    assert _field(running, "tool") == "lcm_status"
    assert _field(running, "session") == "-"
    assert _field(running, "phase") == "handler"
    stack = _field(running, "stack").split("<")
    assert any(frame.endswith(":waiting_status") for frame in stack), stack
    assert stack[-1].startswith("engine.py:") and stack[-1].endswith(":handle_tool_call"), stack

    done = records.only("done")
    assert _field(done, "tool") == "lcm_status"
    assert float(_field(done, "total").rstrip("s")) >= TEST_THRESHOLD_SECONDS
    assert _field(done, "live_messages") == "0"
    assert _sampler_threads() == []


def test_slow_ingest_logs_ingest_phase(engine, records, monkeypatch):
    monkeypatch.setattr(lcm_engine, "_SLOW_TOOL_CALL_SECONDS", TEST_THRESHOLD_SECONDS)
    engine.on_session_start("session-slow", platform="cli", conversation_id="conversation:slow")
    ingested: list[list[dict]] = []

    def waiting_ingest(messages):
        assert records.running_logged.wait(WAIT_TIMEOUT_SECONDS)
        ingested.append(messages)
        return []

    monkeypatch.setattr(engine, "_ingest_messages", waiting_ingest)
    monkeypatch.setattr(lcm_tools, "lcm_status", lambda args, *, engine: "status-ok")
    live = [
        {"role": "user", "content": "first invented line"},
        {"role": "assistant", "content": "second invented line"},
    ]

    assert engine.handle_tool_call("lcm_status", {}, messages=live) == "status-ok"

    assert ingested == [live]
    running = records.only("running")
    assert _field(running, "phase") == "ingest"
    assert _field(running, "session") == "session-slow"
    assert ":waiting_ingest" in _field(running, "stack")
    done = records.only("done")
    assert float(_field(done, "ingest").rstrip("s")) >= TEST_THRESHOLD_SECONDS
    assert _field(done, "live_messages") == "2"
    assert "invented" not in running + done


def test_fast_call_logs_nothing_and_leaves_no_sampler_thread(engine, records, monkeypatch):
    samplers_during_call: list[int] = []

    def fast_status(args, *, engine):
        samplers_during_call.append(len(_sampler_threads()))
        return "status-ok"

    monkeypatch.setattr(lcm_tools, "lcm_status", fast_status)

    assert engine.handle_tool_call("lcm_status", {}) == "status-ok"

    assert samplers_during_call == [1]
    assert _sampler_threads() == []
    assert records.messages == []


def test_raising_handler_propagates_and_leaves_no_sampler_thread(engine, records, monkeypatch):
    samplers_during_call: list[int] = []

    class HandlerFailure(RuntimeError):
        pass

    def failing_status(args, *, engine):
        samplers_during_call.append(len(_sampler_threads()))
        raise HandlerFailure("handler failed")

    monkeypatch.setattr(lcm_tools, "lcm_status", failing_status)

    with pytest.raises(HandlerFailure, match="handler failed"):
        engine.handle_tool_call("lcm_status", {})

    assert samplers_during_call == [1]
    assert _sampler_threads() == []
    assert records.messages == []
