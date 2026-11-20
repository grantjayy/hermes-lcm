"""The two-stage message search statement returns what the one-stage statement returned.

The reference statement below is the single statement ``MessageStore.search``
ran before the two-stage form. It is the oracle: every case runs both on the
same connection with the same filters, order, limit and offset.
"""

import sqlite3

import pytest

from hermes_lcm.search_query import sanitize_fts5_query
from hermes_lcm.store import (
    _MESSAGE_ROLE_BIAS_SQL,
    MessageStore,
    _build_message_search_sql,
    _build_search_order_by,
    _conversation_filter_clause,
    _source_filter_clause,
)

ROW_COUNT = 2100
BASE_TIMESTAMP = 1_700_000_000
SESSIONS = ("session-a", "session-b")
ROLES = ("user", "assistant", "tool")
SOURCES = ("cli", "discord")
CONVERSATIONS = ("conv-north", "conv-south")
FILLER = ("pebble", "orchard", "violet", "canyon", "meadow", "quartz", "willow", "saddle", "thistle")


def _reference_sql(where: list[str], order_by: str) -> str:
    return f"""SELECT m.store_id, m.session_id, m.source, m.role, m.content, m.tool_call_id,
                      m.tool_calls, m.tool_name, m.timestamp, m.token_estimate, m.pinned, m.conversation_id,
                      m.ingested_at, m.observed_at, m.observed_at_source,
                      rank as search_rank,
                      snippet(messages_fts, 0, '>>>', '<<<', '...', 40) as snippet
               FROM messages_fts fts
               JOIN messages m ON m.store_id = fts.rowid
               WHERE {' AND '.join(where)}
               ORDER BY {order_by} LIMIT ? OFFSET ?"""


def _content(i: int) -> str:
    # "lantern" matches every row except multiples of 7 (1800 rows), with a
    # per-row count so bm25 rank differs; filler length varies row length too.
    # Each API case's filter keeps more matches than search()'s candidate cap
    # (500 for these limits), so a wrong stage-one order cannot be repaired by
    # the paging loop reading every candidate.
    parts = [f"entry {i}"]
    if i % 7 != 0:
        parts.extend(["lantern"] * (1 + i % 4))
    if i % 5 == 0:
        parts.append("copper")
    if i % 4 == 0:
        parts.append("amber harbor")
    elif i % 4 == 1:
        parts.append("harbor amber")
    parts.extend(FILLER[: i % 9])
    return " ".join(parts)


def _timestamp(store_id: int) -> int:
    return BASE_TIMESTAMP + store_id * 60


# Module scope: seeding takes several seconds, and every test here only reads.
@pytest.fixture(scope="module")
def store(tmp_path_factory):
    message_store = MessageStore(tmp_path_factory.mktemp("two-stage") / "search.db")
    try:
        for i in range(ROW_COUNT):
            store_id = message_store.append(
                SESSIONS[i % 2],
                {"role": ROLES[i % 3], "content": _content(i)},
                source=SOURCES[(i // 6) % 2],
                conversation_id=CONVERSATIONS[(i // 12) % 2],
            )
            assert store_id == i + 1
        # The timestamp column holds the write time; give each row a distinct
        # value so no two rows tie on the whole sort key.
        with message_store._conn:
            message_store._conn.execute(
                "UPDATE messages SET timestamp = ? + store_id * 60", (BASE_TIMESTAMP,)
            )
        yield message_store
    finally:
        message_store.close()


def _filter(name: str) -> tuple[list[str], list[object]]:
    if name == "none":
        return [], []
    if name == "session":
        return ["m.session_id = ?"], [SESSIONS[1]]
    if name == "role":
        return ["m.role = ?"], ["assistant"]
    if name == "window":
        return ["m.timestamp >= ?", "m.timestamp <= ?"], [_timestamp(150), _timestamp(480)]
    if name == "source":
        clause, args = _source_filter_clause("m.source", "discord")
        assert clause is not None
        return [clause], list(args)
    if name == "conversation":
        clause, args = _conversation_filter_clause("m.conversation_id", CONVERSATIONS[0])
        assert clause is not None
        return [clause], list(args)
    raise ValueError(f"unknown filter {name!r}")


# (query, filter, sort, limit, offset). Each query shape, filter, sort order
# and limit/offset pair appears at least once; the common word with every sort.
STATEMENT_CASES = [
    ("lantern", "none", "recency", 20, 0),
    ("lantern", "none", "recency", 20, 20),
    ("lantern", "session", "recency", 7, 30),
    ("lantern", "role", "relevance", 20, 0),
    ("lantern", "none", "relevance", 20, 20),
    ("lantern", "source", "relevance", 7, 30),
    ("lantern", "window", "hybrid", 20, 0),
    ("lantern", "none", "hybrid", 7, 30),
    ("lantern copper", "source", "recency", 20, 0),
    ("lantern copper", "conversation", "relevance", 7, 30),
    ('"amber harbor"', "none", "recency", 20, 20),
    ('"amber harbor"', "role", "hybrid", 20, 0),
    ('"amber harbor"', "conversation", "recency", 7, 30),
    ('"amber harbor"', "window", "relevance", 20, 0),
]


def _run_both(
    conn: sqlite3.Connection, query: str, filter_name: str, sort: str, limit: int, offset: int
) -> tuple[list[tuple], list[tuple]]:
    safe_query = sanitize_fts5_query(query)
    clauses, filter_args = _filter(filter_name)
    where = ["messages_fts MATCH ?", *clauses]
    args = [safe_query, *filter_args, limit, offset]
    order_by = _build_search_order_by(sort, "m.timestamp", _MESSAGE_ROLE_BIAS_SQL)
    reference = [tuple(row) for row in conn.execute(_reference_sql(where, order_by), args)]
    two_stage = [
        tuple(row)
        for row in conn.execute(_build_message_search_sql(where, order_by), [*args, safe_query])
    ]
    return reference, two_stage


@pytest.mark.parametrize(
    ("query", "filter_name", "sort", "limit", "offset"),
    STATEMENT_CASES,
    ids=[f"{q}-{f}-{s}-{lim}-{off}" for q, f, s, lim, off in STATEMENT_CASES],
)
def test_two_stage_statement_matches_reference(store, query, filter_name, sort, limit, offset):
    reference, two_stage = _run_both(store._conn, query, filter_name, sort, limit, offset)

    assert len(reference) == limit
    if sort == "hybrid":
        # The hybrid ORDER BY reads the current second, and the two statements
        # run at different instants, so compare the rows per id.
        assert {row[0]: row for row in two_stage} == {row[0]: row for row in reference}
    else:
        assert two_stage == reference


def test_search_recency_returns_newest_matching_rows(store):
    results = store.search("lantern", sort="recency", limit=7)

    # Store id 2094 has no match.
    assert [r["store_id"] for r in results] == [2100, 2099, 2098, 2097, 2096, 2095, 2093]


def test_search_recency_with_role_returns_newest_matching_user_rows(store):
    results = store.search("lantern", sort="recency", limit=7, role="user")

    # Store id 2080 is a user row without the word.
    assert [r["store_id"] for r in results] == [2098, 2095, 2092, 2089, 2086, 2083, 2077]


def test_search_recency_with_session_and_time_to_returns_newest_rows_before_cutoff(store):
    results = store.search(
        "lantern",
        sort="recency",
        limit=6,
        session_id=SESSIONS[0],
        time_to=_timestamp(1800),
    )

    # Store id 1793 is a session-a row without the word.
    assert [r["store_id"] for r in results] == [1799, 1797, 1795, 1791, 1789, 1787]
