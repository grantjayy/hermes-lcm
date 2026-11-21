"""Ingest-cursor reconciliation and replay-identity for the LCM engine (WS5 Seam 4).

The ``ReconcileMixin`` holds the machinery that reconciles the persisted store
tail against the active message list after a process restart, plus the stable
replay-identity primitives it relies on. These methods were lifted verbatim out
of ``LCMEngine`` and continue to run bound to the engine instance (``self`` is
the ``LCMEngine``), so they read the engine's runtime state (``_store``,
``_session_id``, ``_config``, ``_ingest_cursor`` is written by the engine from
the value these return) and call back into engine helpers through normal
attribute lookup. ``LCMEngine`` mixes this in, so no call site and no test
changes.

``_PRESERVED_OBJECTIVE_CONTEXT_PREFIX`` lives here (used by the reconciliation
scan) and is re-exported to ``engine.py``; the two tool-call-identity
staticmethods reference the mixin class directly rather than ``LCMEngine`` to
avoid an import cycle (staticmethod resolution is identical).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .externalize import (
    extract_externalized_ref,
    externalized_tool_result_has_persisted_output_marker,
    find_externalized_tool_result_content_for_call,
    load_externalized_payload,
)
from .ingest_protection import (
    _add_inline_persisted_output_generation_metadata,
    _add_inline_persisted_output_identity_metadata,
    _expected_persisted_output_chars,
    _has_inline_persisted_output_generation_metadata,
    _has_lossy_sensitive_redaction,
    _is_hermes_persisted_output_marker,
    _json_has_duplicate_object_keys,
    _persisted_output_marker_identity_digest,
    _persisted_output_saved_path,
    recover_hermes_persisted_output_with_file_stat,
    redact_sensitive_value,
)
from .message_content import normalize_content_value, text_content_for_pattern_matching
from .sanitize import _clean_active_assistant_message
from .store import _normalize_observed_at

import logging

logger = logging.getLogger(__name__)

_PRESERVED_OBJECTIVE_CONTEXT_PREFIX = "[Current user objective preserved from compacted history]"

# A short rebound batch is indistinguishable from a deliberate delta that
# legitimately repeats durable content (a "retry" turn, a fresh singleton
# quirk).  The engine's existing contract preserves those, so the multiset
# replay dedup only takes over once the incoming batch is large enough to be a
# transcript rebuild rather than a delta.  Between the two sizes, a batch that
# repeats a provider id is a retry sequence, not a plain transcript replay, and
# keeps the legacy path so the retry is preserved.
_MULTISET_REPLAY_MIN_MESSAGES = 8
_MULTISET_REPLAY_SHORT_MIN_MESSAGES = 4


class ReconcileMixin:
    @staticmethod
    def _canonicalize_tool_call_identity_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: ReconcileMixin._canonicalize_tool_call_identity_value(val)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [ReconcileMixin._canonicalize_tool_call_identity_value(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped[0] in "[{":
                if _json_has_duplicate_object_keys(value):
                    return value
                try:
                    parsed = json.loads(value)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return value
                if isinstance(parsed, (dict, list)):
                    canonical = ReconcileMixin._canonicalize_tool_call_identity_value(parsed)
                    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            return value
        return value

    @staticmethod
    def _stable_tool_calls_identity(tool_calls: Any) -> str:
        if not tool_calls:
            return ""
        try:
            canonical = ReconcileMixin._canonicalize_tool_call_identity_value(tool_calls)
            return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            return str(tool_calls)

    def _has_durable_persisted_output_replay_identity(self, msg: Dict[str, Any]) -> bool:
        role = str(msg.get("role") or "unknown")
        content = normalize_content_value(msg.get("content")) or ""
        if role != "tool" or not _is_hermes_persisted_output_marker(content):
            return False
        expected_chars = _expected_persisted_output_chars(content)
        persisted_output_source_path = _persisted_output_saved_path(content)
        persisted_output_preview_sha256, allow_redacted_preview_match = self._persisted_output_marker_replay_proof(content)
        if (
            expected_chars is None
            or not persisted_output_source_path
            or not persisted_output_preview_sha256
        ):
            return False
        recovered_with_stat = recover_hermes_persisted_output_with_file_stat(content)
        if recovered_with_stat is None:
            return False
        require_live_file_freshness = True
        durable_content = find_externalized_tool_result_content_for_call(
            tool_call_id=str(msg.get("tool_call_id") or ""),
            session_id=str(msg.get("session_id") or self._session_id or ""),
            expected_chars=expected_chars,
            persisted_output_source_path=persisted_output_source_path,
            persisted_output_preview_sha256=persisted_output_preview_sha256,
            require_persisted_output_file_not_newer=require_live_file_freshness,
            allow_redacted_preview_match=allow_redacted_preview_match,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        if durable_content is None:
            return False
        if recovered_with_stat is not None:
            recovered_content, _file_stat = recovered_with_stat
            if not self._recovered_content_matches_durable_identity(recovered_content, durable_content):
                return False
        return True

    @contextlib.contextmanager
    def _replay_identity_memo_scope(self):
        """Memoise replay identities for the duration of one reconcile pass.

        An identity is a pure function of the message dict (plus files that are
        only ever appended while NEW rows are persisted, which happens after this
        scope closes).  Turn-start reconciliation asks for the identity of the
        same incoming message up to ~6 times and of every stored-tail row again
        for the stale-snapshot probe; memoising by object identity makes the
        pass O(rows) identity computations.  Nested scopes reuse the outer memo.
        """
        if getattr(self, "_replay_identity_memo", None) is not None:
            yield
            return
        self._replay_identity_memo = {}
        try:
            yield
        finally:
            self._replay_identity_memo = None

    def _message_replay_identity(self, msg: Dict[str, Any], *, stored_row: bool = False) -> tuple[str, str, str, str]:
        memo = getattr(self, "_replay_identity_memo", None)
        if memo is None:
            return self._compute_message_replay_identity(msg, stored_row=stored_row)
        key = (id(msg), stored_row)
        hit = memo.get(key)
        # Hold a reference to ``msg`` so its id() cannot be recycled mid-scope.
        if hit is not None and hit[0] is msg:
            return hit[1]
        identity = self._compute_message_replay_identity(msg, stored_row=stored_row)
        memo[key] = (msg, identity)
        return identity

    def _compute_message_replay_identity(self, msg: Dict[str, Any], *, stored_row: bool = False) -> tuple[str, str, str, str]:
        role = str(msg.get("role") or "unknown")
        content = normalize_content_value(msg.get("content")) or ""
        if (
            role == "tool"
            and _is_hermes_persisted_output_marker(content)
            and bool(getattr(self._config, "large_output_externalization_enabled", True))
        ):
            expected_chars = _expected_persisted_output_chars(content)
            persisted_output_source_path = _persisted_output_saved_path(content)
            persisted_output_preview_sha256, allow_redacted_preview_match = self._persisted_output_marker_replay_proof(content)
            durable_content = None
            recovered_with_stat = recover_hermes_persisted_output_with_file_stat(content) if not stored_row else None
            recovered_content = recovered_with_stat[0] if recovered_with_stat is not None else None
            recovered_identity_content = None
            if recovered_content is not None:
                recovered_identity_content = normalize_content_value(
                    redact_sensitive_value(
                        recovered_content,
                        self._config,
                        parse_json_strings=False,
                    )
                )
            require_live_file_freshness = recovered_with_stat is not None

            def live_file_generation_identity() -> str:
                try:
                    live_stat = Path(str(persisted_output_source_path)).stat()
                    return (
                        "[LCM persisted-output live file: "
                        f"path={persisted_output_source_path}; "
                        f"mtime_ns={live_stat.st_mtime_ns}; "
                        f"chars={expected_chars}]"
                    )
                except OSError:
                    return (
                        "[LCM persisted-output live file: "
                        f"path={persisted_output_source_path}; "
                        f"chars={expected_chars}]"
                    )

            if (
                not stored_row
                and expected_chars is not None
                and persisted_output_source_path
                and persisted_output_preview_sha256
                and recovered_with_stat is not None
            ):
                durable_content = find_externalized_tool_result_content_for_call(
                    tool_call_id=str(msg.get("tool_call_id") or ""),
                    session_id=str(msg.get("session_id") or self._session_id or ""),
                    expected_chars=expected_chars,
                    persisted_output_source_path=persisted_output_source_path,
                    persisted_output_preview_sha256=persisted_output_preview_sha256,
                    require_persisted_output_file_not_newer=require_live_file_freshness,
                    allow_redacted_preview_match=allow_redacted_preview_match,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
            if durable_content is not None and (
                recovered_content is None or self._recovered_content_matches_durable_identity(recovered_content, durable_content)
            ):
                content = durable_content
            elif recovered_content is not None:
                stale_durable_content = find_externalized_tool_result_content_for_call(
                    tool_call_id=str(msg.get("tool_call_id") or ""),
                    session_id=str(msg.get("session_id") or self._session_id or ""),
                    expected_chars=expected_chars,
                    persisted_output_source_path=persisted_output_source_path,
                    persisted_output_preview_sha256=persisted_output_preview_sha256,
                    allow_redacted_preview_match=allow_redacted_preview_match,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                if (
                    stale_durable_content is not None
                    and self._recovered_content_matches_durable_identity(recovered_content, stale_durable_content)
                    and not _has_lossy_sensitive_redaction(stale_durable_content)
                    and not _has_lossy_sensitive_redaction(recovered_identity_content)
                ):
                    content = stale_durable_content
                elif stale_durable_content is not None:
                    content = live_file_generation_identity()
                elif recovered_with_stat is not None:
                    content = _add_inline_persisted_output_generation_metadata(
                        _add_inline_persisted_output_identity_metadata(
                            content,
                            _persisted_output_marker_identity_digest(content),
                        ),
                        recovered_with_stat[1],
                    )
                elif recovered_identity_content is not None:
                    content = recovered_identity_content
        tool_calls = msg.get("tool_calls")
        if stored_row:
            session_id = str(msg.get("session_id") or self._session_id or "")
            content = self._restore_ingest_payload_placeholders_in_content_identity(
                content,
                session_id=session_id,
            )
            tool_calls = self._restore_ingest_payload_placeholders_in_value(tool_calls, session_id=session_id)
        ref = extract_externalized_ref(content)
        if ref and "quarantined_assistant_output" not in content:
            payload = load_externalized_payload(
                ref,
                config=self._config,
                hermes_home=self._hermes_home,
            )
            if payload is not None and isinstance(payload.get("content"), str):
                content = payload["content"]
        tool_calls_identity = self._stable_tool_calls_identity(tool_calls)
        return (
            role,
            content,
            str(msg.get("tool_call_id") or ""),
            tool_calls_identity,
        )

    @staticmethod
    def _matches_store_tail_suffix(
        stored_tail: list[tuple[str, str, str, str]],
        candidate_prefix: list[tuple[str, str, str, str]],
    ) -> bool:
        if not candidate_prefix:
            return True
        width = len(candidate_prefix)
        if width > len(stored_tail):
            return False
        # Compare from the END: the reconcile scan calls this once per candidate
        # cursor and a mismatch almost always shows at the newest row, so this
        # fails in O(1) instead of slicing and comparing O(width) rows.
        for offset in range(1, width + 1):
            if stored_tail[-offset] != candidate_prefix[-offset]:
                return False
        return True

    @staticmethod
    def _strip_inline_persisted_output_generation_identity(
        identity: tuple[str, str, str, str],
    ) -> tuple[str, str, str, str]:
        role, content, tool_call_id, tool_calls = identity
        if role != "tool" or not isinstance(content, str):
            return identity
        stripped = re.sub(
            r"\n?\[LCM persisted-output file generation: "
            r"size=\d+; mtime_ns=\d+; ctime_ns=\d+\]\n?(?=</persisted-output>)",
            "\n",
            content,
        )
        return (role, stripped, tool_call_id, tool_calls)

    def _stored_row_has_durable_persisted_output_marker(self, row: Dict[str, Any]) -> bool:
        if str(row.get("role") or "") != "tool":
            return False
        content = normalize_content_value(row.get("content")) or ""
        ref = extract_externalized_ref(content)
        if not ref:
            return False
        return externalized_tool_result_has_persisted_output_marker(
            ref,
            config=self._config,
            hermes_home=self._hermes_home,
        )

    @staticmethod
    def _persisted_output_durable_wildcard_identity(
        identity: tuple[str, str, str, str],
    ) -> tuple[str, str, str, str]:
        role, _content, tool_call_id, tool_calls = identity
        return (role, "[LCM persisted-output durable replay]", tool_call_id, tool_calls)

    def _matches_persisted_output_durable_full_replay(
        self,
        candidate_messages: list[Dict[str, Any]],
        candidate_prefix: list[tuple[str, str, str, str]],
        stored_tail: list[tuple[str, str, str, str]],
        stored_tail_rows: list[Dict[str, Any]] | None,
    ) -> bool:
        if not stored_tail_rows or len(candidate_prefix) != len(stored_tail) or len(candidate_messages) != len(candidate_prefix):
            return False
        transformed_candidate: list[tuple[str, str, str, str]] = []
        transformed_stored: list[tuple[str, str, str, str]] = []
        saw_persisted_output = False
        for candidate_msg, candidate_identity, stored_identity, stored_row in zip(
            candidate_messages,
            candidate_prefix,
            stored_tail,
            stored_tail_rows,
        ):
            candidate_content = normalize_content_value(candidate_msg.get("content")) or ""
            candidate_is_persisted_marker = (
                str(candidate_msg.get("role") or "") == "tool"
                and _is_hermes_persisted_output_marker(candidate_content)
            )
            stored_is_persisted_output = self._stored_row_has_durable_persisted_output_marker(stored_row)
            if candidate_is_persisted_marker or stored_is_persisted_output:
                if (
                    not candidate_is_persisted_marker
                    or not stored_is_persisted_output
                    or not self._has_durable_persisted_output_replay_identity(candidate_msg)
                ):
                    return False
                saw_persisted_output = True
                transformed_candidate.append(self._persisted_output_durable_wildcard_identity(candidate_identity))
                transformed_stored.append(self._persisted_output_durable_wildcard_identity(stored_identity))
                continue
            transformed_candidate.append(candidate_identity)
            transformed_stored.append(stored_identity)
        return saw_persisted_output and transformed_candidate == transformed_stored

    @classmethod
    def _identity_content_for_active_cleanup(cls, content: str) -> Any:
        """Decode canonical stored JSON content before active-cleanup checks.

        Structured assistant content is persisted as deterministic JSON. Active
        replay cleanup sees the original list/dict shape, so restart
        reconciliation has to decode the stored identity before deciding whether
        a durable assistant row could be absent from sanitized active context.
        """
        if not isinstance(content, str):
            return content
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return content
        if isinstance(decoded, (list, dict)) and normalize_content_value(decoded) == content:
            return decoded
        return content

    @classmethod
    def _active_cleanup_replay_identity(
        cls,
        identity: tuple[str, str, str, str],
    ) -> tuple[str, str, str, str] | None:
        role, content, tool_call_id, tool_calls = identity
        if role != "assistant":
            return identity
        msg: dict[str, Any] = {
            "role": role,
            "content": cls._identity_content_for_active_cleanup(content),
        }
        if tool_calls:
            try:
                decoded_tool_calls = json.loads(tool_calls)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded_tool_calls = tool_calls
            msg["tool_calls"] = decoded_tool_calls
        cleaned = _clean_active_assistant_message(msg)
        if cleaned is None:
            return None
        return (
            role,
            normalize_content_value(cleaned.get("content")) or "",
            tool_call_id,
            tool_calls,
        )

    @staticmethod
    def _is_quarantined_assistant_replay_identity(identity: tuple[str, str, str, str]) -> bool:
        role, content, _tool_call_id, _tool_calls = identity
        if role != "assistant":
            return False
        text = str(content or "").strip()
        return bool(
            re.fullmatch(
                r"\[Externalized LCM ingest payload: assistant output quarantined; "
                r"kind=quarantined_assistant_output; "
                r"reason=[A-Za-z0-9_.:/-]+; "
                r"field=[A-Za-z0-9_.:/<>\[\]-]+; "
                r"chars=\d+; bytes=\d+; "
                r"ref=[^\]\s]+\]",
                text,
            )
            or re.fullmatch(
                r"\[LCM active replay placeholder: assistant output quarantined; "
                r"kind=quarantined_assistant_output; "
                r"reason=[A-Za-z0-9_.:/-]+; "
                r"scope=ignored_message_pattern; field=content; "
                r"chars=\d+; bytes=\d+; "
                r"sha256=[0-9a-f]{16}\]",
                text,
            )
        )

    def _stored_tail_for_sanitized_active_replay(
        self,
        stored_tail: list[tuple[str, str, str, str]],
    ) -> list[tuple[str, str, str, str]]:
        """Mirror active-context cleanup for restart replay reconciliation.

        Raw storage remains lossless. This view is used only to reconcile a
        restarted process when the host replays sanitized active context where
        assistant rows may be removed or have internal content stripped.
        """
        sanitized_tail: list[tuple[str, str, str, str]] = []
        for identity in stored_tail:
            cleaned_identity = self._active_cleanup_replay_identity(identity)
            if cleaned_identity is not None:
                sanitized_tail.append(cleaned_identity)
        return sanitized_tail

    def _find_reconciled_cursor_for_store_tail(
        self,
        messages: List[Dict[str, Any]],
        stored_tail: list[tuple[str, str, str, str]],
        *,
        stored_tail_rows: list[Dict[str, Any]] | None = None,
        allow_empty_prefix: bool,
        session_count: int,
        raw_session_count: int,
    ) -> int | None:
        sanitized_replay_tail = self._stored_tail_for_sanitized_active_replay(stored_tail)
        effective_session_count = len(sanitized_replay_tail)
        sanitized_tail_collapsed = len(sanitized_replay_tail) < len(stored_tail)
        boundary_messages = list(stored_tail_rows or [])
        if not boundary_messages:
            for role, content, tool_call_id, tool_calls in stored_tail:
                try:
                    decoded_tool_calls = json.loads(tool_calls) if tool_calls else []
                except (TypeError, ValueError, json.JSONDecodeError):
                    decoded_tool_calls = []
                boundary_messages.append({
                    "role": role,
                    "content": content,
                    "tool_call_id": tool_call_id,
                    "tool_calls": decoded_tool_calls,
                })
        effective_fresh_tail_count = self._fresh_tail_boundary(boundary_messages).count
        empty_prefix_cursor: int | None = None

        # Turn-start cost contract: every per-message predicate and
        # replay identity below is a pure function of ONE message, so compute it
        # ONCE per message and answer each candidate cursor from prefix counts.
        # The previous shape rebuilt every filtered list and recomputed every
        # identity for messages[:cursor] at EACH cursor -- O(n^2) identity
        # computations (JSON canonicalisation + externalized-payload reads) on
        # the post-compaction replay shape, which scans all the way to cursor 0.
        # Semantics are unchanged: each derived value is the same expression the
        # old loop evaluated, just memoised by index.
        n_msgs = len(messages)
        compiled_ignore = bool(self._compiled_ignore_message_patterns)
        identities: list[tuple[str, str, str, str]] = []
        contents: list[str] = []
        is_scaffold: list[bool] = []
        is_visible: list[bool] = []
        is_dropped: list[bool] = []
        is_quarantined: list[bool] = []
        is_marker: list[bool] = []
        is_unrecoverable_marker: list[bool] = []
        is_inline_generation_marker: list[bool] = []
        is_preserved_objective: list[bool] = []
        for msg in messages:
            identity = self._message_replay_identity(msg)
            identities.append(identity)
            scaffold = self._is_replayed_context_scaffold_message(msg)
            is_scaffold.append(scaffold)
            is_visible.append(not scaffold and not self._matches_ignore_message_patterns(msg))
            pattern_text = text_content_for_pattern_matching(msg.get("content")) or ""
            quarantined = self._is_quarantined_assistant_replay_identity(identity)
            is_quarantined.append(quarantined)
            is_dropped.append(
                self._is_volatile_ignored_quarantine_placeholder(msg, pattern_text)
                or self._is_ignored_active_replay_placeholder(msg, pattern_text)
                or (
                    compiled_ignore
                    and quarantined
                    and self._matches_ignore_message_patterns(msg, stored_row=True)
                )
            )
            content = normalize_content_value(msg.get("content")) or ""
            contents.append(content)
            role_text = str(msg.get("role") or "")
            marker = role_text == "tool" and _is_hermes_persisted_output_marker(content)
            is_marker.append(marker)
            is_unrecoverable_marker.append(
                marker and recover_hermes_persisted_output_with_file_stat(content) is None
            )
            is_inline_generation_marker.append(
                marker and _has_inline_persisted_output_generation_metadata(content)
            )
            is_preserved_objective.append(
                role_text != "system"
                and content.lstrip().startswith(_PRESERVED_OBJECTIVE_CONTEXT_PREFIX)
            )

        visible_idx = [i for i in range(n_msgs) if is_visible[i]]
        # "Identity messages" == visible rows that are not dropped placeholders.
        # (The old fallback to the visible list only applied when both lists
        # were empty, so it is the same list.)
        identity_idx = [i for i in visible_idx if not is_dropped[i]]
        visible_ids = [identities[i] for i in visible_idx]
        identity_ids = [identities[i] for i in identity_idx]

        def _prefix_counts(flags: list[bool]) -> list[int]:
            counts = [0] * (n_msgs + 1)
            running = 0
            for i, flag in enumerate(flags):
                if flag:
                    running += 1
                counts[i + 1] = running
            return counts

        identity_member = [False] * n_msgs
        for i in identity_idx:
            identity_member[i] = True
        cnt_visible = _prefix_counts(is_visible)
        cnt_identity = _prefix_counts(identity_member)
        cnt_scaffold = _prefix_counts(is_scaffold)
        cnt_quarantined = _prefix_counts(is_quarantined)
        cnt_dropped = _prefix_counts(is_dropped)
        cnt_objective = _prefix_counts(is_preserved_objective)
        cnt_identity_marker = _prefix_counts(
            [identity_member[i] and is_marker[i] for i in range(n_msgs)]
        )
        cnt_identity_unrecoverable = _prefix_counts(
            [identity_member[i] and is_unrecoverable_marker[i] for i in range(n_msgs)]
        )
        cnt_identity_inline_generation = _prefix_counts(
            [identity_member[i] and is_inline_generation_marker[i] for i in range(n_msgs)]
        )
        cnt_identity_system = _prefix_counts(
            [identity_member[i] and identities[i][0] == "system" for i in range(n_msgs)]
        )
        cnt_identity_user = _prefix_counts(
            [identity_member[i] and identities[i][0] == "user" for i in range(n_msgs)]
        )
        marker_idx = [i for i in range(n_msgs) if is_marker[i]]
        durable_marker_memo: dict[int, bool] = {}

        def _durable_marker(i: int) -> bool:
            if i not in durable_marker_memo:
                durable_marker_memo[i] = self._has_durable_persisted_output_replay_identity(messages[i])
            return durable_marker_memo[i]

        generationless_sanitized_tail: list[tuple[str, str, str, str]] | None = None
        generationless_identity_ids: list[tuple[str, str, str, str]] | None = None
        cleanup_needed_memo: dict[int, bool] = {}

        def _raw_suffix_needs_cleanup(k: int) -> bool:
            if k not in cleanup_needed_memo:
                cleanup_needed_memo[k] = any(
                    self._active_cleanup_replay_identity(identity) != identity
                    for identity in (stored_tail[-k:] if k else [])
                )
            return cleanup_needed_memo[k]

        for cursor in range(n_msgs, -1, -1):
            k_visible = cnt_visible[cursor]
            k_identity = cnt_identity[cursor]
            filtered_candidate_placeholders = k_identity < k_visible
            candidate_has_scaffold_evidence = cnt_scaffold[cursor] > 0
            candidate_has_quarantined_replay_evidence = cnt_quarantined[cursor] > 0
            candidate_prefix = identity_ids[:k_identity]
            if not candidate_prefix:
                empty_prefix_cursor = cursor
                if allow_empty_prefix and (
                    not filtered_candidate_placeholders
                    or candidate_has_scaffold_evidence
                    or candidate_has_quarantined_replay_evidence
                ):
                    return cursor
                continue

            matches_sanitized_tail = (
                len(candidate_prefix) <= len(sanitized_replay_tail)
                and self._matches_store_tail_suffix(sanitized_replay_tail, candidate_prefix)
            )
            matches_raw_tail = self._matches_store_tail_suffix(stored_tail, candidate_prefix)
            matches_visible_sanitized_tail = False
            matches_visible_raw_tail = False
            if filtered_candidate_placeholders and k_visible:
                candidate_visible_prefix = visible_ids[:k_visible]
                matches_visible_sanitized_tail = (
                    len(candidate_visible_prefix) <= len(sanitized_replay_tail)
                    and self._matches_store_tail_suffix(sanitized_replay_tail, candidate_visible_prefix)
                )
                matches_visible_raw_tail = self._matches_store_tail_suffix(
                    stored_tail, candidate_visible_prefix
                )
            candidate_has_unrecoverable_persisted_marker = cnt_identity_unrecoverable[cursor] > 0
            if (
                (matches_visible_sanitized_tail or matches_visible_raw_tail)
                and not candidate_has_unrecoverable_persisted_marker
            ):
                return cursor
            candidate_has_persisted_marker = cnt_identity_marker[cursor] > 0
            matches_durable_persisted_output_full_replay = bool(
                stored_tail_rows
                and len(candidate_prefix) == len(stored_tail)
                and self._matches_persisted_output_durable_full_replay(
                    [messages[i] for i in identity_idx[:k_identity]],
                    candidate_prefix,
                    stored_tail,
                    stored_tail_rows,
                )
            )
            matches_inline_generation_cleanup_tail = False
            if candidate_has_unrecoverable_persisted_marker:
                if generationless_sanitized_tail is None:
                    generationless_sanitized_tail = [
                        self._strip_inline_persisted_output_generation_identity(identity)
                        for identity in sanitized_replay_tail
                    ]
                    generationless_identity_ids = [
                        self._strip_inline_persisted_output_generation_identity(identity)
                        for identity in identity_ids
                    ]
                matches_inline_generation_cleanup_tail = self._matches_store_tail_suffix(
                    generationless_sanitized_tail,
                    generationless_identity_ids[:k_identity],
                )
            raw_suffix_needs_cleanup_equivalence = (
                _raw_suffix_needs_cleanup(len(candidate_prefix)) if matches_raw_tail else False
            )
            if (
                not matches_sanitized_tail
                and not matches_raw_tail
                and not matches_inline_generation_cleanup_tail
                and not matches_durable_persisted_output_full_replay
            ):
                continue

            # Matching a stored suffix is not enough evidence by itself.  A
            # gateway restart may provide only newly arrived delta messages; if
            # the first delta happens to repeat the durable tail, treating that
            # row as replay silently loses it.  Only advance the cursor when the
            # incoming prefix proves replay by covering the full durable session.
            # A system prompt is a strong anchor. Older/minimal transcripts can
            # start directly with user/assistant turns, so multi-row full replay
            # is accepted only when active cleanup did not collapse the durable
            # tail; otherwise a fresh delta can repeat the remaining visible
            # suffix and must be preserved.
            candidate_has_system = cnt_identity_system[cursor] > 0
            candidate_dropped_quarantine_replay_placeholder = cnt_dropped[cursor] > 0
            has_quarantined_singleton_replay = (
                matches_sanitized_tail
                and len(candidate_prefix) == 1
                and effective_session_count == 1
                and self._is_quarantined_assistant_replay_identity(candidate_prefix[0])
                and self._is_quarantined_assistant_replay_identity(sanitized_replay_tail[0])
            )
            candidate_singleton_original_content = (
                contents[identity_idx[0]] if k_identity == 1 else ""
            )
            has_externalized_singleton_replay = (
                matches_raw_tail
                and len(candidate_prefix) == 1
                and raw_session_count == 1
                and bool(extract_externalized_ref(candidate_singleton_original_content))
                and candidate_prefix == stored_tail
            )
            has_persisted_marker_singleton_replay = (
                matches_raw_tail
                and not candidate_has_unrecoverable_persisted_marker
                and len(candidate_prefix) == 1
                and raw_session_count == 1
                and candidate_prefix == stored_tail
                and candidate_prefix[0][0] == "tool"
                and _is_hermes_persisted_output_marker(candidate_singleton_original_content)
            )
            has_durable_persisted_marker_suffix_replay = (
                (matches_sanitized_tail or matches_raw_tail)
                and any(_durable_marker(i) for i in marker_idx if i < cursor)
            )
            has_filtered_full_replay = (
                matches_sanitized_tail
                and candidate_dropped_quarantine_replay_placeholder
                and len(candidate_prefix) >= effective_session_count
                and effective_session_count > 0
            )
            has_inline_generation_cleanup_replay = (
                matches_inline_generation_cleanup_tail
                and candidate_has_unrecoverable_persisted_marker
                and len(candidate_prefix) >= effective_session_count
                and effective_session_count > 0
            )
            has_inline_persisted_generation_suffix_replay = (
                matches_sanitized_tail
                and cnt_identity_inline_generation[cursor] > 0
            )
            if candidate_has_unrecoverable_persisted_marker:
                continue
            has_raw_persisted_marker_exact_replay = (
                candidate_has_persisted_marker
                and not candidate_has_unrecoverable_persisted_marker
                and matches_raw_tail
                and candidate_prefix == stored_tail[-len(candidate_prefix) :]
            )
            has_persisted_marker_specific_replay_evidence = (
                not candidate_has_persisted_marker
                or has_durable_persisted_marker_suffix_replay
                or matches_durable_persisted_output_full_replay
                or has_inline_generation_cleanup_replay
                or has_inline_persisted_generation_suffix_replay
                or has_persisted_marker_singleton_replay
                or has_raw_persisted_marker_exact_replay
            )
            has_effective_full_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_sanitized_tail
                and len(candidate_prefix) >= effective_session_count
                and (
                    candidate_has_system
                    or (effective_session_count > 1 and not sanitized_tail_collapsed)
                    or has_quarantined_singleton_replay
                    or has_filtered_full_replay
                )
            )

            has_scaffold_evidence = candidate_has_scaffold_evidence
            has_raw_full_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_raw_tail
                and not has_scaffold_evidence
                and cursor >= raw_session_count
                and raw_session_count > 1
            )
            has_preserved_objective_scaffold = cnt_objective[cursor] > 0
            candidate_suffix_has_user_turn = cnt_identity_user[cursor] > 0
            has_scaffold_suffix_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_sanitized_tail
                and has_preserved_objective_scaffold
                and not candidate_suffix_has_user_turn
            )
            has_raw_cleanup_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_raw_tail
                and has_scaffold_evidence
                and cursor < n_msgs
                and len(candidate_prefix) >= max(1, effective_fresh_tail_count)
                and raw_suffix_needs_cleanup_equivalence
            )
            if (
                has_effective_full_replay
                or has_externalized_singleton_replay
                or has_persisted_marker_singleton_replay
                or has_durable_persisted_marker_suffix_replay
                or matches_durable_persisted_output_full_replay
                or has_inline_generation_cleanup_replay
                or has_inline_persisted_generation_suffix_replay
                or has_raw_full_replay
                or has_scaffold_suffix_replay
                or has_raw_cleanup_replay
            ):
                return cursor
        return empty_prefix_cursor if allow_empty_prefix else None

    def _record_ingest_reconciliation(
        self,
        *,
        action: str,
        reason: str,
        cursor: int,
        incoming: int,
        session_count: int,
        stored_tail_count: int,
        effective_incoming: int | None = None,
        already_stored: int | None = None,
        persisted: int | None = None,
    ) -> None:
        self._last_ingest_reconciliation = {
            "action": action,
            "reason": reason,
            "cursor": cursor,
            "incoming": incoming,
            "session_count": session_count,
            "stored_tail_count": stored_tail_count,
        }
        if effective_incoming is not None:
            self._last_ingest_reconciliation["effective_incoming"] = effective_incoming
        if already_stored is not None:
            self._last_ingest_reconciliation["already_stored"] = already_stored
        if persisted is not None:
            self._last_ingest_reconciliation["persisted"] = persisted

    def _effective_replay_identities(
        self,
        messages: List[Dict[str, Any]],
    ) -> list[tuple[str, str, str, str]]:
        return [
            self._message_replay_identity(msg)
            for msg in messages
            if not self._is_replayed_context_scaffold_message(msg)
            and not self._matches_ignore_message_patterns(msg)
        ]

    def _is_suspicious_stale_no_overlap_snapshot(
        self,
        incoming_identities: list[tuple[str, str, str, str]],
        stored_tail: list[tuple[str, str, str, str]],
        stored_head: list[tuple[str, str, str, str]],
    ) -> bool:
        """Return true for short stale snapshots with no durable-tail overlap.

        A restarted gateway can hand LCM a stale, short in-memory snapshot from
        the beginning of a longer session.  When that snapshot has no overlap
        with the durable tail, appending it as a delta creates duplicate rows.
        Fail closed only when the short batch is proven stale by matching the
        contiguous durable-store prefix; singleton no-overlap deltas remain
        ambiguous and are preserved.
        """
        if len(incoming_identities) <= 1:
            return False
        if incoming_identities[0][0] != "system":
            return False
        if not stored_tail or len(incoming_identities) >= len(stored_tail):
            return False
        if set(incoming_identities).intersection(stored_tail):
            return False
        if len(incoming_identities) > len(stored_head):
            return False
        return stored_head[: len(incoming_identities)] == incoming_identities

    @staticmethod
    def _tool_call_ids_from_identity(tool_calls_identity: str) -> tuple[str, ...]:
        """Provider ids carried by a canonical ``tool_calls`` identity.

        The id is what makes an assistant tool-call turn and its result unique;
        argument text is deliberately not part of the id set.
        """
        if not tool_calls_identity:
            return ()
        try:
            parsed = json.loads(tool_calls_identity)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ()
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return ()
        ids: list[str] = []
        for call in parsed:
            if not isinstance(call, dict):
                continue
            for key in ("id", "call_id", "tool_call_id"):
                value = call.get(key)
                if isinstance(value, str) and value:
                    ids.append(value)
                    break
        return tuple(sorted(set(ids)))

    @staticmethod
    def _tool_call_ids_from_calls(tool_calls: Any) -> tuple[str, ...]:
        """Provider ids of already-decoded tool calls, without canonicalizing them."""
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]
        if not isinstance(tool_calls, list):
            return ()
        ids: list[str] = []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            for key in ("id", "call_id", "tool_call_id"):
                value = call.get(key)
                if isinstance(value, str) and value:
                    ids.append(value)
                    break
        return tuple(sorted(set(ids)))

    @staticmethod
    def _tool_call_signature_from_identity(tool_calls_identity: str) -> tuple[tuple[str, str], ...]:
        """(provider id, digest of the function name and arguments) of each call in a canonical ``tool_calls`` identity.

        A provider can use an id again for a later call, so two tool-call turns
        are the same call only when the name and the arguments also match.
        Other fields of a call are not part of it.
        """
        if not tool_calls_identity:
            return ()
        try:
            parsed = json.loads(tool_calls_identity)
        except (TypeError, ValueError, json.JSONDecodeError):
            return ()
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return ()
        signature: list[tuple[str, str]] = []
        for call in parsed:
            if not isinstance(call, dict):
                continue
            call_id = ""
            for key in ("id", "call_id", "tool_call_id"):
                value = call.get(key)
                if isinstance(value, str) and value:
                    call_id = value
                    break
            function = call.get("function")
            if not isinstance(function, dict):
                function = {}
            arguments = function.get("arguments", call.get("arguments"))
            if not isinstance(arguments, str):
                arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            name = str(function.get("name") or call.get("name") or "")
            digest = hashlib.sha256(f"{name}\x1f{arguments}".encode("utf-8", "replace")).hexdigest()
            signature.append((call_id, digest))
        return tuple(sorted(signature))

    def _incoming_has_duplicate_provider_ids(self, messages: List[Dict[str, Any]]) -> bool:
        """True when the batch repeats a tool id or assistant call id.

        A plain transcript replay has one row per provider id.  Repeated ids
        mean the batch contains retries and must keep the legacy path, which
        preserves a retried result that reuses its id with new content.
        """
        seen_tool_ids: set[str] = set()
        seen_call_ids: set[str] = set()
        for msg in messages:
            role = str(msg.get("role") or "unknown")
            tool_call_id = str(msg.get("tool_call_id") or "")
            if role == "tool" and tool_call_id:
                if tool_call_id in seen_tool_ids:
                    return True
                seen_tool_ids.add(tool_call_id)
                continue
            call_ids = self._tool_call_ids_from_identity(
                self._stable_tool_calls_identity(msg.get("tool_calls"))
            )
            if role == "assistant" and call_ids:
                if any(call_id in seen_call_ids for call_id in call_ids):
                    return True
                seen_call_ids.update(call_ids)
        return False

    @staticmethod
    def _multiset_text_identity(
        role: str,
        text: str,
        tool_calls_identity: str = "",
    ) -> tuple[str, str]:
        """Normalised, hashed identity for rows that carry no unique provider id.

        Leading/trailing whitespace is tolerated because Hermes reloads
        user/assistant content through ``sanitize_context(content).strip()``
        (hermes_state_messages.py ``_loaded_view_content``) while LCM stored
        the live in-memory string.  Nothing else is normalised: internal
        whitespace and every other byte must still match.  Content is hashed
        so a whole-session scan never retains message bodies.
        """
        digest = hashlib.sha256()
        digest.update(role.encode("utf-8", "surrogatepass"))
        digest.update(b"\x00")
        digest.update((text or "").strip().encode("utf-8", "surrogatepass"))
        if tool_calls_identity:
            digest.update(b"\x00")
            digest.update(tool_calls_identity.encode("utf-8", "surrogatepass"))
        return ("text", digest.hexdigest())

    @staticmethod
    def _tool_call_aliases_from_identity(tool_calls_identity: str) -> frozenset[str]:
        """Every id spelling of every call in a canonical ``tool_calls`` identity.

        A call can carry more than one id (``id`` and ``call_id``), and its tool
        result can answer any of them.
        """
        if not tool_calls_identity:
            return frozenset()
        try:
            parsed = json.loads(tool_calls_identity)
        except (TypeError, ValueError, json.JSONDecodeError):
            return frozenset()
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return frozenset()
        aliases: set[str] = set()
        for call in parsed:
            if not isinstance(call, dict):
                continue
            for key in ("id", "call_id", "tool_call_id"):
                value = call.get(key)
                if isinstance(value, str) and value:
                    aliases.add(value)
        return frozenset(aliases)

    @staticmethod
    def _replay_content_digest(content: str) -> str:
        return hashlib.sha256((content or "").encode("utf-8", "surrogatepass")).hexdigest()

    def _multiset_persisted_indices(
        self,
        messages: List[Dict[str, Any]],
    ) -> tuple[list[int], int, int]:
        """Incoming indexes that the store does not prove it already holds.

        Returns ``(new_indices, stored_rows_scanned, already_stored)``.

        A row is a copy only when the store proves it. Every other row is new,
        so a doubt ends in a stored row and never in a dropped one. A message
        that says the same thing again (a second "Approved") has the same text
        as the first one, so text alone is never proof. A tool that runs again
        can have the same provider id, name and arguments as its first run, so
        a tool call alone is never proof at the end of the list.

        New messages arrive at the end of the list. The stored part of the list
        ends at the last user row whose host timestamp the store holds on a
        user row. Without such a row, the whole list counts as the end. The
        text is not part of this test, because a host can show one user message
        with a note added or removed, and it keeps the timestamp. This relies
        on one property of a host: two user messages with a tool call between
        them carry different timestamps.

        Proof, by kind of row:

        - User row with a host timestamp: the store holds a user row with the
          same timestamp and the same text. The n-th such incoming row needs n
          stored rows. The timestamp alone is not an identity.
        - The last user row of the list with one timestamp is a copy only when
          the last stored user row with that timestamp has its text. A platform
          can deliver an edited message as a new message with the timestamp of
          its first send, and an edit can go back to an earlier text. The store
          then holds that timestamp and text from the first send, in front of
          the row with the text between them.
        - Tool-call turn in the stored part: the store holds a turn with the
          same provider ids and the same calls (function name and arguments).
        - Tool result in the stored part: the store holds its id, and the call
          that it answers in this list is not new. A result answers the nearest
          earlier call in the list that carries its id in any spelling (``id``
          or ``call_id``). The content is not compared here, because a host can
          change the text of an old result. The first stored form stays.
        - Tool invocation at the end (a tool-call turn and the results in this
          list that answer it): the store holds the call, and each of those
          results with the same host timestamp or the same content, after the
          stored user row with that last timestamp. The content is compared as a
          replay identity, so a stored externalized or redacted form still
          matches. One new result makes the whole invocation new. So does a
          call without its result in the list: a host can hand over a tool
          call before the tool ran, and the stored call of an earlier run shows
          nothing about this run. After a user row with a new host timestamp,
          every invocation is new.
        - A provider id proves nothing when the provider uses it again: the
          incoming list holds the id on two calls or on two tool results, or
          the store holds it with two different calls. A row with such an id
          is new.
        - Any other row (assistant text, a user row without a timestamp): the
          store holds a row with the same role, host timestamp and text, or it
          holds the text directly after the stored copy of the row before it,
          or directly before the stored copy of the row after it, and that
          neighbor is a proven copy of one of the kinds above. Each stored
          neighbor pair proves one incoming row, and n incoming rows with one
          text need n stored rows with that text.
        - A row that only a stored neighbor could prove, and that sits after
          the last proven copy of the list, is new. The stored rows cannot tell
          the last answer of a stored turn from a new answer with the same
          text, because an earlier pass can have stored another row with that
          text at the same place.
        """
        skipped: set[int] = set()
        neighbor_keys: dict[int, tuple[Any, ...]] = {}
        text_keys: dict[int, tuple[str, str]] = {}
        stamped_text_keys: dict[int, tuple[Any, ...]] = {}
        call_rows: dict[int, tuple[tuple[str, ...], tuple[Any, ...]]] = {}
        call_results: dict[int, list[int]] = {}
        tool_rows: dict[int, str] = {}
        tool_owner: dict[int, int] = {}
        incoming_call_id_uses: dict[str, int] = {}
        incoming_tool_id_uses: dict[str, int] = {}
        alias_owner: dict[str, int] = {}
        for idx, msg in enumerate(messages):
            if (
                self._is_replayed_context_scaffold_message(msg)
                or self._matches_ignore_message_patterns(msg)
            ):
                # These are never persisted by _ingest_messages, so they can
                # neither be a delta nor proof that something is stored.
                skipped.add(idx)
                continue
            role = str(msg.get("role") or "unknown")
            tool_call_id = str(msg.get("tool_call_id") or "")
            if role == "tool" and tool_call_id:
                tool_rows[idx] = tool_call_id
                neighbor_keys[idx] = ("tool", tool_call_id)
                incoming_tool_id_uses[tool_call_id] = incoming_tool_id_uses.get(tool_call_id, 0) + 1
                owner = alias_owner.get(tool_call_id)
                if owner is not None:
                    tool_owner[idx] = owner
                    call_results[owner].append(idx)
                continue
            _role, content, _tool_call_id, identity_tool_calls = self._message_replay_identity(msg)
            if role == "assistant":
                call_ids = self._tool_call_ids_from_identity(identity_tool_calls)
                if call_ids:
                    signature = self._tool_call_signature_from_identity(identity_tool_calls)
                    call_rows[idx] = (call_ids, signature)
                    call_results[idx] = []
                    neighbor_keys[idx] = ("call", call_ids)
                    for call_id, _form in signature:
                        incoming_call_id_uses[call_id] = incoming_call_id_uses.get(call_id, 0) + 1
                    for alias in self._tool_call_aliases_from_identity(identity_tool_calls) | frozenset(call_ids):
                        alias_owner[alias] = idx
                    continue
            key = self._multiset_text_identity(role, content or "", identity_tool_calls)
            text_keys[idx] = key
            observed_at = _normalize_observed_at(msg.get("timestamp"))
            if observed_at is not None:
                if role == "user":
                    neighbor_keys[idx] = ("user", observed_at, key)
                else:
                    stamped_text_keys[idx] = ("stamped", observed_at, key)
        wanted_text_keys = set(text_keys.values())
        wanted_stamped_text_keys = set(stamped_text_keys.values())
        wanted_tool_ids = set(tool_rows.values())

        # Each "place" below is the number of a stored row in session order.
        stored_tool_ids: set[str] = set()
        stored_tool_times: dict[tuple[str, float], int] = {}
        stored_tool_contents: dict[tuple[str, str], int] = {}
        stored_tool_digests: dict[tuple[str, str], str] = {}
        stored_calls: dict[tuple[tuple[str, ...], tuple[Any, ...]], int] = {}
        stored_call_forms: dict[str, set[str]] = {}
        stored_user_rows: dict[tuple[Any, ...], int] = {}
        stored_user_places: dict[tuple[Any, ...], int] = {}
        stored_user_time_places: dict[float, int] = {}
        stored_stamped_text_rows: dict[tuple[Any, ...], int] = {}
        stored_text_rows: dict[tuple[str, str], int] = {}
        stored_after: dict[tuple[Any, ...], int] = {}
        stored_before: dict[tuple[Any, ...], int] = {}
        previous_neighbor_key: tuple[Any, ...] | None = None
        previous_text_key: tuple[str, str] | None = None
        stored_rows = 0
        for row in self._store.iter_session_message_identities(self._session_id):
            stored_rows += 1
            role = str(row.get("role") or "unknown")
            tool_call_id = str(row.get("tool_call_id") or "")
            neighbor_key: tuple[Any, ...] | None = None
            key: tuple[str, str] | None = None
            if role == "tool" and tool_call_id:
                stored_tool_ids.add(tool_call_id)
                neighbor_key = ("tool", tool_call_id)
                if tool_call_id in wanted_tool_ids:
                    # A session can hold many stored copies of one result, so the
                    # replay identity is built once for each stored form.
                    if row.get("observed_at") is not None:
                        stored_tool_times[(tool_call_id, float(row["observed_at"]))] = stored_rows
                    raw_form = (tool_call_id, self._replay_content_digest(str(row.get("content") or "")))
                    digest = stored_tool_digests.get(raw_form)
                    if digest is None:
                        digest = self._replay_content_digest(
                            self._compute_message_replay_identity(row, stored_row=True)[1]
                        )
                        stored_tool_digests[raw_form] = digest
                    stored_tool_contents[(tool_call_id, digest)] = stored_rows
            else:
                # Read the provider ids straight from the stored tool calls. The
                # canonical identity is built only for a row that needs it: a
                # stored row is computed directly, not through the per-pass memo,
                # because the memo keeps every row that it sees until the pass
                # ends, and this loop reads the whole session.
                call_ids = self._tool_call_ids_from_calls(row.get("tool_calls")) if role == "assistant" else ()
                if call_ids:
                    neighbor_key = ("call", call_ids)
                    if any(call_id in incoming_call_id_uses for call_id in call_ids):
                        signature = self._tool_call_signature_from_identity(
                            self._compute_message_replay_identity(row, stored_row=True)[3]
                        )
                        stored_calls[(call_ids, signature)] = stored_rows
                        for call_id, form in signature:
                            stored_call_forms.setdefault(call_id, set()).add(form)
                else:
                    _role, content, _tool_call_id, identity_tool_calls = self._compute_message_replay_identity(
                        row, stored_row=True
                    )
                    key = self._multiset_text_identity(role, content or "", identity_tool_calls)
                    if key in wanted_text_keys:
                        stored_text_rows[key] = stored_text_rows.get(key, 0) + 1
                    if row.get("observed_at") is not None:
                        if role == "user":
                            neighbor_key = ("user", float(row["observed_at"]), key)
                            stored_user_rows[neighbor_key] = stored_user_rows.get(neighbor_key, 0) + 1
                            stored_user_places[neighbor_key] = stored_rows
                            stored_user_time_places[neighbor_key[1]] = stored_rows
                        else:
                            stamped_key = ("stamped", float(row["observed_at"]), key)
                            if stamped_key in wanted_stamped_text_keys:
                                stored_stamped_text_rows[stamped_key] = stored_stamped_text_rows.get(stamped_key, 0) + 1
            if neighbor_key is None:
                if previous_neighbor_key is not None and key in wanted_text_keys:
                    pair = (previous_neighbor_key, key)
                    stored_after[pair] = stored_after.get(pair, 0) + 1
                previous_neighbor_key, previous_text_key = None, key
            else:
                if previous_text_key is not None and previous_text_key in wanted_text_keys:
                    pair = (previous_text_key, neighbor_key)
                    stored_before[pair] = stored_before.get(pair, 0) + 1
                previous_neighbor_key, previous_text_key = neighbor_key, None

        effective = [idx for idx in range(len(messages)) if idx not in skipped]
        reused_ids = {call_id for call_id, uses in incoming_call_id_uses.items() if uses > 1}
        reused_ids.update(call_id for call_id, uses in incoming_tool_id_uses.items() if uses > 1)
        reused_ids.update(call_id for call_id, forms in stored_call_forms.items() if len(forms) > 1)

        # User rows with a host timestamp. The stored part of the list ends at
        # the last one whose timestamp the store holds.
        proven_copies: set[int] = set()
        seen_user_rows: dict[tuple[Any, ...], int] = {}
        stored_part_end = -1
        stored_part_end_place = 0
        last_row_with_time: dict[float, int] = {}
        for position, idx in enumerate(effective):
            user_row = neighbor_keys.get(idx)
            if user_row is None or user_row[0] != "user":
                continue
            last_row_with_time[user_row[1]] = idx
            seen_user_rows[user_row] = seen_user_rows.get(user_row, 0) + 1
            if seen_user_rows[user_row] <= stored_user_rows.get(user_row, 0):
                proven_copies.add(idx)
            if user_row[1] in stored_user_time_places:
                stored_part_end = position
                stored_part_end_place = stored_user_time_places[user_row[1]]

        # An edited message can go back to an earlier text and keep the
        # timestamp of its first send. The store then holds another text with
        # that timestamp after the row that seems to be the copy, so the row is
        # new. The end of the stored part of the list does not change.
        for idx in last_row_with_time.values():
            user_row = neighbor_keys[idx]
            if idx in proven_copies and stored_user_places[user_row] < stored_user_time_places[user_row[1]]:
                proven_copies.discard(idx)

        def result_id_is_stored(idx: int) -> bool:
            return tool_rows[idx] in stored_tool_ids and tool_rows[idx] not in reused_ids

        def result_is_stored_at_the_end(idx: int) -> bool:
            if not result_id_is_stored(idx):
                return False
            tool_call_id = tool_rows[idx]
            observed_at = _normalize_observed_at(messages[idx].get("timestamp"))
            if observed_at is not None and stored_tool_times.get((tool_call_id, observed_at), 0) > stored_part_end_place:
                return True
            digest = self._replay_content_digest(self._message_replay_identity(messages[idx])[1])
            return stored_tool_contents.get((tool_call_id, digest), 0) > stored_part_end_place

        # Tool-call turns and tool results.
        after_new_user_row = False
        for position, idx in enumerate(effective):
            at_the_end = position > stored_part_end
            if idx in call_rows:
                call = call_rows[idx]
                results = call_results[idx]
                if call not in stored_calls or not reused_ids.isdisjoint(call[0]):
                    continue
                if not at_the_end:
                    proven_copies.add(idx)
                    proven_copies.update(result_idx for result_idx in results if result_id_is_stored(result_idx))
                elif (
                    not after_new_user_row
                    and stored_calls[call] > stored_part_end_place
                    and len(results) >= len(call[0])
                    and all(result_is_stored_at_the_end(result_idx) for result_idx in results)
                ):
                    proven_copies.add(idx)
                    proven_copies.update(results)
            elif idx in tool_rows:
                if idx in tool_owner:
                    continue
                if not at_the_end:
                    if result_id_is_stored(idx):
                        proven_copies.add(idx)
                elif not after_new_user_row and result_is_stored_at_the_end(idx):
                    proven_copies.add(idx)
            elif at_the_end and idx in neighbor_keys:
                after_new_user_row = True

        # Text rows that the store holds with the same host timestamp.
        stamp_proven: set[int] = set()
        seen_stamped_text_rows: dict[tuple[Any, ...], int] = {}
        for idx in effective:
            stamped_key = stamped_text_keys.get(idx)
            if stamped_key is None:
                continue
            seen_stamped_text_rows[stamped_key] = seen_stamped_text_rows.get(stamped_key, 0) + 1
            if seen_stamped_text_rows[stamped_key] <= stored_stamped_text_rows.get(stamped_key, 0):
                stamp_proven.add(idx)

        # Rows without proof of their own: the stored text beside a proven copy.
        # Nothing after the last proven copy of the list is a copy.
        last_proven_position = max(
            (position for position, idx in enumerate(effective) if idx in proven_copies or idx in stamp_proven),
            default=-1,
        )
        used_after: dict[tuple[Any, ...], int] = {}
        used_before: dict[tuple[Any, ...], int] = {}
        used_text_rows: dict[tuple[str, str], int] = {}
        for idx in stamp_proven:
            used_text_rows[text_keys[idx]] = used_text_rows.get(text_keys[idx], 0) + 1
        new_indices: list[int] = []
        already_stored = 0
        for position, idx in enumerate(effective):
            if idx in neighbor_keys:
                if idx in proven_copies:
                    already_stored += 1
                else:
                    new_indices.append(idx)
                continue
            if idx in stamp_proven:
                already_stored += 1
                continue
            key = text_keys[idx]
            proven = False
            if position < last_proven_position and used_text_rows.get(key, 0) < stored_text_rows.get(key, 0):
                if position > 0:
                    before_idx = effective[position - 1]
                    if before_idx in proven_copies:
                        pair = (neighbor_keys[before_idx], key)
                        if used_after.get(pair, 0) < stored_after.get(pair, 0):
                            used_after[pair] = used_after.get(pair, 0) + 1
                            proven = True
                if not proven and position + 1 < len(effective):
                    after_idx = effective[position + 1]
                    if after_idx in proven_copies:
                        pair = (key, neighbor_keys[after_idx])
                        if used_before.get(pair, 0) < stored_before.get(pair, 0):
                            used_before[pair] = used_before.get(pair, 0) + 1
                            proven = True
            if proven:
                used_text_rows[key] = used_text_rows.get(key, 0) + 1
                already_stored += 1
            else:
                new_indices.append(idx)
        return new_indices, stored_rows, already_stored

    def _reconcile_ingest_cursor_from_store(self, messages: List[Dict[str, Any]]) -> int:
        """Infer the in-memory cursor for an existing session after process restart."""
        self._reconcile_multiset_persist_indices = None
        if not self._session_id or not messages:
            return 0

        try:
            session_count = self._store.get_session_count(self._session_id)
        except Exception as exc:  # pragma: no cover - defensive only
            logger.debug("LCM ingest cursor reconciliation count failed: %s", exc)
            return 0
        if session_count <= 0:
            placeholder_budget = self._load_generated_ignored_placeholder_hash_counts()
            placeholder_ordinals = self._load_generated_ignored_placeholder_hash_ordinals()
            if placeholder_budget and placeholder_ordinals:
                consumed: dict[str, int] = {}
                cursor = 0
                for msg in messages:
                    text = text_content_for_pattern_matching(msg.get("content")) or ""
                    digest = self._active_replay_placeholder_digest(text)
                    if not digest:
                        break
                    consumed[digest] = consumed.get(digest, 0) + 1
                    ordinal = consumed[digest]
                    remaining = int(placeholder_budget.get(digest, 0) or 0)
                    if remaining <= 0 or ordinal not in placeholder_ordinals.get(digest, set()):
                        break
                    cursor += 1
                if cursor > 0:
                    self._record_ingest_reconciliation(
                        action="advanced cursor",
                        reason="replayed generated placeholders in empty session",
                        cursor=cursor,
                        incoming=len(messages),
                        session_count=session_count,
                        stored_tail_count=0,
                        effective_incoming=cursor,
                    )
                    return cursor
            return 0

        tail_limit = min(max(len(messages) * 4, 64), session_count)
        stored_rows = self._store.get_session_tail(self._session_id, limit=tail_limit)
        if not stored_rows:
            return 0
        stored_tail_rows = [
            row
            for row in stored_rows
            if not self._matches_ignore_message_patterns(row, stored_row=True)
        ]
        stored_tail = [
            self._message_replay_identity(row, stored_row=True)
            for row in stored_tail_rows
        ]
        cursor = self._find_reconciled_cursor_for_store_tail(
            messages,
            stored_tail,
            stored_tail_rows=stored_tail_rows,
            allow_empty_prefix=True,
            session_count=len(stored_tail),
            raw_session_count=session_count,
        )
        if cursor is not None and cursor > 0:
            scaffold_only_prefix = not self._effective_replay_identities(messages[:cursor])
            reason = (
                "skipped scaffold-only prefix"
                if scaffold_only_prefix
                else "replayed durable tail"
            )
            if scaffold_only_prefix:
                # A scaffold-only prefix (e.g. a "[Recent Summary ...]" block
                # from a compacted session) proves nothing about the rest of
                # the batch: after a rebind the remainder is usually the whole
                # durable fresh tail again.  When the remainder is replay-sized,
                # filter it with the same multiset difference as the ambiguous
                # path instead of persisting it wholesale.  The prefix holds no
                # effective identities, so counting over the full list is the
                # full-replay case.
                remainder_effective = len(self._effective_replay_identities(messages[cursor:]))
                remainder_replay_sized = remainder_effective >= _MULTISET_REPLAY_MIN_MESSAGES or (
                    remainder_effective >= _MULTISET_REPLAY_SHORT_MIN_MESSAGES
                    and not self._incoming_has_duplicate_provider_ids(messages[cursor:])
                )
                if remainder_replay_sized:
                    new_indices, _scanned, already_stored = self._multiset_persisted_indices(messages)
                    new_indices = [idx for idx in new_indices if idx >= cursor]
                    effective_incoming = len(self._effective_replay_identities(messages))
                    if not new_indices:
                        self._record_ingest_reconciliation(
                            action="advanced cursor",
                            reason="skipped scaffold-only prefix; persisted multiset delta",
                            cursor=len(messages),
                            incoming=len(messages),
                            session_count=session_count,
                            stored_tail_count=len(stored_tail),
                            effective_incoming=effective_incoming,
                            already_stored=already_stored,
                            persisted=0,
                        )
                        return len(messages)
                    self._reconcile_multiset_persist_indices = new_indices
                    self._record_ingest_reconciliation(
                        action="persisted delta",
                        reason="skipped scaffold-only prefix; persisted multiset delta",
                        cursor=0,
                        incoming=len(messages),
                        session_count=session_count,
                        stored_tail_count=len(stored_tail),
                        effective_incoming=effective_incoming,
                        already_stored=already_stored,
                        persisted=len(new_indices),
                    )
                    return 0
            self._record_ingest_reconciliation(
                action="advanced cursor",
                reason=reason,
                cursor=cursor,
                incoming=len(messages),
                session_count=session_count,
                stored_tail_count=len(stored_tail),
                effective_incoming=len(self._effective_replay_identities(messages)),
            )
            logger.debug(
                "LCM reconciled ingest cursor after existing-session bind: session=%s cursor=%d incoming=%d stored_tail=%d session_count=%d reason=%s",
                self._session_id,
                cursor,
                len(messages),
                len(stored_tail),
                session_count,
                reason,
            )
            return cursor

        incoming_identities = self._effective_replay_identities(messages)
        stored_head_rows = self._store.get_session_messages(
            self._session_id,
            limit=tail_limit,
        )
        stored_head = [self._message_replay_identity(row, stored_row=True) for row in stored_head_rows]
        # Stale-snapshot proof uses the raw durable prefix.  Ignore-message
        # filters may suppress noisy rows for tail reconciliation, but filtered
        # history alone must not create replay evidence for skipping a batch.
        incoming_has_unproofed_raw_persisted_marker = any(
            str(msg.get("role") or "") == "tool"
            and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
            and recover_hermes_persisted_output_with_file_stat(
                normalize_content_value(msg.get("content")) or ""
            )
            is None
            for msg in messages
        )
        if (
            not incoming_has_unproofed_raw_persisted_marker
            and self._is_suspicious_stale_no_overlap_snapshot(
                incoming_identities,
                stored_tail,
                stored_head,
            )
        ):
            self._record_ingest_reconciliation(
                action="skipped batch",
                reason="skipped stale no-overlap snapshot",
                cursor=len(messages),
                incoming=len(messages),
                session_count=session_count,
                stored_tail_count=len(stored_tail),
                effective_incoming=len(incoming_identities),
            )
            logger.warning(
                "LCM skipped stale no-overlap snapshot after existing-session bind: session=%s incoming=%d effective_incoming=%d stored_tail=%d session_count=%d",
                self._session_id,
                len(messages),
                len(incoming_identities),
                len(stored_tail),
                session_count,
            )
            return len(messages)

        effective_incoming = len(incoming_identities)
        replay_sized = effective_incoming >= _MULTISET_REPLAY_MIN_MESSAGES or (
            effective_incoming >= _MULTISET_REPLAY_SHORT_MIN_MESSAGES
            and not self._incoming_has_duplicate_provider_ids(messages)
        )
        if replay_sized:
            new_indices, stored_rows_scanned, already_stored = self._multiset_persisted_indices(
                messages
            )
            persisted = len(new_indices)
            logger.debug(
                "LCM multiset ingest reconciliation: session=%s incoming=%d effective_incoming=%d stored_rows_scanned=%d already_stored=%d persisted=%d",
                self._session_id,
                len(messages),
                effective_incoming,
                stored_rows_scanned,
                already_stored,
                persisted,
            )
            if not new_indices:
                # Everything in the replay is already durable.  Return the full
                # length so the engine cursor lands exactly where the successful
                # full-replay path leaves it (no prefix bookkeeping involved).
                self._record_ingest_reconciliation(
                    action="advanced cursor",
                    reason="persisted multiset delta",
                    cursor=len(messages),
                    incoming=len(messages),
                    session_count=session_count,
                    stored_tail_count=len(stored_tail),
                    effective_incoming=effective_incoming,
                    already_stored=already_stored,
                    persisted=0,
                )
                return len(messages)

            # Only the indexes selected by the multiset difference are persisted;
            # _ingest_messages applies them in incoming order and still finishes
            # with its in-memory cursor at len(messages).
            self._reconcile_multiset_persist_indices = new_indices
            self._record_ingest_reconciliation(
                action="persisted delta",
                reason="persisted multiset delta",
                cursor=0,
                incoming=len(messages),
                session_count=session_count,
                stored_tail_count=len(stored_tail),
                effective_incoming=effective_incoming,
                already_stored=already_stored,
                persisted=persisted,
            )
            return 0

        self._record_ingest_reconciliation(
            action="persisted batch",
            reason="persisted ambiguous delta",
            cursor=0,
            incoming=len(messages),
            session_count=session_count,
            stored_tail_count=len(stored_tail),
            effective_incoming=effective_incoming,
        )
        return 0

    def _raw_externalized_placeholder_replay_identity(self, msg: Dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(msg.get("role") or "unknown"),
            normalize_content_value(msg.get("content")) or "",
            self._stable_tool_calls_identity(msg.get("tool_calls")),
            str(msg.get("tool_call_id") or ""),
        )

    def _get_store_id_map_for_messages(self, messages: List[Dict[str, Any]]) -> dict[int, int]:
        """Map current raw message objects back to store_ids in stable order.

        Matching starts strictly after ``_last_compacted_store_id`` so repeated
        content from older already-compacted history cannot hijack the mapping.
        Synthetic summary messages simply fail to match and are skipped.  When
        active context has more occurrences of an identical replay identity than
        the store has, the surplus earliest active occurrences are treated as
        synthetic/carry-over and left unmapped so they cannot steal later stored
        literal copies with the same content.
        """
        candidates: list[Dict[str, Any]] = []
        next_candidate_after = self._last_compacted_store_id
        while True:
            page = self._store.get_session_messages_after(
                self._session_id,
                after_store_id=next_candidate_after,
            )
            if not page:
                break
            candidates.extend(page)
            next_candidate_after = page[-1]["store_id"]
        active_identity_counts: dict[tuple[Any, ...], int] = {}
        for msg in messages:
            identity = self._message_replay_identity(msg)
            active_identity_counts[identity] = active_identity_counts.get(identity, 0) + 1
        stored_identity_counts: dict[tuple[Any, ...], int] = {}
        stored_cleanup_identity_counts: dict[tuple[Any, ...], int] = {}
        # Capture each candidate's identity (and its cleanup variant) here - both
        # are already computed for the counts below, so this adds no work. The
        # match-probe loops reuse them instead of recomputing
        # _message_replay_identity(stored_row=True) for every (message, probe)
        # pair. That call is expensive when a stored row carries an externalized
        # payload (JSON canonicalization + a payload-file read), so eliminating
        # the O(candidates^2) recomputes removes repeated disk reads on
        # tool-output-heavy histories. Raw-placeholder identities stay lazy (see
        # the memo below) since most rows never need them.
        stored_identities: list[tuple[Any, ...]] = []
        stored_cleanup_identities: list[Optional[tuple[Any, ...]]] = []
        for stored in candidates:
            identity = self._message_replay_identity(stored, stored_row=True)
            stored_identities.append(identity)
            cleanup_identity = self._active_cleanup_replay_identity(identity)
            stored_cleanup_identities.append(cleanup_identity)
            stored_identity_counts[identity] = stored_identity_counts.get(identity, 0) + 1
            if cleanup_identity is not None:
                stored_cleanup_identity_counts[cleanup_identity] = (
                    stored_cleanup_identity_counts.get(cleanup_identity, 0) + 1
                )

        # Lazily memoize raw-placeholder identities: only the placeholder-ref
        # paths need them, and most histories have few (or none), so computing
        # them on demand keeps the common case free.
        _raw_placeholder_identity_cache: dict[int, tuple[str, str, str, str]] = {}

        def stored_raw_placeholder_identity(probe_idx: int) -> tuple[str, str, str, str]:
            cached = _raw_placeholder_identity_cache.get(probe_idx)
            if cached is None:
                cached = self._raw_externalized_placeholder_replay_identity(candidates[probe_idx])
                _raw_placeholder_identity_cache[probe_idx] = cached
            return cached
        active_surplus_skips: dict[tuple[Any, ...], int] = {}
        generated_surplus_skip_message_ids: set[int] = set()
        generated_placeholder_message_ids = getattr(
            self,
            "_generated_ignored_active_replay_placeholder_message_ids",
            set(),
        )
        for identity, active_count in active_identity_counts.items():
            wanted_cleanup_identity = self._active_cleanup_replay_identity(identity)
            stored_exact = stored_identity_counts.get(identity, 0)
            stored_cleanup = 0
            if wanted_cleanup_identity is not None:
                stored_cleanup = stored_cleanup_identity_counts.get(wanted_cleanup_identity, 0)
            stored_available = max(stored_exact, stored_cleanup)
            if active_count > stored_available:
                surplus_count = active_count - stored_available
                for msg in messages:
                    if surplus_count <= 0:
                        break
                    if id(msg) not in generated_placeholder_message_ids:
                        continue
                    if self._message_replay_identity(msg) != identity:
                        continue
                    generated_surplus_skip_message_ids.add(id(msg))
                    surplus_count -= 1
                if surplus_count > 0:
                    active_surplus_skips[identity] = surplus_count

        placeholder_identity_counts: dict[tuple[str, str, str, str], int] = {}
        for msg in messages:
            msg_content = normalize_content_value(msg.get("content")) or ""
            if msg.get("store_id") is None and self._content_has_externalized_placeholder_ref(msg_content):
                raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
                placeholder_identity_counts[raw_identity] = placeholder_identity_counts.get(raw_identity, 0) + 1
        self._current_compress_placeholder_identity_counts = placeholder_identity_counts

        def find_raw_placeholder_match_index(
            raw_identity: tuple[str, str, str, str],
            start_idx: int,
        ) -> int | None:
            probe_idx = start_idx
            while probe_idx < len(candidates):
                if stored_raw_placeholder_identity(probe_idx) == raw_identity:
                    return probe_idx
                probe_idx += 1
            return None

        def find_message_match_index(msg: Dict[str, Any], start_idx: int) -> int | None:
            msg_content = normalize_content_value(msg.get("content")) or ""
            if msg.get("store_id") is None and self._content_has_externalized_placeholder_ref(msg_content):
                raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
                raw_match_idx = find_raw_placeholder_match_index(raw_identity, start_idx)
                if raw_match_idx is not None:
                    return raw_match_idx

            message_identity = self._message_replay_identity(msg)
            wanted_cleanup_identity = self._active_cleanup_replay_identity(message_identity)
            probe_idx = start_idx
            while probe_idx < len(candidates):
                stored_identity = stored_identities[probe_idx]
                if stored_identity == message_identity:
                    return probe_idx
                if (
                    wanted_cleanup_identity is not None
                    and stored_cleanup_identities[probe_idx] == wanted_cleanup_identity
                ):
                    return probe_idx
                probe_idx += 1
            return None

        def matched_remaining_message_ids(
            message_start_idx: int,
            start_store_idx: int,
            surplus_skips: dict[tuple[Any, ...], int],
        ) -> set[int]:
            matched_message_ids: set[int] = set()
            local_surplus_skips = dict(surplus_skips)
            probe_idx = start_store_idx
            for remaining_msg in messages[message_start_idx:]:
                msg_content = normalize_content_value(remaining_msg.get("content")) or ""
                if (
                    remaining_msg.get("store_id") is None
                    and self._content_has_externalized_placeholder_ref(msg_content)
                ):
                    raw_identity = self._raw_externalized_placeholder_replay_identity(remaining_msg)
                    raw_match_idx = find_raw_placeholder_match_index(raw_identity, probe_idx)
                    if raw_match_idx is not None:
                        matched_message_ids.add(id(remaining_msg))
                        probe_idx = raw_match_idx + 1
                        continue
                message_identity = self._message_replay_identity(remaining_msg)
                if id(remaining_msg) in generated_surplus_skip_message_ids:
                    continue
                surplus = local_surplus_skips.get(message_identity, 0)
                if surplus > 0:
                    local_surplus_skips[message_identity] = surplus - 1
                    continue
                match_idx = find_message_match_index(remaining_msg, probe_idx)
                if match_idx is None:
                    continue
                matched_message_ids.add(id(remaining_msg))
                probe_idx = match_idx + 1
            return matched_message_ids

        ids_by_message_id: dict[int, int] = {}
        store_idx = 0
        for msg_idx, msg in enumerate(messages):
            msg_content = normalize_content_value(msg.get("content")) or ""
            if msg.get("store_id") is None and self._content_has_externalized_placeholder_ref(msg_content):
                raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
                if placeholder_identity_counts.get(raw_identity, 0) > 1:
                    match_idx = find_raw_placeholder_match_index(raw_identity, store_idx)
                    if match_idx is not None:
                        ids_by_message_id[id(msg)] = candidates[match_idx]["store_id"]
                        store_idx = match_idx + 1
                else:
                    # Prefer a later duplicate only when it does not orphan
                    # later active messages that still need monotonic mapping.
                    first_match_idx = find_raw_placeholder_match_index(raw_identity, store_idx)
                    if first_match_idx is not None:
                        baseline_suffix_ids = matched_remaining_message_ids(
                            msg_idx + 1,
                            first_match_idx + 1,
                            active_surplus_skips,
                        )
                    else:
                        baseline_suffix_ids = set()
                    probe_idx = len(candidates) - 1
                    while first_match_idx is not None and probe_idx >= first_match_idx:
                        stored = candidates[probe_idx]
                        if stored_raw_placeholder_identity(probe_idx) == raw_identity:
                            candidate_suffix_ids = matched_remaining_message_ids(
                                msg_idx + 1,
                                probe_idx + 1,
                                active_surplus_skips,
                            )
                            if not baseline_suffix_ids.issubset(candidate_suffix_ids):
                                probe_idx -= 1
                                continue
                            ids_by_message_id[id(msg)] = stored["store_id"]
                            store_idx = probe_idx + 1
                            break
                        probe_idx -= 1
                if id(msg) in ids_by_message_id:
                    continue
            message_identity = self._message_replay_identity(msg)
            if id(msg) in generated_surplus_skip_message_ids:
                continue
            surplus = active_surplus_skips.get(message_identity, 0)
            if surplus > 0:
                active_surplus_skips[message_identity] = surplus - 1
                continue
            match_idx = find_message_match_index(msg, store_idx)
            if match_idx is not None:
                ids_by_message_id[id(msg)] = candidates[match_idx]["store_id"]
                store_idx = match_idx + 1

        return ids_by_message_id
