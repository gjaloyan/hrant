"""Per-conversation message history — the Hermes way, on Hrant's data.

Until 2026-10-10 the agent's memory of the conversation was CONVERSATION:
ONE list of the last 20 exchanges for EVERYONE (owner, wife, brother, WebUI
and supervisor turns mixed), answers cut to 500 characters, rendered as text
into a system prompt that was rebuilt on every turn. Two consequences,
measured on prod:

  * forgetting — a busy afternoon pushed the owner's own morning out of the
    window, and a 3 000-character answer survived as its first 500;
  * cost — the rebuilt system prompt broke the provider's prompt cache, so a
    Telegram turn paid ~77k fresh input tokens; Hermes on the same box, which
    keeps a byte-stable system prompt and appends the conversation as
    messages, paid ~26k for a larger context.

This module keeps one append-only log per conversation thread
(`session_key`) and hands the turn loop real user/assistant messages. The
window is trimmed in CHUNKS: when the history passes its budget, the oldest
turns are dropped down to half the budget, so the prefix the cache keys on
stays identical for many turns instead of shifting by one every turn. What
falls out leaves a short trace (the requests it answered) and remains
reachable through memory search and the nightly consolidation.

Storage: `<knowledge>/threads/<hash>.jsonl` (+ `<hash>.state.json`).
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

DEFAULT_BUDGET_CHARS = 40_000
MAX_TEXT_CHARS = 8_000
MAX_TRACE_CHARS = 1_500
_LOCK = threading.RLock()


def _root() -> Path:
    from . import paths
    return paths.knowledge_dir() / "threads"


def _stem(session_key: str) -> str:
    return hashlib.sha1((session_key or "").encode("utf-8")).hexdigest()[:20]


def budget_chars() -> int:
    try:
        from .config import CONFIG
        v = int((CONFIG._data.get("conversation") or {}).get(
            "history_budget_chars", DEFAULT_BUDGET_CHARS))
        return v if v >= 2_000 else DEFAULT_BUDGET_CHARS
    except Exception:
        return DEFAULT_BUDGET_CHARS


def enabled() -> bool:
    """Kill switch: `conversation.threaded_history: false` in config.yaml
    restores the old text-in-system-prompt behaviour without a deploy."""
    try:
        from .config import CONFIG
        return bool((CONFIG._data.get("conversation") or {}).get(
            "threaded_history", True))
    except Exception:
        return True


def _clip(text: str, n: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= n else text[:n] + " …[cut]"


class ThreadStore:
    def __init__(self, root: Optional[Path] = None):
        self._root_override = root

    @property
    def root(self) -> Path:
        return self._root_override or _root()

    def _paths(self, session_key: str) -> tuple[Path, Path]:
        stem = _stem(session_key)
        return self.root / f"{stem}.jsonl", self.root / f"{stem}.state.json"

    def _read_state(self, path: Path) -> dict:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def _write_state(self, path: Path, state: dict) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def _entries(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict):
                out.append(row)
        return out

    # ── write ─────────────────────────────────────────────────────────

    def append(self, session_key: str, *, user: str, answer: str,
               attachments: Optional[list] = None, trace: str = "",
               turn_id: str = "", session_id: str = "",
               speaker_id: str = "") -> None:
        """One finished exchange. Never raises."""
        if not session_key or not (user or "").strip():
            return
        try:
            log_path, state_path = self._paths(session_key)
            row = {
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "user": _clip(user, MAX_TEXT_CHARS),
                "answer": _clip(answer, MAX_TEXT_CHARS),
            }
            if attachments:
                row["attachments"] = [str(a) for a in attachments if a][:16]
            if trace:
                row["trace"] = _clip(trace, MAX_TRACE_CHARS)
            if turn_id:
                row["turn_id"] = turn_id
            with _LOCK:
                self.root.mkdir(parents=True, exist_ok=True)
                state = self._read_state(state_path)
                if session_id and state.get("session_id") not in (None, session_id):
                    # The user started a new session (WebUI "new chat"):
                    # the model starts from a clean thread, the log stays.
                    state = {"window_start": len(self._entries(log_path))}
                if session_id:
                    state["session_id"] = session_id
                if speaker_id:
                    state["speaker_id"] = speaker_id
                state.setdefault("session_key", session_key)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._write_state(state_path, state)
        except Exception as e:
            log.debug("threads.append failed: %s", e)

    def seed_if_empty(self, session_key: str, turns: list[dict]) -> int:
        """Start a thread that has never been written from the old shared
        buffer, so switching to threads does not erase the conversation in
        progress. Returns how many exchanges were copied. Never raises."""
        try:
            log_path, _state = self._paths(session_key)
            if log_path.exists():
                return 0
            n = 0
            for t in turns or []:
                if (t.get("user") or "").strip():
                    self.append(session_key, user=t.get("user") or "",
                                answer=t.get("answer") or "")
                    n += 1
            if n == 0:
                # Remember that there was nothing, so we do not try again.
                self.root.mkdir(parents=True, exist_ok=True)
                log_path.touch()
            return n
        except Exception as e:
            log.debug("threads.seed_if_empty failed: %s", e)
            return 0

    # ── read ──────────────────────────────────────────────────────────

    def messages(self, session_key: str, *, budget: Optional[int] = None,
                 session_id: str = "") -> list[dict]:
        """[{role, content}] for the model, oldest first. Never raises."""
        if not session_key:
            return []
        budget = budget or budget_chars()
        try:
            log_path, state_path = self._paths(session_key)
            with _LOCK:
                entries = self._entries(log_path)
                state = self._read_state(state_path)
                if session_id and state.get("session_id") not in (None, session_id):
                    return []  # a new session that has not spoken yet
                start = int(state.get("window_start") or 0)
                window = entries[start:]
                sizes = [_entry_chars(e) for e in window]
                if sum(sizes) > budget:
                    # Trim in a chunk, down to half the budget, so the
                    # cached prefix survives the next several turns.
                    keep, total = 0, 0
                    for size in reversed(sizes):
                        if total + size > budget // 2 and keep:
                            break
                        total += size
                        keep += 1
                    dropped = window[:len(window) - keep]
                    start += len(dropped)
                    state["window_start"] = start
                    state["earlier"] = _merge_earlier(state.get("earlier"), dropped)
                    self._write_state(state_path, state)
                    window = entries[start:]
                earlier = state.get("earlier") or ""
        except Exception as e:
            log.debug("threads.messages failed: %s", e)
            return []
        out: list[dict] = []
        if earlier:
            out.append({"role": "user", "content": (
                "[Earlier in this conversation — trimmed from the window; "
                "background only, not a request]\n" + earlier)})
            out.append({"role": "assistant", "content": "(noted)"})
        for e in window:
            out.append({"role": "user", "content": _render_user(e)})
            out.append({"role": "assistant", "content": _render_answer(e)})
        return out


def iter_day(store: "ThreadStore", since: str, until: str = "9999"):
    """(session_key, speaker_id, rows) for every thread with exchanges whose
    `ts` ("YYYY-MM-DD HH:MM:SS") falls in [since, until). For the nightly
    review. Never raises."""
    try:
        files = sorted(store.root.glob("*.jsonl"))
    except Exception:
        return
    for log_path in files:
        try:
            state = store._read_state(log_path.with_suffix(".state.json"))
            rows = [r for r in store._entries(log_path)
                    if since <= str(r.get("ts") or "") < until]
        except Exception:
            continue
        if rows:
            yield (state.get("session_key") or log_path.stem,
                   state.get("speaker_id") or "", rows)


def _entry_chars(e: dict) -> int:
    return len(_render_user(e)) + len(_render_answer(e))


def _render_user(e: dict) -> str:
    text = e.get("user") or ""
    atts = e.get("attachments") or []
    if atts:
        text += "\n[attached: " + ", ".join(f"sha256={a}" for a in atts) + "]"
    return text


def _render_answer(e: dict) -> str:
    text = e.get("answer") or "(no answer recorded)"
    if e.get("trace"):
        text += "\n\n[what I did that turn: " + e["trace"] + "]"
    return text


def _merge_earlier(previous: Optional[str], dropped: list[dict]) -> str:
    """A deterministic trace of what fell out: the requests, briefly. The
    nightly consolidation is where a real summary belongs."""
    lines = [l for l in (previous or "").splitlines() if l.strip()]
    for e in dropped:
        lines.append(f"- {e.get('ts', '')[:16]} {(e.get('user') or '')[:160]}".rstrip())
    return "\n".join(lines[-30:])


def trace_digest(tool_names: list[str], media_paths: list[str]) -> str:
    """Compact record of a turn's tools for the next turn's history."""
    if not tool_names and not media_paths:
        return ""
    counts: dict[str, int] = {}
    for n in tool_names:
        counts[n] = counts.get(n, 0) + 1
    parts = [n if c == 1 else f"{n} ×{c}" for n, c in counts.items()]
    out = ", ".join(parts)
    if media_paths:
        out += "; delivered " + ", ".join(media_paths[:5])
    return out


THREADS = ThreadStore()
