"""Structured user-question primitive (AskUserQuestion analogue).

User request (May 2026): give the agent a clean way to ask for a
decision instead of writing free-form "should I X or Y?" prose. The
shape mirrors Anthropic's `AskUserQuestion` block — a single
question, a short header, optional explanation of why the answer is
needed, 2-4 options (each with label + description), an optional
recommended/default index, single- or multi-select.

How a turn uses it:

  1. The agent calls the `ask_user(...)` builtin tool with the
     structured payload. The tool persists a `PendingQuestion` JSON
     blob under `<data_dir>/jobs/questions/<question_id>.json` and
     returns the sentinel:

       { "ok": true, "awaiting_input": true,
         "question_id": "q-abcdef", "note": "..." }

  2. `unified_agent.run_unified` detects the sentinel in the tool
     loop's last tool result and short-circuits the turn — instead
     of letting the LLM compose more text, it ends with an
     `AgentAnswer` whose `.question` field points at the pending
     payload. The SSE stream emits a `question` event so the WebUI
     can render an interactive card immediately.

  3. Telegram channel renders the question via inline-keyboard
     buttons (one per option). WebUI renders a clean `AskUserCard`.

  4. User picks an option → the channel posts the choice through
     `/api/chat/answer-question` (WebUI) or a callback handler
     (Telegram) → a fresh agent turn fires with the user's choice
     as the new user message. The conversation history naturally
     carries the previous turn's question text, so the agent picks
     up without extra plumbing.

The store is file-backed (same atomic-rename pattern as background
jobs) so a service restart doesn't lose an in-flight question.
"""
from __future__ import annotations

import json
import logging
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .. import paths


log = logging.getLogger(__name__)

# Sentinel key inside a tool's JSON return that tells `run_unified`
# "stop and surface this question to the user". Carries the question
# id so the loop knows what to attach to the AgentAnswer.
AWAITING_INPUT_KEY = "awaiting_input"


@dataclass
class QuestionOption:
    """One choice the user can pick. Mirrors the AskUserQuestion
    option shape (label + description) plus an `id` we use for
    callback-data so the channel can identify what was clicked."""
    id: str
    label: str
    description: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PendingQuestion:
    """One question waiting for the user's answer."""
    question_id: str
    asked_at: float
    question: str
    why: str
    # Short label (≤12 chars by convention) shown as a chip/tag in
    # the question card header. Helps the user spot "auth method"
    # vs "library choice" at a glance when multiple questions
    # surface in a session.
    header: str
    options: list[dict]
    multi_select: bool
    default_option_id: str
    # Who asked + where to route the answer back to. Mirrors the
    # background-job `original_speaker_id` / `original_chat_id`
    # fields — same purpose, different store.
    asker_speaker_id: str
    asker_chat_id: Optional[int]
    channel: str
    # Audit trail. Empty until the user picks.
    answered: bool = False
    answer_at: Optional[float] = None
    # The chosen option id (or comma-separated list when
    # multi_select=True).
    answer_choice: str = ""
    # Free-text fallback (e.g. user picked "Other" on Telegram).
    answer_text: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PendingQuestion":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid})


_STORE_LOCK = threading.RLock()


class QuestionStore:
    """File-backed store. One JSON file per question_id under
    `<data_dir>/jobs/questions/`. Concurrent writes (the user clicks
    a Telegram button while the WebUI also submits) are serialized
    via the module-level lock + atomic .tmp rename."""

    def __init__(self, root: Optional[Path] = None):
        self._root_override = root

    @property
    def _root(self) -> Path:
        if self._root_override is not None:
            return self._root_override
        return paths.data_dir(require=False) / "jobs" / "questions"

    def _path(self, question_id: str) -> Path:
        # Sanitize: only the secrets.token_hex shape we generate is
        # allowed through, so this is belt-and-suspenders against
        # someone passing "../" via the API.
        safe = "".join(
            c for c in (question_id or "")
            if c.isalnum() or c in "-_"
        )
        if not safe:
            raise ValueError("question_id is empty / invalid")
        return self._root / f"{safe}.json"

    def get(self, question_id: str) -> Optional[PendingQuestion]:
        try:
            p = self._path(question_id)
        except ValueError:
            return None
        if not p.exists():
            return None
        try:
            with _STORE_LOCK:
                raw = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(
                "question store: failed to load %s: %s", question_id, e,
            )
            return None
        if not isinstance(raw, dict):
            return None
        try:
            return PendingQuestion.from_dict(raw)
        except Exception as e:
            log.warning("question store: bad shape for %s: %s", question_id, e)
            return None

    def put(self, q: PendingQuestion) -> None:
        with _STORE_LOCK:
            p = self._path(q.question_id)
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            tmp.write_text(
                json.dumps(q.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(p)

    def mark_answered(
        self,
        question_id: str,
        *,
        choice: str,
        text: str = "",
    ) -> Optional[PendingQuestion]:
        """Idempotent: a second click after answer lands returns the
        existing record without double-firing the resume turn."""
        with _STORE_LOCK:
            q = self.get(question_id)
            if q is None:
                return None
            if q.answered:
                return q
            q.answered = True
            q.answer_at = time.time()
            q.answer_choice = choice or ""
            q.answer_text = (text or "").strip()
            self.put(q)
            return q

    def gc_old(self, *, older_than_days: int = 7) -> int:
        """Delete questions older than N days — answered (by answer time) AND
        abandoned/never-answered (by ask time). Called from the bg-job
        watchdog's daily sweep so the store doesn't grow unboundedly. Returns
        the number of files removed.

        Audit follow-up: every `ask_user` call writes a JSON blob; without
        cleanup the store accumulates one stale file per question forever.
        Pre-fix only ANSWERED questions were collected, so abandoned ones lived
        forever (15-day-old open questions seen on prod, polluting the
        open-list and leaving stale keyboards). 7-day default gives plenty of
        time to revisit a question but caps long-term cruft.
        """
        cutoff = time.time() - max(1, int(older_than_days)) * 86400
        removed = 0
        try:
            root = self._root
            if not root.exists():
                return 0
            with _STORE_LOCK:
                for path in root.glob("*.json"):
                    if path.name.endswith(".tmp"):
                        continue
                    try:
                        raw = json.loads(path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if not isinstance(raw, dict):
                        continue
                    # Answered -> expire by answer time; abandoned -> by ask
                    # time. Both past the cutoff are dead weight.
                    answered = bool(raw.get("answered"))
                    ts = (raw.get("answer_at") if answered
                          else raw.get("asked_at")) or 0
                    if ts >= cutoff:
                        continue
                    try:
                        path.unlink()
                        removed += 1
                    except Exception as e:
                        log.debug(
                            "question gc: unlink %s failed: %s", path, e,
                        )
        except Exception as e:
            log.warning("question gc sweep failed: %s", e)
        return removed

    def supersede_open(self, speaker_id: str, keep_qid: str) -> int:
        """Mark a speaker's OTHER open questions answered (superseded) so only
        the latest is live. A cascade of questions otherwise leaves a pile of
        live keyboards that cross-wire resume turns (tapping a stale one fires
        the wrong resume — the "My choice id: N doesn't match" failures). This
        does NOT trigger a resume; it just flips the flag, so taps on a
        superseded keyboard get an 'already answered' toast. Returns the count.
        """
        if not speaker_id:
            return 0
        n = 0
        with _STORE_LOCK:
            for q in self.list_open(limit=100):
                if q.question_id == keep_qid:
                    continue
                if (q.asker_speaker_id or "") != speaker_id:
                    continue
                q.answered = True
                q.answer_at = time.time()
                q.answer_choice = ""
                q.answer_text = "(superseded by a newer question)"
                self.put(q)
                n += 1
        return n

    def list_open(self, *, limit: int = 50) -> list[PendingQuestion]:
        """All un-answered questions, newest first. Used by the
        WebUI to restore an in-flight question on page reload."""
        out: list[PendingQuestion] = []
        try:
            root = self._root
            if not root.exists():
                return []
            for path in sorted(root.glob("*.json"), reverse=True):
                if path.name.endswith(".tmp"):
                    continue
                try:
                    raw = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(raw, dict):
                    continue
                try:
                    q = PendingQuestion.from_dict(raw)
                except Exception:
                    continue
                if not q.answered:
                    out.append(q)
                if len(out) >= limit:
                    break
        except Exception as e:
            log.warning("question store: list_open failed: %s", e)
        return out


STORE = QuestionStore()


# ─── Telegram callback dispatcher (q: prefix) ─────────────────────


def _register_telegram_callback() -> None:
    """Wire `q:<question_id>:<option_id>` inline-keyboard clicks
    into the tg_interactive dispatcher. Hits on every bot startup —
    `register_callback_handler` is idempotent so re-registering is
    a no-op.

    On click:
      1. Look up the question, refuse if it's missing or already
         answered (avoid double-firing the resume turn).
      2. Mark answered with the chosen option_id.
      3. Synthesize a "user message" describing the choice + fire a
         fresh agent turn under the original speaker_id so the
         conversation continues with proper memory routing.
      4. Reply text to the chat with the agent's new answer.

    The owner-only role check is reused — only the original asker
    (or another owner) can pick an answer; a stranger clicking the
    button gets a toast and the keyboard stays put.
    """
    from .. import tg_interactive as _tg
    from .. import roles as _roles

    def _handler(parts: list[str], ctx: dict):
        if not parts:
            return _tg.CallbackResult(ok=False, toast="malformed callback")
        clicker_id = ctx.get("clicker_speaker_id") or ""
        if not _roles.is_owner(clicker_id):
            return _tg.CallbackResult(
                ok=False,
                toast="only the owner can answer this question",
                clear_keyboard=False,
            )
        qid = parts[0]
        choice = parts[1] if len(parts) > 1 else ""
        q = STORE.get(qid)
        if q is None:
            return _tg.CallbackResult(
                ok=False, toast="question not found", clear_keyboard=True,
            )
        if q.answered:
            return _tg.CallbackResult(
                ok=False, toast="already answered", clear_keyboard=True,
            )
        marked = STORE.mark_answered(qid, choice=choice, text="")
        if marked is None:
            return _tg.CallbackResult(
                ok=False, toast="couldn't persist your choice",
                clear_keyboard=False,
            )
        # Resolve the chosen option's label for the resume turn.
        chosen_label = ""
        for opt in marked.options:
            if opt.get("id") == choice:
                chosen_label = opt.get("label") or ""
                break
        user_msg = (
            f"My choice: {chosen_label}" if chosen_label
            else f"My choice id: {choice}"
        )
        # Fire the next turn on a background thread so the callback
        # ACK lands in Telegram fast (sub-100ms) — the agent answer
        # arrives as a separate reply_text after the run completes.
        import threading as _th
        from .. import agent as _agent_mod
        from ..sessions import normalize_speaker as _norm
        speaker = _norm(marked.asker_speaker_id or clicker_id)

        def _runner():
            try:
                ag = _agent_mod.Agent()
                res = ag.run(
                    user_msg,
                    channel="telegram",
                    speaker_id=speaker,
                )
                # Send the reply via the channel layer. Reuse the
                # `_send_with_buttons` helper on the live Telegram
                # channel — same path the supervisor uses for its
                # final DM, so formatting + chunking + HTML-fallback
                # behaviour all match.
                from .. import channels as _ch
                cm = getattr(_ch, "CHANNELS", None)
                if cm is None or marked.asker_chat_id is None:
                    return
                for _ch_obj in (getattr(cm, "channels", {}) or {}).values():
                    if not hasattr(_ch_obj, "_send_with_buttons"):
                        continue
                    try:
                        _ch_obj._send_with_buttons(
                            marked.asker_chat_id,
                            (res.answer or "").strip()[:4000] or "(empty)",
                            None,
                        )
                        break
                    except Exception as e:
                        log.warning(
                            "ask_user resume DM via %r failed: %s",
                            type(_ch_obj).__name__, e,
                        )
            except Exception as e:
                log.warning("ask_user resume turn crashed: %s", e)

        _th.Thread(
            target=_runner,
            daemon=True,
            name=f"ask-user-resume-{qid}",
        ).start()
        # Edit the original question card so the buttons disappear
        # and a "✓ You picked: …" line replaces them.
        edited = (
            f"❓ <b>{q.header or 'Question'}</b>\n\n"
            f"{q.question}\n\n"
            f"✓ <b>You picked:</b> {chosen_label or choice}"
        )
        return _tg.CallbackResult(
            ok=True,
            toast="recorded — agent working…",
            edited_text=edited,
            clear_keyboard=True,
        )

    _tg.register_callback_handler("q", _handler)


# Module-load side effect — same pattern as skills._register_skill_callback().
try:
    _register_telegram_callback()
except Exception as _e:
    log.warning("ask_user: TG callback registration failed: %s", _e)


# ─── Helper — build a PendingQuestion from a tool call ─────────────


def create_question(
    *,
    question: str,
    options: list[dict],
    why: str = "",
    header: str = "",
    multi_select: bool = False,
    default_option_id: str = "",
    asker_speaker_id: str = "",
    asker_chat_id: Optional[int] = None,
    channel: str = "",
) -> PendingQuestion:
    """Build + persist a PendingQuestion. The tool handler calls
    this; the agent gets back the question_id sentinel."""
    # Normalize options. Each must have `label`; description is
    # optional. We assign each option a stable `id` so callback-data
    # / endpoint requests can name a choice without leaking labels
    # (which may contain emoji / Unicode that breaks URL encoding).
    norm_options: list[dict] = []
    for i, opt in enumerate(options or []):
        if not isinstance(opt, dict):
            continue
        label = (opt.get("label") or "").strip()
        if not label:
            continue
        opt_id = (opt.get("id") or "").strip() or f"opt-{i}"
        norm_options.append(
            QuestionOption(
                id=opt_id,
                label=label,
                description=(opt.get("description") or "").strip(),
            ).to_dict()
        )
    if len(norm_options) < 2:
        raise ValueError(
            "ask_user: need at least 2 options "
            f"(got {len(norm_options)})"
        )
    if len(norm_options) > 6:
        raise ValueError(
            "ask_user: at most 6 options supported "
            f"(got {len(norm_options)}); split into sub-questions"
        )
    qid = "q-" + secrets.token_hex(6)
    q = PendingQuestion(
        question_id=qid,
        asked_at=time.time(),
        question=(question or "").strip(),
        why=(why or "").strip(),
        header=(header or "").strip()[:32],
        options=norm_options,
        multi_select=bool(multi_select),
        default_option_id=(default_option_id or "").strip(),
        asker_speaker_id=(asker_speaker_id or "").strip(),
        asker_chat_id=asker_chat_id,
        channel=(channel or "").strip(),
    )
    STORE.put(q)
    # One live question per speaker: a new question supersedes the speaker's
    # prior open ones so a cascade can't leave stale keyboards that cross-wire
    # resume turns.
    try:
        STORE.supersede_open(q.asker_speaker_id, q.question_id)
    except Exception as e:
        log.debug("supersede_open failed: %s", e)
    return q
