"""Dev-mode helpers: redact file contents out of an agent prompt and
persist a per-request capture under `dev/` so the user can inspect
what was actually sent to the LLM without dumping kilobytes of
file content back into the UI response.

Two pieces:

  * `redact_prompt(text)` — replace the BODIES of known sections
    (SOUL / IDENTITY / USER PROFILE / CORE MEMORY / NOTES / RECENT
    COMMITS / EXTRACTED IDENTIFIERS / capabilities catalog / etc.)
    with `[file: <path>, <N> chars]` markers. Other text passes
    through untouched, so things like the THINKING plan and the
    USER REQUEST stay visible.
  * `save_dev_capture(...)` — write the redacted call list to
    `dev/<timestamp>_<request_id>.json` for offline inspection.
    The directory is gitignored; the user prunes manually.
"""
from __future__ import annotations
import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
DEV_DIR = ROOT / "dev"


# Section header → replacement reference.
# Headers appear at column 0 in built prompts. We capture the body
# until the next `# `-prefixed header (or end of string) and substitute
# a one-line marker that names the file (or source) it came from.
#
# Order matters: more specific headers first, so a `# IDENTITY` block
# doesn't accidentally absorb the next `# AGENT NAME OVERRIDE`.
_SECTION_FILES: tuple[tuple[str, str], ...] = (
    ("# SOUL",                      "knowledge/identity/soul.md"),
    ("# IDENTITY",                  "knowledge/identity/identity.md"),
    ("# USER PROFILE",              "knowledge/identity/user.md"),
    ("# AGENT NAME OVERRIDE",       "<derived from identity.md ## Имя>"),
    ("# LANGUAGE OVERRIDE",         "<derived from user.md ## Язык общения>"),
    ("# CORE MEMORY",               "knowledge/core_memory.md"),
    ("CORE MEMORY:",                "knowledge/core_memory.md"),  # _think variant
    ("# CURRENT PROJECT",           "<knowledge/projects/.current>"),
    ("# GOALS",                     "knowledge/goals.json"),
    ("# ACTIVE GOALS",              "knowledge/goals.json"),
    ("# SHORT-TERM MEMORY",         "<MEMORY.recall_block>"),
    ("# RECENT COMMITS",            "<git log --oneline -50>"),
    ("# RECENT TURNS",              "knowledge/conversation.json"),
    ("# RECENT CONVERSATION",       "knowledge/conversation.json"),
    ("# NOTES",                     "knowledge/profession/* (loaded notes)"),
    ("# MY CAPABILITIES",           "<capabilities block>"),
    ("MY CAPABILITIES:",            "<capabilities block>"),  # _think variant
    ("# SKILLS",                    "<skills catalog>"),
    ("# SKILL CATALOG",             "<skills catalog>"),
    ("# THINKING (prepared by the reasoning module)",
                                    "<thinking module output>"),
    ("EXTRACTED IDENTIFIERS",       "<verifier auto-extracted from tool_context>"),
    ("TOOL OUTPUTS",                "<tool_context: outputs of read_file / web_search / etc.>"),
    ("SOURCE NOTES",                "<verifier: notes_text>"),
)

# `(?ms)` — multiline + dot-matches-all is built into each pattern below.
# Body extends until either:
#   (a) the next line that starts with "# " followed by an UPPERCASE
#       letter (= our section convention), or
#   (b) end of string.
_HEADER_BODY_RE_TEMPLATE = (
    r"(?P<header>{header})\s*\n"            # header + newline
    r"(?P<body>.*?)"                         # body (non-greedy)
    r"(?=\n#\s+[A-ZА-Я]|\Z)"                 # next ## header or EOF
)


def _length_marker(file_ref: str, body: str) -> str:
    """Build a `[file: path, NNN chars]` placeholder."""
    n = len(body.rstrip())
    return f"[file: {file_ref}, {n} chars]"


def redact_prompt(text: str, *, hard_cap_chars: int = 12000) -> str:
    """Replace file-content section bodies with reference markers.

    Untouched text (e.g. `# USER REQUEST`, the actual user message,
    `# CRITIQUE`) survives so dev-mode shows real per-turn content.
    Also caps the final string at `hard_cap_chars` to prevent
    pathological growth from multiple long sections that don't
    match any known header — the cap is applied AFTER redaction so
    the redacted markers themselves don't push a clean prompt over
    the limit.
    """
    if not text:
        return ""
    out = text
    for header, file_ref in _SECTION_FILES:
        # Header has special regex chars in some entries (parens, slashes);
        # escape it. `re.escape` does the right thing.
        pattern = re.compile(
            _HEADER_BODY_RE_TEMPLATE.format(header=re.escape(header)),
            re.MULTILINE | re.DOTALL,
        )
        out = pattern.sub(
            lambda m, ref=file_ref: f"{m.group('header')}\n{_length_marker(ref, m.group('body'))}",
            out,
        )
    if len(out) > hard_cap_chars:
        omitted = len(out) - hard_cap_chars
        out = out[:hard_cap_chars] + f"\n…[+{omitted} chars truncated by dev-mode cap]"
    return out


# --- on-disk capture -------------------------------------------------------


_DEV_LOCK = threading.Lock()


def _ensure_dev_dir() -> None:
    DEV_DIR.mkdir(parents=True, exist_ok=True)


def save_dev_capture(
    *,
    request_id: str,
    question: str,
    llm_calls: Iterable[dict],
    answer_preview: str,
    confidence: int,
) -> Path | None:
    """Persist a redacted capture for one Agent.run() invocation.

    Filename `dev/<YYYY-MM-DD_HH-MM-SS>_<short-id>.json`. Returns the
    written path, or None if writing failed (rare — disk full, locked).
    Best-effort: never raises.

    Skipped under pytest: tests fire many `Agent().run(...)` calls
    against mock routers, and persisting each run produced bursts of
    20-40 mock captures per `pytest tests/` invocation that swamped
    real-usage entries in dev/. PYTEST_CURRENT_TEST is the canonical
    pytest env signal; AGI_DISABLE_DEV_CAPTURE is an explicit override
    for users who want this off entirely.
    """
    import os
    if os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("AGI_DISABLE_DEV_CAPTURE"):
        return None
    try:
        _ensure_dev_dir()
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = DEV_DIR / f"{ts}_{request_id[:8]}.json"
        payload = {
            "ts": ts,
            "request_id": request_id,
            "question": question[:500],
            "answer_preview": (answer_preview or "")[:600],
            "confidence": confidence,
            "llm_calls": list(llm_calls),
        }
        with _DEV_LOCK:
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return path
    except Exception:
        return None


def new_request_id() -> str:
    """Short, sortable-by-creation request id for dev capture."""
    return uuid.uuid4().hex[:12]
