"""Telegram message formatting — markdown→HTML conversion + Hermes-style
clean blocks (answer body / thinking trace footer / stats block).

Telegram accepts a small subset of HTML via `parse_mode="HTML"`:
  <b>, <strong>, <i>, <em>, <u>, <s>, <strike>, <del>, <span>,
  <a href>, <code>, <pre>, <pre><code class="language-...">,
  <blockquote>, <tg-emoji>

We intentionally support a SUBSET that maps cleanly from
markdown the LLM tends to emit:

  **bold**         → <b>bold</b>
  *italic*         → <i>italic</i>      (only when paired)
  ~~strike~~       → <s>strike</s>
  `inline code`    → <code>...</code>
  ```code block``` → <pre>...</pre>     (multi-line, with optional language)
  [text](url)      → <a href="url">text</a>

Everything else gets HTML-escaped so user-supplied text can't
break the parser. The converter is conservative — when in doubt
it leaves the source as-is (with escaping) rather than producing
half-formed markup that 400s the Telegram API.
"""
from __future__ import annotations

import html as _html
import re as _re
from typing import Any, Iterable


# ─── markdown → Telegram HTML ──────────────────────────────────────


# Telegram-allowed HTML tags. Anything else gets escaped on output.
_ALLOWED_TAGS = frozenset({
    "b", "strong", "i", "em", "u", "s", "strike", "del",
    "code", "pre", "a", "blockquote", "tg-spoiler", "tg-emoji", "span",
})


# Multi-line fenced code block: ```lang\n...\n``` or ```\n...\n```
_FENCED_CODE_RE = _re.compile(
    r"```([a-zA-Z0-9_+-]+)?\n([\s\S]*?)```",
    _re.MULTILINE,
)


# Inline code: `...` (no nested backticks)
_INLINE_CODE_RE = _re.compile(r"`([^`\n]+)`")


# Bold: **...** (non-greedy, single line)
_BOLD_RE = _re.compile(r"\*\*([^\*\n]+?)\*\*")


# Strike: ~~...~~
_STRIKE_RE = _re.compile(r"~~([^~\n]+?)~~")


# Italic: *...* (single asterisk pair, avoid matching ** by checking neighbors)
# Run AFTER bold.
_ITALIC_RE = _re.compile(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)")


# Markdown link: [text](url)
_LINK_RE = _re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")


# Setext-ish headers (# ... at start of line) → bold
_HEADER_RE = _re.compile(r"^(#{1,6})\s+(.+?)\s*$", _re.MULTILINE)


def markdown_to_telegram_html(text: str) -> str:
    """Convert a markdown-ish answer body to Telegram HTML.

    Strategy: pull code blocks OUT first (replace with placeholders so
    their content isn't touched by other rules), escape everything,
    re-apply inline markup, then put code blocks back as <pre> /
    <code>.

    Idempotent on pure plain text — if the input has no markdown
    markers it comes back HTML-escaped only.
    """
    if not text:
        return ""

    # Pull fenced code blocks out first.
    code_blocks: list[tuple[str, str]] = []

    def _stash_fenced(m: _re.Match) -> str:
        lang = (m.group(1) or "").strip()
        body = m.group(2)
        code_blocks.append((lang, body))
        return f"\x00FENCED{len(code_blocks) - 1}\x00"

    text = _FENCED_CODE_RE.sub(_stash_fenced, text)

    # Pull inline `code` segments out too — they shouldn't be
    # touched by bold/italic.
    inline_codes: list[str] = []

    def _stash_inline(m: _re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x00INLINE{len(inline_codes) - 1}\x00"

    text = _INLINE_CODE_RE.sub(_stash_inline, text)

    # Now safe to HTML-escape the body.
    text = _html.escape(text, quote=False)

    # Apply inline markup. Order matters: bold (**) before italic (*).
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _STRIKE_RE.sub(r"<s>\1</s>", text)
    text = _ITALIC_RE.sub(r"<i>\1</i>", text)

    # Markdown links — only http(s) for safety. Other schemes are
    # left as plain text.
    def _link_sub(m: _re.Match) -> str:
        label = m.group(1)
        url = m.group(2)
        if not url.startswith(("http://", "https://", "tg://")):
            return m.group(0)  # leave as-is, escaped
        # Re-escape label since we escaped earlier and label was
        # part of that pass — actually no, label is from the
        # *escaped* text now, so it's already safe. But re-escape
        # url's quotes defensively.
        url_safe = url.replace("\"", "%22")
        return f'<a href="{url_safe}">{label}</a>'

    text = _LINK_RE.sub(_link_sub, text)

    # Markdown headers → bold lines (Telegram has no header tag).
    text = _HEADER_RE.sub(r"<b>\2</b>", text)

    # Restore inline code segments.
    for i, code in enumerate(inline_codes):
        escaped = _html.escape(code, quote=False)
        text = text.replace(
            f"\x00INLINE{i}\x00", f"<code>{escaped}</code>"
        )

    # Restore fenced code blocks.
    for i, (lang, body) in enumerate(code_blocks):
        escaped_body = _html.escape(body, quote=False)
        if lang:
            lang_safe = _re.sub(r"[^a-zA-Z0-9_+-]", "", lang)[:20]
            block = (
                f'<pre><code class="language-{lang_safe}">'
                f"{escaped_body}</code></pre>"
            )
        else:
            block = f"<pre>{escaped_body}</pre>"
        text = text.replace(f"\x00FENCED{i}\x00", block)

    return text


def escape_html(text: str) -> str:
    """Public escape for callers that build their own HTML segments
    (footer labels, error messages, etc.)."""
    return _html.escape(text or "", quote=False)


# ─── Tool-call icons + arg-preview (Hermes-style live trace) ──────


# Hermes-like emoji per tool. Used in the streaming "thinking block"
# so each line is instantly recognisable. Unknown tools fall back
# to a neutral 🔹 — the row still appears, just without a custom
# icon.
TOOL_ICON: dict[str, str] = {
    # File / code I/O
    "read_file": "📄",
    "view_file": "📄",
    "write_file": "📝",
    "edit_file": "📝",
    "patch": "🔧",
    "save_to_workspace": "💾",
    # Search / navigation
    "search_files": "🔍",
    "grep": "🔍",
    "search": "🔍",
    "search_knowledge": "🔍",
    "glob": "🔍",
    "list_files": "📂",
    "locate_symbol": "📍",
    # Shell / process
    "terminal_exec": "⌨️",
    "execute_code": "🐍",
    "run_python": "🐍",
    "sandbox_exec": "🛡️",
    "start_background_job": "⚙️",
    "list_background_jobs": "⚙️",
    "get_background_job": "⚙️",
    # Network
    "web_search": "🌐",
    "fetch_url": "🔗",
    "search_package": "📦",
    # Skills + meta
    "list_skills": "📚",
    "load_skill": "📚",
    "skill_view": "📚",
    "skill_manage": "📚",
    "propose_skill": "📚",
    # Media
    "analyze_image": "🖼️",
    "preprocess_video": "🎬",
    # Schedule / cron / messages
    "schedule_message": "⏰",
    "cronjob": "⏰",
    # Config
    "set_setting": "⚙️",
    "save_user_fact": "🗒️",
    # Roles + access
    "grant_telegram_access": "🔑",
    "revoke_telegram_access": "🔒",
    "approve_pairing": "🤝",
    "list_pending_pairings": "🤝",
    "list_telegram_access": "🤝",
    # Self-mod / delegation
    "propose_self_modification": "🛠️",
    "delegate": "👥",
}


# Per-tool "primary argument" — the most informative one to show as
# the preview after the tool name. Anything not in this map falls
# back to the first non-empty arg value.
_TOOL_PRIMARY_ARG: dict[str, str] = {
    "read_file": "path",
    "view_file": "path",
    "write_file": "path",
    "edit_file": "path",
    "patch": "path",
    "save_to_workspace": "filename",
    "search_files": "pattern",
    "grep": "pattern",
    "search": "query",
    "search_knowledge": "query",
    "glob": "pattern",
    "list_files": "path",
    "locate_symbol": "symbol",
    "terminal_exec": "command",
    "execute_code": "code",
    "run_python": "code",
    "sandbox_exec": "command",
    "start_background_job": "label",
    "list_background_jobs": "status",
    "get_background_job": "job_id",
    "web_search": "query",
    "fetch_url": "url",
    "search_package": "name",
    "list_skills": "tag",
    "load_skill": "name",
    "skill_view": "name",
    "skill_manage": "name",
    "propose_skill": "name",
    "analyze_image": "question",
    "preprocess_video": "sha",
    "schedule_message": "when",
    "cronjob": "action",
    "set_setting": "key",
    "save_user_fact": "fact",
    "grant_telegram_access": "user_id",
    "revoke_telegram_access": "user_id",
    "approve_pairing": "code_or_user_id",
    "list_pending_pairings": "",
    "list_telegram_access": "",
    "propose_self_modification": "description",
    "delegate": "role",
}


_TOOL_ARG_PREVIEW_MAX = 48  # chars, after which we ellipse


def tool_icon(name: str) -> str:
    """Return the emoji icon for a tool name. Falls back to 🔹 for
    unknown tools."""
    return TOOL_ICON.get(name or "", "🔹")


def arg_preview(name: str, args: Any) -> str:
    """Render a short preview of the tool's primary argument value
    for the streaming "thinking block". Returns "" when args is
    empty / missing — caller renders just `<icon> <name>` then.
    """
    if not args:
        return ""
    if not isinstance(args, dict):
        # Some agents send args as a JSON string. Try to parse.
        if isinstance(args, str):
            v = args.strip()
            if len(v) > _TOOL_ARG_PREVIEW_MAX:
                v = v[:_TOOL_ARG_PREVIEW_MAX - 3] + "..."
            return v
        return ""
    primary_key = _TOOL_PRIMARY_ARG.get(name or "", "")
    val: Any = None
    if primary_key:
        val = args.get(primary_key)
    if val is None or val == "":
        # Fall back to the first non-empty arg value.
        for k, v in args.items():
            if v not in (None, "", [], {}):
                val = v
                break
    if val is None:
        return ""
    s = str(val).strip()
    # Collapse newlines to spaces — single-line preview only.
    s = s.replace("\n", " ").replace("\r", " ")
    if len(s) > _TOOL_ARG_PREVIEW_MAX:
        s = s[:_TOOL_ARG_PREVIEW_MAX - 3] + "..."
    return s


def format_tool_entry_line(icon: str, name: str, preview: str, count: int = 1) -> str:
    """Single line in the streaming thinking block. Plain text (no
    HTML) because the streamer doesn't currently pass parse_mode."""
    arg_part = f': "{preview}"' if preview else ""
    suffix = f" (×{count})" if count > 1 else ""
    return f"{icon} {name}{arg_part}{suffix}"


# ─── Hermes-style answer + footer assembly ─────────────────────────


# Visual divider between answer body and the metadata footer.
# Single em-rule is cleaner than the old ━━━ banner and reads
# nicer on both light and dark Telegram themes.
_DIVIDER = "─────────"


def render_answer_with_footer(
    *,
    answer_html: str,
    trace_footer: str = "",
    stats_block: str = "",
) -> str:
    """Compose a Telegram HTML message for the agent's final answer.

    Layout:
      <answer body, HTML-escaped + markdown-converted>
      ─────────
      <trace footer> (optional, plain HTML)
      <stats block>  (optional, plain HTML)

    Footers are joined under a single divider so the user sees a
    clear separation between "what I asked for" and "how it was
    computed".
    """
    parts: list[str] = []
    body = (answer_html or "").rstrip()
    if body:
        parts.append(body)
    tail_parts = [p.strip() for p in (trace_footer, stats_block) if (p or "").strip()]
    if tail_parts:
        parts.append(f"<i>{_DIVIDER}</i>")
        parts.extend(tail_parts)
    return "\n\n".join(parts)


def format_trace_footer(trace_items: Iterable, total_time_s: float = 0.0) -> str:
    """Hermes-style thinking trace summary.

    Single line listing the distinct stages (chat → solve → verify),
    plus a tools summary line if any tools ran.

    `trace_items` is an iterable of trace-step objects with `.event`
    and optional `.tool_call.name`. Robust to the dict / model_dump
    forms the agent produces.
    """
    items = list(trace_items or [])
    if not items:
        return ""
    seen_stages: list[str] = []
    seen_set: set[str] = set()
    tool_counts: dict[str, int] = {}
    for s in items:
        ev = _attr(s, "event") or ""
        if ev.startswith("tool"):
            tc = _attr(s, "tool_call")
            tn = _attr(tc, "name") if tc else None
            if tn:
                tool_counts[tn] = tool_counts.get(tn, 0) + 1
            continue
        if ev and ev not in seen_set:
            seen_set.add(ev)
            seen_stages.append(ev)
    lines: list[str] = []
    if seen_stages:
        chain = " → ".join(seen_stages[:8])
        if len(seen_stages) > 8:
            chain += f" → … (+{len(seen_stages) - 8})"
        if total_time_s > 0:
            lines.append(
                f"🧠 <b>{escape_html(chain)}</b>  "
                f"<i>({len(items)} steps · {total_time_s:.1f}s)</i>"
            )
        else:
            lines.append(
                f"🧠 <b>{escape_html(chain)}</b>  "
                f"<i>({len(items)} steps)</i>"
            )
    if tool_counts:
        tools = ", ".join(
            f"<code>{escape_html(n)}</code>×{c}"
            for n, c in sorted(tool_counts.items())
        )
        lines.append(f"🔧 {tools}")
    return "\n".join(lines)


def format_stats_block(token_usage: Any) -> str:
    """Hermes-style token / call summary.

    One condensed line for the typical case; second line breaks
    out per-stage breakdown when the input was non-trivial (≥ 5k
    tokens AND more than one stage).
    """
    if token_usage is None:
        return ""
    tu = token_usage
    total = _attr(tu, "total_tokens", 0)
    if not total:
        return ""
    in_ = _attr(tu, "input_tokens", 0)
    out = _attr(tu, "output_tokens", 0)
    calls = _attr(tu, "llm_calls", 0)
    cache_read = _attr(tu, "cache_read_tokens", 0)
    cache_create = _attr(tu, "cache_creation_tokens", 0)
    # Format with thousands separator for readability.
    parts = [
        f"🔢 <b>{total:,}</b> tok",
        f"<i>in {in_:,} · out {out:,}</i>",
    ]
    if cache_read or cache_create:
        cache_parts = []
        if cache_read:
            cache_parts.append(f"r {cache_read:,}")
        if cache_create:
            cache_parts.append(f"w {cache_create:,}")
        parts.append(f"<i>cache {' · '.join(cache_parts)}</i>")
    parts.append(f"<i>{calls} call{'s' if calls != 1 else ''}</i>")
    head_line = "  ·  ".join(parts)

    lines = [head_line]
    # Per-stage breakdown when input is non-trivial.
    stages = _attr(tu, "by_stage", None) or {}
    if isinstance(stages, dict) and len(stages) > 1 and in_ >= 5_000:
        top = list(stages.items())[:3]
        seg = []
        for name, s in top:
            s_in = (s or {}).get("input_tokens", 0)
            pct = (s_in / in_ * 100) if in_ else 0
            seg.append(
                f"<code>{escape_html(str(name))}</code> {s_in:,} ({pct:.0f}%)"
            )
        lines.append(f"📊 {' · '.join(seg)}")
    return "\n".join(lines)


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Robust attribute lookup. Works for objects with attrs AND
    for dicts (token usage sometimes arrives as the model_dump
    form). Returns `default` if neither shape has the key."""
    if obj is None:
        return default
    if hasattr(obj, name):
        return getattr(obj, name, default)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default
