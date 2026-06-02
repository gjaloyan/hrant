"""Agent identity management: soul.md, identity.md, user.md.

These three files always live in `knowledge/identity/` and are always
loaded into context:

  * soul.md     — character, values, communication style. Edited by a
                  human (or by the agent itself on a deliberate change).
  * identity.md — who I am, what I can do, what I don't do. The anchor
                  of self-definition.
  * user.md     — what I know about the user: communication language,
                  preferences, personal facts, interaction rules.
                  Updated automatically when the user shares something
                  about themselves or asks to be remembered.

The default files are created on first launch. The user may edit them
by hand — the agent respects manual edits and never rewrites the files
wholesale.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Literal

from .config import CONFIG

UserFactCategory = Literal["language", "style", "about_user", "rule"]

_DEFAULT_SOUL = """\
# Soul
*Character, values, and tone of the agent. Always in context.*

## Role
I am a self-learning AI assistant and companion to the user.
Not just a tool, but a partner who helps, learns, and grows
together with the user.

## Character
- Warm and human, without fake politeness.
- Direct and honest: if I don't know something, I say so plainly.
- Curious: I genuinely want to understand things deeply.
- Respectful of the user, their time, and their choices.

## Communication style
- Brief by default; go into detail only when it's truly needed.
- No clichés like "of course!", "great question!", or long disclaimers.
- Mirror the user's language: reply in whatever language the user writes in.
- For small talk — respond like a human, in one or two sentences.
- For tasks — be structured and ground answers in sources.

## Principles
- I answer only based on what I know. I flag assumptions honestly.
- I remember facts about the user and apply them.
- I remember communication preferences and follow them without reminders.
- I learn from every dialogue, but I don't turn every remark into research.
"""

_DEFAULT_IDENTITY = """\
# Identity
*Who I am. Always in context.*

## Me
- Self-learning agent — the user's local AI assistant.
- My source code is a Python project on disk. I CAN read it via `read_file`.
- I have: knowledge base (topic notes), core memory (persistent facts),
  soul (character), user profile (what I know about the user).
- I can learn: search the web, create notes, verify myself,
  accumulate experience for future fine-tuning.

## My concrete capabilities
- **Tools**: web_search, fetch_url, read_file, run_python +
  any additional ones from skills and MCP servers. Full list is always
  in the MY CAPABILITIES block in the system prompt.
- **Skills**: declarative plugins in `backend/skills/`. Each skill is
  an instruction + (optionally) its own tool. Activated by triggers.
- **MCP servers**: connected via config.yaml, provide external tools.
- **Self-analysis**: when asked about myself (architecture, code,
  improvements), I MUST call `read_file` on my source code first,
  then draw conclusions. Without reading the code, any claim about
  my own implementation is a hallucination.

## What I do well
- Maintain conversation warmly and briefly.
- Answer questions based on notes and cite sources.
- Learn new topics on demand or automatically when knowledge is lacking.
- Remember what the user asks me to remember, and follow it.
- Manage projects: context, decisions, issues.

## What I don't do
- Never fabricate facts for plausibility.
- Never make claims about my own code without reading the file.
- Never apply "deep analysis" to casual chitchat.
- Never ignore or "forget" user preferences.
- Never turn every short message into an excuse to create a note.
"""

_DEFAULT_USER = """\
# User Profile
*What I know about the user. Updated automatically when the user
shares something about themselves or about how to communicate with them.*

## Language
(not specified)

## Style and tone
(not specified)

## About the user
(not specified)

## Interaction rules
(not specified)
"""


# Canonical (English) section header that each fact category is written
# under in user.md.
_SECTION_BY_CATEGORY: dict[str, str] = {
    "language": "## Language",
    "style": "## Style and tone",
    "about_user": "## About the user",
    "rule": "## Interaction rules",
}

# Legacy Russian headers from profiles created before the English
# migration. The section extractors and add_user_fact accept these so
# existing on-disk user_profile.md files keep working with no data
# migration — a fact for an old profile lands under its existing
# Russian header instead of spawning a duplicate English section.
_LEGACY_HEADERS_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "language": ("## Язык общения", "## Язык"),
    "style": ("## Стиль и тон",),
    "about_user": ("## О пользователе",),
    "rule": ("## Правила взаимодействия",),
}

# Empty-section placeholders, English (current) + Russian (legacy).
_PLACEHOLDERS = ("(not specified)", "(пока не указано)")


def _acceptable_headers(category: str) -> tuple[str, ...]:
    """All header spellings a category may appear under in a profile,
    English first (preferred for new writes) then legacy Russian."""
    primary = _SECTION_BY_CATEGORY.get(category, _SECTION_BY_CATEGORY["about_user"])
    return (primary, *_LEGACY_HEADERS_BY_CATEGORY.get(category, ()))


_DEFAULT_SPEAKER_FOR_LEGACY_USER_MD = "webui:default"


# Patterns for lines that the memory extractor sometimes lands in
# user_profile.md but actually describe AGENT behavior, not the user.
# Example pollution (real, from a production install):
#     ## Правила взаимодействия
#     - Respond to the name Hrant.  _(добавлено 2026-05-14)_
# Inside USER PROFILE that line reads as "the user is named Hrant",
# which conflates with the actual agent name from identity.md and
# the model starts addressing the user with the agent's own name.
# Identity is the ONLY source of truth for the agent's name — we
# never want this in user_profile.
_AGENT_BEHAVIOR_PATTERNS = (
    "respond to the name",
    "answer to the name",
    "reply as ",
    "your name is",
    "you are called",
    "you answer to",
    "откликаешься на имя",
    "тебя зовут",  # only when describing the AGENT, not the user
)


def _looks_like_agent_behavior_line(line: str) -> bool:
    """True for bullet lines that describe how the AGENT should behave
    (especially agent-name rules) rather than facts about the user.

    Only fires on bullet lines (`-` / `*`) inside markdown lists; section
    headers and prose are left alone."""
    s = line.strip()
    if not s or not s.startswith(("-", "*")):
        return False
    body = s.lstrip("-*").strip().lower()
    if not body:
        return False
    # "тебя зовут <name>" inside user_profile is ambiguous — keep
    # variants that name the user explicitly. The bug only fires when
    # the name matches identity.md's agent name, but the read-side
    # filter can't know which name belongs to which side. So strip
    # all "тебя зовут" lines from user_profile too — the user's name
    # belongs in `User is named X` / `User's name is X` bullets which
    # don't match this pattern. A false-positive removal degrades to
    # "agent doesn't see this particular phrasing"; the canonical
    # `User is named X` bullet (if present) survives.
    for pat in _AGENT_BEHAVIOR_PATTERNS:
        if pat in body:
            return True
    return False


def _strip_agent_behavior_lines(text: str) -> str:
    """Drop bullet lines that look like agent-behavior rules
    accidentally written into user_profile.md. Section headers and
    prose are preserved verbatim."""
    return "\n".join(
        ln for ln in text.splitlines()
        if not _looks_like_agent_behavior_line(ln)
    )


def _sanitize_speaker_for_path(speaker_id: str) -> str:
    """Convert a speaker_id ('telegram:123', 'webui:default', …) into
    a filesystem-safe filename stem. Colons forbidden on Windows;
    only `[a-z0-9_-]` survives, everything else becomes `_`."""
    import re as _re
    return _re.sub(r"[^a-z0-9_-]+", "_", speaker_id.lower()).strip("_") or "default"


class IdentityManager:
    def __init__(self, base_dir: Path | None = None):
        self.dir = Path(base_dir or CONFIG.knowledge["base_dir"]) / "identity"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.soul_path = self.dir / "soul.md"
        self.identity_path = self.dir / "identity.md"
        # Legacy single-user_profile path. From Phase 10 onwards the
        # WebUI default speaker (`webui:default`) lives here, every
        # other speaker gets its own file under `profiles/`.
        self.user_path = self.dir / "user.md"
        self.profiles_dir = self.dir / "profiles"
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        # History for each user-profile file: per-file timestamped
        # snapshots. The original layout used `_history/user_*.md` for
        # the single user.md; we keep that path for `webui:default` and
        # add `_history/<sanitized>_*.md` for other speakers.
        self.history_dir = self.dir / "_history"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_defaults()

    def _user_path_for(self, speaker_id: str | None) -> Path:
        """Return the per-speaker user_profile file path. The WebUI
        default keeps the legacy `user.md` location; other speakers
        (each Telegram user, future channels) get their own file
        under `profiles/<sanitized>.md`."""
        sp = speaker_id or _DEFAULT_SPEAKER_FOR_LEGACY_USER_MD
        if sp == _DEFAULT_SPEAKER_FOR_LEGACY_USER_MD:
            return self.user_path
        return self.profiles_dir / f"{_sanitize_speaker_for_path(sp)}.md"

    def _snapshot_user_profile(self, speaker_id: str | None = None) -> Path | None:
        """Snapshot the current per-speaker user_profile into _history/.

        Filename: `user_<ts>.md` for the WebUI default (back-compat);
        `<sanitized_speaker>_<ts>.md` for everyone else."""
        path = self._user_path_for(speaker_id)
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sp = speaker_id or _DEFAULT_SPEAKER_FOR_LEGACY_USER_MD
        if sp == _DEFAULT_SPEAKER_FOR_LEGACY_USER_MD:
            prefix = "user"
        else:
            prefix = _sanitize_speaker_for_path(sp)
        snap = self.history_dir / f"{prefix}_{ts}.md"
        i = 1
        while snap.exists():
            snap = self.history_dir / f"{prefix}_{ts}_{i}.md"
            i += 1
        try:
            snap.write_text(text, encoding="utf-8")
        except Exception:
            return None
        return snap

    def list_user_versions(self) -> list[dict]:
        """Version history of user.md, newest first."""
        if not self.history_dir.exists():
            return []
        out: list[dict] = []
        for f in sorted(self.history_dir.glob("user_*.md"), reverse=True):
            stem = f.stem.removeprefix("user_")
            stamp_part = stem.split("_")[0] + "_" + stem.split("_")[1] if "_" in stem else stem
            try:
                dt = datetime.strptime(stamp_part, "%Y%m%d_%H%M%S")
                human = dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                human = stem
            out.append({"timestamp": human, "path": str(f), "size": f.stat().st_size})
        return out

    def _ensure_defaults(self) -> None:
        if not self.soul_path.exists():
            self.soul_path.write_text(_DEFAULT_SOUL, encoding="utf-8")
        if not self.identity_path.exists():
            self.identity_path.write_text(_DEFAULT_IDENTITY, encoding="utf-8")
        if not self.user_path.exists():
            self.user_path.write_text(_DEFAULT_USER, encoding="utf-8")

    # ---------- read ----------
    def soul(self) -> str:
        return self.soul_path.read_text(encoding="utf-8")

    def identity(self) -> str:
        return self.identity_path.read_text(encoding="utf-8")

    def user_profile(self, speaker_id: str | None = None) -> str:
        """Read the per-speaker user_profile.md. On first read for a
        non-default speaker, the file is auto-created from the
        _DEFAULT_USER template so the profile is editable from the
        moment a new speaker arrives.

        Agent-behavior rules misclassified as user facts (e.g.
        `Respond to the name Hrant.`) are stripped from the returned
        text. They live on disk for audit history (so the user can
        see what was extracted and correct the extractor), but they
        must not reach the prompt — there they read as 'the user is
        named Hrant' and trigger the cross-name confusion bug.
        Extraction-side suppression is in memory_extractor.py."""
        path = self._user_path_for(speaker_id)
        if not path.exists():
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(_DEFAULT_USER, encoding="utf-8")
            except Exception:
                return _DEFAULT_USER
        return _strip_agent_behavior_lines(path.read_text(encoding="utf-8"))

    def list_speaker_profiles(self) -> list[dict]:
        """Every user_profile file present on disk. Returns
        [{speaker_id, path, size, modified}, …] sorted by modified
        newest-first. Used by the WebUI to list profiles in the
        User Profile tab so the user can switch between them."""
        out: list[dict] = []
        # Legacy webui:default.
        if self.user_path.exists():
            try:
                st = self.user_path.stat()
                out.append({
                    "speaker_id": _DEFAULT_SPEAKER_FOR_LEGACY_USER_MD,
                    "path": str(self.user_path),
                    "size": st.st_size,
                    "modified": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                })
            except OSError:
                pass
        # Per-speaker files under profiles/.
        if self.profiles_dir.exists():
            for f in self.profiles_dir.glob("*.md"):
                # Re-derive a "best guess" speaker_id from the sanitized
                # filename (e.g. telegram_123 -> telegram:123). Best-effort:
                # the agent stores the canonical speaker_id at write time
                # but the legacy on-disk shape only carries the stem.
                stem = f.stem
                if "_" in stem:
                    head, _, tail = stem.partition("_")
                    speaker_id = f"{head}:{tail}"
                else:
                    speaker_id = f"unknown:{stem}"
                try:
                    st = f.stat()
                    out.append({
                        "speaker_id": speaker_id,
                        "path": str(f),
                        "size": st.st_size,
                        "modified": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    })
                except OSError:
                    continue
        return sorted(out, key=lambda r: r["modified"], reverse=True)

    @staticmethod
    def _extract_section(text: str, headers: tuple[str, ...]) -> str:
        """Pull the body of the first matching `## <header>` section.

        Empty if no matching header is found, the section is empty, or
        the section holds only an empty placeholder.
        """
        lines = text.splitlines()
        try:
            start = next(
                i for i, ln in enumerate(lines)
                if ln.strip() in headers
            )
        except StopIteration:
            return ""
        end = len(lines)
        for j in range(start + 1, len(lines)):
            if lines[j].startswith("## "):
                end = j
                break
        body_lines = [
            ln for ln in lines[start + 1 : end]
            if ln.strip() and ln.strip() not in _PLACEHOLDERS
        ]
        return "\n".join(body_lines).strip()

    @classmethod
    def _extract_language_section(cls, profile_text: str) -> str:
        """Pull the body of the `## Language` section from user.md
        (legacy Russian `## Язык общения` / `## Язык` also accepted)."""
        return cls._extract_section(
            profile_text, ("## Language", "## Язык общения", "## Язык"),
        )

    # Canonical agent name. Used as the LAST-RESORT fallback in
    # `agent_name()` when neither identity.md nor any other source
    # reveals the name. The brand-level fact that this agent is called
    # Hrant is asserted in pyproject.toml and the CLI banner; this
    # constant keeps the runtime in sync if a user's identity.md gets
    # accidentally cleaned out.
    DEFAULT_AGENT_NAME = "Hrant"

    @classmethod
    def _extract_name_section(cls, identity_text: str) -> str:
        """Pull the body of the `## Имя` / `## Name` section from
        identity.md. Live installs sometimes ship a template with a
        differently-labelled self section (`## Me`, `## I`, …) — the
        caller falls back to scanning the whole file via
        `_scan_agent_name` when this returns empty."""
        return cls._extract_section(
            identity_text, ("## Имя", "## Name"),
        )

    @classmethod
    def _scan_agent_name(cls, identity_text: str) -> str:
        """Last-resort scan: pick the agent's own name out of free-form
        identity.md prose. Looks for the same phrasing the user might
        write in a fresh template ('My name is X', 'Меня зовут X',
        'I am X', 'I'm X — a…'). Returns the captured name or ''."""
        import re as _re
        patterns = (
            r"[Mm]y\s+name\s+is\s+\*{0,2}([A-Z][A-Za-zА-Яа-яЁё\-]+)",
            r"[Мм]еня\s+зовут\s+\*{0,2}([A-Za-zА-Яа-яЁё\-]+)",
            r"\bI\s+am\s+\*{0,2}([A-Z][A-Za-zА-Яа-яЁё\-]+)",
            r"\bI'?m\s+\*{0,2}([A-Z][A-Za-zА-Яа-яЁё\-]+)\s+[—–-]",
        )
        for pat in patterns:
            m = _re.search(pat, identity_text)
            if m:
                candidate = m.group(1).strip("* ")
                # Filter out generic words a sloppy regex could grab.
                # "I am a..." → "a"; "I am Self-learning" → "Self-learning"
                # both shouldn't pass.
                low = candidate.lower()
                if low in {"a", "an", "the", "self", "your", "still", "always"}:
                    continue
                if "-" in candidate and low.split("-", 1)[0] in {"self"}:
                    continue
                return candidate
        return ""

    def agent_name(self) -> str:
        """Canonical resolution of THIS agent's name.

        Order of precedence:
          1. The body of `## Имя` / `## Name` in identity.md (the
             curated section). Already-known phrases like 'My name is
             Hrant' inside that body get cleaned by the caller.
          2. A free-form scan of identity.md for 'My name is X' /
             'Меня зовут X' patterns. Survives templates whose self
             section is labelled `## Me` / `## I` / similar.
          3. `DEFAULT_AGENT_NAME` ('Hrant'). The brand-level fact —
             the CLI is `hrant`, the systemd unit is `hrant.service`,
             pyproject.toml says so. If identity.md is silent, it's
             still the agent's name."""
        body = self._extract_name_section(self.identity())
        if body:
            named = self._scan_agent_name(body)
            if named:
                return named
            # Section exists but no recognisable pattern — take the
            # first non-empty line (commonly the bare name itself).
            for ln in body.splitlines():
                cleaned = ln.strip().strip("*-• ").strip()
                if cleaned and len(cleaned) < 40:
                    return cleaned
        # Fall through to whole-file scan for installs whose self
        # section is labelled differently (the production install
        # we just audited uses `## Me`).
        scanned = self._scan_agent_name(self.identity())
        if scanned:
            return scanned
        return self.DEFAULT_AGENT_NAME

    @classmethod
    def _extract_user_name(cls, profile_text: str) -> str:
        """Best-effort: pull the user's first name from user_profile.md.

        Looks at the `## О пользователе` / `## About` / `## Name`
        sections and tries to extract a name. The body is free-form
        prose so we use a few simple patterns. Empty when nothing
        matches — the caller then omits the USER NAME block instead
        of guessing.
        """
        import re as _re
        body = cls._extract_section(
            profile_text,
            (
                "## Имя", "## Name",
                "## О пользователе", "## About",
                "## About the user",
            ),
        )
        # Patterns: "User's name is X", "User is X", "Меня зовут X",
        # "Пользователя зовут X", "User is named X". Stop at comma /
        # period / end-of-line — names are short.
        for pat in (
            r"[Uu]ser'?s?\s+name\s+is\s+([A-Z][A-Za-zА-Яа-яЁё\-]+)",
            r"[Uu]ser\s+is\s+named\s+([A-Z][A-Za-zА-Яа-яЁё\-]+)",
            r"[Uu]ser\s+is\s+([A-Z][A-Za-zА-Яа-яЁё\-]+)\s*[,.]",
            r"[Мм]еня\s+зовут\s+([A-Za-zА-Яа-яЁё\-]+)",
            r"[Пп]ользователя\s+зовут\s+([A-Za-zА-Яа-яЁё\-]+)",
        ):
            m = _re.search(pat, body)
            if m:
                return m.group(1).strip()
        return ""

    def preamble(self, *, speaker_id: str | None = None) -> str:
        """Block mixed into the system prompt of every conversational call.

        Order: character first, then self-definition, then user profile.
        Core memory is appended separately by the agent (CoreMemory owns it).

        `speaker_id` picks the per-speaker user_profile file. Default
        (None) → WebUI default (legacy `user.md`).

        If the user profile pins a response language, append a
        LANGUAGE OVERRIDE block at the END so it carries the most weight
        in the model's context.
        """
        identity_text = self.identity().strip()
        profile_text = self.user_profile(speaker_id=speaker_id).strip()
        out = (
            "# SOUL\n"
            f"{self.soul().strip()}\n\n"
            "# IDENTITY\n"
            f"{identity_text}\n\n"
            "# USER PROFILE\n"
            f"{profile_text}\n"
        )
        # NAMES block: place at the END (highest model attention) so the
        # agent never confuses its OWN name with the USER'S name. Both
        # names are stated explicitly with `you` / `the user` labels.
        # The agent name comes from agent_name() which falls back to
        # the canonical DEFAULT_AGENT_NAME when identity.md is silent —
        # the prior version only rendered the block if `## Имя`/`## Name`
        # was present, and an install with `## Me` got NO name at all.
        agent_name = self.agent_name()
        user_name = self._extract_user_name(profile_text)
        if agent_name or user_name:
            out += "\n# NAMES — DO NOT CONFUSE\n"
            if agent_name:
                out += (
                    "YOUR name (the assistant's name) — the user "
                    "addresses YOU by this name; if asked 'who are "
                    "you' you answer with this name:\n"
                    f"**{agent_name}**\n\n"
                )
            if user_name:
                out += (
                    f"USER'S name — the person you are talking to is "
                    f"named **{user_name}**. Address the user as "
                    f"'{user_name}' (or by a pronoun in their language). "
                    f"NEVER address the user by YOUR own name. The user "
                    f"is NOT named the same as you.\n\n"
                )
            out += (
                "Rule: if the user says 'I am X' or 'my name is X' or "
                "'you are Y, I am X', then X is the USER'S name (update "
                "your understanding) and Y is YOUR name. Do not flip "
                "these. Do not call the user by your own name.\n\n"
                "Voice: when you write messages, speak in FIRST PERSON "
                "(\"I will remember\", \"я запомню\") — not in third "
                "person about yourself (\"Hrant will remember\", "
                "\"Hrant запомнит\"). Your name is for OTHERS to use; "
                "you refer to yourself as 'I' / 'я'.\n"
            )
        from .config import CONFIG
        forced = (CONFIG.response_language or "mirror").lower()
        lang_body = self._extract_language_section(profile_text)
        out += (
            "\n# RESPONSE LANGUAGE\n"
            "Think, plan, and reason internally in clean ENGLISH — "
            "your private deliberation, tool inputs, structured outputs, "
            "scratch work, and debug text all stay English. Keeps "
            "cognition consistent across multilingual users.\n\n"
        )
        if forced not in ("mirror", ""):
            # Explicit deployment-level override (opt-in via config).
            lang_name = {
                "en": "English", "ru": "Russian", "fr": "French",
                "de": "German", "es": "Spanish", "it": "Italian",
                "pt": "Portuguese", "zh": "Chinese", "ja": "Japanese",
                "uk": "Ukrainian", "hy": "Armenian", "tr": "Turkish",
            }.get(forced, forced)
            out += (
                f"Write your final user-facing answer ONLY in {lang_name}, "
                "regardless of the user's input language or any profile "
                "language pin. This is a deployment-level override.\n"
            )
        elif lang_body:
            # Profile-pinned reply language wins over mirroring.
            out += (
                "Write your final user-facing answer in the language "
                "pinned in the USER PROFILE Language section below — "
                "that pin wins over mirroring:\n"
                f"{lang_body}\n"
            )
        else:
            # Default: mirror the user's input language.
            out += (
                "Write your final user-facing answer in the SAME language "
                "the user wrote their last message in (mirror it). "
                "Russian -> Russian, English -> English, French -> French, "
                "and so on. Always think in English internally; only the "
                "user-facing reply switches language.\n"
            )
        return out

    # ---------- write to user.md ----------
    @staticmethod
    def _normalize_fact(s: str) -> str:
        """Canonical form for dedup comparison.

        Strips the bullet marker, the trailing `_(added YYYY-MM-DD)_`
        timestamp (and the legacy Russian `_(добавлено …)_` form),
        leading/trailing whitespace and punctuation, and lowercases.
        Two bullets that say the same thing in the same language
        collapse to the same key. Cross-language duplicates (EN vs RU
        phrasing of the same fact) still pass — semantic dedup is a
        separate problem we don't tackle here.
        """
        import re as _re
        s = s.strip()
        # drop leading '- ' or '* '
        s = _re.sub(r"^[-*]\s*", "", s)
        # drop the auto-added timestamp marker (English + legacy Russian)
        s = _re.sub(r"\s*_\([Дд]обавлено\s+\d{4}-\d{2}-\d{2}\)_\s*$", "", s)
        s = _re.sub(r"\s*_\(added\s+\d{4}-\d{2}-\d{2}\)_\s*$", "", s)
        # collapse whitespace
        s = _re.sub(r"\s+", " ", s).strip()
        # strip trailing punctuation that varies between phrasings
        s = s.rstrip(".!?;:,")
        return s.lower()

    def add_user_fact(
        self,
        fact: str,
        category: UserFactCategory = "about_user",
        *,
        speaker_id: str | None = None,
    ) -> str:
        """Add a fact to the per-speaker user_profile file.

        Default (speaker_id=None) targets the WebUI legacy `user.md`.
        Pass `speaker_id="telegram:123"` to write into THAT user's
        profile so facts about your wife don't leak into your own
        profile (and vice versa).

        Dedup: same fact (after canonicalisation) won't be re-added.
        Empty placeholders are removed when the first real fact lands."""
        fact = fact.strip()
        if not fact:
            return ""
        stamp = datetime.now().strftime("%Y-%m-%d")
        bullet = f"- {fact}  _(added {stamp})_"
        # Accept legacy Russian headers so facts land in an existing
        # section instead of spawning a duplicate English one; new
        # sections are created with the English header.
        headers = _acceptable_headers(category)
        new_key = self._normalize_fact(fact)

        target_path = self._user_path_for(speaker_id)
        text = self.user_profile(speaker_id=speaker_id)
        lines = text.splitlines()
        # find section (any accepted spelling)
        try:
            sec_idx = next(
                i for i, line in enumerate(lines) if line.strip() in headers
            )
        except StopIteration:
            # missing section — append at file end under the English header
            self._snapshot_user_profile(speaker_id=speaker_id)
            lines += ["", headers[0], bullet]
            target_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return bullet

        end_idx = len(lines)
        for j in range(sec_idx + 1, len(lines)):
            if lines[j].startswith("## "):
                end_idx = j
                break

        for existing in lines[sec_idx + 1 : end_idx]:
            stripped = existing.strip()
            if not stripped or stripped in _PLACEHOLDERS:
                continue
            if self._normalize_fact(stripped) == new_key:
                return existing

        self._snapshot_user_profile(speaker_id=speaker_id)
        body = [
            ln for ln in lines[sec_idx + 1 : end_idx]
            if ln.strip() not in _PLACEHOLDERS
        ]
        while body and not body[-1].strip():
            body.pop()
        body.append(bullet)
        body.append("")

        new_lines = lines[: sec_idx + 1] + body + lines[end_idx:]
        target_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return bullet

    def set_user_profile(self, content: str, *, speaker_id: str | None = None) -> Path:
        """Replace the entire per-speaker user_profile file. Snapshots
        the previous content into _history/ first."""
        self._snapshot_user_profile(speaker_id=speaker_id)
        path = self._user_path_for(speaker_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


IDENTITY = IdentityManager()
