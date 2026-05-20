"""Skills system: декларативные плагины для агента.

Скилл — это директория `backend/skills/<name>/` с двумя возможными файлами:

  SKILL.md   (обязательно)  — фронтматтер + инструкции для LLM
  handler.py (опционально)  — Python-модуль с функцией `register(registry)`,
                              которая добавляет инструменты в ToolRegistry.

Структура SKILL.md:

  ---
  name: pdf_summary
  description: Read a local PDF and produce a structured summary.
  triggers: [pdf, summarize, документ, конспект]
  when_to_use: |
    User asks to summarize, extract, or analyze a local PDF/DOCX file.
  required_tools: [pdfinfo, pypdf]   # optional, see H1
  ---

  # PDF Summary
  Подробные инструкции для LLM в свободной форме: как пользоваться,
  какие шаги, какие инструменты звать, в каком формате отдавать ответ.

Скилл может регистрировать инструменты — тогда они становятся
доступны в обычном tool-use loop. Может и не регистрировать —
тогда это просто блок инструкций, который подмешивается в system prompt
при матче.
"""
from __future__ import annotations
import importlib.util
import math as _math
import re as _re
import shutil as _shutil
import threading as _threading
from collections import Counter as _Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from .tool_registry import ToolRegistry, get_registry


@dataclass
class Skill:
    name: str
    description: str
    triggers: list[str] = field(default_factory=list)
    when_to_use: str = ""
    body: str = ""           # содержимое после frontmatter
    path: Path = field(default_factory=Path)
    # Phase 12B: which tier this skill came from. "builtin" = ships
    # with the engine repo (backend/skills/), "user" = installed by
    # the owner into data_dir (~/.hrant/data/skills/). The latter
    # survives `hrant update`.
    source: str = "builtin"
    enabled: bool = True
    # H1: dependencies the skill expects to be available before it
    # can run. Resolved against (ToolRegistry name) | (binary on PATH)
    # | (importable Python module), in that order. Skills with
    # missing tools are still listed in the catalog but marked
    # `[NEEDS: ...]` so the LLM sees the gap and can propose_install
    # rather than blindly invoke a missing binary.
    #
    # H1-rev: stored as list of dicts `{"name": str, "manager": str|None}`.
    # Strings in YAML are normalised into `{"name": "...", "manager": None}`
    # so the dict form is the only shape callers ever see.
    required_tools: list[dict] = field(default_factory=list)
    # H2: broader topical keywords, complementary to triggers.
    # Matched with WORD-BOUNDARY semantics (`\bvideo\b`) so they don't
    # false-fire on substrings the way triggers do. Triggers stay for
    # narrow intentional phrases; tags are for the topic shape.
    tags: list[str] = field(default_factory=list)

    def matches(self, text: str) -> bool:
        """Triggers (substring, legacy) + tags (word-boundary).

        Triggers match like before — `pdf` fires on `pdf_summary`. Tags
        require a word boundary — `video` fires on `video editor` but
        NOT on `subdivide`. This keeps trigger semantics stable while
        letting tags be broader topical keywords without false-firing.
        """
        if not text:
            return False
        low = text.lower()
        if self.triggers and any(t.lower() in low for t in self.triggers):
            return True
        if self.tags:
            for tag in self.tags:
                tag_low = tag.lower().strip()
                if not tag_low:
                    continue
                if _re.search(rf"\b{_re.escape(tag_low)}\b", low):
                    return True
        return False

    def system_block(self, missing_tools: Optional[list[str]] = None) -> str:
        """Блок, который подмешивается в system prompt при активации.

        `missing_tools`, если передан и непустой, добавляет видимое
        предупреждение сверху — модель должна либо `propose_install`,
        либо пивотнуться, а не звать заведомо отсутствующий бинарь.
        """
        parts = [f"## SKILL: {self.name}", self.description.strip()]
        if missing_tools:
            parts.append(
                f"\n⚠️ **MISSING TOOLS:** {', '.join(missing_tools)} — "
                f"this skill requires them but they are not available. "
                f"Either call `propose_install` to add them, or pivot to "
                f"a different approach."
            )
        if self.when_to_use:
            parts.append(f"\n*When to use:* {self.when_to_use.strip()}")
        if self.body.strip():
            parts.append(f"\n{self.body.strip()}")
        return "\n".join(parts)


def _parse_skill_md(path: Path) -> Optional[Skill]:
    """Парсит SKILL.md. Возвращает None, если файл невалидный."""
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    header = text[3:end].strip()
    body = text[end + 4 :].strip()
    try:
        meta = yaml.safe_load(header) or {}
    except yaml.YAMLError:
        return None
    name = str(meta.get("name", path.parent.name)).strip()
    if not name:
        return None
    return Skill(
        name=name,
        description=str(meta.get("description", "")).strip(),
        triggers=[str(t) for t in (meta.get("triggers") or [])],
        when_to_use=str(meta.get("when_to_use", "")).strip(),
        body=body,
        path=path.parent,
        required_tools=_normalize_required_tools(meta.get("required_tools")),
        tags=[str(t) for t in (meta.get("tags") or [])],
    )


def _normalize_required_tools(raw) -> list[dict]:
    """Accept the two YAML shapes for `required_tools` and emit a
    uniform list of `{"name": str, "manager": str|None}` dicts.

    Shape A — plain strings:
        required_tools: [ffmpeg, pypdf]
    Shape B — dicts with explicit manager:
        required_tools:
          - name: ffmpeg
            manager: apt
          - name: pypdf
            manager: pip
    Shape C — mixed (allowed):
        required_tools:
          - pypdf
          - {name: ffmpeg, manager: apt}

    Anything malformed (no `name`, empty) is silently dropped — a
    bad entry shouldn't break loading the whole skill.
    """
    out: list[dict] = []
    for entry in (raw or []):
        if isinstance(entry, str):
            n = entry.strip()
            if n:
                out.append({"name": n, "manager": None})
            continue
        if isinstance(entry, dict):
            n = str(entry.get("name", "")).strip()
            if not n:
                continue
            m = entry.get("manager")
            m = str(m).strip() if m else None
            out.append({"name": n, "manager": m or None})
            continue
        # Anything else (None, int, list-of-list) is silently skipped.
    return out


# H1: tool availability probe. Public so unified_agent / WebUI can
# call it without recreating the resolution logic.

def _tool_is_available(name: str, registry: Optional[ToolRegistry] = None) -> bool:
    """Probe whether `name` is reachable.

    Resolution order:
      1. Registered tool in the ToolRegistry (e.g. 'read_file').
      2. Binary on PATH (e.g. 'ffmpeg', 'libreoffice').
      3. Importable Python module (e.g. 'PIL', 'pypdf').

    Returns False on any miss — no exceptions propagate.
    """
    if not name:
        return False
    reg = registry if registry is not None else get_registry()
    if name in reg.tools:
        return True
    if _shutil.which(name):
        return True
    try:
        spec = importlib.util.find_spec(name)
        if spec is not None:
            return True
    except Exception:
        return False
    return False


# ─── H4: lightweight semantic match (TF-IDF cosine) ──────────────────
#
# Goal: when triggers + tags miss, fall back to a fuzzy match over the
# skill's metadata bag (name + description + tags + triggers +
# when_to_use). Implemented as TF-IDF cosine — no new dependencies,
# no embeddings, no model download, no API call. It catches paraphrases
# that share vocabulary with the skill metadata (e.g. user types "слить
# логотип на видео" and the skill body mentions "logo" + "video" +
# "delogo"), but NOT pure synonym pairs ("movie" ↔ "video") — that
# requires real embeddings, which we'd add only once the catalogue
# grows past ~30 skills and the keyword approach starts missing.

_SEMANTIC_STOPWORDS = frozenset({
    # English
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "to", "from",
    "with", "for", "by", "at", "is", "are", "was", "were", "be", "been",
    "this", "that", "these", "those", "it", "its", "as", "if", "do", "does",
    "did", "have", "has", "had", "will", "would", "can", "could", "should",
    "i", "you", "he", "she", "we", "they", "me", "my", "your", "our", "their",
    # Russian
    "и", "в", "на", "с", "со", "по", "до", "от", "у", "о", "об", "за",
    "из", "к", "не", "что", "как", "это", "тот", "так", "вот", "там",
    "то", "я", "ты", "он", "она", "мы", "вы", "они", "мне", "тебе",
    "мой", "твой", "наш", "ваш", "их", "его", "её", "ну", "же", "ли",
    "бы", "был", "была", "было", "был", "есть", "быть",
})


def _semantic_tokenize(text: str) -> list[str]:
    """Tokenize for the semantic index.

    Steps:
      1. Lowercase.
      2. Replace underscores and hyphens with spaces (so
         `video_overlay_remove` yields three tokens, not one).
      3. Extract alphanumeric runs as tokens (Unicode-aware).
      4. Drop tokens shorter than 2 chars or in the stopword list.

    No stemming — added cost > marginal recall for our catalogue size.
    """
    if not text:
        return []
    text = text.lower()
    text = _re.sub(r"[_\-]+", " ", text)
    return [
        tok
        for tok in _re.findall(r"\w+", text)
        if len(tok) >= 2 and tok not in _SEMANTIC_STOPWORDS
    ]


class _SemanticIndex:
    """TF-IDF cosine index over the skill catalogue.

    Per skill, we concatenate (name, description, tags, triggers,
    when_to_use) into a bag of tokens. The index is rebuilt on every
    `SkillsManager.load()` so renames / new skills / disabled flags
    take effect on the next turn.

    For our catalogue size (≤50 skills) the math is so cheap there's
    no benefit to caching beyond the per-process build.
    """

    def __init__(self) -> None:
        self._docs: dict[str, list[str]] = {}     # name -> tokens
        self._df: _Counter = _Counter()           # term -> doc-freq
        self._N: int = 0
        self._vectors: dict[str, dict[str, float]] = {}
        # I1: protect read/write of the index from cross-thread races.
        # `SkillsManager.load()` (writer) and `run_unified` → semantic_match
        # (reader) can run on different threads — without this, a
        # Telegram callback that toggles a skill mid-search can clear
        # `_vectors` while `search()` is iterating it and raise
        # RuntimeError: dictionary changed size during iteration.
        self._lock = _threading.RLock()

    def build(self, skills: list[Skill]) -> None:
        # I1: assemble the new state in locals first, then swap atomically
        # under the lock. That way readers never see a half-built index.
        new_docs: dict[str, list[str]] = {}
        new_df: _Counter = _Counter()
        for sk in skills:
            text = " ".join([
                sk.name or "",
                sk.description or "",
                " ".join(sk.tags or []),
                " ".join(sk.triggers or []),
                sk.when_to_use or "",
            ])
            tokens = _semantic_tokenize(text)
            new_docs[sk.name] = tokens
            for term in set(tokens):
                new_df[term] += 1
        new_N = len(new_docs)
        # Compute the vectors with the new corpus before swapping in.
        new_vectors: dict[str, dict[str, float]] = {}
        for name, tokens in new_docs.items():
            new_vectors[name] = self._tf_idf_with_corpus(tokens, new_df, new_N)
        with self._lock:
            self._docs = new_docs
            self._df = new_df
            self._N = new_N
            self._vectors = new_vectors

    @staticmethod
    def _tf_idf_with_corpus(
        tokens: list[str],
        df: _Counter,
        N: int,
    ) -> dict[str, float]:
        """Pure helper — same math as `_tf_idf` but takes the corpus
        DF + N as args so `build` can compute vectors against a fresh
        corpus before publishing it via the lock."""
        if not tokens:
            return {}
        counts = _Counter(tokens)
        max_tf = max(counts.values())
        vec: dict[str, float] = {}
        for term, tf in counts.items():
            d = df.get(term, 0)
            tf_norm = tf / max_tf
            idf = _math.log((1 + N) / (1 + d)) + 1.0
            vec[term] = tf_norm * idf
        return vec

    def search(
        self,
        query: str,
        *,
        limit: int = 3,
        min_score: float = 0.15,
    ) -> list[tuple[str, float]]:
        """Top-K skills by cosine similarity, filtered by min_score."""
        # I1: snapshot the index under the lock. After this point we
        # iterate the local references — even if `build()` runs on
        # another thread it can't mutate what we're walking.
        with self._lock:
            if not self._docs:
                return []
            df_snap = self._df
            N_snap = self._N
            vectors_snap = list(self._vectors.items())
        q_tokens = _semantic_tokenize(query)
        if not q_tokens:
            return []
        q_vec = self._tf_idf_with_corpus(q_tokens, df_snap, N_snap)
        if not q_vec:
            return []
        q_norm = _math.sqrt(sum(v * v for v in q_vec.values()))
        if q_norm == 0:
            return []
        results: list[tuple[str, float]] = []
        for name, doc_vec in vectors_snap:
            d_norm = _math.sqrt(sum(v * v for v in doc_vec.values()))
            if d_norm == 0:
                continue
            dot = sum(q_vec.get(t, 0.0) * w for t, w in doc_vec.items())
            score = dot / (q_norm * d_norm)
            if score >= min_score:
                results.append((name, score))
        results.sort(key=lambda x: -x[1])
        return results[:limit]


def _load_handler(skill_dir: Path, registry: ToolRegistry) -> None:
    """Если рядом с SKILL.md лежит handler.py с функцией register(registry) —
    вызываем её. Это даёт скиллу возможность добавить свои tools."""
    handler_path = skill_dir / "handler.py"
    if not handler_path.exists():
        return
    spec = importlib.util.spec_from_file_location(
        f"backend.skills._loaded.{skill_dir.name}", handler_path
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except Exception as e:
        # Не падаем — скилл просто не будет иметь инструментов.
        print(f"[skills] handler load error in {skill_dir.name}: {e}")
        return
    register = getattr(module, "register", None)
    if callable(register):
        try:
            register(registry)
        except Exception as e:
            print(f"[skills] handler register() failed in {skill_dir.name}: {e}")


def _user_skills_dir() -> Path:
    """Per-install user skills, separate from the engine repo so they
    survive `hrant update`. Sits inside data_dir like every other piece
    of persisted user state."""
    from . import paths as _paths
    return _paths.data_dir(require=False) / "skills"


def _disabled_path() -> Path:
    """Soft kill-switch list: skill names that exist on disk but the
    user has marked disabled in the WebUI. Stored separately so a
    disable-then-re-enable round-trip doesn't risk corrupting the
    skill's own files."""
    from . import paths as _paths
    return _paths.data_dir(require=False) / "skills_disabled.json"


def _load_disabled() -> set[str]:
    """Read `skills_disabled.json`. M2: tolerate stale or malformed
    files but log them — a stale `vid` entry from an old propose_skill
    smoke shouldn't silently disable a real skill named `vid` years
    later, and a corrupted JSON shouldn't pass as "nothing disabled"
    without a warning.
    """
    import json
    import logging as _l
    log = _l.getLogger(__name__)
    p = _disabled_path()
    if not p.exists():
        return set()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        log.warning(
            "skills_disabled.json is corrupted (%s); treating as empty. "
            "Owner: edit %s manually or delete it.", e, p,
        )
        return set()
    except Exception as e:
        log.warning("skills_disabled.json read failed: %s", e)
        return set()
    if not isinstance(raw, dict):
        log.warning(
            "skills_disabled.json: expected JSON object, got %s; "
            "treating as empty. Owner: edit %s.", type(raw).__name__, p,
        )
        return set()
    disabled_raw = raw.get("disabled")
    if disabled_raw is None:
        return set()
    if not isinstance(disabled_raw, list):
        log.warning(
            "skills_disabled.json: 'disabled' must be a list, got %s; "
            "treating as empty. Owner: edit %s.",
            type(disabled_raw).__name__, p,
        )
        return set()
    # Filter to non-empty strings only.
    out: set[str] = set()
    for entry in disabled_raw:
        if isinstance(entry, str) and entry.strip():
            out.add(entry.strip())
        else:
            log.debug(
                "skills_disabled.json: ignoring non-string entry %r", entry,
            )
    return out


def _save_disabled(disabled: set[str]) -> None:
    import json
    p = _disabled_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"disabled": sorted(disabled)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


class SkillsManager:
    """Two-tier discovery:

    - Built-in skills live in `backend/skills/` (engine repo, ships
      with `hrant update`, read-only at runtime).
    - User skills live in `~/.hrant/data/skills/` (per-install,
      writable from the WebUI, preserved across engine updates).

    When the same `name` appears in both tiers the USER tier wins —
    that's how an owner can override a built-in's body without
    touching the engine repo. Disabled skills (in `skills_disabled.json`)
    are still loaded into the manifest but `enabled=False` and they
    DO NOT contribute their triggers / handlers / system blocks.
    """

    def __init__(
        self,
        skills_dir: Optional[Path] = None,
        user_skills_dir: Optional[Path] = None,
    ):
        self.dir = Path(
            skills_dir or Path(__file__).resolve().parent / "skills"
        )
        self.dir.mkdir(parents=True, exist_ok=True)
        # User dir resolved lazily — paths.data_dir() can raise pre-init
        # but skills aren't needed before the first turn anyway.
        self._user_dir_override = user_skills_dir
        self.skills: list[Skill] = []
        self._loaded = False
        # H4: TF-IDF semantic-match fallback. Rebuilt on every load().
        self._semantic_index = _SemanticIndex()

    @property
    def user_dir(self) -> Path:
        return self._user_dir_override or _user_skills_dir()

    def _scan_dir(
        self,
        d: Path,
        source: str,
        disabled: set[str],
        registry: ToolRegistry,
    ) -> list[Skill]:
        if not d.exists():
            return []
        out: list[Skill] = []
        for child in sorted(d.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue
            sk = _parse_skill_md(skill_md)
            if sk is None:
                continue
            sk.source = source
            sk.enabled = sk.name not in disabled
            out.append(sk)
            # Only register handlers for ENABLED skills. A disabled
            # skill's handler.py won't add its tools to the registry.
            if sk.enabled:
                _load_handler(child, registry)
        return out

    def load(self, registry: Optional[ToolRegistry] = None) -> list[Skill]:
        """Scan both built-in and user dirs. User overrides built-in
        by name. Idempotent — safe to re-call after edits."""
        import logging as _l
        log = _l.getLogger(__name__)
        registry = registry or get_registry()
        disabled = _load_disabled()
        builtin = self._scan_dir(self.dir, "builtin", disabled, registry)
        try:
            user = self._scan_dir(self.user_dir, "user", disabled, registry)
        except Exception:
            # data_dir may not be available in tests pre-init.
            user = []
        by_name: dict[str, Skill] = {s.name: s for s in builtin}
        # User overrides builtin.
        for s in user:
            by_name[s.name] = s
        self.skills = sorted(by_name.values(), key=lambda s: s.name)
        # M2: detect stale entries in skills_disabled.json — names
        # listed as disabled but with no matching skill on disk.
        # Common cause: an old propose_skill smoke test left a name
        # behind and the user-tier skill dir got cleaned up later,
        # leaving the disabled flag dangling. Warn so the owner can
        # clean it up; do NOT auto-remove (might be a legitimate
        # placeholder for a soon-to-be-added skill).
        if disabled:
            known = set(by_name.keys())
            stale = sorted(disabled - known)
            if stale:
                log.warning(
                    "skills_disabled.json has %d stale entr%s (no matching "
                    "skill on disk): %s. Owner: edit %s to clean them up.",
                    len(stale), "y" if len(stale) == 1 else "ies",
                    ", ".join(stale), _disabled_path(),
                )
        # H4: rebuild semantic index over the enabled set. Building over
        # all skills would let disabled ones rank against the query and
        # produce ghost suggestions; better to only index what's actually
        # callable.
        try:
            self._semantic_index.build([s for s in self.skills if s.enabled])
        except Exception:
            # Index failure is non-fatal — keyword match still works.
            self._semantic_index = _SemanticIndex()
        self._loaded = True
        return self.skills

    def reload(self, registry: Optional[ToolRegistry] = None) -> list[Skill]:
        """Force re-scan even if already loaded. Called by the WebUI
        after an edit / install / enable-toggle."""
        self._loaded = False
        return self.load(registry)

    def ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def list(self) -> list[Skill]:
        self.ensure_loaded()
        return list(self.skills)

    def match(self, text: str) -> list[Skill]:
        """Triggers from ENABLED skills only."""
        self.ensure_loaded()
        return [s for s in self.skills if s.enabled and s.matches(text)]

    def semantic_match(
        self,
        text: str,
        *,
        limit: int = 3,
        min_score: float = 0.15,
    ) -> list[Skill]:
        """TF-IDF fallback over (name + description + tags + triggers
        + when_to_use). Used when `match()` returns empty but the
        request might still be topical. Returns at most `limit`
        ENABLED skills above `min_score`, ranked by cosine similarity.

        H4 caveat: this is fuzzy keyword match, NOT real semantic
        similarity. It won't bridge pure synonyms ('movie' ↔ 'video')
        — for that you'd need embeddings. It's good enough at our
        catalogue size (≤50 skills) and avoids the model-download +
        per-turn API-call cost embeddings would carry.
        """
        self.ensure_loaded()
        hits = self._semantic_index.search(text, limit=limit, min_score=min_score)
        out: list[Skill] = []
        for name, _score in hits:
            sk = self.get(name)
            if sk is not None and sk.enabled:
                out.append(sk)
        return out

    def semantic_match_scored(
        self,
        text: str,
        *,
        limit: int = 3,
        min_score: float = 0.15,
    ) -> list[tuple[Skill, float]]:
        """Same as `semantic_match()` but returns (skill, score) pairs
        so callers (and tests) can inspect the ranking signal."""
        self.ensure_loaded()
        hits = self._semantic_index.search(text, limit=limit, min_score=min_score)
        out: list[tuple[Skill, float]] = []
        for name, score in hits:
            sk = self.get(name)
            if sk is not None and sk.enabled:
                out.append((sk, score))
        return out

    def find_proposal_duplicates(
        self,
        *,
        name: str,
        description: str,
        triggers: list[str] | None = None,
        when_to_use: str = "",
        body: str = "",
        min_score: float = 0.55,
        limit: int = 3,
    ) -> list[tuple[Skill, float]]:
        """Find existing skills whose semantic signature is too close
        to a proposed new one.

        Used by `propose_skill` to refuse a duplicate before it lands.
        Without this, a post-turn skill_reflection that misses the
        existing-catalog hint produces a second skill with a different
        name covering the same workflow (e.g. an active
        `background-job-status` skill plus a fresh proposal for
        `check-bench-status` that does the same thing).

        Score is Jaccard overlap over the tokenized text
        (name + description + tags + triggers + when_to_use) — the
        same tokenizer the semantic index uses. Considers BOTH enabled
        and disabled skills because a duplicate disabled skill is
        still a duplicate. The same-name overwrite path (clean_name
        equals an existing skill's name) is NOT flagged — that's the
        explicit merge channel skill_reflection is supposed to take.
        """
        self.ensure_loaded()
        proposal_text = " ".join(filter(None, [
            name or "",
            description or "",
            " ".join(triggers or []),
            when_to_use or "",
            body or "",
        ]))
        proposal_tokens = _semantic_tokenize(proposal_text)
        if not proposal_tokens:
            return []
        proposal_set: set[str] = set(proposal_tokens)
        clean_name = "".join(
            c if c.isalnum() or c in "_-" else "_" for c in (name or "")
        ).strip("_")
        out: list[tuple[Skill, float]] = []
        for sk in self.skills:
            # Same-name proposal is an explicit overwrite, not a
            # duplicate — let it through.
            if sk.name == clean_name:
                continue
            sk_text = " ".join([
                sk.name or "",
                sk.description or "",
                " ".join(sk.tags or []),
                " ".join(sk.triggers or []),
                sk.when_to_use or "",
            ])
            sk_tokens = set(_semantic_tokenize(sk_text))
            if not sk_tokens:
                continue
            inter = proposal_set & sk_tokens
            union = proposal_set | sk_tokens
            score = len(inter) / max(1, len(union))
            if score >= min_score:
                out.append((sk, score))
        out.sort(key=lambda p: p[1], reverse=True)
        return out[:limit]

    def get(self, name: str) -> Optional[Skill]:
        self.ensure_loaded()
        for s in self.skills:
            if s.name == name:
                return s
        return None

    def missing_tools_for(
        self,
        skill: Skill,
        registry: Optional[ToolRegistry] = None,
    ) -> list[str]:
        """Return the subset of `skill.required_tools` not currently
        resolvable on this host, as a list of names. Empty list =
        ready to run.

        Walks the normalised dict shape (`{"name": ..., "manager": ...}`)
        — `_normalize_required_tools` guarantees this layout at parse
        time, so callers always see dicts here. Returns just the
        names, since the existing surfaces (catalog NEEDS marker,
        system_block warning) care only about what's missing, not
        which manager will install it.
        """
        if not skill.required_tools:
            return []
        reg = registry if registry is not None else get_registry()
        out: list[str] = []
        for entry in skill.required_tools:
            # Schema invariant: every entry is a dict after parse-time
            # normalisation. Anything else is a parser bug; surface it
            # rather than silently coercing.
            assert isinstance(entry, dict), (
                f"required_tools entry not dict (schema bug): {entry!r}"
            )
            name = (entry.get("name") or "").strip()
            if not name:
                continue
            if not _tool_is_available(name, reg):
                out.append(name)
        return out

    def missing_tools_with_manager_for(
        self,
        skill: Skill,
        registry: Optional[ToolRegistry] = None,
    ) -> list[dict]:
        """Same as `missing_tools_for` but returns dicts with the
        resolved manager: `[{"name": "ffmpeg", "manager": "apt"}, ...]`.

        Manager resolution: explicit hint on the frontmatter wins;
        otherwise `installer.resolve_manager_for(name)` guesses based
        on the known-binary list (apt for ffmpeg/libreoffice/qpdf/...,
        pip for everything else).

        Used by the auto-propose path in unified_agent to fire
        `installer.propose` per missing tool with the right manager.
        """
        from . import installer as _installer

        if not skill.required_tools:
            return []
        reg = registry if registry is not None else get_registry()
        out: list[dict] = []
        for entry in skill.required_tools:
            # Schema invariant — see missing_tools_for. After H1-rev,
            # entries are always dicts; the legacy `list[str]` shape
            # is dead code post-parse.
            assert isinstance(entry, dict), (
                f"required_tools entry not dict (schema bug): {entry!r}"
            )
            name = (entry.get("name") or "").strip()
            manager = (entry.get("manager") or "").strip() or None
            if not name:
                continue
            if _tool_is_available(name, reg):
                continue
            if not manager:
                manager = _installer.resolve_manager_for(name)
            out.append({"name": name, "manager": manager})
        return out

    def catalog_block(self, registry: Optional[ToolRegistry] = None) -> str:
        """Catalog of ENABLED skills only. Skills with `required_tools`
        that aren't resolvable on this host are marked `[NEEDS: ...]`
        so the LLM can `propose_install` or pivot rather than silently
        try to call a missing binary."""
        self.ensure_loaded()
        active = [s for s in self.skills if s.enabled]
        if not active:
            return ""
        lines = [
            "# AVAILABLE SKILLS",
            "(Detailed instructions activate automatically on trigger match.)",
        ]
        for s in active:
            triggers = ", ".join(s.triggers) if s.triggers else "—"
            meta_parts = [f"triggers: {triggers}"]
            if s.tags:
                meta_parts.append(f"tags: {', '.join(s.tags)}")
            line = f"- **{s.name}**: {s.description}  _({'; '.join(meta_parts)})_"
            missing = self.missing_tools_for(s, registry=registry)
            if missing:
                line += f"  ⚠️ **[NEEDS: {', '.join(missing)}]**"
            lines.append(line)
        return "\n".join(lines)

    # ---- write paths (used by the API) ----

    def set_enabled(self, name: str, enabled: bool) -> Optional[Skill]:
        """Toggle the soft kill-switch. Returns the updated skill or
        None if the name doesn't exist."""
        self.ensure_loaded()
        existing = self.get(name)
        if existing is None:
            return None
        disabled = _load_disabled()
        if enabled:
            disabled.discard(name)
        else:
            disabled.add(name)
        _save_disabled(disabled)
        self.reload()
        return self.get(name)

    def upsert_user_skill(self, name: str, content: str) -> Skill:
        """Create or update a user skill from raw SKILL.md content.
        Owner-only path — caller enforces. Writes to data_dir; never
        touches the engine repo."""
        clean = "".join(c if c.isalnum() or c in "_-" else "_" for c in name).strip("_")
        if not clean:
            raise ValueError("skill name must contain at least one alphanumeric")
        target = self.user_dir / clean / "SKILL.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self.reload()
        sk = self.get(clean)
        if sk is None:
            raise ValueError("written file did not parse as a valid skill")
        return sk

    def delete_user_skill(self, name: str) -> bool:
        """Remove a user-tier skill from disk. Built-in skills are
        NOT deletable from the WebUI (would be re-shipped by the
        next `hrant update` anyway). Returns True if removed."""
        target = self.user_dir / name
        if not target.exists():
            return False
        import shutil
        shutil.rmtree(target)
        self.reload()
        return True


SKILLS = SkillsManager()


# ─── Self-improvement loop: propose a new skill ─────────────────────


# Subscribers fire when `propose()` creates a new draft skill. The
# Telegram bridge subscribes to DM the owner with an inline-button
# approval prompt; the WebUI could subscribe too.
_ON_SKILL_PROPOSED: list = []


def register_on_skill_proposed(fn) -> None:
    """Idempotent subscribe to skill-proposed events."""
    if fn not in _ON_SKILL_PROPOSED:
        _ON_SKILL_PROPOSED.append(fn)


def _fire_skill_proposed(skill: Skill) -> None:
    for fn in list(_ON_SKILL_PROPOSED):
        try:
            fn(skill)
        except Exception as e:
            import logging as _l
            _l.getLogger(__name__).warning(
                "skill-proposed callback %s failed: %s", fn, e,
            )


def _render_skill_md(
    *,
    name: str,
    description: str,
    triggers: list[str] | None,
    when_to_use: str,
    body: str,
) -> str:
    """Build a SKILL.md text from structured inputs. YAML frontmatter
    is intentionally minimal — name/description/triggers/when_to_use —
    so a human reading the proposal in Telegram or the WebUI sees
    exactly the fields the registry cares about."""
    trig_yaml = "[" + ", ".join(repr(t) for t in (triggers or [])) + "]"
    fm_lines = [
        "---",
        f"name: {name}",
        f"description: {description!r}",
        f"triggers: {trig_yaml}",
    ]
    wtu = (when_to_use or "").strip()
    if wtu:
        # Multi-line when_to_use through YAML block-literal so the
        # author can write a paragraph without quoting issues.
        fm_lines.append("when_to_use: |")
        for ln in wtu.splitlines():
            fm_lines.append(f"  {ln}")
    fm_lines.append("---")
    return "\n".join(fm_lines) + "\n\n" + (body or "").lstrip()


def propose(
    *,
    name: str,
    description: str,
    triggers: list[str] | None = None,
    when_to_use: str = "",
    body: str = "",
    requester: str = "webui:default",
) -> Optional[Skill]:
    """Create a new draft skill from a structured request.

    Self-improvement loop entry point. The agent calls this after
    successfully completing a non-trivial workflow that future turns
    could reuse. The new skill is written to the user-tier directory
    (`~/.hrant/data/skills/<name>/SKILL.md`) BUT IS DISABLED by
    default — the owner must press 'Activate' in Telegram (or
    enable in the WebUI) before it goes live. That's the safety
    backstop: an agent-authored skill can't auto-execute on the
    next turn without explicit human OK.

    Returns the freshly-loaded Skill (disabled state) or None on
    persistence failure.
    """
    clean_name = "".join(
        c if c.isalnum() or c in "_-" else "_" for c in (name or "")
    ).strip("_")
    if not clean_name:
        return None
    content = _render_skill_md(
        name=clean_name,
        description=description or "",
        triggers=triggers,
        when_to_use=when_to_use,
        body=body,
    )
    try:
        sk = SKILLS.upsert_user_skill(clean_name, content)
    except Exception as e:
        import logging as _l
        _l.getLogger(__name__).warning("propose(): upsert failed: %s", e)
        return None
    # Disable by default — owner must explicitly activate.
    SKILLS.set_enabled(clean_name, False)
    sk = SKILLS.get(clean_name)
    if sk is None:
        return None
    _fire_skill_proposed(sk)
    return sk


# ─── Inline-keyboard callback dispatcher (skill: prefix) ────────────


def _register_skill_callback() -> None:
    """Wire skill-proposal approval into the tg_interactive dispatcher.

    callback_data shapes:
      - skill:enable:<name>  activate (remove from disabled.json)
      - skill:show:<name>    return the SKILL.md body as a followup
      - skill:delete:<name>  remove the user-tier skill from disk

    Owner-only. Non-owner clickers get a refusal toast.
    """
    from . import tg_interactive as _tg
    from . import roles as _roles_mod

    def _handler(parts, ctx):
        if not parts:
            return _tg.CallbackResult(ok=False, toast="malformed callback")
        clicker_id = ctx.get("clicker_speaker_id") or ""
        if not _roles_mod.is_owner(clicker_id):
            return _tg.CallbackResult(
                ok=False,
                toast="only the owner can manage skills",
                clear_keyboard=False,
            )
        action = parts[0]
        if action == "enable":
            if len(parts) < 2:
                return _tg.CallbackResult(ok=False, toast="malformed enable")
            sname = parts[1]
            sk = SKILLS.set_enabled(sname, True)
            if sk is None:
                return _tg.CallbackResult(
                    ok=False, toast=f"skill {sname!r} not found",
                    clear_keyboard=False,
                )
            return _tg.CallbackResult(
                ok=True,
                edited_text=(
                    f"✅ <b>Skill activated</b>\n"
                    f"<code>{_tg.escape_html(sname)}</code> is now live."
                ),
                toast="activated",
            )
        if action == "show":
            if len(parts) < 2:
                return _tg.CallbackResult(ok=False, toast="malformed show")
            sname = parts[1]
            sk = SKILLS.get(sname)
            if sk is None:
                return _tg.CallbackResult(
                    ok=False, toast=f"skill {sname!r} not found",
                    clear_keyboard=False,
                )
            body = sk.body or "(empty body)"
            if len(body) > 3700:
                body = body[:3700] + "\n…[truncated]"
            return _tg.CallbackResult(
                ok=True,
                toast="body sent",
                followup_text=(
                    f"📄 <b>SKILL: {_tg.escape_html(sname)}</b>\n"
                    f"<pre>{_tg.escape_html(body)}</pre>"
                ),
                clear_keyboard=False,
            )
        if action == "delete":
            if len(parts) < 2:
                return _tg.CallbackResult(ok=False, toast="malformed delete")
            sname = parts[1]
            ok = SKILLS.delete_user_skill(sname)
            if not ok:
                return _tg.CallbackResult(
                    ok=False, toast=f"skill {sname!r} not found or not user-tier",
                    clear_keyboard=False,
                )
            return _tg.CallbackResult(
                ok=True,
                edited_text=(
                    f"❌ <b>Skill deleted</b>\n"
                    f"<code>{_tg.escape_html(sname)}</code> removed from disk."
                ),
                toast="deleted",
            )
        return _tg.CallbackResult(ok=False, toast=f"unknown action {action!r}")

    _tg.register_callback_handler("skill", _handler)


_register_skill_callback()
