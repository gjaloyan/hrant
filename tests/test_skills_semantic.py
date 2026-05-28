"""Tests for H4 — semantic match fallback (TF-IDF cosine).

Pinned behaviour:
  - `_semantic_tokenize` lowercases, splits underscores/hyphens,
    and drops tokens shorter than 2 chars (length filter only; no
    stopword list — common words are down-weighted by TF-IDF IDF).
  - `_SemanticIndex.build(skills)` indexes (name + description +
    tags + triggers + when_to_use) per skill. `search(query)` returns
    top-K (name, score) pairs above min_score, ranked by cosine.
  - `SkillsManager.semantic_match(text)` returns Skill objects for
    ENABLED skills only. Disabled skills don't pollute the index.
  - `semantic_match` is a FALLBACK — `match()` always wins; semantic
    surfaces in `unified_agent` only when trigger/tag match is empty.
  - Empty / whitespace / pure-stopword queries return an empty list.
  - The index handles the actual builtin skills sensibly (calc on
    "посчитай", summarize_pdf on "конспект документа").
"""
from __future__ import annotations
import textwrap
from pathlib import Path

import pytest

from backend.skills import (
    SkillsManager,
    Skill,
    _SemanticIndex,
    _semantic_tokenize,
)
from backend.tool_registry import ToolRegistry


@pytest.fixture
def skills_dir(tmp_path, monkeypatch):
    """Isolated skills root + isolated disabled.json so this dev box's
    stale `skills_disabled.json` (e.g. 'vid' left over from an old
    propose_skill smoke) doesn't bleed into the test."""
    d = tmp_path / "skills"
    d.mkdir()
    fake_disabled = tmp_path / "skills_disabled.json"
    monkeypatch.setattr(
        "backend.skills._disabled_path",
        lambda: fake_disabled,
    )
    return d


def _write_skill(root: Path, name: str, frontmatter: str, body: str = "") -> Path:
    sk_dir = root / name
    sk_dir.mkdir()
    (sk_dir / "SKILL.md").write_text(
        f"---\n{frontmatter.strip()}\n---\n\n{body}",
        encoding="utf-8",
    )
    return sk_dir


# ─── _semantic_tokenize ──────────────────────────────────────────────


def test_tokenize_lowercases_and_splits_words():
    assert _semantic_tokenize("Hello World") == ["hello", "world"]


def test_tokenize_splits_underscore_and_hyphen():
    assert _semantic_tokenize("video_overlay-remove") == [
        "video", "overlay", "remove",
    ]


def test_tokenize_keeps_common_words_no_stopword_filter():
    """No stopword list: common words survive; only 1-char tokens are dropped
    by the length filter. TF-IDF IDF down-weights them at index time."""
    toks = _semantic_tokenize("the video is a movie")
    assert "video" in toks
    assert "movie" in toks
    # Common English words now survive (no stopword list)
    assert "the" in toks
    assert "is" in toks
    # Single-char token "a" is still dropped by the length filter
    assert "a" not in toks


def test_tokenize_drops_stopwords_russian():
    toks = _semantic_tokenize("удалить логотип с видео")
    assert "удалить" in toks
    assert "логотип" in toks
    assert "видео" in toks
    assert "с" not in toks


def test_tokenize_drops_short_tokens():
    """1-char tokens are noise."""
    toks = _semantic_tokenize("a b cd ef")
    assert "cd" in toks
    assert "ef" in toks
    assert "a" not in toks
    assert "b" not in toks


def test_tokenize_handles_empty_text():
    assert _semantic_tokenize("") == []
    assert _semantic_tokenize("   ") == []


# ─── _SemanticIndex math ─────────────────────────────────────────────


def _mk(name: str, **kw) -> Skill:
    return Skill(name=name, description=kw.get("description", ""),
                 triggers=kw.get("triggers", []),
                 tags=kw.get("tags", []),
                 when_to_use=kw.get("when_to_use", ""))


def test_index_search_finds_obvious_match():
    idx = _SemanticIndex()
    idx.build([
        _mk("video_logo", description="Remove a logo overlay from a video",
            tags=["video", "logo", "overlay", "ffmpeg"]),
        _mk("calc", description="Evaluate arithmetic expressions",
            tags=["math", "arithmetic"]),
    ])
    hits = idx.search("remove logo from video")
    assert hits
    assert hits[0][0] == "video_logo"
    assert hits[0][1] > 0.15


def test_index_search_returns_empty_for_unrelated_query():
    idx = _SemanticIndex()
    idx.build([
        _mk("calc", description="Evaluate arithmetic expressions",
            tags=["math", "arithmetic"]),
    ])
    hits = idx.search("compose a haiku about clouds")
    assert hits == []


def test_index_search_ranks_by_score_descending():
    idx = _SemanticIndex()
    idx.build([
        _mk("video_logo",
            description="Remove a logo overlay from a video using ffmpeg",
            tags=["video", "logo", "overlay", "ffmpeg", "delogo"]),
        _mk("pdf_summary",
            description="Summarize a PDF document",
            tags=["pdf", "summary"]),
    ])
    hits = idx.search("logo overlay video ffmpeg")
    assert hits
    # video_logo should beat pdf_summary on this query.
    assert hits[0][0] == "video_logo"
    if len(hits) > 1:
        assert hits[0][1] > hits[1][1]


def test_index_respects_limit():
    idx = _SemanticIndex()
    idx.build([
        _mk(f"s{i}", description=f"video processing skill {i}",
            tags=["video", "processing"])
        for i in range(5)
    ])
    hits = idx.search("video", limit=2)
    assert len(hits) <= 2


def test_index_respects_min_score():
    idx = _SemanticIndex()
    idx.build([
        _mk("vid", description="video processing", tags=["video"]),
    ])
    # Query share is small; min_score=0.99 should filter it out.
    hits = idx.search("video", min_score=0.99)
    assert hits == []


def test_index_search_empty_query():
    idx = _SemanticIndex()
    idx.build([_mk("vid", description="video", tags=["video"])])
    assert idx.search("") == []
    # Words with no vocabulary overlap with the index yield no hits.
    assert idx.search("the and a is") == []


def test_index_search_no_docs():
    """Querying an empty index doesn't crash."""
    idx = _SemanticIndex()
    assert idx.search("anything") == []


def test_index_idf_downweights_common_terms():
    """A term appearing in every doc should contribute less than a
    term unique to one doc. This is the whole point of IDF."""
    idx = _SemanticIndex()
    idx.build([
        _mk("a", description="common common common unique_to_a",
            tags=[]),
        _mk("b", description="common common common", tags=[]),
        _mk("c", description="common common common", tags=[]),
    ])
    hits = idx.search("unique_to_a common")
    assert hits
    # 'a' should rank first because of its unique term.
    assert hits[0][0] == "a"


# ─── SkillsManager.semantic_match() ──────────────────────────────────


def test_semantic_match_returns_skills_not_names(skills_dir):
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: Remove a logo overlay from a video using ffmpeg
        triggers: [delogo]
        tags: [video, logo, overlay, ffmpeg]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    result = sm.semantic_match("remove logo from a video")
    assert result
    assert isinstance(result[0], Skill)
    assert result[0].name == "vid"


def test_semantic_match_skips_disabled_skills(skills_dir):
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: Remove a logo overlay from a video using ffmpeg
        triggers: [delogo]
        tags: [video, logo, overlay, ffmpeg]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    sm.set_enabled("vid", False)
    # After disabling, the index rebuild excludes 'vid' so it can't
    # appear in semantic results.
    assert sm.semantic_match("remove logo from a video") == []


def test_semantic_match_empty_query_returns_empty(skills_dir):
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: video work
        triggers: [video]
        tags: [video]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    assert sm.semantic_match("") == []


def test_semantic_match_scored_returns_scores(skills_dir):
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: Remove a logo overlay from a video
        triggers: [delogo]
        tags: [video, logo, overlay]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    pairs = sm.semantic_match_scored("remove logo from video")
    assert pairs
    sk, score = pairs[0]
    assert sk.name == "vid"
    assert 0.0 < score <= 1.0


@pytest.mark.skip(
    reason="2026-05-21: keyword-based Skill.matches() removed; "
    "semantic_match is now the only auto-routing surface"
)
def test_semantic_match_is_fallback_not_replacement_for_match(skills_dir):
    """Skills must still match by trigger — semantic_match is purely
    additive. This pins the contract: both APIs exist, callers decide."""
    _write_skill(skills_dir, "vid", textwrap.dedent("""
        name: vid
        description: video work
        triggers: [конспект]
        tags: [video]
    """))
    sm = SkillsManager(skills_dir=skills_dir)
    sm.load(registry=ToolRegistry())
    # trigger match still works.
    assert [s.name for s in sm.match("сделай конспект")] == ["vid"]


# ─── bundled skills get sensible suggestions ────────────────────────


def test_bundled_calc_surfaces_for_arithmetic_phrase():
    """End-to-end: the shipped `calc` skill should rank for an
    arithmetic-flavoured query that doesn't hit its triggers."""
    from backend import skills as mod
    mod.SKILLS._loaded = False
    mod.SKILLS.skills = []
    mod.SKILLS.ensure_loaded()
    pairs = mod.SKILLS.semantic_match_scored(
        "evaluate this arithmetic expression",
        limit=3,
    )
    names = [s.name for s, _ in pairs]
    assert "calc" in names


def test_bundled_summarize_pdf_surfaces_for_summary_phrase():
    from backend import skills as mod
    mod.SKILLS._loaded = False
    mod.SKILLS.skills = []
    mod.SKILLS.ensure_loaded()
    pairs = mod.SKILLS.semantic_match_scored(
        "make a structured summary out of this document",
        limit=3,
    )
    names = [s.name for s, _ in pairs]
    assert "summarize_pdf" in names


# ─── I1: concurrent rebuild vs search must not crash ───────────────


def test_concurrent_rebuild_vs_search_does_not_crash():
    """Stress: thread A repeatedly searches, thread B repeatedly
    rebuilds the index. Before I1, this raised
    `RuntimeError: dictionary changed size during iteration` from
    inside search() when build() mid-cleared `_vectors`.
    Now build() swaps atomically under a lock and search() snapshots
    before iterating."""
    import threading
    from backend.skills import _SemanticIndex, Skill

    idx = _SemanticIndex()

    def make_skills(n):
        return [
            Skill(name=f"s{i}", description=f"video processing skill {i}",
                  tags=["video", "ffmpeg"])
            for i in range(n)
        ]

    # Initial population so search has something to find.
    idx.build(make_skills(10))

    errors: list = []
    stop = threading.Event()

    def reader():
        try:
            for _ in range(200):
                if stop.is_set():
                    return
                idx.search("video processing")
        except Exception as e:  # pragma: no cover — failure marker
            errors.append(("reader", type(e).__name__, str(e)))

    def writer():
        try:
            for i in range(50):
                if stop.is_set():
                    return
                idx.build(make_skills(10 + (i % 5)))
        except Exception as e:  # pragma: no cover — failure marker
            errors.append(("writer", type(e).__name__, str(e)))

    threads = [
        threading.Thread(target=reader) for _ in range(3)
    ] + [threading.Thread(target=writer) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    stop.set()

    assert not errors, f"concurrent rebuild/search raised: {errors}"


def test_semantic_index_exposes_lock():
    """Pin the API surface — `_SemanticIndex` has a `_lock` attribute
    that's an RLock (re-entrancy lets build use it without
    deadlocking on internal calls)."""
    from backend.skills import _SemanticIndex
    idx = _SemanticIndex()
    assert hasattr(idx, "_lock")
    # RLock instances expose `.acquire()` / `.release()` and support
    # nested with-blocks from the same thread.
    with idx._lock:
        with idx._lock:
            pass  # nested acquire works → it's a RLock
