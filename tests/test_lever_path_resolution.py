"""resolve_knowledge_path — surgical fix for the relative-path
bug found in the 2026-05-27 audit #5.

23 autonomic levers had `DEFAULT_X = Path("knowledge/...")`. The
relative path resolved against the service's CWD
(`/home/hrant/hrant/`) instead of the user's data dir
(`~/.hrant/data/`). Result: 8 days of phantom-path writes that
were invisible to the rest of the agent.
"""
from __future__ import annotations

from pathlib import Path


def test_absolute_path_returned_untouched(tmp_path):
    from backend.autonomic.lever import resolve_knowledge_path
    abs_path = tmp_path / "explicit" / "file.jsonl"
    assert resolve_knowledge_path(abs_path) == abs_path


def test_knowledge_relative_rerooted(monkeypatch, tmp_path):
    """Path starting with `knowledge/` gets rerooted under
    `paths.knowledge_dir()`."""
    from backend import paths
    from backend.autonomic.lever import resolve_knowledge_path
    monkeypatch.setattr(paths, "knowledge_dir", lambda: tmp_path / "data" / "knowledge")
    out = resolve_knowledge_path("knowledge/autonomic/model_eval_log.jsonl")
    assert out == tmp_path / "data" / "knowledge" / "autonomic" / "model_eval_log.jsonl"


def test_non_knowledge_relative_falls_back_to_cwd(tmp_path):
    """Relative paths that DON'T start with `knowledge/` go through
    `.resolve()` — legacy behaviour for anything else."""
    from backend.autonomic.lever import resolve_knowledge_path
    out = resolve_knowledge_path("other/file.txt")
    assert out.is_absolute()


def test_string_input_accepted():
    from backend.autonomic.lever import resolve_knowledge_path
    # Just check that string input doesn't raise.
    out = resolve_knowledge_path("knowledge/foo.json")
    assert isinstance(out, Path)


def test_path_input_accepted():
    from backend.autonomic.lever import resolve_knowledge_path
    out = resolve_knowledge_path(Path("knowledge/foo.json"))
    assert isinstance(out, Path)


def test_resolve_is_idempotent_across_levers(monkeypatch, tmp_path):
    """Two different levers resolving the same logical path land at
    the same absolute location."""
    from backend import paths
    from backend.autonomic.lever import resolve_knowledge_path
    monkeypatch.setattr(paths, "knowledge_dir", lambda: tmp_path / "k")
    a = resolve_knowledge_path("knowledge/autonomic/model_eval_log.jsonl")
    b = resolve_knowledge_path("knowledge/autonomic/model_eval_log.jsonl")
    assert a == b
