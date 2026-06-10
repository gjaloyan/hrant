"""Atomic-write helper contract — Bundle B (Fix C3).

The audit found ~13 JSON stores used direct `path.write_text(json.dumps(...))`.
A crash between truncate and final write left the file zero-length or
torn; the next `_load` either reset the store to empty (losing data) or
hit a JSONDecodeError. `backend.paths.write_atomic_json` extracts the
proven atomic-rename pattern so every JSON writer in the codebase goes
through one battle-tested path.

Pinned behaviour:
  - The writer NEVER writes directly to `path`; it writes to a `.tmp`
    sibling and then `replace`s.
  - An exception raised mid-write leaves the prior version of `path`
    intact (we test by forcing a `_load` after a faked OS failure).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_write_atomic_json_uses_tmp_rename(tmp_path: Path):
    """The helper writes to a `.tmp` sibling and renames to the
    target. Final file content matches the input dict."""
    from backend.paths import write_atomic_json

    target = tmp_path / "store.json"
    payload = {"a": 1, "b": [2, 3], "c": "hi"}

    write_atomic_json(target, payload)

    assert target.exists()
    # The .tmp file must NOT remain after a successful write.
    assert not (tmp_path / "store.json.tmp").exists()
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk == payload


def test_write_atomic_json_tmp_then_replace(tmp_path: Path):
    """The .tmp file is created BEFORE the rename, then renamed.
    We mock Path.replace to capture the source path and verify
    the tmp + rename ordering directly."""
    from backend.paths import write_atomic_json

    target = tmp_path / "store.json"
    expected_tmp = tmp_path / "store.json.tmp"
    captured: dict = {}

    original_replace = Path.replace

    def _spy(self, other):
        captured["replace_src"] = Path(self)
        captured["replace_dst"] = Path(other)
        # The tmp must already exist on disk at the moment replace fires.
        captured["tmp_existed_pre_replace"] = expected_tmp.exists()
        return original_replace(self, other)

    with patch.object(Path, "replace", _spy):
        write_atomic_json(target, {"x": 1})

    assert captured["replace_src"] == expected_tmp
    assert captured["replace_dst"] == target
    assert captured["tmp_existed_pre_replace"] is True
    # After the replace, only the final file remains.
    assert target.exists()
    assert not expected_tmp.exists()


def test_write_atomic_json_no_torn_file_on_crash(tmp_path: Path):
    """If the writer is interrupted mid-write (here: replace raises),
    the previously-saved file is preserved unchanged. The `.tmp`
    sibling may stay behind but the user's data file is intact."""
    from backend.paths import write_atomic_json

    target = tmp_path / "store.json"
    # Seed a prior known-good version.
    target.write_text(
        json.dumps({"version": "previous"}, ensure_ascii=False),
        encoding="utf-8",
    )

    def _explode(self, other):
        raise OSError("simulated crash between truncate and rename")

    with patch.object(Path, "replace", _explode):
        with pytest.raises(OSError):
            write_atomic_json(target, {"version": "next"})

    # Prior file is intact — no torn write reached the real path.
    assert target.exists()
    on_disk = json.loads(target.read_text(encoding="utf-8"))
    assert on_disk == {"version": "previous"}


def test_write_atomic_json_creates_parent_dirs(tmp_path: Path):
    """If the parent dir doesn't exist, the helper creates it."""
    from backend.paths import write_atomic_json

    target = tmp_path / "deep" / "nested" / "store.json"
    assert not target.parent.exists()

    write_atomic_json(target, {"ok": True})

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def test_write_atomic_json_indent_none_compact(tmp_path: Path):
    """`indent=None` produces the compact single-line shape that
    vector_store et al. rely on to keep file size down."""
    from backend.paths import write_atomic_json

    target = tmp_path / "store.json"
    write_atomic_json(target, {"a": 1, "b": 2}, indent=None)
    text = target.read_text(encoding="utf-8")
    assert "\n" not in text.strip()


def test_write_atomic_json_chmod_when_mode_given(tmp_path: Path):
    """When `mode` is passed, the final file ends up with that mode
    (POSIX only — best-effort on Windows). Smoke test by passing
    0o600 and asserting the call doesn't raise; the post-rename file
    must exist."""
    from backend.paths import write_atomic_json

    target = tmp_path / "secret.json"
    write_atomic_json(target, {"k": "v"}, mode=0o600)
    assert target.exists()


def test_conversation_save_uses_atomic_write(tmp_path: Path, monkeypatch):
    """ConversationMemory._save routes through write_atomic_json.

    Sanity check that the migration in conversation.py landed —
    monkeypatch the helper and assert _save calls it with the right
    payload shape."""
    from backend import conversation as conv_mod

    captured: dict = {}

    def _spy(path, data, **kwargs):
        captured["path"] = Path(path)
        captured["data"] = data

    monkeypatch.setattr(conv_mod, "write_atomic_json", _spy)
    mem = conv_mod.ConversationMemory(path=tmp_path / "conv.json")
    mem.add_turn("hi", "hey", intent="chat", channel="webui")

    assert captured["path"] == tmp_path / "conv.json"
    # Data shape: list of turn dicts, our single new turn is the
    # only entry (the file didn't exist before).
    assert isinstance(captured["data"], list)
    assert len(captured["data"]) == 1
    assert captured["data"][0]["user"] == "hi"
