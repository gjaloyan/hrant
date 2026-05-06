"""Round 12 — workspace mirror so the agent can actually `read_file` on
the things the user uploads.

Before this, attachments were saved as `<sha>.bin` and the only handle
the agent had was a generic "[ATTACHMENT NOTICE]" — calling
`read_file("contract.pdf")` failed because the file was at
`knowledge/attachments/<hash>.bin`. Now every upload also lands at
`workspace/inbox/<original_filename>` and the agent's prompt lists the
exact path.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from backend.workspace import (
    INBOX, OUTBOX, NOTES,
    WorkspaceManager,
)


# --- safe_filename ---------------------------------------------------------


def test_safe_filename_strips_path_separators(tmp_path):
    ws = WorkspaceManager(root=tmp_path)
    assert ws.safe_filename("../etc/passwd") == "passwd"
    assert ws.safe_filename("a/b/c.txt") == "c.txt"
    assert ws.safe_filename("a\\b\\c.txt") == "c.txt"


def test_safe_filename_replaces_control_and_reserved_chars(tmp_path):
    ws = WorkspaceManager(root=tmp_path)
    out = ws.safe_filename('weird<name>:"|.txt')
    # Each disallowed char becomes _; extension preserved.
    assert "<" not in out and ">" not in out and ":" not in out
    assert out.endswith(".txt")


def test_safe_filename_falls_back_when_empty(tmp_path):
    ws = WorkspaceManager(root=tmp_path)
    out = ws.safe_filename("", sha="abcdef0123456789", default="file")
    assert out.startswith("file_")
    # Short sha (8 chars) appended.
    assert "abcdef01" in out


def test_safe_filename_truncates_but_keeps_extension(tmp_path):
    ws = WorkspaceManager(root=tmp_path)
    long = "a" * 300 + ".pdf"
    out = ws.safe_filename(long)
    assert out.endswith(".pdf")
    assert len(out) <= 120 + 1  # cap + the dot


# --- mirror_attachment -----------------------------------------------------


def _write_blob(p: Path, data: bytes) -> str:
    import hashlib
    p.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def test_mirror_creates_inbox_file_with_original_name(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    blob = tmp_path / "blob.bin"
    sha = _write_blob(blob, b"hello world")
    out = ws.mirror_attachment(
        sha=sha, original_name="hello.txt", blob_path=blob, kind="file",
    )
    assert out == ws.root / INBOX / "hello.txt"
    assert out.read_bytes() == b"hello world"


def test_mirror_writes_meta_sidecar(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    blob = tmp_path / "blob.bin"
    sha = _write_blob(blob, b"x")
    out = ws.mirror_attachment(
        sha=sha, original_name="thing.pdf", blob_path=blob,
        kind="file", mime="application/pdf",
    )
    meta_path = out.with_name(out.name + ".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["sha256"] == sha
    assert meta["mime"] == "application/pdf"
    assert meta["original_name"] == "thing.pdf"


def test_mirror_idempotent_for_same_sha_and_name(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    blob = tmp_path / "blob.bin"
    sha = _write_blob(blob, b"same bytes")
    p1 = ws.mirror_attachment(sha=sha, original_name="a.txt", blob_path=blob, kind="file")
    p2 = ws.mirror_attachment(sha=sha, original_name="a.txt", blob_path=blob, kind="file")
    assert p1 == p2
    # No `_<sha>.txt` collision file made — single inbox entry.
    inbox_files = [p for p in (ws.root / INBOX).iterdir() if p.is_file() and not p.name.endswith(".meta.json")]
    assert len(inbox_files) == 1


def test_mirror_disambiguates_same_name_different_bytes(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    blob1 = tmp_path / "b1.bin"
    blob2 = tmp_path / "b2.bin"
    sha1 = _write_blob(blob1, b"version one")
    sha2 = _write_blob(blob2, b"version two")
    p1 = ws.mirror_attachment(sha=sha1, original_name="report.txt", blob_path=blob1, kind="file")
    p2 = ws.mirror_attachment(sha=sha2, original_name="report.txt", blob_path=blob2, kind="file")
    assert p1 != p2
    assert p1.name == "report.txt"
    # Second got a sha-suffix; both files coexist.
    assert sha2[:8] in p2.name
    assert p1.read_bytes() == b"version one"
    assert p2.read_bytes() == b"version two"


def test_mirror_rejects_path_traversal_in_filename(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    blob = tmp_path / "blob.bin"
    sha = _write_blob(blob, b"x")
    p = ws.mirror_attachment(
        sha=sha, original_name="../../../etc/passwd", blob_path=blob, kind="file",
    )
    # Resulting file MUST be inside the inbox dir — no parent escape.
    assert (ws.root / INBOX) in p.parents
    assert p.name == "passwd"


# --- save_outbox -----------------------------------------------------------


def test_save_outbox_writes_file(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    p = ws.save_outbox("draft.md", "# hello")
    assert p == ws.root / OUTBOX / "draft.md"
    assert p.read_text(encoding="utf-8") == "# hello"


def test_save_outbox_collision_gets_timestamp_suffix(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    p1 = ws.save_outbox("draft.md", "v1")
    p2 = ws.save_outbox("draft.md", "v2")
    assert p1 != p2
    # Both files live; original keeps the bare name.
    assert p1.name == "draft.md"
    assert p2.name != "draft.md"
    assert p2.read_text(encoding="utf-8") == "v2"


def test_save_outbox_overwrite_true_replaces(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    p1 = ws.save_outbox("draft.md", "v1")
    p2 = ws.save_outbox("draft.md", "v2", overwrite=True)
    assert p1 == p2
    assert p2.read_text(encoding="utf-8") == "v2"


def test_save_outbox_rejects_inbox_subdir(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    with pytest.raises(ValueError):
        ws.save_outbox("evil.txt", "x", subdir="inbox")


def test_save_outbox_to_notes(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    p = ws.save_outbox("idea.md", "scratch", subdir=NOTES)
    assert p == ws.root / NOTES / "idea.md"


def test_save_outbox_rejects_oversize_content(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    # 10 MB cap — 11 MB content rejected.
    big = "a" * (11 * 1024 * 1024)
    with pytest.raises(ValueError):
        ws.save_outbox("huge.txt", big)


# --- list_subtree ---------------------------------------------------------


def test_list_subtree_returns_files_newest_first(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    p_old = ws.save_outbox("a.txt", "old")
    time.sleep(0.05)
    p_new = ws.save_outbox("b.txt", "new")
    items = ws.list_subtree(OUTBOX)
    names = [i.path.name for i in items]
    assert names == ["b.txt", "a.txt"]


def test_list_subtree_skips_meta_sidecars(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    blob = tmp_path / "blob.bin"
    sha = _write_blob(blob, b"x")
    ws.mirror_attachment(sha=sha, original_name="a.pdf", blob_path=blob, kind="file")
    items = ws.list_subtree(INBOX)
    assert len(items) == 1
    assert items[0].path.name == "a.pdf"


# --- sweep_old -------------------------------------------------------------


def test_sweep_old_deletes_files_older_than_retention(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    blob = tmp_path / "blob.bin"
    sha = _write_blob(blob, b"x")
    p_old = ws.mirror_attachment(sha=sha, original_name="old.txt", blob_path=blob, kind="file")
    # Backdate the inbox file 100 days into the past.
    cutoff = (datetime.utcnow() - timedelta(days=100)).timestamp()
    os.utime(p_old, (cutoff, cutoff))
    meta = p_old.with_name(p_old.name + ".meta.json")
    if meta.exists():
        os.utime(meta, (cutoff, cutoff))
    deleted = ws.sweep_old(
        inbox_retention_days=90,
        outbox_retention_days=0,
        notes_retention_days=0,
    )
    assert deleted[INBOX] == 1
    assert not p_old.exists()
    assert not meta.exists()


def test_sweep_old_zero_retention_disables_subtree(tmp_path):
    ws = WorkspaceManager(root=tmp_path / "ws")
    p = ws.save_outbox("keep.txt", "x")
    cutoff = (datetime.utcnow() - timedelta(days=10000)).timestamp()
    os.utime(p, (cutoff, cutoff))
    deleted = ws.sweep_old(
        inbox_retention_days=0,
        outbox_retention_days=0,  # disabled
        notes_retention_days=0,
    )
    assert deleted[OUTBOX] == 0
    assert p.exists()


# --- AttachmentStore ↔ workspace integration ------------------------------


def test_attachment_save_creates_workspace_mirror(tmp_path, monkeypatch):
    """Saving an attachment must produce a workspace mirror at
    `workspace/inbox/<original_name>` and record `workspace_path` on
    the index entry."""
    from backend import workspace as ws_mod
    from backend.attachments import AttachmentStore

    ws_root = tmp_path / "ws"
    ws_mod._WORKSPACE_INSTANCE = ws_mod.WorkspaceManager(root=ws_root)
    store = AttachmentStore(root=tmp_path / "att")
    rec = store.save(b"document body", "text/plain", filename="doc.txt", kind="file")
    assert rec.workspace_path  # populated
    mirror = ws_root / INBOX / "doc.txt"
    assert mirror.exists()
    assert mirror.read_bytes() == b"document body"
    # Cleanup module singleton so other tests aren't tainted.
    ws_mod._WORKSPACE_INSTANCE = None


def test_attachment_save_failure_to_mirror_doesnt_break_upload(
    tmp_path, monkeypatch,
):
    """If the workspace mirror fails (disk full, perms, etc.) the
    upload still succeeds — workspace is convenience, attachment store
    is source of truth."""
    from backend import workspace as ws_mod
    from backend.attachments import AttachmentStore

    class _Boom:
        def mirror_attachment(self, **_kw):
            raise OSError("simulated disk failure")

        def relative_to_repo(self, _p):
            return ""

    monkeypatch.setattr(ws_mod, "get_workspace", lambda: _Boom())
    store = AttachmentStore(root=tmp_path / "att")
    rec = store.save(b"text", "text/plain", filename="x.txt", kind="file")
    assert rec.sha256
    assert rec.workspace_path == ""  # mirror failed, but upload survived


def test_attachment_old_record_without_workspace_path_loads_fine(tmp_path):
    """Backward-compat: existing index entries pre-this-feature must
    deserialise without crashing on the missing `workspace_path` field."""
    from backend.attachments import Attachment
    rec = {
        "sha256": "a" * 64,
        "kind": "file",
        "mime_type": "text/plain",
        "size": 5,
        "filename": "x.txt",
        "transcript": "",
        "created": "2025-01-01T00:00:00Z",
        # Notably: no workspace_path key.
    }
    att = Attachment.from_record(rec)
    assert att.workspace_path == ""
    assert att.sha256 == "a" * 64


# --- save_to_workspace tool registration ----------------------------------


def test_save_to_workspace_tool_registered():
    from backend.tool_registry import get_registry
    r = get_registry()
    assert "save_to_workspace" in r.tools
    schema = r.tools["save_to_workspace"].input_schema
    # Required args present.
    assert "filename" in schema.get("properties", {})
    assert "content" in schema.get("properties", {})
    # subdir restricted to outbox|notes (no inbox writes from the tool).
    subdir_enum = schema["properties"]["subdir"].get("enum", [])
    assert set(subdir_enum) == {"outbox", "notes"}


def test_save_to_workspace_handler_returns_path(tmp_path):
    from backend import workspace as ws_mod
    from backend.builtin_tools import _save_to_workspace_handler

    ws_mod._WORKSPACE_INSTANCE = ws_mod.WorkspaceManager(root=tmp_path / "ws")
    out = json.loads(_save_to_workspace_handler("note.md", "hello", "notes"))
    assert out["ok"] is True
    assert out["path"].endswith("note.md")
    assert (tmp_path / "ws" / NOTES / "note.md").read_text(encoding="utf-8") == "hello"
    ws_mod._WORKSPACE_INSTANCE = None


def test_save_to_workspace_handler_rejects_inbox():
    from backend.builtin_tools import _save_to_workspace_handler
    out = json.loads(_save_to_workspace_handler("x.txt", "y", "inbox"))
    assert out["ok"] is False
    assert "subdir" in out["error"].lower()


# --- Agent attachment marker mentions workspace paths --------------------


def test_attachment_marker_lists_workspace_paths(tmp_path, monkeypatch):
    """The marker that goes into the LLM prompt must list workspace paths
    so the model can `read_file` them directly."""
    from backend import workspace as ws_mod
    from backend.attachments import AttachmentStore
    from backend.agent import Agent

    ws_mod._WORKSPACE_INSTANCE = ws_mod.WorkspaceManager(root=tmp_path / "ws")
    store = AttachmentStore(root=tmp_path / "att")
    rec = store.save(b"resume body", "text/plain", filename="resume.txt", kind="file")

    # Patch the agent's ATTACHMENTS lookup to use our test store.
    import backend.agent as agent_mod
    monkeypatch.setattr(agent_mod, "_loaded_skills", lambda: None, raising=False)

    # The marker only depends on `_attachments` and the global ATTACHMENTS,
    # so we patch that singleton inline rather than build the whole agent.
    import backend.attachments as att_mod
    monkeypatch.setattr(att_mod, "ATTACHMENTS", store)

    class _Stub(Agent):
        def __init__(self):  # bypass full init
            self._attachments = [rec.sha256]

    marker = _Stub()._attachment_marker()
    assert "ATTACHMENT NOTICE" in marker
    assert "resume.txt" in marker
    # Workspace path is in the marker so read_file finds it.
    assert "workspace" in marker.lower()
    assert "inbox" in marker
    ws_mod._WORKSPACE_INSTANCE = None


# --- Config plumbing -------------------------------------------------------


def test_workspace_config_has_defaults():
    from backend.config import CONFIG
    ws = CONFIG.workspace
    assert ws.get("inbox_retention_days") == 90
    assert ws.get("outbox_retention_days") == 0
    assert ws.get("notes_retention_days") == 0
    assert ws.get("root", "").endswith("workspace") or "workspace" in ws.get("root", "")
