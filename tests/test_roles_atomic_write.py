"""Tests for the 2026-05-23 atomic roles.json write (audit Important #9).

Pre-fix: `_save` did `p.write_text(...)` — non-atomic. A power-loss
between truncate and full body would zero the file. The loader's
catch then falls back to `{"owner_speaker_ids": [DEFAULT_SPEAKER]}`
and SILENTLY DEMOTES every trusted/owner Telegram speaker to guest.

Post-fix: write via `.tmp` + `replace()` — the on-disk path always
points at a fully-written JSON body."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def isolated_roles(tmp_path, monkeypatch):
    from backend.config import CONFIG
    monkeypatch.setitem(CONFIG.knowledge, "base_dir", str(tmp_path / "kb"))
    from backend import roles as _roles
    # Clear any module-level cache that might survive between tests.
    yield _roles


def test_save_writes_via_tmp_and_replace(isolated_roles, monkeypatch):
    """Capture the order of operations: a tmp file is created, then
    replace() rotates it onto the real path. The real path must
    NEVER be opened with truncate-write."""
    real_writes: list[str] = []
    tmp_writes: list[str] = []
    real_text_writes_to_target: list[str] = []
    target = isolated_roles._roles_path()

    orig_write_text = Path.write_text

    def _capture_write_text(self, content, *a, **kw):
        if str(self) == str(target):
            real_text_writes_to_target.append(str(self))
        if str(self).endswith(".tmp"):
            tmp_writes.append(str(self))
        else:
            real_writes.append(str(self))
        return orig_write_text(self, content, *a, **kw)

    with patch.object(Path, "write_text", _capture_write_text):
        isolated_roles._save({
            "owner_speaker_ids": ["webui:default", "telegram:1"],
            "speakers": {"telegram:2": {"role": "trusted"}},
        })

    # Critical assertion: the real target was NEVER directly
    # write_text'd. Only the .tmp was, then it got renamed.
    assert real_text_writes_to_target == []
    # Sanity: a .tmp was written (the staging file)
    assert any(t.endswith(".json.tmp") for t in tmp_writes)


def test_save_produces_valid_json_on_disk(isolated_roles):
    """Round-trip: write a state, read back via load — match."""
    payload = {
        "owner_speaker_ids": ["webui:default", "telegram:42"],
        "speakers": {"telegram:99": {"role": "trusted", "label": "Wife"}},
    }
    isolated_roles._save(payload)
    on_disk = json.loads(isolated_roles._roles_path().read_text(encoding="utf-8"))
    assert on_disk["owner_speaker_ids"] == payload["owner_speaker_ids"]
    assert on_disk["speakers"] == payload["speakers"]


def test_save_leaves_no_tmp_after_success(isolated_roles):
    isolated_roles._save({"owner_speaker_ids": ["webui:default"], "speakers": {}})
    parent = isolated_roles._roles_path().parent
    leftovers = list(parent.glob("*.tmp"))
    assert leftovers == []


def test_save_atomic_under_crash(isolated_roles):
    """Simulate a power-loss between tmp-write and rename. The real
    roles.json must remain intact with whatever WAS there before."""
    # First, write a valid baseline.
    isolated_roles._save({
        "owner_speaker_ids": ["webui:default", "telegram:1"],
        "speakers": {"telegram:2": {"role": "trusted"}},
    })
    target = isolated_roles._roles_path()
    baseline = target.read_text(encoding="utf-8")

    # Now sabotage replace() to simulate crash AFTER tmp write,
    # BEFORE the rename can finish.
    with patch.object(Path, "replace", side_effect=OSError("simulated crash")):
        with pytest.raises(OSError):
            isolated_roles._save({
                "owner_speaker_ids": ["webui:default"],  # would be a demotion
                "speakers": {},
            })

    # The baseline file is UNCHANGED — atomic write held.
    after = target.read_text(encoding="utf-8")
    assert after == baseline
    parsed = json.loads(after)
    assert "telegram:1" in parsed["owner_speaker_ids"]
    assert parsed["speakers"]["telegram:2"]["role"] == "trusted"
