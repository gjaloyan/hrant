"""Audit follow-up: sensitive config files must land on disk as 0o600.

The files involved hold API keys (providers.json, oauth_tokens.json),
OAuth refresh tokens, paired-device codes (pairing.json), chat ID
maps (telegram_chat_ids.json), and operator-private analytics
(access_log.json). Pre-fix they were created with the umask default
(0o644 on most Linux distros) which means any local user could read
them. After the fix:

  - `paths.write_secret_json` chmods the .tmp pre-rename to 0o600
    so the atomic swap preserves owner-only mode.
  - `paths.secure_existing_file` tightens an existing world-readable
    file on boot.
  - Each sensitive writer routes through `write_secret_json`.

On Windows `os.chmod` is largely a no-op (POSIX mode bits don't map
to NTFS ACLs); we skip mode-bit assertions there but still run the
behavior tests for atomicity.
"""
from __future__ import annotations

import json
import os
import stat
import sys

import pytest


# ─── Helper-level tests ───────────────────────────────────────────


def test_write_secret_json_creates_file_at_0o600(tmp_path):
    """The helper writes the JSON and the resulting file has only
    owner read/write bits set (no group, no other)."""
    from backend.paths import write_secret_json

    p = tmp_path / "secret.json"
    write_secret_json(p, {"k": "v"})
    assert p.exists()
    loaded = json.loads(p.read_text(encoding="utf-8"))
    assert loaded == {"k": "v"}
    if sys.platform != "win32":
        mode = stat.S_IMODE(p.stat().st_mode)
        assert mode == 0o600, (
            f"sensitive file must be 0o600, got 0o{mode:o}"
        )


def test_write_secret_json_overwrites_world_readable_predecessor(tmp_path):
    """An existing 0o644 file (created by older code or by hand) gets
    rewritten by `write_secret_json` to 0o600. Pre-fix this was the
    common production state: the file was world-readable forever
    because chmod was never called."""
    from backend.paths import write_secret_json

    p = tmp_path / "secret.json"
    p.write_text("{}", encoding="utf-8")
    if sys.platform != "win32":
        os.chmod(p, 0o644)
        assert stat.S_IMODE(p.stat().st_mode) == 0o644

    write_secret_json(p, {"new": "content"})
    if sys.platform != "win32":
        mode = stat.S_IMODE(p.stat().st_mode)
        assert mode == 0o600, (
            f"write_secret_json must tighten an existing world-readable "
            f"file, got 0o{mode:o}"
        )


def test_write_secret_json_leaves_no_tmp_sibling(tmp_path):
    """Atomic via .tmp + rename — the .tmp sibling must NOT remain
    on disk after a successful write."""
    from backend.paths import write_secret_json

    p = tmp_path / "secret.json"
    write_secret_json(p, {"k": "v"})
    leftover = p.with_suffix(p.suffix + ".tmp")
    assert not leftover.exists()


def test_secure_existing_file_tightens_644_to_600(tmp_path):
    """Boot-time sweep: an existing 0o644 file gets retightened to
    0o600 without rewriting its contents."""
    if sys.platform == "win32":
        pytest.skip("POSIX mode bits don't apply on Windows")
    from backend.paths import secure_existing_file

    p = tmp_path / "old.json"
    p.write_text('{"keep": "me"}', encoding="utf-8")
    os.chmod(p, 0o644)
    assert stat.S_IMODE(p.stat().st_mode) == 0o644

    secure_existing_file(p)
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    # Contents preserved.
    assert json.loads(p.read_text(encoding="utf-8")) == {"keep": "me"}


def test_secure_existing_file_no_op_on_missing(tmp_path):
    """Missing file → silent no-op (boot sweep shouldn't blow up
    on configs that don't exist yet)."""
    from backend.paths import secure_existing_file
    secure_existing_file(tmp_path / "nonexistent.json")  # must not raise


# ─── Writer wiring — each sensitive writer routes through the helper ──


def test_save_providers_writes_at_0o600(tmp_path, monkeypatch):
    """`_save_providers` is the providers.json writer. It MUST emit
    0o600. This is the single biggest audit finding — providers.json
    holds raw API keys."""
    if sys.platform == "win32":
        pytest.skip("POSIX mode bits don't apply on Windows")
    monkeypatch.setattr(
        "backend.providers.PROVIDERS_PATH", tmp_path / "providers.json"
    )
    from backend.providers import _save_providers

    _save_providers([{"id": "anthropic-1", "type": "anthropic", "api_key": "sk-test"}])
    p = tmp_path / "providers.json"
    mode = stat.S_IMODE(p.stat().st_mode)
    assert mode == 0o600, f"providers.json should be 0o600, got 0o{mode:o}"


def test_save_chat_ids_writes_at_0o600(tmp_path, monkeypatch):
    """`save_chat_ids` writes telegram_chat_ids.json — maps Telegram
    user_id → chat_id, so leak = doxxing risk."""
    if sys.platform == "win32":
        pytest.skip("POSIX mode bits don't apply on Windows")
    p = tmp_path / "telegram_chat_ids.json"
    monkeypatch.setattr("backend.contacts._chat_ids_path", lambda: p)
    from backend.contacts import save_chat_ids
    save_chat_ids({"222": {"chat_id": 12345, "username": "u", "label": "x"}})
    mode = stat.S_IMODE(p.stat().st_mode)
    assert mode == 0o600


def test_save_tts_config_writes_at_0o600(tmp_path, monkeypatch):
    """`tts.save_config` writes tts_config.json."""
    if sys.platform == "win32":
        pytest.skip("POSIX mode bits don't apply on Windows")
    p = tmp_path / "tts_config.json"
    monkeypatch.setattr("backend.tts._config_path", lambda: p)
    from backend.tts import save_config
    save_config({"backend": "edge_tts"})
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_save_transcriber_config_writes_at_0o600(tmp_path, monkeypatch):
    """`transcriber.save_config` writes transcriber_config.json."""
    if sys.platform == "win32":
        pytest.skip("POSIX mode bits don't apply on Windows")
    p = tmp_path / "transcriber_config.json"
    monkeypatch.setattr("backend.transcriber._config_path", lambda: p)
    from backend.transcriber import save_config
    save_config({"backend": "openai_whisper"})
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_active_model_save_writes_at_0o600(tmp_path, monkeypatch):
    """ActiveModel persists to active_model.json. Not as sensitive as
    keys, but leaks operator choices — audit lists it as 0o600."""
    if sys.platform == "win32":
        pytest.skip("POSIX mode bits don't apply on Windows")
    p = tmp_path / "active_model.json"
    monkeypatch.setattr("backend.providers.ACTIVE_MODEL_PATH", p)
    # ActiveModel._save reads ACTIVE_MODEL_PATH at the module scope,
    # so we re-import to pick up the patched value, then construct a
    # fresh instance.
    import importlib
    import backend.providers as prov_mod
    importlib.reload(prov_mod)
    # Re-apply the patch after reload (reload re-resolves the module).
    monkeypatch.setattr(prov_mod, "ACTIVE_MODEL_PATH", p)
    am = prov_mod.ActiveModel()
    am._data = {"provider_id": "anthropic-1", "model": "claude-x"}
    am._save()
    assert p.exists()
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_pairing_save_writes_at_0o600(tmp_path):
    """`PairingStore._save` writes pairing.json — contains time-
    bounded pairing codes used to enroll new devices. Audit: 0o600."""
    if sys.platform == "win32":
        pytest.skip("POSIX mode bits don't apply on Windows")
    from backend.access import PairingStore
    p = tmp_path / "pairing.json"
    store = PairingStore.__new__(PairingStore)
    # Hand-construct just enough state for _save() to run.
    store.path = p
    import threading
    store._lock = threading.RLock()
    store._save({"requests": []})
    assert p.exists()
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_oauth_tokens_save_writes_at_0o600(tmp_path, monkeypatch):
    """OAuthTokenStore._save writes oauth_tokens.json — refresh
    tokens, expiry, access tokens. The most leak-sensitive file."""
    if sys.platform == "win32":
        pytest.skip("POSIX mode bits don't apply on Windows")
    p = tmp_path / "oauth_tokens.json"
    monkeypatch.setattr("backend.providers.OAUTH_TOKENS_PATH", p)
    from backend.providers import OAuthTokenStore
    store = OAuthTokenStore.__new__(OAuthTokenStore)
    import threading
    store._lock = threading.RLock()
    store._tokens = {"anthropic-1": {"access_token": "secret"}}
    store._save()
    assert p.exists()
    assert stat.S_IMODE(p.stat().st_mode) == 0o600


def test_access_log_save_writes_at_0o600(tmp_path):
    """KnowledgeManager._write_json writes access_log.json (and
    other internal JSON). Audit lists access_log.json at 0o600
    because it reveals operator topic-access patterns."""
    if sys.platform == "win32":
        pytest.skip("POSIX mode bits don't apply on Windows")
    from backend.knowledge_manager import KnowledgeManager
    base = tmp_path / "knowledge"
    base.mkdir()
    km = KnowledgeManager.__new__(KnowledgeManager)
    km.base = base
    p = base / "access_log.json"
    km._write_json(p, {"topic-a": 3})
    assert p.exists()
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
