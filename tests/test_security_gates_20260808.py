"""Security gates closed after the 2026-08-08 system audit.

All three were measured against the live prod box before being fixed:

  * A guest Telegram speaker asked for the channels config and got the live
    bot token back in plain text. `read_file` was the one dangerous tool
    without a handler-level owner gate; its only defence was
    `roles.permissions_block`, a soft prompt, in a codebase whose stated
    lesson is that soft prompts do not hold.
  * `_media_path_is_safe` allowed anything under data_dir, and that is where
    the secrets live — `.env` and `knowledge/channels.json` both passed. Since
    `_strip_and_send_media` removes the MEDIA: line from the visible reply,
    the exfiltration would have left no trace in the message body.
  * Revoking someone by setting their role to `guest` was silently overridden
    by their lingering entry in the legacy `allowed_users` list, and the
    running bot read that list once at startup — so the revoke reported
    success, changed nothing, and would not have taken effect until the next
    deploy either.
"""
from __future__ import annotations

import json

import pytest


# ── read_file is owner-only ───────────────────────────────────────────

def _as(speaker: str):
    from backend.roles import set_current_speaker
    return set_current_speaker(speaker)


def test_read_file_refuses_a_non_owner(tmp_path):
    from backend.roles import reset_current_speaker
    import backend.builtin_tools as bt

    secret = tmp_path / "channels.json"
    secret.write_text('{"bot_token": "SUPER-SECRET"}', encoding="utf-8")

    token = _as("telegram:999999999")           # a guest
    try:
        out = bt._read_file_handler(str(secret))
    finally:
        reset_current_speaker(token)

    assert "SUPER-SECRET" not in out
    payload = json.loads(out)
    assert payload.get("ok") is False or "owner" in json.dumps(payload).lower()


def test_read_file_still_works_for_the_owner(tmp_path):
    from backend.roles import reset_current_speaker
    import backend.builtin_tools as bt

    f = tmp_path / "notes.txt"
    f.write_text("hello owner", encoding="utf-8")
    bt.FILE_CACHE.clear() if hasattr(bt.FILE_CACHE, "clear") else None

    token = _as("webui:default")                # always owner
    try:
        out = bt._read_file_handler(str(f))
    finally:
        reset_current_speaker(token)
    assert "hello owner" in out


# ── MEDIA: attachments come from the delivery dirs only ───────────────

@pytest.mark.parametrize("name", [".env", "channels.json", "roles.json",
                                  "pairing.json", ".anything"])
def test_secrets_and_dotfiles_are_never_attachable(tmp_path, name):
    from backend.channels import _media_path_is_safe
    import tempfile
    from pathlib import Path

    # Put it in the MOST permissive allowed root there is — the tempdir.
    # Even there, a secret name or a dotfile must be refused.
    root = Path(tempfile.gettempdir())
    p = root / f"audit_test_{name}"
    target = root / name
    target.write_text("x", encoding="utf-8")
    try:
        assert _media_path_is_safe(target) is False
    finally:
        target.unlink(missing_ok=True)
        p.unlink(missing_ok=True)


def test_the_whole_data_dir_is_no_longer_an_attachment_source(monkeypatch, tmp_path):
    """The regression that mattered: data_dir/knowledge/<file> used to pass.

    pytest's tmp_path lives UNDER the system tempdir, which is itself an
    allowed delivery root, so the tempdir has to be pointed elsewhere or the
    test passes for the wrong reason."""
    import tempfile
    from backend import channels as ch
    elsewhere = tmp_path.parent / "not-the-temp-root"
    elsewhere.mkdir(exist_ok=True)
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(elsewhere))
    monkeypatch.setattr(ch.paths, "data_dir", lambda require=False: tmp_path)
    private = tmp_path / "knowledge"
    private.mkdir(parents=True)
    f = private / "some_private_state.json"
    f.write_text("{}", encoding="utf-8")
    assert ch._media_path_is_safe(f) is False


def test_outbox_is_still_a_valid_delivery_source(monkeypatch, tmp_path):
    from backend import channels as ch
    monkeypatch.setattr(ch.paths, "data_dir", lambda require=False: tmp_path)
    outbox = tmp_path / "workspace" / "outbox"
    outbox.mkdir(parents=True)
    f = outbox / "invoice.pdf"
    f.write_bytes(b"%PDF-1.4")
    assert ch._media_path_is_safe(f) is True


def test_host_files_are_still_refused():
    from backend.channels import _media_path_is_safe
    from pathlib import Path
    assert _media_path_is_safe(Path("/etc/passwd")) is False


# ── deny beats allow ──────────────────────────────────────────────────

def test_an_explicit_guest_role_overrides_the_legacy_allow_list(monkeypatch):
    """Revoking by role must not be undone by a stale allowed_users entry."""
    from backend import access
    monkeypatch.setattr(
        access, "_roles_list_for_test", None, raising=False)
    import backend.roles as roles_mod
    monkeypatch.setattr(roles_mod, "list_roles", lambda: {
        "speakers": {"telegram:999999999": {"role": "guest"}}})
    monkeypatch.setattr(roles_mod, "role_of", lambda sid: "guest")

    d = access.is_telegram_allowed(999999999, legacy_allowed=["999999999"])
    assert d.allowed is False
    assert "roles.json" in d.reason


def test_the_legacy_list_still_admits_someone_with_no_explicit_role(monkeypatch):
    """Deny-beats-allow must not break the migration path for users who were
    only ever in the legacy list."""
    from backend import access
    import backend.roles as roles_mod
    monkeypatch.setattr(roles_mod, "list_roles", lambda: {"speakers": {}})
    monkeypatch.setattr(roles_mod, "role_of", lambda sid: "guest")

    d = access.is_telegram_allowed(555, legacy_allowed=["555"])
    assert d.allowed is True
    assert d.role == "trusted"


def test_the_bot_rereads_the_allow_list_per_message(monkeypatch):
    """The running bot used to hold a constructor snapshot, so a revoke did
    not take effect until the next deploy."""
    from backend.channels import TelegramBot
    import backend.channels as ch

    bot = TelegramBot.__new__(TelegramBot)
    bot.channel_id = "tg1"
    bot.allowed_users = ["111", "222"]          # the startup snapshot

    monkeypatch.setattr(ch, "get_channel",
                        lambda cid: {"config": {"allowed_users": ["111"]}})
    assert bot._current_allowed_users() == ["111"]

    def _boom(cid):
        raise RuntimeError("config unreadable")

    monkeypatch.setattr(ch, "get_channel", _boom)
    # A read failure must fall back to the snapshot, never widen access and
    # never lock the owner out.
    assert bot._current_allowed_users() == ["111", "222"]
