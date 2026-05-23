"""Tests for the 2026-05-23 LogBus secret-redaction fix (audit Critical #3).

Pre-fix: `publish_tool_event(args={...})` wrote raw args into the
ring buffer AND the daily-rotating JSONL file on disk (7-day retention).
If the LLM ever called `set_setting("anthropic_api_key", "sk-ant-...")`
or any tool taking a token / password as an arg, that secret would
persist unredacted for a week.

Post-fix: `_redact_tool_args` substitutes `'<redacted>'` for any arg
whose key name contains a sensitive substring (api_key, secret,
token, password, etc.), AND for the `value` arg of set_setting /
save_user_fact when the `key` arg names a sensitive setting."""
from __future__ import annotations

import pytest


@pytest.fixture
def clean_bus(tmp_path, monkeypatch):
    from backend import log_bus as _lb
    monkeypatch.setattr(_lb, "_logs_dir", lambda: tmp_path)
    _lb.BUS.clear()
    yield _lb.BUS
    _lb.BUS.clear()


def test_redact_obviously_named_key(clean_bus):
    from backend.log_bus import publish_tool_event
    publish_tool_event(
        name="some_tool",
        args={"path": "/tmp/x", "api_key": "sk-ant-xxxxx"},
    )
    rows = clean_bus.tail()
    assert len(rows) == 1
    args = rows[0]["meta"]["args"]
    assert args["api_key"] == "<redacted>"
    assert args["path"] == "/tmp/x"  # non-sensitive arg untouched


def test_redact_variants_of_secret_naming(clean_bus):
    from backend.log_bus import publish_tool_event
    publish_tool_event(name="t", args={
        "ANTHROPIC_API_KEY": "sk-1",
        "openai_apikey": "sk-2",
        "user_password": "hunter2",
        "bearer_token": "xyz",
        "client_secret": "abc",
        "refresh_token": "def",
        "private_key": "-----BEGIN…",
        "auth_header": "Bearer abc",
        "credential": "raw",
    })
    rows = clean_bus.tail()
    assert all(v == "<redacted>" for v in rows[0]["meta"]["args"].values())


def test_set_setting_value_redacted_when_key_is_sensitive(clean_bus):
    """The killer case: set_setting takes `(key, value)`. The PARAM
    NAME `value` is innocent, but if `key="anthropic_api_key"` then
    the VALUE is the secret. Redact based on the value of the key
    arg, not just its name."""
    from backend.log_bus import publish_tool_event
    publish_tool_event(
        name="set_setting",
        args={"key": "anthropic_api_key", "value": "sk-ant-LIVE-key"},
    )
    rows = clean_bus.tail()
    args = rows[0]["meta"]["args"]
    # The setting NAME ("anthropic_api_key") is not itself secret —
    # surfacing "set_setting was called for anthropic_api_key" is
    # useful for forensics. Only the value gets redacted.
    assert args["key"] == "anthropic_api_key"
    assert args["value"] == "<redacted>"


def test_set_setting_value_NOT_redacted_for_normal_key(clean_bus):
    """A non-sensitive setting (e.g. tts_voice="alloy") must NOT be
    redacted — only when the key NAME selects a sensitive setting."""
    from backend.log_bus import publish_tool_event
    publish_tool_event(
        name="set_setting",
        args={"key": "tts_voice", "value": "alloy"},
    )
    rows = clean_bus.tail()
    args = rows[0]["meta"]["args"]
    assert args["key"] == "tts_voice"
    assert args["value"] == "alloy"


def test_non_sensitive_tool_args_passthrough(clean_bus):
    """Most tools don't have sensitive args. Make sure passthrough
    is byte-equal — otherwise the cost would be a constantly-mutated
    args dict in every log entry."""
    from backend.log_bus import publish_tool_event
    publish_tool_event(name="read_file", args={
        "path": "/etc/issue", "start_line": 1, "end_line": 5,
    })
    rows = clean_bus.tail()
    args = rows[0]["meta"]["args"]
    assert args == {"path": "/etc/issue", "start_line": 1, "end_line": 5}


def test_redaction_does_not_mutate_caller_args(clean_bus):
    """The publisher must not modify the dict the caller owns."""
    from backend.log_bus import publish_tool_event
    original = {"api_key": "sk-xxxxx"}
    publish_tool_event(name="t", args=original)
    assert original == {"api_key": "sk-xxxxx"}


def test_redact_helper_tolerates_non_dict():
    """If a future caller passes args=None or args="oops", the
    helper must return safely without crashing the publish path."""
    from backend.log_bus import _redact_tool_args
    assert _redact_tool_args("t", None) is None  # type: ignore[arg-type]
    assert _redact_tool_args("t", "oops") == "oops"  # type: ignore[arg-type]
    assert _redact_tool_args("t", []) == []  # type: ignore[arg-type]
