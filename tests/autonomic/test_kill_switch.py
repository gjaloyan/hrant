from pathlib import Path

import pytest

from backend.autonomic.kill_switch import KillSwitch


def test_missing_file_means_disabled(tmp_path: Path):
    ks = KillSwitch(tmp_path / "ENABLED")
    assert ks.is_enabled() is False


def test_true_content_enabled(tmp_path: Path):
    p = tmp_path / "ENABLED"
    p.write_text("true")
    ks = KillSwitch(p)
    assert ks.is_enabled() is True


def test_false_content_disabled(tmp_path: Path):
    p = tmp_path / "ENABLED"
    p.write_text("false")
    ks = KillSwitch(p)
    assert ks.is_enabled() is False


def test_whitespace_and_case_insensitive(tmp_path: Path):
    p = tmp_path / "ENABLED"
    p.write_text("  TRUE\n")
    ks = KillSwitch(p)
    assert ks.is_enabled() is True


def test_unknown_value_is_disabled(tmp_path: Path):
    p = tmp_path / "ENABLED"
    p.write_text("maybe")
    ks = KillSwitch(p)
    assert ks.is_enabled() is False


def test_enable_disable(tmp_path: Path):
    ks = KillSwitch(tmp_path / "ENABLED")
    ks.enable()
    assert ks.is_enabled() is True
    ks.disable()
    assert ks.is_enabled() is False
