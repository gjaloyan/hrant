"""Тесты реестра версий модели."""
from pathlib import Path

from backend.model_versions import ModelVersionRegistry


def test_initial_seed(tmp_kb):
    reg = ModelVersionRegistry(path=tmp_kb.base / "model_versions.json")
    state = reg.list()
    assert state.current == "v0"
    assert len(state.versions) == 1
    assert state.versions[0].tag == "v0"


def test_register_and_switch(tmp_kb):
    reg = ModelVersionRegistry(path=tmp_kb.base / "model_versions.json")
    reg.register("v1", "my-agent-v1", examples_count=120)
    reg.register("v2", "my-agent-v2", examples_count=240)
    msg = reg.switch("v2")
    assert "v2" in msg
    assert reg.current().tag == "v2"


def test_rollback(tmp_kb):
    reg = ModelVersionRegistry(path=tmp_kb.base / "model_versions.json")
    reg.register("v1", "my-agent-v1", examples_count=120)
    reg.switch("v1")
    msg = reg.rollback()
    assert "v0" in msg
    assert reg.current().tag == "v0"


def test_next_tag(tmp_kb):
    reg = ModelVersionRegistry(path=tmp_kb.base / "model_versions.json")
    assert reg.next_tag() == "v1"
    reg.register("v1", "x", examples_count=0)
    assert reg.next_tag() == "v2"
