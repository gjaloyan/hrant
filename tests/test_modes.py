"""Тесты MODE-пресетов: каждый режим применяется правильно."""
from pathlib import Path

import pytest
import yaml

from backend.config import MODE_PRESETS, Config


def _write_config(tmp_path: Path, mode: str, extra: dict | None = None) -> Path:
    data: dict = {"mode": mode}
    if extra:
        data.update(extra)
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


# ---------- sanity для каждого режима ----------
@pytest.mark.parametrize("mode", list(MODE_PRESETS.keys()))
def test_each_mode_loads(tmp_path, mode):
    cfg_path = _write_config(tmp_path, mode)
    cfg = Config(path=cfg_path)
    assert cfg.mode == mode
    assert cfg.model_a.get("model")  # Claude всегда есть
    assert cfg.verification.get("always_use_model_a") is True


# ---------- local_full ----------
def test_local_full(tmp_path):
    cfg = Config(path=_write_config(tmp_path, "local_full"))
    assert cfg.mode == "local_full"
    assert cfg.model_b is not None
    assert "qwen" in cfg.model_b["model"]
    assert cfg.training_location == "local"
    assert cfg.finetune_enabled is True
    assert cfg.router["fallback_to_local"] is True
    assert cfg.router["auto_shift_after_finetune"] is True


# ---------- cloud_finetune ----------
def test_cloud_finetune(tmp_path):
    cfg = Config(path=_write_config(tmp_path, "cloud_finetune"))
    assert cfg.training_location == "cloud"
    assert cfg.finetune_enabled is True
    assert cfg.model_b is not None  # Qwen всё равно запускается локально для inference
    assert cfg.finetune.get("cloud_recipe") is not None


# ---------- local_cpu ----------
def test_local_cpu(tmp_path):
    cfg = Config(path=_write_config(tmp_path, "local_cpu"))
    assert cfg.training_location == "local_cpu"
    assert cfg.model_b is not None
    assert "1.5b" in cfg.model_b["model"].lower()
    # shift отключён для слабой CPU-модели
    assert cfg.router["auto_shift_after_finetune"] is False
    assert cfg.finetune.get("cpu_warning")


# ---------- claude_only ----------
def test_claude_only(tmp_path):
    cfg = Config(path=_write_config(tmp_path, "claude_only"))
    assert cfg.mode == "claude_only"
    assert cfg.model_b is None
    assert cfg.training_location == "disabled"
    assert cfg.finetune_enabled is False
    assert cfg.router["fallback_to_local"] is False
    assert cfg.router["auto_shift_after_finetune"] is False


# ---------- user override ----------
def test_user_override_model_a(tmp_path):
    cfg = Config(
        path=_write_config(
            tmp_path,
            "local_full",
            {"model_a": {"model": "claude-opus-4-6"}},
        )
    )
    assert cfg.model_a["model"] == "claude-opus-4-6"
    # остальные дефолты preset должны остаться
    assert cfg.model_a["api_key_env"] == "ANTHROPIC_API_KEY"


def test_user_override_router_budget(tmp_path):
    cfg = Config(
        path=_write_config(
            tmp_path,
            "local_full",
            {"router": {"daily_api_budget_usd": 50.0}},
        )
    )
    assert cfg.router["daily_api_budget_usd"] == 50.0
    # shift_schedule из preset сохранился
    assert "v0" in cfg.router["shift_schedule"]


def test_unknown_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="Неизвестный mode"):
        Config(path=_write_config(tmp_path, "nonsense"))


# ---------- router + claude_only ----------
def test_router_in_claude_only_mode(tmp_path, monkeypatch):
    """В claude_only роутер должен рассматривать Ollama как постоянно недоступную."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    cfg_path = _write_config(tmp_path, "claude_only")

    # переинициализируем глобальный CONFIG на временный yaml
    from backend import config as cfg_mod
    from backend import llm as llm_mod

    new_cfg = Config(path=cfg_path)
    monkeypatch.setattr(cfg_mod, "CONFIG", new_cfg)
    monkeypatch.setattr(llm_mod, "CONFIG", new_cfg)

    # Создадим router — должен работать без Ollama
    r = llm_mod.DualModelRouter(state_path=tmp_path / "router_state.json")
    assert r.cfg_b is None
    assert r._ollama_available() is False

    stats = r.stats()
    assert stats["mode"] == "claude_only"
    assert stats["model_b_id"] is None
    assert stats["model_b_available"] is False
