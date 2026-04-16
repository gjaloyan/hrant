"""Тесты гейтов training_location в FineTunePipeline."""
from unittest.mock import patch

import pytest
import yaml

from backend.config import Config


def _cfg(tmp_path, mode):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"mode": mode}), encoding="utf-8")
    return Config(path=p)


@pytest.fixture
def swap_config(monkeypatch):
    """Утилита: переинициализирует глобальный CONFIG на указанный режим."""
    def _swap(new_cfg):
        from backend import config as cfg_mod
        from backend import finetune_pipeline as pipe_mod
        from backend import llm as llm_mod
        monkeypatch.setattr(cfg_mod, "CONFIG", new_cfg)
        monkeypatch.setattr(pipe_mod, "CONFIG", new_cfg)
        monkeypatch.setattr(llm_mod, "CONFIG", new_cfg)
    return _swap


# ---------- claude_only: run_full_pipeline должен падать понятно ----------
def test_claude_only_blocks_run_full(tmp_kb, tmp_path, swap_config):
    swap_config(_cfg(tmp_path, "claude_only"))
    from backend.finetune_pipeline import FineTunePipeline

    pipe = FineTunePipeline()
    with pytest.raises(RuntimeError, match="fine-tune выключен"):
        pipe.run_full_pipeline()


# ---------- cloud_finetune: run_full_pipeline redirect на export-cloud ----------
def test_cloud_mode_blocks_local_training(tmp_kb, tmp_path, swap_config):
    swap_config(_cfg(tmp_path, "cloud_finetune"))
    from backend.finetune_pipeline import FineTunePipeline

    pipe = FineTunePipeline()
    with pytest.raises(RuntimeError, match="training_location=cloud"):
        pipe.run_full_pipeline()


def _seed_pairs_and_bypass_dedup(n: int, monkeypatch) -> None:
    """Кладёт n примеров и отключает дедуп curator — для тестов пайплайна."""
    from backend.finetune import store as finetune_store
    from backend.finetune_curator import FinetuneDataCurator

    for i in range(n):
        finetune_store().add(
            question=f"как подключить устройство №{i} в проекте?",
            answer=f"ответ {i}: " + ("деталь " * 30),
            source_notes=[f"topic_{i}"],
            confidence=92,
            project=None,
        )
    # Отключаем дедуп — иначе похожие шаблоны теряются
    monkeypatch.setattr(
        FinetuneDataCurator, "_is_duplicate", lambda self, c, accepted: False
    )


# ---------- local_full: run_full_pipeline доходит до train_with_unsloth ----------
def test_local_full_invokes_training(tmp_kb, tmp_path, swap_config, monkeypatch):
    swap_config(_cfg(tmp_path, "local_full"))
    from backend.finetune_pipeline import FineTunePipeline

    _seed_pairs_and_bypass_dedup(55, monkeypatch)

    pipe = FineTunePipeline()
    with patch.object(pipe, "train_with_unsloth") as mock_train, \
         patch.object(pipe, "register_with_ollama", return_value="qwen-agent-v1"):
        mock_train.return_value = tmp_path / "fake.gguf"
        name = pipe.run_full_pipeline()

    assert name == "qwen-agent-v1"
    mock_train.assert_called_once()


# ---------- cloud export: пакет реально создаётся ----------
def test_export_for_cloud_creates_package(tmp_kb, tmp_path, swap_config, monkeypatch):
    swap_config(_cfg(tmp_path, "cloud_finetune"))
    from backend.finetune_pipeline import FineTunePipeline

    _seed_pairs_and_bypass_dedup(55, monkeypatch)

    pipe = FineTunePipeline()
    pkg = pipe.export_for_cloud()

    assert pkg.exists() and pkg.is_dir()
    files = sorted(p.name for p in pkg.iterdir())
    assert "train.jsonl" in files
    assert "val.jsonl" in files
    assert "train_script.py" in files
    assert "config.json" in files
    assert "README_CLOUD.md" in files

    # README должен содержать инструкции
    readme = (pkg / "README_CLOUD.md").read_text(encoding="utf-8")
    assert "unsloth" in readme.lower()
    assert "import-gguf" in readme


# ---------- import_gguf: вызывает ollama create ----------
def test_import_gguf_calls_ollama(tmp_kb, tmp_path, swap_config):
    swap_config(_cfg(tmp_path, "cloud_finetune"))
    from backend.finetune_pipeline import FineTunePipeline

    fake_gguf = tmp_path / "unsloth.Q4_K_M.gguf"
    fake_gguf.write_bytes(b"fake")

    pipe = FineTunePipeline()
    with patch("backend.finetune_pipeline.subprocess.run") as mock_run:
        name = pipe.import_gguf(str(fake_gguf), tag="v1")

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "ollama"
    assert args[1] == "create"
    assert name.endswith("-v1")


def test_import_gguf_missing_file_raises(tmp_kb, tmp_path, swap_config):
    swap_config(_cfg(tmp_path, "cloud_finetune"))
    from backend.finetune_pipeline import FineTunePipeline

    pipe = FineTunePipeline()
    with pytest.raises(FileNotFoundError):
        pipe.import_gguf("/nonexistent/model.gguf", tag="v1")
