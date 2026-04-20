import asyncio

import pytest

from backend.autonomic.startup import (
    build_scheduler,
    start_autonomic_scheduler,
    stop_autonomic_scheduler,
)


@pytest.mark.asyncio
async def test_start_and_stop_autonomic(tmp_path, monkeypatch):
    ks = tmp_path / "ENABLED"
    ks.write_text("true")
    monkeypatch.setenv("AUTONOMIC_ENABLED_PATH", str(ks))
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "0.05")
    monkeypatch.setenv("AUTONOMIC_KNOWLEDGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTONOMIC_ERROR_LOG_PATH", str(tmp_path / "error_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_LEVER_LOG_PATH", str(tmp_path / "lever_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_PENDING_PATH", str(tmp_path / "pending.jsonl"))
    monkeypatch.setenv("AUTONOMIC_TICK_LOG_PATH", str(tmp_path / "tick_log.jsonl"))

    from backend.autonomic.levers import clear_registry
    clear_registry()
    bundle = build_scheduler()
    await start_autonomic_scheduler(bundle)
    assert bundle.scheduler.is_running() is True
    await asyncio.sleep(0.1)
    await stop_autonomic_scheduler(bundle)
    assert bundle.scheduler.is_running() is False
    clear_registry()


@pytest.mark.asyncio
async def test_stop_is_idempotent(tmp_path, monkeypatch):
    ks = tmp_path / "ENABLED"
    ks.write_text("true")
    monkeypatch.setenv("AUTONOMIC_ENABLED_PATH", str(ks))
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "0.05")
    monkeypatch.setenv("AUTONOMIC_KNOWLEDGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTONOMIC_ERROR_LOG_PATH", str(tmp_path / "error_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_LEVER_LOG_PATH", str(tmp_path / "lever_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_PENDING_PATH", str(tmp_path / "pending.jsonl"))
    monkeypatch.setenv("AUTONOMIC_TICK_LOG_PATH", str(tmp_path / "tick_log.jsonl"))

    from backend.autonomic.levers import clear_registry
    clear_registry()
    bundle = build_scheduler()
    await start_autonomic_scheduler(bundle)
    await stop_autonomic_scheduler(bundle)
    await stop_autonomic_scheduler(bundle)
    assert bundle.scheduler.is_running() is False
    clear_registry()


def test_build_scheduler_uses_real_tick_by_default(tmp_path, monkeypatch):
    ks = tmp_path / "ENABLED"
    ks.write_text("true")
    monkeypatch.setenv("AUTONOMIC_ENABLED_PATH", str(ks))
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "0.05")
    monkeypatch.setenv("AUTONOMIC_KNOWLEDGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTONOMIC_ERROR_LOG_PATH", str(tmp_path / "error_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_LEVER_LOG_PATH", str(tmp_path / "lever_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_PENDING_PATH", str(tmp_path / "pending.jsonl"))
    monkeypatch.setenv("AUTONOMIC_TICK_LOG_PATH", str(tmp_path / "tick_log.jsonl"))

    from backend.autonomic.levers import clear_registry
    clear_registry()

    from backend.autonomic.startup import build_scheduler
    bundle = build_scheduler()
    assert bundle is not None
    assert bundle.scheduler._on_tick.__name__ == "_tick"
    clear_registry()


def test_build_scheduler_registers_all_d03_levers(tmp_path, monkeypatch):
    ks = tmp_path / "ENABLED"
    ks.write_text("true")
    monkeypatch.setenv("AUTONOMIC_ENABLED_PATH", str(ks))
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "0.05")
    monkeypatch.setenv("AUTONOMIC_KNOWLEDGE_ROOT", str(tmp_path))
    monkeypatch.setenv("AUTONOMIC_ERROR_LOG_PATH", str(tmp_path / "error_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_LEVER_LOG_PATH", str(tmp_path / "lever_log.jsonl"))
    monkeypatch.setenv("AUTONOMIC_PENDING_PATH", str(tmp_path / "pending.jsonl"))
    monkeypatch.setenv("AUTONOMIC_TICK_LOG_PATH", str(tmp_path / "tick_log.jsonl"))

    from backend.autonomic.levers import LeverRegistry, clear_registry
    clear_registry()

    from backend.autonomic.startup import build_scheduler
    build_scheduler()
    names = LeverRegistry.instance().names()
    assert "FIRE_SERVER_HEALTH" in names
    assert "FIRE_ERROR_TRIAGE" in names
    assert "FIRE_SELF_HEAL" in names
    assert "FIRE_SERVICE_REPAIR" in names
    assert "FIRE_INTEGRITY_HEARTBEAT" in names
    assert "FIRE_GOAL_PROPOSE" in names
    assert "FIRE_MEMORY_CONSOLIDATION" in names
    clear_registry()
