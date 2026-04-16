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

    sched = build_scheduler()
    await start_autonomic_scheduler(sched)
    assert sched.is_running() is True
    await asyncio.sleep(0.1)
    await stop_autonomic_scheduler(sched)
    assert sched.is_running() is False


@pytest.mark.asyncio
async def test_stop_is_idempotent(tmp_path, monkeypatch):
    ks = tmp_path / "ENABLED"
    ks.write_text("true")
    monkeypatch.setenv("AUTONOMIC_ENABLED_PATH", str(ks))
    monkeypatch.setenv("AUTONOMIC_TICK_SECONDS", "0.05")

    sched = build_scheduler()
    await start_autonomic_scheduler(sched)
    await stop_autonomic_scheduler(sched)
    await stop_autonomic_scheduler(sched)
    assert sched.is_running() is False
