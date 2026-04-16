import asyncio
from pathlib import Path

import pytest

from backend.autonomic.kill_switch import KillSwitch
from backend.autonomic.scheduler import AutonomicScheduler


@pytest.mark.asyncio
async def test_scheduler_fires_tick_handler(tmp_path: Path):
    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("true")
    counter = {"n": 0}

    def handler():
        counter["n"] += 1

    sched = AutonomicScheduler(
        kill_switch=KillSwitch(ks_path),
        on_tick=handler,
        tick_interval_seconds=0.05,
    )
    await sched.start()
    await asyncio.sleep(0.2)
    await sched.stop()
    assert counter["n"] >= 3


@pytest.mark.asyncio
async def test_scheduler_respects_kill_switch(tmp_path: Path):
    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("false")
    counter = {"n": 0}

    def handler():
        counter["n"] += 1

    sched = AutonomicScheduler(
        kill_switch=KillSwitch(ks_path),
        on_tick=handler,
        tick_interval_seconds=0.05,
    )
    await sched.start()
    await asyncio.sleep(0.2)
    await sched.stop()
    assert counter["n"] == 0


@pytest.mark.asyncio
async def test_scheduler_handler_exception_does_not_kill_loop(tmp_path: Path):
    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("true")
    calls = {"n": 0}

    def handler():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")

    sched = AutonomicScheduler(
        kill_switch=KillSwitch(ks_path),
        on_tick=handler,
        tick_interval_seconds=0.05,
    )
    await sched.start()
    await asyncio.sleep(0.2)
    await sched.stop()
    assert calls["n"] >= 2


@pytest.mark.asyncio
async def test_scheduler_stop_is_graceful(tmp_path: Path):
    ks_path = tmp_path / "ENABLED"
    ks_path.write_text("true")
    sched = AutonomicScheduler(
        kill_switch=KillSwitch(ks_path),
        on_tick=lambda: None,
        tick_interval_seconds=0.05,
    )
    await sched.start()
    await asyncio.sleep(0.1)
    await sched.stop()
    assert sched.is_running() is False
