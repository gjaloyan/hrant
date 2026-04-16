"""FIRE_SERVER_HEALTH — checks disk/memory/CPU thresholds."""
from __future__ import annotations

from typing import Any

import psutil

from ..lever import Lever
from ..types import (
    Cost,
    LeverCategory,
    LeverReport,
    LeverSafety,
    LeverStatus,
    StateSnapshot,
    utcnow,
)

DEFAULT_DISK_MIN_GB = 1.0
DEFAULT_MEMORY_MIN_GB = 0.5
DEFAULT_CPU_MAX_LOAD = 4.0


class FIRE_SERVER_HEALTH(Lever):
    name = "FIRE_SERVER_HEALTH"
    category = LeverCategory.IMMUNE
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=0.05)
    required_context: list[str] = ["state"]

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        state = context.get("state")
        if state is not None:
            disk_free_gb = state.disk_free_gb
            memory_free_gb = state.memory_free_gb
            cpu_load_1m = state.cpu_load_1m
        else:
            disk_free_gb = psutil.disk_usage(".").free / (1024 ** 3)
            memory_free_gb = psutil.virtual_memory().available / (1024 ** 3)
            try:
                cpu_load_1m = float(psutil.getloadavg()[0])
            except (AttributeError, OSError):
                cpu_load_1m = psutil.cpu_percent(interval=None) / 100.0

        disk_min = float(params.get("disk_min_gb", DEFAULT_DISK_MIN_GB))
        mem_min = float(params.get("memory_min_gb", DEFAULT_MEMORY_MIN_GB))
        cpu_max = float(params.get("cpu_max_load", DEFAULT_CPU_MAX_LOAD))

        issues: list[str] = []
        if disk_free_gb < disk_min:
            issues.append(f"disk_low:{disk_free_gb:.2f}gb<{disk_min}gb")
        if memory_free_gb < mem_min:
            issues.append(f"memory_low:{memory_free_gb:.2f}gb<{mem_min}gb")
        if cpu_load_1m > cpu_max:
            issues.append(f"cpu_high:{cpu_load_1m:.2f}>{cpu_max}")

        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=LeverStatus.SUCCESS,
            outcome={
                "disk_free_gb": round(disk_free_gb, 2),
                "memory_free_gb": round(memory_free_gb, 2),
                "cpu_load_1m": round(cpu_load_1m, 2),
                "issues": issues,
            },
            reason=(
                f"server_health_ok:{len(issues)}_issues"
                if issues
                else "server_health_ok"
            ),
        )
