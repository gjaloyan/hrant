"""FIRE_SERVICE_REPAIR — whitelist-gated systemctl restart + verify."""
from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any

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

log = logging.getLogger(__name__)

_PLATFORM_SUPPORTED = sys.platform.startswith("linux")

SERVICE_WHITELIST: set[str] = {"ollama", "docker", "mcp", "tmp_cleanup"}


class FIRE_SERVICE_REPAIR(Lever):
    name = "FIRE_SERVICE_REPAIR"
    category = LeverCategory.IMMUNE
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost(seconds=5.0)
    required_context: list[str] = []

    def preconditions(self, state: StateSnapshot) -> bool:
        return True

    def run(self, params: dict[str, Any], context: dict[str, Any]) -> LeverReport:
        started = utcnow()
        service = str(params.get("service", ""))
        max_attempts = int(params.get("max_attempts", 1))

        if service not in SERVICE_WHITELIST:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.BLOCKED_BY_SAFETY,
                outcome={"service": service},
                reason=f"service_not_in_whitelist:{service}",
            )

        if not _PLATFORM_SUPPORTED:
            return LeverReport(
                lever=self.name,
                params=dict(params),
                started_at=started,
                finished_at=utcnow(),
                status=LeverStatus.SKIPPED,
                outcome={"service": service, "platform": sys.platform},
                reason="platform_unsupported",
            )

        attempts = 0
        final_status_active = False
        final_stdout = ""
        while attempts < max_attempts:
            attempts += 1
            try:
                subprocess.run(
                    ["systemctl", "restart", service],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except Exception as exc:
                log.warning("systemctl restart %s raised: %s", service, exc)
            try:
                status = subprocess.run(
                    ["systemctl", "status", service],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                final_stdout = status.stdout
                final_status_active = "active (running)" in status.stdout
            except Exception as exc:
                log.warning("systemctl status %s raised: %s", service, exc)
                final_status_active = False
            if final_status_active:
                break

        status_code = LeverStatus.SUCCESS if final_status_active else LeverStatus.ESCALATED
        return LeverReport(
            lever=self.name,
            params=dict(params),
            started_at=started,
            finished_at=utcnow(),
            status=status_code,
            outcome={
                "service": service,
                "attempts": attempts,
                "final_status_active": final_status_active,
                "journal_tail": final_stdout[-500:],
            },
            reason=(
                f"repaired:{service}"
                if final_status_active
                else f"repair_failed:{service}"
            ),
        )
