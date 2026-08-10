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

# Units this lever may touch. Rebuilt 2026-08-09 from what actually exists on
# the box: the previous list was {"ollama","docker","mcp","tmp_cleanup"} and
# was 0-for-4 — `mcp` and `tmp_cleanup` are not units in either manager,
# `docker` is a SYSTEM unit unreachable as uid 1000 (polkit denies it), and
# `ollama` is a name COLLISION between a healthy system unit and a
# permanently crash-looping user one, so a bare `systemctl restart ollama`
# would bounce the healthy copy and never touch the broken one.
#
# Entries are bare unit names; the manager is taken from the failed-unit
# record, never guessed, so the collision cannot recur.
SERVICE_WHITELIST: set[str] = {
    "hrant", "lightrag", "piper-api", "whisper-api",
    "llama-bge-m3-embeddings", "llama-qwen35-9b-uncensored",
    "hermes-gateway", "hermes-dashboard",
}


def _unit_base(unit: str) -> str:
    """`user:lightrag.service` -> `lightrag`."""
    return unit.split(":", 1)[-1].removesuffix(".service")


def _systemctl(manager: str, *args: str, timeout: float = 30.0):
    """Run systemctl against the RIGHT manager. The old code shelled a bare
    `systemctl`, which always means the system manager — so every user unit
    was unreachable and every ambiguous name resolved to the wrong one."""
    import subprocess as _sp
    argv = ["systemctl"] + (["--user"] if manager == "user" else []) + list(args)
    return _sp.run(argv, capture_output=True, text=True, timeout=timeout)


def _active_enter(manager: str, unit: str) -> str:
    """ActiveEnterTimestamp, or "" — the only honest proof a restart happened.
    Substring-matching `systemctl status` for "active (running)" cannot tell
    "I restarted it" from "something else was already running under that
    name", which is how a polkit-denied restart logged "repaired:ollama"."""
    try:
        r = _systemctl(manager, "show", unit, "-p", "ActiveEnterTimestamp",
                       "--value", timeout=10.0)
        return (r.stdout or "").strip() if r.returncode == 0 else ""
    except Exception:
        return ""


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
        max_attempts = int(params.get("max_attempts", 1))

        # Take the unit from the TICK STATE, not from static rule params
        # (2026-08-09). params-only meant the rule had to name a hardcoded
        # service, so the lever could never repair whatever had actually
        # failed — and the only writable predicate was `True`, i.e. restart
        # that one service on a timer whether or not it was healthy.
        target = str(params.get("service", ""))
        manager = "system"
        if not target:
            state = context.get("state")
            failed = list(getattr(state, "failed_services", []) or [])
            for entry in failed:
                mgr, _, unit = entry.partition(":")
                if _unit_base(entry) in SERVICE_WHITELIST:
                    manager, target = mgr or "system", unit or entry
                    break
            if not target:
                return LeverReport(
                    lever=self.name, params=dict(params), started_at=started,
                    finished_at=utcnow(), status=LeverStatus.SKIPPED,
                    outcome={"failed_services": failed},
                    reason=("no_failed_whitelisted_service" if failed
                            else "no_failed_services"),
                )
        service = _unit_base(target)

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

        unit = target if target.endswith(".service") else f"{service}.service"
        attempts = 0
        final_status_active = False
        final_stdout = ""
        rc = None
        before = _active_enter(manager, unit)
        while attempts < max_attempts:
            attempts += 1
            try:
                r = _systemctl(manager, "restart", unit)
                rc = r.returncode
                final_stdout = ((r.stdout or "") + (r.stderr or ""))[:2000]
            except Exception as exc:
                log.warning("systemctl restart %s raised: %s", unit, exc)
                rc = None
            # A restart the manager REFUSED is not a repair. The old code
            # discarded this returncode entirely, then grepped `systemctl
            # status` for "active (running)" — so a polkit denial against an
            # already-running unit was logged as a successful repair.
            if rc != 0:
                final_status_active = False
                continue
            after = _active_enter(manager, unit)
            is_active = False
            try:
                a = _systemctl(manager, "is-active", unit, timeout=10.0)
                is_active = (a.stdout or "").strip() == "active"
            except Exception:
                is_active = False
            # Proof = it is running AND its active-since stamp MOVED.
            final_status_active = is_active and bool(after) and after != before
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
