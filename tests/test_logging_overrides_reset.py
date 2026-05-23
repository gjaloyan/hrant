"""Tests for the 2026-05-23 logging-override reset on profile switch
(audit Important #10).

Pre-fix: `_apply_logging_overrides` only RAISED log levels — never
reset them. Switching from a profile with `backend.unified_agent=DEBUG`
to a profile without that override left the module at DEBUG forever
(until process restart). Disk usage in logs/ + sensitive-info
leakage grew unbounded.

Post-fix: on every apply, restore any module touched by the
PREVIOUS overlay (snapshot taken at boot) before applying the new
one. Root logger also resets to its boot-time level."""
from __future__ import annotations

import logging

import pytest


@pytest.fixture
def clean_logging():
    """Capture pre-test state of root + a few module loggers, restore
    after. Tests can mutate freely without leaking levels across
    the run."""
    pre_root = logging.getLogger().level
    pre_levels = {
        name: logging.getLogger(name).level
        for name in (
            "backend.unified_agent",
            "backend.job_supervisor",
            "test.logging_override",
        )
    }
    yield
    logging.getLogger().setLevel(pre_root)
    for name, level in pre_levels.items():
        logging.getLogger(name).setLevel(level)


def test_apply_raises_then_reset_drops_back(clean_logging):
    """Switch profile A (DEBUG on unified_agent) → profile B (empty
    overlay). After B applies, unified_agent must be back at its
    boot level, not stuck at DEBUG."""
    from backend import main
    # Re-snapshot in case earlier tests bumped a module — capture
    # the BOOT level for backend.unified_agent.
    logger = logging.getLogger("backend.unified_agent")
    boot_level = logger.level
    main._LOG_LEVEL_BOOT_SNAPSHOT["backend.unified_agent"] = boot_level

    main._apply_logging_overrides({
        "modules": {"backend.unified_agent": "DEBUG"},
    })
    assert logger.getEffectiveLevel() == logging.DEBUG

    # Now apply an EMPTY overlay — should reset.
    main._apply_logging_overrides({})
    assert logger.level == boot_level


def test_apply_resets_root_when_switching_to_empty(clean_logging):
    from backend import main
    root = logging.getLogger()
    boot_root = root.level
    main._LOG_LEVEL_BOOT_SNAPSHOT["__root__"] = boot_root

    main._apply_logging_overrides({"root": "DEBUG"})
    assert root.level == logging.DEBUG

    main._apply_logging_overrides({})
    assert root.level == boot_root


def test_apply_switching_to_different_modules_resets_old(clean_logging):
    """Profile A → unified_agent=DEBUG. Profile B → job_supervisor=DEBUG.
    After B: unified_agent back to boot, job_supervisor at DEBUG."""
    from backend import main
    unified = logging.getLogger("backend.unified_agent")
    sup = logging.getLogger("backend.job_supervisor")
    main._LOG_LEVEL_BOOT_SNAPSHOT["backend.unified_agent"] = unified.level
    main._LOG_LEVEL_BOOT_SNAPSHOT["backend.job_supervisor"] = sup.level
    unified_boot = unified.level
    sup_boot = sup.level

    main._apply_logging_overrides({
        "modules": {"backend.unified_agent": "DEBUG"},
    })
    assert unified.getEffectiveLevel() == logging.DEBUG

    main._apply_logging_overrides({
        "modules": {"backend.job_supervisor": "DEBUG"},
    })
    assert unified.level == unified_boot
    assert sup.getEffectiveLevel() == logging.DEBUG


def test_apply_none_overlay_resets(clean_logging):
    """Passing None as the overlay is the same as the empty case."""
    from backend import main
    logger = logging.getLogger("backend.unified_agent")
    boot = logger.level
    main._LOG_LEVEL_BOOT_SNAPSHOT["backend.unified_agent"] = boot

    main._apply_logging_overrides({"modules": {"backend.unified_agent": "ERROR"}})
    assert logger.level == logging.ERROR

    main._apply_logging_overrides(None)
    assert logger.level == boot


def test_apply_repeated_same_overlay_idempotent(clean_logging):
    """Applying the same overlay twice should leave the loggers at
    the requested level both times."""
    from backend import main
    logger = logging.getLogger("backend.unified_agent")
    main._LOG_LEVEL_BOOT_SNAPSHOT["backend.unified_agent"] = logger.level

    overlay = {"modules": {"backend.unified_agent": "WARNING"}}
    main._apply_logging_overrides(overlay)
    main._apply_logging_overrides(overlay)
    assert logger.level == logging.WARNING


def test_snapshot_captures_root_at_boot():
    """The snapshot dict must include `__root__` after the boot
    block ran — pin via direct read."""
    from backend import main
    assert "__root__" in main._LOG_LEVEL_BOOT_SNAPSHOT
    assert isinstance(main._LOG_LEVEL_BOOT_SNAPSHOT["__root__"], int)
