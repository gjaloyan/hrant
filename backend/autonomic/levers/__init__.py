"""Autonomic lever registry (singleton)."""
from __future__ import annotations

from threading import Lock
from typing import Iterable

from ..lever import Lever
from ..types import LeverCategory


class LeverRegistry:
    _instance: "LeverRegistry | None" = None
    _lock: Lock = Lock()

    def __init__(self) -> None:
        self._levers: dict[str, Lever] = {}

    @classmethod
    def instance(cls) -> "LeverRegistry":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def register(self, lever_cls: type[Lever]) -> None:
        instance = lever_cls()
        if instance.name in self._levers:
            raise ValueError(f"Lever {instance.name!r} already registered")
        self._levers[instance.name] = instance

    def get(self, name: str) -> Lever | None:
        return self._levers.get(name)

    def all(self) -> Iterable[Lever]:
        return list(self._levers.values())

    def names(self) -> list[str]:
        return list(self._levers.keys())

    def by_category(self, category: LeverCategory) -> list[Lever]:
        return [l for l in self._levers.values() if l.category == category]

    def clear(self) -> None:
        self._levers.clear()


def register_lever(lever_cls: type[Lever]) -> None:
    LeverRegistry.instance().register(lever_cls)


def get_lever(name: str) -> Lever | None:
    return LeverRegistry.instance().get(name)


def list_levers() -> list[str]:
    return LeverRegistry.instance().names()


def clear_registry() -> None:
    """Testing helper — wipe all registrations."""
    LeverRegistry.instance().clear()


def register_default_immune_levers() -> None:
    from .error_triage import FIRE_ERROR_TRIAGE
    from .self_heal import FIRE_SELF_HEAL
    from .server_health import FIRE_SERVER_HEALTH
    from .service_repair import FIRE_SERVICE_REPAIR
    register_lever(FIRE_SERVER_HEALTH)
    register_lever(FIRE_ERROR_TRIAGE)
    register_lever(FIRE_SELF_HEAL)
    register_lever(FIRE_SERVICE_REPAIR)


def register_default_autonomic_levers() -> None:
    from .capability_scan import FIRE_CAPABILITY_SCAN
    from .goal_propose import FIRE_GOAL_PROPOSE
    from .graph_maintenance import FIRE_GRAPH_MAINTENANCE
    from .integrity_heartbeat import FIRE_INTEGRITY_HEARTBEAT
    from .memory_consolidation import FIRE_MEMORY_CONSOLIDATION
    from .note_curation import FIRE_NOTE_CURATION
    from .proactive_learn import FIRE_PROACTIVE_LEARN
    from .self_study import FIRE_SELF_STUDY
    register_lever(FIRE_INTEGRITY_HEARTBEAT)
    register_lever(FIRE_GOAL_PROPOSE)
    register_lever(FIRE_MEMORY_CONSOLIDATION)
    register_lever(FIRE_CAPABILITY_SCAN)
    register_lever(FIRE_SELF_STUDY)
    register_lever(FIRE_GRAPH_MAINTENANCE)
    register_lever(FIRE_PROACTIVE_LEARN)
    register_lever(FIRE_NOTE_CURATION)
