import pytest

from backend.autonomic.lever import Lever
from backend.autonomic.levers import (
    LeverRegistry,
    clear_registry,
    get_lever,
    list_levers,
    register_lever,
)
from backend.autonomic.types import Cost, LeverCategory, LeverReport, LeverSafety, LeverStatus, utcnow


class LeverA(Lever):
    name = "A"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost()
    required_context: list[str] = []
    def preconditions(self, state): return True
    def run(self, params, context):
        return LeverReport(
            lever=self.name, params=params,
            started_at=utcnow(), finished_at=utcnow(),
            status=LeverStatus.SUCCESS, outcome={}, reason="",
        )


class LeverB(LeverA):
    name = "B"
    category = LeverCategory.IMMUNE


@pytest.fixture(autouse=True)
def reset():
    clear_registry()
    yield
    clear_registry()


def test_register_and_get():
    register_lever(LeverA)
    lever = get_lever("A")
    assert isinstance(lever, LeverA)


def test_duplicate_registration_raises():
    register_lever(LeverA)
    with pytest.raises(ValueError, match="already registered"):
        register_lever(LeverA)


def test_get_missing_returns_none():
    assert get_lever("MISSING") is None


def test_list_levers_returns_all_names():
    register_lever(LeverA)
    register_lever(LeverB)
    assert sorted(list_levers()) == ["A", "B"]


def test_list_by_category():
    register_lever(LeverA)
    register_lever(LeverB)
    reg = LeverRegistry.instance()
    autonomic = reg.by_category(LeverCategory.AUTONOMIC)
    immune = reg.by_category(LeverCategory.IMMUNE)
    assert [l.name for l in autonomic] == ["A"]
    assert [l.name for l in immune] == ["B"]


def test_registry_is_singleton():
    reg1 = LeverRegistry.instance()
    reg2 = LeverRegistry.instance()
    assert reg1 is reg2


def test_immune_levers_are_auto_registered():
    from backend.autonomic.levers import register_default_immune_levers
    clear_registry()
    register_default_immune_levers()
    reg = LeverRegistry.instance()
    names = reg.names()
    assert "FIRE_SERVER_HEALTH" in names
    assert "FIRE_ERROR_TRIAGE" in names
    assert "FIRE_SELF_HEAL" in names
    assert "FIRE_SERVICE_REPAIR" in names
    clear_registry()


def test_autonomic_levers_are_auto_registered():
    from backend.autonomic.levers import register_default_autonomic_levers
    clear_registry()
    register_default_autonomic_levers()
    reg = LeverRegistry.instance()
    names = reg.names()
    assert "FIRE_INTEGRITY_HEARTBEAT" in names
    assert "FIRE_GOAL_PROPOSE" in names
    assert "FIRE_MEMORY_CONSOLIDATION" in names
    clear_registry()


def test_autonomic_levers_include_d04_cohort():
    from backend.autonomic.levers import register_default_autonomic_levers
    clear_registry()
    register_default_autonomic_levers()
    reg = LeverRegistry.instance()
    names = reg.names()
    assert "FIRE_INTEGRITY_HEARTBEAT" in names
    assert "FIRE_GOAL_PROPOSE" in names
    assert "FIRE_MEMORY_CONSOLIDATION" in names
    assert "FIRE_CAPABILITY_SCAN" in names
    assert "FIRE_SELF_STUDY" in names
    clear_registry()


def test_autonomic_levers_include_d05_cohort():
    from backend.autonomic.levers import register_default_autonomic_levers
    clear_registry()
    register_default_autonomic_levers()
    reg = LeverRegistry.instance()
    names = reg.names()
    assert "FIRE_INTEGRITY_HEARTBEAT" in names
    assert "FIRE_GOAL_PROPOSE" in names
    assert "FIRE_MEMORY_CONSOLIDATION" in names
    assert "FIRE_CAPABILITY_SCAN" in names
    assert "FIRE_SELF_STUDY" in names
    assert "FIRE_NOTE_CURATION" in names
    assert "FIRE_GRAPH_MAINTENANCE" in names
    assert "FIRE_PROACTIVE_LEARN" in names
    clear_registry()
