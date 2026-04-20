import json
from pathlib import Path

import pytest

from backend.autonomic.lever import Lever
from backend.autonomic.safety import SafetyDecision, SafetyGate
from backend.autonomic.types import Cost, LeverCategory, LeverReport, LeverSafety, LeverStatus, utcnow


class GreenLever(Lever):
    name = "G"
    category = LeverCategory.AUTONOMIC
    safety = LeverSafety.GREEN
    executor = "python"
    estimated_cost = Cost()
    required_context: list[str] = []
    def preconditions(self, state): return True
    def run(self, params, context):
        return LeverReport(lever=self.name, params=params, started_at=utcnow(),
                           finished_at=utcnow(), status=LeverStatus.SUCCESS,
                           outcome={}, reason="")


class YellowLever(GreenLever):
    name = "Y"
    safety = LeverSafety.YELLOW


class RedLever(GreenLever):
    name = "R"
    safety = LeverSafety.RED


@pytest.fixture
def gate(tmp_path: Path) -> SafetyGate:
    return SafetyGate(pending_approvals_path=tmp_path / "pending.jsonl")


def test_green_allowed(gate: SafetyGate):
    decision = gate.evaluate(GreenLever(), params={"x": 1})
    assert decision == SafetyDecision.ALLOW


def test_yellow_queued(gate: SafetyGate):
    decision = gate.evaluate(YellowLever(), params={"x": 2})
    assert decision == SafetyDecision.QUEUE_FOR_APPROVAL


def test_red_blocked(gate: SafetyGate):
    decision = gate.evaluate(RedLever(), params={})
    assert decision == SafetyDecision.BLOCK


def test_yellow_writes_pending_approval(gate: SafetyGate, tmp_path: Path):
    gate.evaluate(YellowLever(), params={"x": 2})
    lines = (tmp_path / "pending.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["lever"] == "Y"
    assert entry["params"] == {"x": 2}
    assert "requested_at" in entry


def test_red_does_not_queue(gate: SafetyGate, tmp_path: Path):
    gate.evaluate(RedLever(), params={})
    assert (tmp_path / "pending.jsonl").read_text() == ""


def test_list_pending(gate: SafetyGate):
    gate.evaluate(YellowLever(), params={"a": 1})
    gate.evaluate(YellowLever(), params={"b": 2})
    pending = gate.list_pending()
    assert len(pending) == 2
    assert pending[0]["params"] == {"a": 1}
    assert pending[1]["params"] == {"b": 2}


import re


def test_queue_writes_unique_id_per_entry(gate: SafetyGate):
    gate.evaluate(YellowLever(), {"a": 1})
    gate.evaluate(YellowLever(), {"b": 2})

    entries = gate.list_pending()
    assert len(entries) == 2
    assert "id" in entries[0] and "id" in entries[1]
    assert entries[0]["id"] != entries[1]["id"]
    assert re.match(r"^[0-9a-f]{12}$", entries[0]["id"])


def test_remove_pending_removes_matching_entry(gate: SafetyGate):
    gate.evaluate(YellowLever(), {"a": 1})
    gate.evaluate(YellowLever(), {"b": 2})
    entries = gate.list_pending()
    target_id = entries[0]["id"]

    assert gate.remove_pending(target_id) is True
    remaining = gate.list_pending()
    assert len(remaining) == 1
    assert remaining[0]["id"] != target_id


def test_remove_pending_unknown_id_returns_false(gate: SafetyGate):
    assert gate.remove_pending("nonexistent") is False
