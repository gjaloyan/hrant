"""Core types for the autonomic subsystem."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class LeverSafety(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


class LeverCategory(str, Enum):
    AUTONOMIC = "autonomic"
    TELEMETRY = "telemetry"
    IMMUNE = "immune"
    BODY = "body"
    META = "meta"


class LeverStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"
    ESCALATED = "escalated"
    BLOCKED_BY_SAFETY = "blocked_by_safety"
    NOT_EXECUTED = "not_executed"


class TickDecisionSource(str, Enum):
    L0_REFLEX = "L0_reflex"
    L0_IMMUNE = "L0_immune"
    L1_ROUTER = "L1_router"
    L2_DIAGNOSER = "L2_diagnoser"
    L3_ESCALATION = "L3_escalation"


@dataclass
class Cost:
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    usd: float = 0.0

    def __add__(self, other: "Cost") -> "Cost":
        return Cost(
            tokens_in=self.tokens_in + other.tokens_in,
            tokens_out=self.tokens_out + other.tokens_out,
            seconds=self.seconds + other.seconds,
            usd=self.usd + other.usd,
        )


@dataclass
class StateSnapshot:
    taken_at: datetime
    uptime_seconds: float
    disk_free_gb: float
    memory_free_gb: float
    cpu_load_1m: float
    last_run: dict[str, datetime]
    recent_errors: list[dict[str, Any]]
    pending_approvals: int
    kb_notes_count: int
    kb_graph_nodes: int


@dataclass
class LeverReport:
    lever: str
    params: dict[str, Any]
    started_at: datetime
    finished_at: datetime
    status: LeverStatus
    outcome: dict[str, Any]
    cost: Cost = field(default_factory=Cost)
    reason: str = ""
    follow_ups: list[str] = field(default_factory=list)

    def to_jsonl(self) -> str:
        payload = {
            "lever": self.lever,
            "params": self.params,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "status": self.status.value,
            "outcome": self.outcome,
            "cost": asdict(self.cost),
            "reason": self.reason,
            "follow_ups": self.follow_ups,
        }
        return json.dumps(payload, ensure_ascii=False) + "\n"

    @classmethod
    def from_jsonl(cls, line: str) -> "LeverReport":
        data = json.loads(line)
        return cls(
            lever=data["lever"],
            params=data["params"],
            started_at=datetime.fromisoformat(data["started_at"]),
            finished_at=datetime.fromisoformat(data["finished_at"]),
            status=LeverStatus(data["status"]),
            outcome=data["outcome"],
            cost=Cost(**data["cost"]),
            reason=data.get("reason", ""),
            follow_ups=data.get("follow_ups", []),
        )


@dataclass
class TickDecision:
    source: TickDecisionSource
    lever: str | None
    params: dict[str, Any]
    reason: str
    rule_name: str | None = None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
