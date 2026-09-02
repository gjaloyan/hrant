import json
from datetime import datetime, timezone
from pathlib import Path

from backend.autonomic.levers.integrity_heartbeat import FIRE_INTEGRITY_HEARTBEAT
from backend.autonomic.types import LeverCategory, LeverSafety, LeverStatus, StateSnapshot


def _snapshot(**overrides) -> StateSnapshot:
    base = dict(
        taken_at=datetime.now(timezone.utc),
        uptime_seconds=10.0,
        disk_free_gb=100.0,
        memory_free_gb=8.0,
        cpu_load_1m=0.5,
        last_run={},
        recent_errors=[],
        pending_approvals=0,
        kb_notes_count=0,
        kb_graph_nodes=0,
    )
    base.update(overrides)
    return StateSnapshot(**base)


def test_integrity_heartbeat_metadata():
    lever = FIRE_INTEGRITY_HEARTBEAT()
    assert lever.name == "FIRE_INTEGRITY_HEARTBEAT"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_integrity_heartbeat_empty_knowledge_is_clean(tmp_path: Path):
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["orphan_files"] == []
    assert report.outcome["dead_entries"] == []
    assert report.outcome["index_count"] == 0
    assert report.outcome["file_count"] == 0


def test_integrity_heartbeat_detects_orphan_file(tmp_path: Path):
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    notes_dir = tmp_path / "profession"
    notes_dir.mkdir()
    (notes_dir / "python.md").write_text("# Python\n", encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.outcome["orphan_files"] == ["profession/python.md"]
    assert report.outcome["dead_entries"] == []
    assert "drift" in report.reason


def test_integrity_heartbeat_detects_dead_entry(tmp_path: Path):
    index = {"profession/ghost.md": {"topic": "ghost"}}
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.outcome["dead_entries"] == ["profession/ghost.md"]


def test_integrity_heartbeat_excludes_system_dirs(tmp_path: Path):
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")
    for excluded in ("_history", "autonomic", "immune", "identity"):
        d = tmp_path / excluded
        d.mkdir()
        (d / "note.md").write_text("x", encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.outcome["orphan_files"] == []
    assert report.outcome["file_count"] == 0


def test_integrity_heartbeat_reports_ok_when_matched(tmp_path: Path):
    index = {"profession/python.md": {"topic": "python"}}
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    (tmp_path / "profession").mkdir()
    (tmp_path / "profession" / "python.md").write_text("# Python\n", encoding="utf-8")
    lever = FIRE_INTEGRITY_HEARTBEAT()
    report = lever.run({"knowledge_root": str(tmp_path)}, {})
    assert report.outcome["orphan_files"] == []
    assert report.outcome["dead_entries"] == []
    assert report.reason == "integrity_ok"


def test_integrity_heartbeat_preconditions_always_true():
    lever = FIRE_INTEGRITY_HEARTBEAT()
    assert lever.preconditions(_snapshot()) is True


from unittest.mock import patch

from backend.autonomic.levers.goal_propose import FIRE_GOAL_PROPOSE


def test_goal_propose_metadata():
    lever = FIRE_GOAL_PROPOSE()
    assert lever.name == "FIRE_GOAL_PROPOSE"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "python"


def test_goal_propose_skips_when_gaps_file_missing(tmp_path: Path):
    lever = FIRE_GOAL_PROPOSE()
    report = lever.run({"gaps_path": str(tmp_path / "missing.json")}, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_gaps"


def test_goal_propose_skips_when_gaps_file_empty(tmp_path: Path):
    gaps_path = tmp_path / "gaps.json"
    gaps_path.write_text("{}", encoding="utf-8")
    lever = FIRE_GOAL_PROPOSE()
    report = lever.run({"gaps_path": str(gaps_path)}, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_gaps"


def test_goal_propose_delegates_to_goals_manager(tmp_path: Path):
    gaps_path = tmp_path / "gaps.json"
    gaps_path.write_text(json.dumps({
        "python_async": {"topic": "python_async", "count": 3, "last": "2026-04-15"},
        "rust_ownership": {"topic": "rust_ownership", "count": 2, "last": "2026-04-16"},
    }), encoding="utf-8")

    captured: dict = {}

    class _FakeGoal:
        def __init__(self, description: str):
            self.description = description
            self.id = "id_" + description[:5]

    def _fake_suggest(gaps, max_goals=3):
        captured["gaps"] = gaps
        captured["max_goals"] = max_goals
        return [_FakeGoal(f"Learn about: {g['topic']}") for g in gaps[:max_goals]]

    lever = FIRE_GOAL_PROPOSE()
    with patch("backend.autonomic.levers.goal_propose.GOALS") as mock_goals:
        mock_goals.suggest_from_gaps.side_effect = _fake_suggest
        report = lever.run({"gaps_path": str(gaps_path), "max_goals": 2}, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["proposed"] == 2
    assert report.outcome["gap_count"] == 2
    assert captured["max_goals"] == 2
    assert {g["topic"] for g in captured["gaps"]} == {"python_async", "rust_ownership"}


def test_goal_propose_preconditions_true():
    lever = FIRE_GOAL_PROPOSE()
    assert lever.preconditions(_snapshot()) is True


from backend.autonomic.levers.memory_consolidation import FIRE_MEMORY_CONSOLIDATION


def _fake_claude_response() -> dict:
    return {
        "session_summary": "User discussed Python async patterns.",
        "user_profile_facts": [
            {"summary": "User prefers Python over Java", "confidence": 0.9, "category": "preference"},
            {"summary": "User lives in Yerevan", "confidence": 0.95, "category": "location"},
        ],
        "durable_facts": [
            {
                "summary": "tomatoes cost 2 USD/kg in Armenia",
                "triples": [["tomato", "costs_in", "armenia"]],
                "tags": ["price", "armenia"],
                "category": "price",
                "confidence": 0.95,
            }
        ],
        "topic_threads": ["python async", "armenian prices"],
    }


def _write_sessions(path: Path, sessions: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"current_id": "x", "sessions": sessions}, ensure_ascii=False), encoding="utf-8")


def _minimal_session(sid: str, turns: int = 1, consolidated: bool = False) -> dict:
    s = {
        "id": sid,
        "started": "2026-04-14 00:00:00",
        "ended": "2026-04-14 01:00:00",
        "title": f"session-{sid}",
        "archived": False,
        "turns": [
            {"ts": "2026-04-14 00:00:00", "user": "hello", "answer": "hi", "intent": "chat"}
            for _ in range(turns)
        ],
    }
    if consolidated:
        s["consolidated"] = True
    return s


def test_memory_consolidation_metadata():
    lever = FIRE_MEMORY_CONSOLIDATION()
    assert lever.name == "FIRE_MEMORY_CONSOLIDATION"
    assert lever.category == LeverCategory.AUTONOMIC
    assert lever.safety == LeverSafety.GREEN
    assert lever.executor == "claude"


def test_memory_consolidation_skips_when_all_consolidated(tmp_path: Path):
    sessions_path = tmp_path / "sessions.json"
    _write_sessions(sessions_path, [_minimal_session("a", consolidated=True)])
    lever = FIRE_MEMORY_CONSOLIDATION()
    assert lever.preconditions(_snapshot()) is True
    report = lever.run({
        "sessions_path": str(sessions_path),
        "user_md_path": str(tmp_path / "user.md"),
        "memory_facts_path": str(tmp_path / "memory_facts.jsonl"),
    }, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_unconsolidated_sessions"


def test_memory_consolidation_writes_to_three_tiers(tmp_path: Path):
    sessions_path = tmp_path / "sessions.json"
    user_md_path = tmp_path / "user.md"
    facts_path = tmp_path / "memory_facts.jsonl"
    _write_sessions(sessions_path, [_minimal_session("a", turns=2)])
    user_md_path.write_text("# User Profile\n\n## О пользователе\n- existing fact\n", encoding="utf-8")

    lever = FIRE_MEMORY_CONSOLIDATION()
    with patch("backend.autonomic.levers.memory_consolidation.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_claude_response()
        report = lever.run({
            "sessions_path": str(sessions_path),
            "user_md_path": str(user_md_path),
            "memory_facts_path": str(facts_path),
            "max_sessions": 5,
        }, {})

    assert report.status == LeverStatus.SUCCESS
    assert report.outcome["sessions_processed"] == 1
    assert report.outcome["profile_added"] == 2
    assert report.outcome["facts_added"] == 1

    user_md_content = user_md_path.read_text(encoding="utf-8")
    assert "User prefers Python over Java" in user_md_content
    assert "User lives in Yerevan" in user_md_content

    facts_lines = facts_path.read_text(encoding="utf-8").splitlines()
    assert len(facts_lines) == 1
    fact = json.loads(facts_lines[0])
    assert fact["summary"] == "tomatoes cost 2 USD/kg in Armenia"

    saved = json.loads(sessions_path.read_text(encoding="utf-8"))
    assert saved["sessions"][0]["consolidated"] is True
    assert saved["sessions"][0]["summary"] == "User discussed Python async patterns."


def test_memory_consolidation_dedups_profile_facts(tmp_path: Path):
    sessions_path = tmp_path / "sessions.json"
    user_md_path = tmp_path / "user.md"
    facts_path = tmp_path / "memory_facts.jsonl"
    _write_sessions(sessions_path, [_minimal_session("a")])
    user_md_path.write_text(
        "# User Profile\n\n## О пользователе\n- user prefers python over java (existing)\n",
        encoding="utf-8",
    )

    lever = FIRE_MEMORY_CONSOLIDATION()
    with patch("backend.autonomic.levers.memory_consolidation.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_claude_response()
        report = lever.run({
            "sessions_path": str(sessions_path),
            "user_md_path": str(user_md_path),
            "memory_facts_path": str(facts_path),
        }, {})

    assert report.outcome["profile_added"] == 1


def test_memory_consolidation_dedups_durable_facts(tmp_path: Path):
    sessions_path = tmp_path / "sessions.json"
    user_md_path = tmp_path / "user.md"
    facts_path = tmp_path / "memory_facts.jsonl"
    _write_sessions(sessions_path, [_minimal_session("a")])
    existing = {
        "summary": "tomatoes cost 2 USD/kg in Armenia",
        "triples": [["tomato", "costs_in", "armenia"]],
        "tags": ["price"],
        "category": "price",
        "confidence": 1.0,
        "ts": "2026-04-10 00:00:00",
        "source_turn": "",
    }
    facts_path.write_text(json.dumps(existing) + "\n", encoding="utf-8")

    lever = FIRE_MEMORY_CONSOLIDATION()
    with patch("backend.autonomic.levers.memory_consolidation.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_claude_response()
        report = lever.run({
            "sessions_path": str(sessions_path),
            "user_md_path": str(user_md_path),
            "memory_facts_path": str(facts_path),
        }, {})

    assert report.outcome["facts_added"] == 0
    facts_lines = facts_path.read_text(encoding="utf-8").splitlines()
    assert len(facts_lines) == 1


def test_memory_consolidation_caps_at_max_sessions(tmp_path: Path):
    sessions_path = tmp_path / "sessions.json"
    user_md_path = tmp_path / "user.md"
    facts_path = tmp_path / "memory_facts.jsonl"
    _write_sessions(sessions_path, [_minimal_session(f"s{i}") for i in range(10)])

    lever = FIRE_MEMORY_CONSOLIDATION()
    with patch("backend.autonomic.levers.memory_consolidation.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_claude_response()
        report = lever.run({
            "sessions_path": str(sessions_path),
            "user_md_path": str(user_md_path),
            "memory_facts_path": str(facts_path),
            "max_sessions": 3,
        }, {})

    assert report.outcome["sessions_processed"] == 3
    saved = json.loads(sessions_path.read_text(encoding="utf-8"))
    consolidated_count = sum(1 for s in saved["sessions"] if s.get("consolidated"))
    assert consolidated_count == 3


def test_memory_consolidation_skips_empty_sessions(tmp_path: Path):
    sessions_path = tmp_path / "sessions.json"
    user_md_path = tmp_path / "user.md"
    facts_path = tmp_path / "memory_facts.jsonl"
    _write_sessions(sessions_path, [_minimal_session("a", turns=0)])

    lever = FIRE_MEMORY_CONSOLIDATION()
    report = lever.run({
        "sessions_path": str(sessions_path),
        "user_md_path": str(user_md_path),
        "memory_facts_path": str(facts_path),
    }, {})
    assert report.status == LeverStatus.SKIPPED
    assert report.reason == "no_unconsolidated_sessions"


def test_memory_consolidation_follow_ups_includes_topic_threads(tmp_path: Path):
    sessions_path = tmp_path / "sessions.json"
    user_md_path = tmp_path / "user.md"
    facts_path = tmp_path / "memory_facts.jsonl"
    _write_sessions(sessions_path, [_minimal_session("a")])

    lever = FIRE_MEMORY_CONSOLIDATION()
    with patch("backend.autonomic.levers.memory_consolidation.router") as mock_router:
        mock_router.return_value.call_json.return_value = _fake_claude_response()
        report = lever.run({
            "sessions_path": str(sessions_path),
            "user_md_path": str(user_md_path),
            "memory_facts_path": str(facts_path),
        }, {})

    assert "python async" in report.follow_ups
    assert "armenian prices" in report.follow_ups


def test_consolidation_marks_do_not_discard_concurrent_turns(tmp_path):
    """The lever must not write back a file it read minutes ago.

    It loads sessions.json, spends one LLM call per session, then saves.
    The session store appends turns to that same file throughout, so the
    old whole-blob write meant whichever writer finished last discarded
    the other's work.
    """
    import json
    from backend.autonomic.levers.memory_consolidation import (
        FIRE_MEMORY_CONSOLIDATION,
    )

    path = tmp_path / "sessions.json"
    path.write_text(json.dumps({"sessions": [
        {"id": "a", "turns": [{"user": "one"}]},
    ]}), encoding="utf-8")

    lever = FIRE_MEMORY_CONSOLIDATION()
    stale = lever._load_sessions(path)          # what the run started from
    assert len(stale["sessions"]) == 1

    # While the LLM calls run, the store adds a turn and a whole session.
    path.write_text(json.dumps({"sessions": [
        {"id": "a", "turns": [{"user": "one"}, {"user": "two"}]},
        {"id": "b", "turns": [{"user": "three"}]},
    ]}), encoding="utf-8")

    lever._mark_consolidated(path, {"a": "a summary"})

    after = json.loads(path.read_text(encoding="utf-8"))["sessions"]
    by_id = {s["id"]: s for s in after}
    assert set(by_id) == {"a", "b"}, "the session added meanwhile was dropped"
    assert len(by_id["a"]["turns"]) == 2, "the turn added meanwhile was dropped"
    assert by_id["a"]["consolidated"] is True
    assert by_id["a"]["summary"] == "a summary"
    assert not by_id["b"].get("consolidated")
