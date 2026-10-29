"""Rules that keep their history, and skills that are looked after.

Prod, 2026-10-29: the "stay silent when there is no signal" rule was
written on 10-24 and broken again on 10-25 and 10-28 without anything
noticing, every rule carried the date of the last merge and the evidence
"merged", and 23 of 25 switched-on agent skills had not been loaded once in
347 turns.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

from backend import learned_rules as lr


def _seed(p, rules):
    p.write_text(json.dumps({"rules": rules, "history": []}), encoding="utf-8")


# ── broken rules ──────────────────────────────────────────────────────

def test_a_rule_broken_on_two_later_nights_is_escalated_once(tmp_path):
    p = tmp_path / "rules.json"
    text = "When a monitoring task finds no trigger, send no status message."
    _seed(p, [{"text": text, "added": "2026-10-24", "evidence": ["2026-10-24: x"]}])

    assert lr.record_violations([(text, "sent a status")], night="2026-10-25", path=p) == []
    assert lr.record_violations([(text, "again")], night="2026-10-25", path=p) == [], \
        "a night counts once"
    newly = lr.record_violations([(text, "a third time")], night="2026-10-28", path=p)
    assert [r["text"] for r in newly] == [text]
    assert lr.rules(p)[0]["escalated"] == "2026-10-28"
    assert lr.record_violations([(text, "still")], night="2026-10-29", path=p) == [], \
        "a rule is escalated once"
    assert len(lr.rules(p)[0]["violations"]) == 3


def test_the_night_a_rule_was_written_does_not_count_against_it(tmp_path):
    p = tmp_path / "rules.json"
    text = "Interpret short replies using the preceding topic."
    _seed(p, [{"text": text, "added": "2026-10-24"}])
    lr.record_violations([(text, "same incident, seen in the overlap")],
                         night="2026-10-24", path=p)
    assert lr.rules(p)[0]["violations"] == []


def test_counting_a_violation_is_not_a_change_of_the_rule_set(tmp_path):
    p = tmp_path / "rules.json"
    text = "Report the first run time exactly as the tool returned it."
    _seed(p, [{"text": text, "added": "2026-10-20"}])
    lr.record_violations([(text, "wrong time")], night="2026-10-21", path=p)
    assert json.loads(p.read_text(encoding="utf-8"))["history"] == []


def test_rules_from_before_this_change_load_in_the_new_shape(tmp_path):
    p = tmp_path / "rules.json"
    _seed(p, [{"text": "Merged long ago and nobody knows from what.", "added": "2026-10-29",
               "evidence": "merged"},
              {"text": "Written with a quote for evidence.", "added": "2026-10-14",
               "evidence": "[2026-10-13 10:00] the user said so"}])
    a, b = lr.rules(p)
    assert a["evidence"] == [] and a["violations"] == []
    assert b["evidence"] == ["[2026-10-13 10:00] the user said so"]


# ── merging keeps provenance ──────────────────────────────────────────

def test_a_merge_keeps_the_earliest_date_the_evidence_and_the_violations(tmp_path, monkeypatch):
    p = tmp_path / "rules.json"
    monkeypatch.setattr(lr, "MAX_RULES", 2)
    _seed(p, [
        {"text": "Modify an existing schedule directly instead of asking again.",
         "added": "2026-10-14", "evidence": ["2026-10-14: asked the time twice"],
         "violations": [{"night": "2026-10-25", "what": "asked again"}]},
        {"text": "Execute a fully specified scheduled task without clarifying questions.",
         "added": "2026-10-25", "evidence": ["2026-10-25: asked the topic"]},
    ])
    out = lr.add(["Generate images with generate_image instead of describing them."],
                 evidence="[10-28] described a room", path=p,
                 merge=lambda cur, new, n: [
                     {"text": "Change existing schedules and run specified scheduled tasks "
                              "without asking again.", "from": ["C1", "C2"]},
                     {"text": "Generate images with generate_image instead of describing them.",
                      "from": ["N1"]}])
    assert out["merged"] is True
    merged, fresh = lr.rules(p)
    assert merged["added"] == "2026-10-14"
    assert len(merged["evidence"]) == 2
    assert [v["night"] for v in merged["violations"]] == ["2026-10-25"]
    assert fresh["evidence"] and "described a room" in fresh["evidence"][0]


def test_a_merge_that_names_no_sources_falls_back_to_the_wording(tmp_path, monkeypatch):
    p = tmp_path / "rules.json"
    monkeypatch.setattr(lr, "MAX_RULES", 1)
    _seed(p, [{"text": "Verify the returned run time before reporting a schedule.",
               "added": "2026-10-25", "evidence": ["2026-10-25: 25:30"]}])
    lr.add(["Keep replies to small talk under two sentences."], path=p,
           merge=lambda cur, new, n: ["Verify the returned run time before reporting a schedule!"])
    (rule,) = lr.rules(p)
    assert rule["added"] == "2026-10-25" and rule["evidence"] == ["2026-10-25: 25:30"]


def test_without_a_merge_the_least_seen_rule_gives_way_not_the_oldest(tmp_path, monkeypatch):
    p = tmp_path / "rules.json"
    monkeypatch.setattr(lr, "MAX_RULES", 2)
    _seed(p, [
        {"text": "Never offer to accept passwords or login codes from the user.",
         "added": "2026-10-14", "evidence": ["2026-10-14: offered to log in"],
         "violations": [{"night": "2026-10-20", "what": "x"}]},
        {"text": "Prefer the manufacturer's manual over forum posts for specs.",
         "added": "2026-10-26"},
    ])
    lr.add(["Run long downloads as background jobs instead of waiting in the turn."], path=p)
    texts = [r["text"] for r in lr.rules(p)]
    assert any("passwords" in t for t in texts), "the confirmed old rule stays"
    assert not any("manufacturer" in t for t in texts)
    assert "background jobs" in texts[-1]


# ── the nightly review ────────────────────────────────────────────────

@pytest.fixture
def night(tmp_path, monkeypatch):
    from backend import threads
    from backend.threads import ThreadStore
    from backend.consolidation import config as ccfg
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path / "data"))
    store = ThreadStore(root=tmp_path / "threads")
    monkeypatch.setattr(threads, "THREADS", store)
    monkeypatch.setattr(lr, "_path", lambda: tmp_path / "rules.json")
    monkeypatch.setattr(ccfg, "user_md_path", lambda: tmp_path / "missing.md")
    monkeypatch.setattr(ccfg, "profile_path_for_speaker", lambda s: tmp_path / "missing2.md")
    store.append("tg:1", speaker_id="telegram:1", user="проверь PEPE",
                 answer="Сигнала нет, всё спокойно.")
    return tmp_path


RULE = "When a scheduled check finds nothing new, answer [SILENT] with no status message."


def test_the_review_counts_a_repeat_against_its_rule_and_escalates(night, monkeypatch):
    from backend.consolidation import review
    _seed(night / "rules.json", [
        {"text": "Interpret short replies using the preceding topic.", "added": "2026-01-01"},
        {"text": RULE, "added": "2026-01-01",
         "violations": [{"night": "2026-01-02", "what": "sent a status"}]}])
    seen = {}

    def _fake(system, user, *, max_tokens):
        seen["user"] = user
        return {"summary": "s",
                "issues": [{"kind": "ignored_preference", "what": "Sent a no-signal status.",
                            "evidence": "[10-28 09:00] Сигнала нет", "breaks_rule": 2}],
                "lessons": [], "report": "Снова написал без сигнала."}

    monkeypatch.setattr(review, "_call_json", _fake)
    sent = []
    monkeypatch.setattr(review, "_notify_owner", lambda out: sent.append(out))
    out = review.run_review()
    assert f"2. {RULE}" in seen["user"], "the review sees the rules numbered"
    assert out["rules_broken"] == [RULE]
    assert [r["text"] for r in out["escalated"]] == [RULE]
    assert lr.rules()[1]["escalated"]
    assert sent and sent[0]["escalated"]


def test_a_dry_run_counts_nothing(night, monkeypatch):
    from backend.consolidation import review
    _seed(night / "rules.json", [{"text": RULE, "added": "2026-01-01"}])
    monkeypatch.setattr(review, "_call_json", lambda *a, **k: {
        "issues": [{"kind": "other", "what": "w", "breaks_rule": "1"}], "lessons": []})
    out = review.run_review(dry_run=True, notify=False)
    assert out["rules_broken"] == [RULE]
    assert lr.rules()[0]["violations"] == []


def test_the_morning_report_names_escalations_and_skill_upkeep(monkeypatch):
    from backend import roles, scheduled_messages
    from backend.consolidation import review
    monkeypatch.setattr(roles, "primary_owner", lambda: "telegram:1")
    monkeypatch.setattr(scheduled_messages, "list_pending", lambda: [])
    texts = []
    monkeypatch.setattr(scheduled_messages, "schedule", lambda **kw: texts.append(kw["text"]))
    review._notify_owner({
        "report": "r",
        "escalated": [{"text": RULE, "violations": [{"night": "2026-10-25"},
                                                     {"night": "2026-10-28"}]}],
        "skills_disabled": ["corewars-warrior-optimize"],
        "skills_changed": [{"name": "crypto-asset-analysis", "added": 12, "removed": 3,
                            "backup": "SKILL.md.bak-20261029-122744",
                            "sample": ["Check ETF flows."]}],
    })
    (text,) = texts
    assert "2026-10-25, 2026-10-28" in text and "check in code" in text and RULE in text
    assert "corewars-warrior-optimize" in text
    assert "crypto-asset-analysis" in text and "+12" in text and "Check ETF flows." in text


# ── skills ────────────────────────────────────────────────────────────

@pytest.fixture
def skills(tmp_path, monkeypatch):
    from backend import skills as _skills
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("HRANT_DATA_DIR", str(data))
    builtin = tmp_path / "builtin"
    builtin.mkdir()
    mgr = _skills.SkillsManager(skills_dir=builtin)
    monkeypatch.setattr(_skills, "SKILLS", mgr)
    monkeypatch.setattr(_skills, "_ON_SKILL_PROPOSED", [])
    return {"data": data, "builtin": builtin, "mgr": mgr}


def _write(d, name, body="Body."):
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill {name}.\ntriggers: [{name}]\n---\n\n{body}\n",
        encoding="utf-8")


def _usage(data):
    return json.loads((data / "skills_usage.json").read_text(encoding="utf-8"))


def test_a_skill_unused_for_the_window_is_switched_off_and_a_builtin_never(skills):
    from backend import skill_upkeep
    mgr = skills["mgr"]
    _write(mgr.user_dir / "old-skill", "old-skill")
    _write(mgr.user_dir / "new-skill", "new-skill")
    _write(skills["builtin"] / "calc", "calc")
    long_ago = (date.today() - timedelta(days=skill_upkeep.STALE_DAYS + 5)).isoformat()
    (skills["data"] / "skills_usage.json").write_text(json.dumps({
        "old-skill": {"first_seen": long_ago, "last_used": long_ago},
        "calc": {"first_seen": long_ago}}), encoding="utf-8")

    assert skill_upkeep.disable_stale() == ["old-skill"]
    assert mgr.get("old-skill").enabled is False
    assert mgr.get("calc").enabled is True
    assert "new-skill" in _usage(skills["data"]), "an unseen skill starts its clock today"
    assert mgr.get("new-skill").enabled is True


def test_switching_a_skill_back_on_restarts_its_clock(skills):
    from backend import skill_upkeep
    mgr = skills["mgr"]
    _write(mgr.user_dir / "seasonal", "seasonal")
    long_ago = (date.today() - timedelta(days=skill_upkeep.STALE_DAYS + 5)).isoformat()
    (skills["data"] / "skills_usage.json").write_text(json.dumps({
        "seasonal": {"first_seen": long_ago}}), encoding="utf-8")
    assert skill_upkeep.disable_stale() == ["seasonal"]
    mgr.set_enabled("seasonal", True)
    assert skill_upkeep.disable_stale() == []


def test_a_dry_run_switches_nothing_off_and_writes_nothing(skills):
    from backend import skill_upkeep
    mgr = skills["mgr"]
    _write(mgr.user_dir / "fresh", "fresh")
    assert skill_upkeep.disable_stale(dry_run=True) == []
    assert not (skills["data"] / "skills_usage.json").exists()


def test_loading_a_skill_records_its_use(skills):
    from backend.builtin_tools import _load_skill_handler
    _write(skills["mgr"].user_dir / "used", "used")
    assert json.loads(_load_skill_handler("used"))["ok"] is True
    row = _usage(skills["data"])["used"]
    assert row["uses"] == 1 and row["last_used"] == date.today().isoformat()


def test_a_rewrite_of_a_switched_on_skill_is_kept_and_reported_once(skills):
    from backend import skill_upkeep
    from backend import skills as _skills
    _skills.propose(name="crypto", description="d", body="Step one.\nStep two.")
    skills["mgr"].set_enabled("crypto", True)
    sdir = skills["mgr"].user_dir / "crypto"

    _skills.propose(name="crypto", description="d", body="Step one.\nStep two.\nCheck ETF flows.")
    (bak,) = list(sdir.glob("SKILL.md.bak-*"))
    assert "Check ETF flows." not in bak.read_text(encoding="utf-8")
    assert skills["mgr"].get("crypto").enabled is True, "still silent and still on"

    _skills.propose(name="crypto", description="d", body="Step one.\nCheck ETF flows.\nCheck the Fed.")
    assert len(list(sdir.glob("SKILL.md.bak-*"))) == 1, "one backup until the next report"
    (entry,) = skill_upkeep.take_changes()
    assert entry["name"] == "crypto" and entry["rewrites"] == 2
    assert entry["added"] == 2 and entry["removed"] == 1, "measured against the version the owner saw"
    assert skill_upkeep.take_changes() == [], "reported once"


def test_a_rewrite_of_a_switched_off_skill_is_not_reported(skills):
    from backend import skill_upkeep
    from backend import skills as _skills
    _skills.propose(name="draft", description="d", body="One.")
    _skills.propose(name="draft", description="d", body="Two.")
    assert skill_upkeep.take_changes() == []
    assert not list((skills["mgr"].user_dir / "draft").glob("SKILL.md.bak-*"))


def test_a_rewrite_undone_before_the_report_leaves_nothing_to_report(skills):
    from backend import skill_upkeep
    from backend import skills as _skills
    _skills.propose(name="flip", description="d", body="Original.")
    skills["mgr"].set_enabled("flip", True)
    _skills.propose(name="flip", description="d", body="Changed.")
    _skills.propose(name="flip", description="d", body="Original.")
    assert skill_upkeep.take_changes() == []
