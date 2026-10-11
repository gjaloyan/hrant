"""The nightly review: the day's conversations in, tomorrow's rules out.

The owner's design (2026-10-10): the check that makes the agent better is
one honest look at the whole day, not a judge on every turn. Before this,
lessons came only from jobs marked "failed" (a wrong answer is still a
"completed" job), went to a jammed self-mod queue, and profiles were only
ever appended to — 19.5k characters read on every reply.
"""
from __future__ import annotations

import json
import time

import pytest

from backend import learned_rules as lr


# ── learned rules ─────────────────────────────────────────────────────

def test_rules_are_added_rendered_and_revertible(tmp_path):
    p = tmp_path / "rules.json"
    lr.add(["Generate images with generate_image when the user asks for a design."],
           path=p)
    text = lr.render(p)
    assert text.startswith("# LEARNED RULES")
    assert "generate_image" in text
    lr.add(["Answer in Russian unless the user chose another language."], path=p)
    assert len(lr.rules(p)) == 2
    assert lr.revert_last(p) is True
    assert len(lr.rules(p)) == 1


def test_a_restated_rule_is_skipped(tmp_path):
    p = tmp_path / "rules.json"
    lr.add(["Check the conversation history before asking the user to resend a photo."], path=p)
    out = lr.add(["Check the conversation history before asking the user to re-send a photo!"], path=p)
    assert out["added"] == [] and len(lr.rules(p)) == 1


def test_rules_stay_within_budget_and_long_ones_are_cut(tmp_path):
    p = tmp_path / "rules.json"
    distinct = [
        "Generate pictures with generate_image when someone asks for a design or a render.",
        "Before asking for a resend, look for the attachment in the conversation history.",
        "Quote exchange rates with the source and the time they were read.",
        "Keep replies to small talk under two sentences.",
        "Set reminders in the user's local time zone and repeat the time back.",
        "Name the court and case number exactly as the registry prints them.",
        "When speech recognition is garbled, quote what was heard and ask once.",
        "Never claim a file was sent unless a MEDIA line is in the answer.",
        "Prefer the manufacturer's manual over forum posts for specifications.",
        "Run long downloads as background jobs instead of waiting in the turn.",
        "Ask at most one clarifying question per request, with concrete options.",
        "Check service status with systemctl before saying a server is down.",
        "Translate Armenian legal terms but keep the original term in brackets.",
        "Summarise a long article in five bullet points unless asked otherwise.",
        "Rule number 14: confirm a destructive command with the owner before running it, "
        "and state exactly which files or services it will touch and how to undo it.",
    ]
    for text in distinct:
        lr.add([text], path=p)
    rs = lr.rules(p)
    assert len(rs) <= lr.MAX_RULES
    assert all(len(r["text"]) <= lr.MAX_RULE_CHARS for r in rs)
    assert sum(len(r["text"]) for r in rs) <= lr.MAX_TOTAL_CHARS
    assert "Rule number 14" in rs[-1]["text"], "the newest rule must survive"


def test_an_llm_merge_replaces_the_set_when_over_budget(tmp_path, monkeypatch):
    p = tmp_path / "rules.json"
    monkeypatch.setattr(lr, "MAX_RULES", 2)
    lr.add(["First rule about verifying facts before stating them."], path=p)
    lr.add(["Second rule about using generate_image for visual requests."], path=p)
    out = lr.add(["Third rule about keeping answers short on simple questions."], path=p,
                 merge=lambda cur, new, n: ["Merged rule A covering facts and images.",
                                            "Third rule about keeping answers short."])
    assert out["merged"] is True
    assert [r["text"] for r in lr.rules(p)][0].startswith("Merged rule A")


# ── profile compaction ────────────────────────────────────────────────

def test_a_bloated_profile_is_rewritten_and_the_language_pin_survives(tmp_path, monkeypatch):
    from backend.consolidation import review
    p = tmp_path / "telegram_1.md"
    old = ("# User Profile\n\n## Язык общения\n- Respond to user in Russian always.\n\n"
           "## О пользователе\n- User is named Gor.\n\n"
           + "\n".join(f"## 2026-08-{i:02d} — daily update\n\nTransient detail {i}." * 3
                       for i in range(1, 28)))
    p.write_text(old, encoding="utf-8")

    def _fake(system, user, *, max_tokens):
        return {"profile_markdown": "# User Profile\n## Language\nPREFERS ENGLISH (wrong!)\n\n"
                                    "## About the user\n- Gor; drives a 2001 Subaru Forester."
                                    + " Pharmacist." * 10}

    monkeypatch.setattr(review, "_call_json", _fake)
    done = review.compact_profile(p, "")
    new = p.read_text(encoding="utf-8")
    assert done and done["after"] < done["before"]
    assert "Respond to user in Russian always." in new
    assert "PREFERS ENGLISH" not in new
    assert "daily update" not in new
    assert list(tmp_path.glob("telegram_1.md.bak-*")), "a backup must be kept"


def test_a_broken_rewrite_leaves_the_profile_alone(tmp_path, monkeypatch):
    from backend.consolidation import review
    p = tmp_path / "u.md"
    original = "# User Profile\n" + "x" * 9000
    p.write_text(original, encoding="utf-8")
    monkeypatch.setattr(review, "_call_json", lambda *a, **k: {"profile_markdown": "oops"})
    assert review.compact_profile(p, "") is None
    assert p.read_text(encoding="utf-8") == original


# ── the review step ───────────────────────────────────────────────────

@pytest.fixture
def day(tmp_path, monkeypatch):
    from backend import threads
    from backend.threads import ThreadStore
    store = ThreadStore(root=tmp_path / "threads")
    monkeypatch.setattr(threads, "THREADS", store)
    monkeypatch.setattr(lr, "_path", lambda: tmp_path / "rules.json")
    store.append("tg:1", speaker_id="telegram:1",
                 user="можешь сгенерировать фотки комнаты?",
                 answer="❓ Какой генератор изображений использовать?")
    store.append("tg:1", speaker_id="telegram:1",
                 user="ты опять спрашиваешь вместо того чтобы сделать",
                 answer="Извини.")
    return store


def test_the_review_reads_the_day_and_writes_tomorrows_rule(day, monkeypatch):
    from backend.consolidation import review
    seen = {}

    def _fake(system, user, *, max_tokens):
        if "nightly self-review" in system:
            seen["user"] = user
            return {"summary": "One request went badly.",
                    "issues": [{"kind": "asked_instead_of_doing",
                                "what": "Asked which image tool to use.",
                                "evidence": "можешь сгенерировать"}],
                    "lessons": ["When asked to make pictures, call generate_image instead of asking which service to use."],
                    "report": "Вчера я спросил вместо того, чтобы нарисовать. Исправил."}
        return {}

    monkeypatch.setattr(review, "_call_json", _fake)
    sent = []
    monkeypatch.setattr(review, "_notify_owner", lambda out: sent.append(out))
    out = review.run_review()
    assert "можешь сгенерировать фотки" in seen["user"]
    assert "REPORT_LANGUAGE" in seen["user"]
    assert out["lessons_added"] and "generate_image" in lr.render()
    assert sent and "нарисовать" in sent[0]["report"]


def test_a_dry_run_proposes_but_writes_nothing(day, monkeypatch):
    from backend.consolidation import review
    monkeypatch.setattr(review, "_call_json", lambda system, user, *, max_tokens: {
        "lessons": ["When asked to make pictures, call generate_image instead of asking."],
        "report": "x"} if "nightly self-review" in system else {})
    monkeypatch.setattr(review, "_notify_owner", lambda out: pytest.fail("dry run notified"))
    out = review.run_review(dry_run=True)
    assert out["lessons_proposed"] and out["lessons_added"] == []
    assert lr.render() == ""


def test_a_quiet_day_changes_nothing(tmp_path, monkeypatch):
    from backend import threads
    from backend.threads import ThreadStore
    from backend.consolidation import review
    monkeypatch.setattr(threads, "THREADS", ThreadStore(root=tmp_path / "t"))
    monkeypatch.setattr(lr, "_path", lambda: tmp_path / "rules.json")
    monkeypatch.setattr(review, "_call_json",
                        lambda *a, **k: pytest.fail("no conversations, no LLM call"))
    monkeypatch.setattr(review, "_notify_owner", lambda out: pytest.fail("nothing to report"))
    from backend.consolidation import config as ccfg
    monkeypatch.setattr(ccfg, "user_md_path", lambda: tmp_path / "missing.md")
    out = review.run_review()
    assert out["lessons_added"] == [] and lr.render() == ""


def test_the_pipeline_runs_the_review_and_no_longer_appends_profiles():
    import inspect
    from backend.consolidation import pipeline
    src = inspect.getsource(pipeline.run)
    assert "_review.run_review(" in src
    assert pipeline._PROFILES_BY_REVIEW is True


def test_learned_rules_reach_the_system_prompt():
    import inspect
    import backend.unified_agent as ua
    assert inspect.getsource(ua).count("_learned_render()") == 2


# ── it runs at night ──────────────────────────────────────────────────

def test_the_run_waits_for_the_night(monkeypatch):
    from backend.consolidation import config as ccfg, scheduler, state
    monkeypatch.setattr(state, "load", lambda: state.ConsolidationState(
        last_run_at=time.time() - 21 * 3600))
    monkeypatch.setattr(state, "cooldown_remaining_seconds", lambda st: 0)
    monkeypatch.setattr(scheduler, "_in_night_window", lambda now: False)
    should, reason = scheduler._should_fire()
    assert should is False and "night" in reason
    # A missed night is caught up, not skipped.
    monkeypatch.setattr(state, "load", lambda: state.ConsolidationState(
        last_run_at=time.time() - ccfg.CATCHUP_AFTER_SECONDS - 60))
    should, reason = scheduler._should_fire()
    assert "night" not in reason


def test_a_run_the_afternoon_before_does_not_shut_the_night(monkeypatch):
    """The first night never ran (2026-10-11): the last run was at 16:06 and
    a 20h cooldown kept 03:00-06:00 closed. A run as late as 18:00 must leave
    the next window open."""
    from backend.consolidation import gather, scheduler, state
    now = time.time()
    monkeypatch.setattr(state, "load", lambda: state.ConsolidationState(
        last_run_at=now - 9 * 3600))  # 18:00 the day before, now 03:00
    monkeypatch.setattr(scheduler, "_in_night_window", lambda t: True)
    monkeypatch.setattr(gather, "last_activity_ts", lambda: now - 5 * 3600)
    should, reason = scheduler._should_fire(now)
    assert should is True, reason


def test_the_night_window_is_read_in_the_users_time_zone(monkeypatch):
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from backend.consolidation import scheduler
    import backend.settings as settings
    monkeypatch.setattr(settings, "user_timezone", lambda: "Asia/Yerevan")
    at_4am = datetime(2026, 10, 11, 4, 0, tzinfo=ZoneInfo("Asia/Yerevan")).timestamp()
    at_4pm = datetime(2026, 10, 11, 16, 0, tzinfo=ZoneInfo("Asia/Yerevan")).timestamp()
    assert scheduler._in_night_window(at_4am) is True
    assert scheduler._in_night_window(at_4pm) is False


# ── learning no longer holds the reply ────────────────────────────────

def test_post_answer_learning_runs_off_the_reply_path(monkeypatch):
    import threading
    import backend.unified_agent as ua
    monkeypatch.setattr(ua, "_LEARN_AFTER_ANSWER", True)
    started, release = threading.Event(), threading.Event()

    def _slow(**kw):
        started.set()
        release.wait(5)

    t0 = time.time()
    ua._after_answer(_slow, x=1)
    assert time.time() - t0 < 0.5, "the caller must not wait for learning"
    assert started.wait(2)
    release.set()


def test_the_three_learning_hooks_go_through_after_answer():
    import inspect
    import backend.unified_agent as ua
    src = inspect.getsource(ua.run_unified)
    for hook in ("_commit_turn_memory,", "maybe_capture_correction,",
                 "_ML.analyze_failure,"):
        assert f"_after_answer(\n                {hook}" in src or \
               f"_after_answer(\n            {hook}" in src, hook


def test_test_harness_conversations_are_not_reviewed(tmp_path, monkeypatch):
    from backend.consolidation import review
    from backend.threads import ThreadStore
    store = ThreadStore(root=tmp_path / "t")
    store.append("webui:smoke", speaker_id="webui:smoke", user="smoke test", answer="ok")
    store.append("tg:1", speaker_id="telegram:1", user="real question", answer="real answer")
    text, stats = review.day_transcript("2000-01-01", store=store)
    assert "real question" in text and "smoke test" not in text
    assert stats["speakers"] == ["telegram:1"]
