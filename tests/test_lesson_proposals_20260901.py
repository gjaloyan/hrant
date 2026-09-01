"""A behavioural lesson must become a change, not a wish.

Prod 2026-09-01: 88 of 97 active goals had NO subtasks — no plan, nothing
executable — so nothing could act on them and the stale sweep retired them
after 14 days. Three months of that produced 447 distinct suggestions and 5
approvals.

A lesson is a RULE, and rules live in prompt_modules.py. Proposing the
actual edit puts it in the queue that already has a Telegram approval flow,
a readable diff, and an apply path that compiles and runs the tests.
"""
import pytest

from backend import lesson_proposals as lp
from backend.prompt_modules import MODULES


class FakeProp:
    def __init__(self, title, status="pending"):
        self.title = title
        self.description = title
        self.status = status


def test_the_anchor_is_really_in_the_module():
    """The whole edit hinges on this literal string existing. If the module
    body is reworded and the anchor goes, every proposal fails to apply with
    'old_code not found' — silently, two weeks later, at approval time."""
    assert lp.ANCHOR in MODULES["m11_lessons"].body


def test_the_module_is_always_loaded():
    from backend.prompt_modules import DEFAULT_ORDER
    assert MODULES["m11_lessons"].always_on
    assert "m11_lessons" in DEFAULT_ORDER, "the rules would never be read"


def test_the_edit_inserts_above_the_anchor_and_keeps_it():
    edit = lp.build_edit("check the path exists before naming it")
    assert edit["old_code"] == lp.ANCHOR
    assert edit["new_code"].endswith(lp.ANCHOR), (
        "the anchor must survive, or only one lesson could ever be added")
    assert edit["new_code"].startswith("- check the path exists")


def test_applying_twice_keeps_both_lessons():
    """Simulate the apply engine's literal replace, twice."""
    body = MODULES["m11_lessons"].body
    first = lp.build_edit("always verify a claim before stating it")
    body = body.replace(first["old_code"], first["new_code"], 1)
    second = lp.build_edit("never promise an action you did not take")
    body = body.replace(second["old_code"], second["new_code"], 1)

    lessons = lp.existing_lessons(body)
    assert len(lessons) == 2, lessons
    assert lp.ANCHOR in body


def test_evidence_rides_along_as_a_comment():
    edit = lp.build_edit("cite the source", evidence="seen 9x in verifier")
    assert "seen 9x in verifier" in edit["new_code"]
    # It must not become part of the rule the model reads as instruction.
    assert "<!--" in edit["new_code"]


def test_a_rule_already_present_is_not_proposed_again():
    body = MODULES["m11_lessons"].body.replace(
        lp.ANCHOR, "- Always verify a claim before stating it.\n\n" + lp.ANCHOR)
    why = lp.already_known("always verify a claim before stating it", body, [])
    assert why and "already a rule" in why


def test_a_rule_already_waiting_is_not_proposed_again():
    """The meta-learner rephrases the same complaint endlessly — that is how
    509 archived goals still counted as 447 'distinct' ones."""
    pending = [FakeProp("Lesson: always verify a claim before stating it")]
    why = lp.already_known("Always verify a claim before stating it.",
                           MODULES["m11_lessons"].body, pending)
    assert why and "already proposed" in why


def test_a_rejected_proposal_does_not_block_a_new_one():
    pending = [FakeProp("Lesson: always verify a claim", status="rejected")]
    assert lp.already_known("always verify a claim",
                            MODULES["m11_lessons"].body, pending) is None


def test_a_genuinely_different_lesson_gets_through():
    pending = [FakeProp("Lesson: always verify a claim before stating it")]
    assert lp.already_known(
        "prefer running the command over describing it",
        MODULES["m11_lessons"].body, pending) is None


@pytest.mark.parametrize("junk", ["", "   ", "too short"])
def test_junk_is_refused(junk):
    assert lp.propose_lesson(junk) is None


# ── the meta-learner no longer files wishes ────────────────────────────

def _meta(monkeypatch, tmp_path):
    """A meta-learner whose two sinks are observable."""
    import importlib
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend import config, goals, meta_learner
    importlib.reload(config)
    importlib.reload(goals)
    importlib.reload(meta_learner)

    added = []
    monkeypatch.setattr(meta_learner.GOALS, "add",
                        lambda **kw: added.append(kw))
    return meta_learner, added


def test_a_prompt_lesson_becomes_a_proposal_not_a_planless_goal(
        monkeypatch, tmp_path):
    ml, added = _meta(monkeypatch, tmp_path)
    made = []
    monkeypatch.setattr("backend.lesson_proposals.propose_lesson",
                        lambda d, **k: made.append(d) or object())

    ml.META_LEARNER._auto_fix({
        "fix_action": "improve_prompt",
        "fix_detail": "verify a file path exists before naming it in an answer",
        "severity": 8,
    })
    assert made, "the lesson was not proposed"
    assert not added, f"a goal was filed anyway: {added}"


def test_a_pattern_becomes_a_proposal_too(monkeypatch, tmp_path):
    """372 of the 509 swept goals came from this path, and it never
    attached a plan to any of them."""
    ml, added = _meta(monkeypatch, tmp_path)
    made = []
    monkeypatch.setattr("backend.lesson_proposals.propose_lesson",
                        lambda d, **k: made.append(d) or object())
    monkeypatch.setattr(ml.META_LEARNER, "_save_patterns", lambda: None)

    class _R:
        @staticmethod
        def call_json(*a, **k):
            return {"patterns": [{
                "pattern": "states file paths it never checked",
                "priority": 9, "frequency": 6,
                "suggested_fix": "check the path exists before naming it",
            }]}

    monkeypatch.setattr(ml, "router", lambda: _R)
    monkeypatch.setattr(
        ml.META_LEARNER, "_read_log",
        lambda **k: [{"question": "q", "error": "x",
                      "analysis": {"root_cause": "unchecked claim",
                                   "error_pattern": "states unverified paths"}}] * 5)
    ml.META_LEARNER.extract_patterns()

    assert made == ["check the path exists before naming it"]
    assert not added, f"a planless goal was filed anyway: {added}"


def test_a_pattern_with_no_fix_text_still_keeps_the_observation(
        monkeypatch, tmp_path):
    """Falling back to a goal is right when there is nothing to propose —
    losing the observation entirely would be worse than a wish."""
    ml, added = _meta(monkeypatch, tmp_path)
    monkeypatch.setattr(ml.META_LEARNER, "_save_patterns", lambda: None)

    class _R:
        @staticmethod
        def call_json(*a, **k):
            return {"patterns": [{"pattern": "something odd",
                                  "priority": 9, "frequency": 3,
                                  "suggested_fix": ""}]}

    monkeypatch.setattr(ml, "router", lambda: _R)
    monkeypatch.setattr(
        ml.META_LEARNER, "_read_log",
        lambda **k: [{"question": "q", "error": "x",
                      "analysis": {"root_cause": "unchecked claim",
                                   "error_pattern": "states unverified paths"}}] * 5)
    ml.META_LEARNER.extract_patterns()
    assert added, "the observation was dropped"


def test_no_module_and_no_proposal_files_nothing_planless(
        monkeypatch, tmp_path):
    """The exact 88-goal case: no guessable module, and the lesson was a
    duplicate. Filing a goal with subtasks=None would recreate the graveyard."""
    ml, added = _meta(monkeypatch, tmp_path)
    monkeypatch.setattr("backend.lesson_proposals.propose_lesson",
                        lambda d, **k: None)
    monkeypatch.setattr(ml.META_LEARNER, "_guess_target_module",
                        staticmethod(lambda d: ""))

    ml.META_LEARNER._auto_fix({
        "fix_action": "improve_prompt",
        "fix_detail": "be more careful about unverified claims in general",
        "severity": 6,
    })
    assert not added, f"a goal with no plan was filed: {added}"


# ── the prompt must not grow without bound ─────────────────────────────

def test_an_empty_lessons_module_costs_nothing():
    """It starts empty and fills only as the owner approves. A heading and
    an insertion anchor billed on every turn to say nothing is exactly the
    bloat the 20:1 input ratio complains about."""
    from backend.prompt_modules import build_prompt, _is_empty_collector

    assert _is_empty_collector(MODULES["m11_lessons"].body)
    assert "LESSONS LEARNED" not in build_prompt()


def test_a_module_with_a_lesson_is_included():
    from backend.prompt_modules import _is_empty_collector

    body = MODULES["m11_lessons"].body.replace(
        lp.ANCHOR, "- Verify before you claim.\n\n" + lp.ANCHOR)
    assert not _is_empty_collector(body)


def test_a_real_module_is_never_dropped():
    from backend.prompt_modules import DEFAULT_ORDER, _is_empty_collector

    for name in DEFAULT_ORDER:
        if name == "m11_lessons":
            continue
        assert not _is_empty_collector(MODULES[name].body), name


def test_the_module_has_a_ceiling():
    """Unbounded growth here would recreate the runaway queue in the prompt
    instead of in goals.json."""
    full = MODULES["m11_lessons"].body.replace(
        lp.ANCHOR,
        "".join(f"- lesson number {i} about something.\n"
                for i in range(lp.MAX_LESSONS)) + "\n" + lp.ANCHOR)
    why = lp.already_known("an entirely unrelated new rule about timing",
                           full, [])
    assert why and "full" in why


def test_below_the_ceiling_still_accepts():
    nearly = MODULES["m11_lessons"].body.replace(
        lp.ANCHOR,
        "".join(f"- lesson number {i} about something.\n"
                for i in range(lp.MAX_LESSONS - 1)) + "\n" + lp.ANCHOR)
    assert lp.already_known("an entirely unrelated new rule about timing",
                            nearly, []) is None


def test_the_edit_really_applies_to_the_real_file():
    """The apply engine does a literal replace on the SOURCE FILE and
    refuses an ambiguous match. If the anchor ever appears twice — or the
    patched file stops parsing — every lesson proposal fails at approval
    time, two weeks after it was written, with 'old_code not found'.
    """
    import ast
    from pathlib import Path

    src = Path("backend/prompt_modules.py").read_text(encoding="utf-8")
    assert src.count(lp.ANCHOR) == 1, "ambiguous anchor: apply would refuse"

    edit = lp.build_edit("verify a path before naming it", "seen 6x")
    patched = src.replace(edit["old_code"], edit["new_code"], 1)

    ast.parse(patched)                       # still valid Python
    assert patched.count(lp.ANCHOR) == 1     # room for the next lesson
    assert "- verify a path before naming it." in patched


def test_a_russian_lesson_is_translated_before_it_reaches_the_prompt(
        monkeypatch):
    """The prompt is English-only. The meta-learner writes in the language
    of the conversation it learned from, and the goals on prod carried
    fixes like 'Требовать проверять каждое дело по официальному источнику'."""
    seen = {}
    monkeypatch.setattr("backend.meaning_translate.to_english",
                        lambda t, **k: "Check every case against an official source")

    class Store:
        _LOCK = __import__("threading").RLock()
        _proposals: list = []

        def _save(self):
            pass

    monkeypatch.setattr("backend.self_modifier.SELF_MODIFIER", Store())
    monkeypatch.setattr("backend.self_modifier._fire_proposal_created",
                        lambda p: seen.setdefault("p", p))

    made = lp.propose_lesson(
        "Требовать проверять каждое дело по официальному источнику")
    assert made is not None
    assert "official source" in made.new_code
    assert "Требовать" not in made.new_code


def test_an_untranslatable_lesson_is_dropped_not_written_raw(monkeypatch):
    monkeypatch.setattr("backend.meaning_translate.to_english",
                        lambda t, **k: t)          # translation failed
    assert lp.propose_lesson("Требовать проверять каждое дело") is None
