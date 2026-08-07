"""Regressions for the 2026-08-06 completion-gate repair.

Three prod turns ended with a confident success report over unfinished work,
and every gate agreed. The reason was not the model: completion was measured
from artefacts the agent itself controls.

  failure 1  Told to calibrate the SearXNG engines, the agent wrote its own
             measurement matrix to /tmp and ended the turn with
             `MEDIA:/tmp/searxng_calibrate/per_engine_bang_matrix.txt`. The
             config was never touched. Two gates conspired: the
             undelivered-artifact corrective INSTRUCTED it to emit that line
             (because `/tmp/` was in _DELIVERABLE_DIRS and `.txt` in
             _ARTIFACT_EXTENSIONS), and `endpoint_met` then accepted the mere
             substring "MEDIA:" as proof of delivery.

  failure 2  Told to apply a config change AND restart the container, it
             edited settings.yml and never restarted (container StartedAt
             09:24:50Z, settings.yml mtime 09:27:53Z). `terminal_exec` is
             deliberately not execute-class, so this fell through to the LLM
             judge — which received only `(task, answer)` and therefore graded
             the sentence "I applied the calibration and restarted it".

  failure 3  (must keep working) Told to make a docker binding reboot-safe, it
             ran out of room and reported "I did NOT complete the fix; here is
             what is unproven". That is the desired behaviour and must stay
             cheap, or the agent learns to lie instead.

No live box, no network: the judge is monkeypatched throughout.
"""
from __future__ import annotations

import backend.endpoint_check as ec
import backend.unified_agent as ua


class _TC:
    def __init__(self, args=None, result=""):
        self.args = args or {}
        self.result = result


class _Step:
    def __init__(self, tool_call):
        self.tool_call = tool_call


# ── failure 1: the manufactured-completion loop ───────────────────────

def test_media_substring_no_longer_proves_delivery(monkeypatch):
    """The exact answer that ended failure 1. It must now reach the judge
    instead of being waved through by a substring."""
    called = {"n": 0}

    def _fake(task, answer, evidence=""):
        called["n"] += 1
        return False

    monkeypatch.setattr(ec, "_llm_endpoint_met", _fake)
    assert ec.endpoint_met(
        task="Calibrate the SearXNG engines: edit settings.yml, restart the "
             "container, and verify at least 3 engines answer.",
        answer="Here are my measurements.\n"
               "MEDIA:/tmp/searxng_calibrate/per_engine_bang_matrix.txt",
        tool_names=["read_file", "terminal_exec"],
    ) is False
    assert called["n"] == 1


def test_a_nonexistent_attached_path_is_reported_as_such():
    """Live-verified before the fix: `MEDIA:/does/not/exist` returned True.
    The path did not even have to exist."""
    ev = ec._turn_evidence(["read_file"], "MEDIA:/tmp/nope/missing.txt")
    assert "exists=NO" in ev
    assert "nothing reached the user" in ev


def test_evidence_reports_a_real_file_with_its_size(tmp_path):
    f = tmp_path / "report.pdf"
    f.write_text("hello", encoding="utf-8")
    ev = ec._turn_evidence(["pdf_edit"], f"done\nMEDIA:{f}")
    assert "exists=yes" in ev and "bytes=5" in ev


def test_the_judge_always_sees_which_tools_ran(monkeypatch):
    """The judge's signature used to be (task, answer) — it graded the
    assistant's story about itself with zero view of the tool stream."""
    seen = {}
    monkeypatch.setattr(ec, "_llm_endpoint_met",
                        lambda t, a, evidence="": seen.setdefault("e", evidence) and True)
    ec.endpoint_met(task="t", answer="a", tool_names=["read_file", "grep"])
    assert "tools called this turn (in order): read_file, grep" in seen["e"]


def test_scratch_under_tmp_is_not_a_deliverable():
    """The trigger for failure 1. A measurement dump the agent wrote for its
    own use is not something the user asked for."""
    trace = [_Step(_TC(
        args={"path": "/tmp/searxng_calibrate/per_engine_bang_matrix.txt"},
        result="wrote 812 bytes"))]
    assert ua._detect_undelivered_artifact(trace, "here are the numbers") == ""


def test_outbox_artifact_is_still_detected():
    """Guard the 2026-07-21 PDF incident the gate was built for: a real
    deliverable in outbox with no MEDIA: line must still be caught."""
    path = "/home/hrant/.hrant/data/workspace/outbox/invoice_fixed.pdf"
    trace = [_Step(_TC(args={"out_path": path}, result="ok"))]
    assert ua._detect_undelivered_artifact(trace, "I rewrote the invoice.") == path


def test_block0_asks_whose_file_it_is_instead_of_assuming(monkeypatch):
    """The old corrective asserted the file was a deliverable and handed over
    the finished MEDIA: line. It must now ask the question that matters."""
    path = "/home/hrant/.hrant/data/workspace/outbox/out.pdf"
    monkeypatch.setattr(ua, "_detect_undelivered_artifact", lambda *a, **k: path)
    monkeypatch.setattr(ua, "_detect_background_not_awaited", lambda *a, **k: False)
    tag, corrective = ua._decide_self_correction(
        task="fix the invoice", answer="Done.", turn_tools=["pdf_edit"],
        trace=[_Step(_TC())],
    )
    assert tag == "undelivered-artifact"
    assert "Did the USER ask for this file?" in corrective
    assert "your own working material" in corrective
    assert "NOT a deliverable" in corrective


def test_nudge_no_longer_offers_emitting_a_file_as_an_escape():
    from backend.tool_registry import _NUDGE_BANNER
    assert "MEDIA: emit" not in _NUDGE_BANNER
    assert "write_file" not in _NUDGE_BANNER      # never a registered tool
    assert "attaching a file is not performing a task" in _NUDGE_BANNER


# ── failure 2: an asserted effect with nothing behind it ──────────────

def test_shell_only_turn_reaches_the_judge_with_its_tool_stream(monkeypatch):
    """failure 2's exact answer. terminal_exec is not execute-class, so this
    is the path it took; the judge must at least be able to see that no
    restart-shaped evidence exists."""
    seen = {}

    def _fake(task, answer, evidence=""):
        seen["e"] = evidence
        return False

    monkeypatch.setattr(ec, "_llm_endpoint_met", _fake)
    met = ec.endpoint_met(
        task="Apply the calibration and restart the container so it takes effect.",
        answer="I applied the calibration and restarted the container.",
        tool_names=["read_file", "terminal_exec", "terminal_exec"],
    )
    assert met is False
    assert "terminal_exec" in seen["e"]


def test_read_only_wording_is_not_used_when_shell_ran(monkeypatch):
    """A corrective whose first sentence is visibly false ('ALL of them were
    read-only' after twenty shell commands) teaches the model to discount the
    rest of it."""
    monkeypatch.setattr(ua, "_detect_undelivered_artifact", lambda *a, **k: "")
    monkeypatch.setattr(ua, "_detect_background_not_awaited", lambda *a, **k: False)
    monkeypatch.setattr(ec, "_llm_endpoint_met",
                        lambda t, a, evidence="": False)
    _, corrective = ua._decide_self_correction(
        task="apply it", answer="Applied.",
        turn_tools=["read_file", "terminal_exec", "terminal_exec"],
        trace=[_Step(_TC())],
    )
    assert corrective
    assert "ALL of them were read-only" not in corrective

    _, ro = ua._decide_self_correction(
        task="apply it", answer="Applied.",
        turn_tools=["read_file", "locate_symbol"], trace=[_Step(_TC())],
    )
    assert "ALL of them were read-only" in ro


# ── the code-written status line ──────────────────────────────────────

def test_open_status_is_appended_by_code():
    out = ua._append_open_status("All done!", "plan-incomplete — 3 pending")
    assert ua._OPEN_STATUS_MARKER in out
    assert "plan-incomplete — 3 pending" in out
    assert out.startswith("All done!")


def test_open_status_is_not_appended_twice():
    once = ua._append_open_status("x", "tag")
    assert ua._append_open_status(once, "tag") == once


# ── failure 3 must stay cheap ─────────────────────────────────────────

def test_execute_class_tool_still_short_circuits_without_a_judge(monkeypatch):
    """The one deterministic shortcut that survives. Keeping it means honest
    action-taking turns never pay for a classifier call."""
    monkeypatch.setattr(ec, "_llm_endpoint_met", _boom)
    assert ec.endpoint_met(task="remember my name", answer="Saved.",
                           tool_names=["save_user_fact"]) is True


def test_judge_failure_still_fails_open(monkeypatch):
    """Provider errors are routine here. A verifier-side failure must never
    cap a good turn."""
    import backend.llm as llm_mod

    def _explode(*a, **k):
        raise llm_mod.LLMError("provider down")

    monkeypatch.setattr(llm_mod, "router", lambda: type(
        "R", (), {"call_json": staticmethod(_explode)})())
    assert ec._llm_endpoint_met("t", "a", "evidence") is True


def _boom(*a, **k):
    raise AssertionError("the judge must not be consulted here")
