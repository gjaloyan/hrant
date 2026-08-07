"""A file the owner asked for, produced but never attached, is a failed task.

Prod incident 2026-07-21: the agent correctly rewrote an invoice PDF into
workspace/outbox/armenia_ci_gor_jaloyan_pe.pdf (verified on disk afterwards:
the new recipient and tax number were both present) and answered without a
MEDIA: line. The owner received nothing and reported the task as broken.
"""
from __future__ import annotations

from backend.unified_agent import (
    _decide_self_correction,
    _detect_undelivered_artifact,
)


class _TC:
    def __init__(self, name, args=None, result=""):
        self.name = name
        self.args = args or {}
        self.result = result


class _Step:
    def __init__(self, tool_call):
        self.tool_call = tool_call


OUT = "/home/hrant/.hrant/data/workspace/outbox/invoice.pdf"


def test_detects_produced_file_with_no_media_line():
    trace = [_Step(_TC("terminal_exec", {"command": f"python3 edit.py > {OUT}"},
                       f"wrote {OUT}"))]
    assert _detect_undelivered_artifact(trace, "Done, the PDF is ready.") == OUT


def test_media_line_means_delivered():
    trace = [_Step(_TC("terminal_exec", {"command": "x"}, f"wrote {OUT}"))]
    answer = f"Here is the invoice.\nMEDIA:{OUT}"
    assert _detect_undelivered_artifact(trace, answer) == ""


def test_mentioning_the_path_in_prose_is_not_delivery():
    # The Jul-21 answer named the path and still delivered nothing.
    trace = [_Step(_TC("terminal_exec", {"command": "x"}, f"saved to {OUT}"))]
    answer = f"The file was created at {OUT}."
    assert _detect_undelivered_artifact(trace, answer) == OUT


def test_no_artifact_no_correction():
    trace = [_Step(_TC("read_file", {"path": "/home/hrant/hrant/backend/llm.py"}))]
    assert _detect_undelivered_artifact(trace, "Here is what the code does.") == ""


def test_scratch_paths_outside_deliverable_dirs_are_ignored():
    trace = [_Step(_TC("terminal_exec", {"command": "x"},
                       "wrote /home/hrant/hrant/backend/notes.md"))]
    assert _detect_undelivered_artifact(trace, "Updated the module.") == ""


def test_empty_trace_is_safe():
    assert _detect_undelivered_artifact(None, "text") == ""
    assert _detect_undelivered_artifact([], "text") == ""


def test_self_correction_fires_with_the_media_instruction():
    trace = [_Step(_TC("terminal_exec", {"command": "x"}, f"wrote {OUT}"))]
    tag, corrective = _decide_self_correction(
        task="change the pdf", answer="Done.", turn_tools=["terminal_exec"],
        trace=trace,
    )
    assert tag == "undelivered-artifact"
    assert f"MEDIA:{OUT}" in corrective
    # Reworded 2026-08-06: the corrective now ASKS whether the file is the
    # user's before assuming it is. Assuming it is how the agent's own scratch
    # measurements got shipped as the deliverable.
    assert "Did the USER ask for this file?" in corrective
    assert "is a failed task" in corrective


def test_self_correction_silent_when_delivered():
    trace = [_Step(_TC("terminal_exec", {"command": "x"}, f"wrote {OUT}"))]
    tag, corrective = _decide_self_correction(
        task="change the pdf", answer=f"Done.\nMEDIA:{OUT}",
        turn_tools=["terminal_exec"], trace=trace,
    )
    assert tag != "undelivered-artifact"
