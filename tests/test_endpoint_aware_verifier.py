"""Endpoint-aware verifier confidence.

Audit 2026-05-27 prod state: the terminal-bench turn (17 inspect
calls, no `start_background_job`, no deliverable) received
confidence=75 from the verifier. The verifier checks whether
*claims* in the answer are *verifiable*, not whether the answer
*delivered* against the user's request shape.

For action-verb requests ("run X", "запусти Y") the new
post-hoc check requires at least one execute-class tool call OR
a MEDIA: file-delivery line in the answer. Otherwise the
confidence is capped at 30 and an explicit
'endpoint_not_met' contradiction is added.

Non-action requests (chat, recall, "what is X?") are unaffected.
"""
from __future__ import annotations


def test_detects_english_action_verbs():
    from backend.endpoint_check import _detect_action_verbs
    for s in (
        "run terminal-bench",
        "launch the build",
        "send it to my wife",
        "set the rate to +25%",
        "install httpx",
        "create the adapter",
        "fix the verifier",
    ):
        assert _detect_action_verbs(s), f"missed action verb in: {s!r}"


def test_detects_russian_action_verbs():
    from backend.endpoint_check import _detect_action_verbs
    for s in (
        "запусти бенч",
        "сделай мне сайт",
        "отправь ей сообщение",
        "измени голос на быстрый",
        "ускорь TTS",
        "поставь tts.rate +25%",
    ):
        assert _detect_action_verbs(s), f"missed action verb in: {s!r}"


def test_does_not_flag_question_requests():
    from backend.endpoint_check import _detect_action_verbs
    for s in (
        "what is python",
        "how does ffmpeg work",
        "tell me about HPLC",
        "что это значит",
        "как работает router",
        "почему confidence упала",
        "hi how are you",
    ):
        assert not _detect_action_verbs(s), f"false positive on: {s!r}"


def test_endpoint_met_when_no_action_verbs():
    """Non-action requests don't need any execute tool. Verifier
    should not penalise a chat turn for absence of execute calls."""
    from backend.endpoint_check import endpoint_met
    assert endpoint_met(
        task="what's the capital of france",
        answer="Paris.",
        tool_names=[],
    ) is True


def test_endpoint_met_when_execute_tool_in_trace():
    """Action verb + execute-class tool → endpoint met."""
    from backend.endpoint_check import endpoint_met
    assert endpoint_met(
        task="запусти бенч на 5 задачах",
        answer="started job j-7a3c",
        tool_names=["define_task_endpoint", "start_background_job"],
    ) is True


def test_endpoint_missed_when_action_verb_but_only_inspection():
    """The 2026-05-26 terminal-bench failure mode: action verb,
    17 terminal_exec / read_file calls, no execute-class call.
    `terminal_exec` is *not* in EXECUTE_TOOLS — it's ambiguous."""
    from backend.endpoint_check import endpoint_met
    assert endpoint_met(
        task="can you run terminal-bench for you",
        answer="есть только подготовка и проверки",
        tool_names=["load_skill", "load_tool_bundle",
                    "terminal_exec", "terminal_exec",
                    "read_file", "terminal_exec"],
    ) is False


def test_endpoint_met_when_media_line_in_answer():
    """Even without a recognised execute tool, a MEDIA: line in
    the answer signals delivery and counts as endpoint-met."""
    from backend.endpoint_check import endpoint_met
    assert endpoint_met(
        task="сделай мне видео без лого",
        answer="done\n\nMEDIA:/home/x/.hrant/data/workspace/outbox/clip.mp4",
        tool_names=["read_file", "preprocess_video", "run_python"],
    ) is True


def test_terminal_exec_alone_does_not_count_as_execute():
    """terminal_exec is ambiguous (it can be `which X` or
    `pip install Y`). For endpoint detection we ignore it — to
    register as endpoint-met it must be paired with a clear
    execute tool OR a MEDIA: delivery."""
    from backend.endpoint_check import endpoint_met
    assert endpoint_met(
        task="run the build",
        answer="checked status",
        tool_names=["terminal_exec", "terminal_exec"],
    ) is False


def test_honest_diagnose_then_propose_passes_endpoint_check():
    """The 2026-05-28 smoke ('can you benchmark yourself? if not,
    propose how to make it possible') exposed a false-positive:
    'make' triggered action-verb, but the correct answer was
    diagnose-then-propose (no execute, no MEDIA). Treating it as
    endpoint_not_met punished the right behavior. This pattern
    must pass."""
    from backend.endpoint_check import endpoint_met
    answer = (
        "I cannot benchmark myself directly on Terminal-Bench right "
        "now, but I can explain why and propose a concrete path.\n\n"
        "## Option A: minimal adapter (~30 min)\n"
        "A thin CLI wrapper that accepts Harbor's task JSON...\n\n"
        "## Option B: full integration\n"
        "Add a --harbor-mode flag...\n\n"
        "Recommendation: Option A first."
    )
    assert endpoint_met(
        task="can you benchmark yourself? if not, propose how to "
             "make it possible",
        answer=answer,
        tool_names=[],
    ) is True


def test_honest_diagnose_without_proposal_still_fails():
    """A bare 'I can't' with no alternative offered is NOT a
    legitimate diagnose-then-propose response — that's the old
    failure mode the verifier should still catch."""
    from backend.endpoint_check import endpoint_met
    assert endpoint_met(
        task="run terminal-bench",
        answer="I cannot do that.",
        tool_names=[],
    ) is False


def test_russian_diagnose_propose_recognised():
    """The honest-diagnose pattern works for Russian responses too."""
    from backend.endpoint_check import endpoint_met
    answer = (
        "Я не могу запустить бенч прямо сейчас, потому что нет "
        "CLI-обёртки. Предлагаю Вариант A — написать тонкий "
        "адаптер."
    )
    assert endpoint_met(
        task="запусти бенч",
        answer=answer,
        tool_names=[],
    ) is True


def test_run_python_alone_does_not_count_as_execute():
    """run_python is the workaround-route that the 2026-05-27 smoke
    test exposed: agent dodged `start_background_job` for "run
    terminal-bench" by executing tasks inline with run_python and
    claimed success. Like terminal_exec, run_python is ambiguous
    (compute 2+2 vs write a file), so the endpoint check must NOT
    accept it as a stand-in for proper job control."""
    from backend.endpoint_check import endpoint_met
    assert endpoint_met(
        task="запусти terminal-bench",
        answer="ran 3 tasks",
        tool_names=["load_skill", "terminal_exec", "run_python",
                    "run_python", "read_file", "run_python"],
    ) is False


def test_cap_confidence_helper_caps_only_when_endpoint_missed():
    """The verifier-side helper drops confidence to <=30 when the
    endpoint isn't met. When met (or non-action), the original
    confidence passes through unchanged."""
    from backend.endpoint_check import cap_confidence_for_endpoint

    # Action verb + no execute tool → capped.
    assert cap_confidence_for_endpoint(
        task="запусти бенч", answer="...",
        tool_names=["terminal_exec"], confidence=75,
    ) <= 30

    # Action verb + execute tool → passthrough.
    assert cap_confidence_for_endpoint(
        task="запусти бенч", answer="...",
        tool_names=["start_background_job"], confidence=75,
    ) == 75

    # Non-action question → passthrough.
    assert cap_confidence_for_endpoint(
        task="what is python", answer="...",
        tool_names=[], confidence=85,
    ) == 85


def test_cap_confidence_does_not_inflate():
    """The cap is a floor on the high side: confidence already
    below 30 stays below 30, not bumped up."""
    from backend.endpoint_check import cap_confidence_for_endpoint
    assert cap_confidence_for_endpoint(
        task="запусти бенч", answer="...",
        tool_names=[], confidence=20,
    ) == 20
