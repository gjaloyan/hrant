"""Machine turns must not evict what the person actually said.

Measured 2026-08-12, from the owner's own conversation.

He asked for OCR model options; the agent researched Hugging Face and
recommended `Graf-J/captcha-conv-transformer-finetuned`. Four turns later he
said "choose a small model yourself" and it chose Qwen2.5-Coder — a CODE
model, to read CAPTCHAs — and he had to write: "you suggest me a 'graf j'
model but choice qwen2.5? why? you forget we need a capcha reading model".

It had not ignored him. The prompt carried `context_block(n=6)`, and by then
those six turns were:

    My choice: Выбери сам небольшую модель
    My choice: Да, продолжай Graf-J
    [background job bg-af877e93c810 completed]
    ...

Its own recommendation was already gone. Each `ask_user` costs TWO slots (the
question and the "My choice: …" reply) and every finished background job costs
another, so a working session evicts itself: the machinery survives and the
conversation does not.
"""
import pytest

from backend.conversation import _is_machine_turn, _keep_the_conversation


def _t(user, intent="task"):
    return {"user": user, "intent": intent, "answer": "ok"}


def test_the_research_turn_survives_a_burst_of_jobs():
    """The exact measured shape."""
    turns = [
        _t("find the best captcha model"),
        _t("My choice: pick one yourself"),
        _t("BACKGROUND_JOB_COMPLETED: bg-1", "supervisor"),
        _t("BACKGROUND_JOB_COMPLETED: bg-2", "supervisor"),
        _t("BACKGROUND_JOB_COMPLETED: bg-3", "supervisor"),
        _t("BACKGROUND_JOB_COMPLETED: bg-4", "supervisor"),
    ]
    kept = [t["user"] for t in _keep_the_conversation(turns, 6)]
    assert "find the best captcha model" in kept


def test_a_few_machine_turns_are_still_kept_for_continuity():
    """A finished job IS context — the fix is priority, not exclusion."""
    turns = [_t("do the thing")] + [
        _t(f"BACKGROUND_JOB_COMPLETED: bg-{i}", "supervisor") for i in range(5)
    ]
    kept = _keep_the_conversation(turns, 6)
    machine = [t for t in kept if _is_machine_turn(t)]
    assert 1 <= len(machine) <= 2


def test_chronological_order_is_preserved():
    turns = [
        _t("first"),
        _t("BACKGROUND_JOB_COMPLETED: bg-1", "supervisor"),
        _t("second"),
    ]
    kept = [t["user"] for t in _keep_the_conversation(turns, 6)]
    assert kept.index("first") < kept.index("second")


def test_a_pure_conversation_is_untouched():
    turns = [_t(f"msg {i}") for i in range(4)]
    assert _keep_the_conversation(turns, 6) == turns


def test_the_newest_human_turns_win_when_over_budget():
    turns = [_t(f"msg {i}") for i in range(20)]
    kept = [t["user"] for t in _keep_the_conversation(turns, 3)]
    assert kept == ["msg 17", "msg 18", "msg 19"]


@pytest.mark.parametrize("turn, machine", [
    ({"user": "hello", "intent": "task"}, False),
    ({"user": "BACKGROUND_JOB_COMPLETED: bg-x", "intent": "task"}, True),
    ({"user": "anything", "intent": "supervisor"}, True),
    ({"user": "My choice: yes", "intent": "task"}, False),
])
def test_machine_turns_are_identified(turn, machine):
    """A `My choice:` reply is the HUMAN — it costs a slot but it is his."""
    assert _is_machine_turn(turn) is machine


def test_empty_and_degenerate_inputs():
    assert _keep_the_conversation([], 6) == []
    assert _keep_the_conversation([_t("x")], 0) == []


def test_the_turn_prompt_asks_for_a_working_window():
    """Six was enough for chat and far too few for work."""
    import inspect
    import backend.unified_agent as ua
    src = inspect.getsource(ua.run_unified)
    assert "context_block(n=10" in src


def test_recent_actually_applies_the_filter(tmp_path):
    """Guard the WIRING, not just the helper.

    The first version of this file tested `_keep_the_conversation` directly
    and stayed green against a mutation that restored the old `turns[-n:]` —
    i.e. it survived the exact bug it exists to prevent. This goes through
    `recent()`, the path the prompt actually uses."""
    from backend.conversation import ConversationMemory

    cm = ConversationMemory(path=tmp_path / "c.json", max_turns=50)
    cm.add_turn("find the best captcha model", "I recommend Graf-J",
                session_key="s1")
    # MORE machine turns than the window, so a naive tail slice really does
    # evict the human turn — otherwise the test passes either way, which is
    # how the first version of it survived its own bug.
    for i in range(9):
        cm.add_turn(f"BACKGROUND_JOB_COMPLETED: bg-{i}", "retry launched",
                    intent="supervisor", session_key="s1")

    got = [t["user"] for t in cm.recent(6, session_key="s1")]
    assert "find the best captcha model" in got, (
        "the human turn was evicted by machine turns")


def test_the_context_block_carries_the_recommendation(tmp_path):
    """End to end: what the prompt receives."""
    from backend.conversation import ConversationMemory

    cm = ConversationMemory(path=tmp_path / "c.json", max_turns=50)
    cm.add_turn("find the best captcha model", "I recommend Graf-J",
                session_key="s1")
    for i in range(9):
        cm.add_turn(f"BACKGROUND_JOB_COMPLETED: bg-{i}", "retry",
                    intent="supervisor", session_key="s1")

    block = cm.context_block(n=6, session_key="s1")
    assert "Graf-J" in block
