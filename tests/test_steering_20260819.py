"""A message sent mid-task must reach the task, not open a second one.

Reported by the owner: he wrote to the agent while a task was executing,
and it opened a new task instead of taking the correction into the
running one. The withdrawn instruction kept executing beside its own
correction.

The design decision under test is that nothing is classified. A mid-turn
message may be a correction, an addition, or an unrelated request, and no
text rule separates them — every attempt is keyword routing wearing a
hat. All of them are handed to the running turn verbatim; the model
decides. What the code must guarantee is narrower and checkable: the
message reaches the turn, it reaches the right turn, and it is never
silently dropped.
"""
import pytest

from backend import steering


@pytest.fixture(autouse=True)
def _clean_registry():
    """Module state is process-global; a leak between tests would let one
    test's turn answer another's message."""
    steering._queues.clear()
    steering._owners.clear()
    steering._orphans.clear()
    yield
    steering._queues.clear()
    steering._owners.clear()
    steering._orphans.clear()


# ── routing to the running turn ─────────────────────────────────────

def test_a_message_finds_the_turn_already_serving_that_speaker():
    steering.register_turn("job-1", "tg:gor")
    assert steering.active_job_for("tg:gor") == "job-1"


def test_no_running_turn_means_no_steering():
    """Then the caller must start a normal turn — the default path."""
    assert steering.active_job_for("tg:gor") is None
    assert steering.enqueue("", "hello") is False


def test_speakers_cannot_steer_each_others_work():
    steering.register_turn("job-1", "tg:gor")
    steering.register_turn("job-2", "tg:someone-else")
    assert steering.active_job_for("tg:someone-else") == "job-2"
    assert steering.active_job_for("tg:gor") == "job-1"


def test_a_closed_turn_stops_accepting():
    """Otherwise a message parks against a job that will never read it."""
    steering.register_turn("job-1", "tg:gor")
    steering.close_turn("job-1")
    assert steering.active_job_for("tg:gor") is None
    assert steering.enqueue("job-1", "too late") is False


# ── delivery into the running turn ──────────────────────────────────

def test_the_turn_receives_what_was_sent():
    steering.register_turn("job-1", "tg:gor")
    steering.enqueue("job-1", "no, the second case", speaker_id="tg:gor")
    got = steering.take("job-1")
    assert [m.text for m in got] == ["no, the second case"]


def test_a_message_is_delivered_once():
    """Re-showing it every tool result would drown the turn in its own
    backlog and read as the user repeating themselves."""
    steering.register_turn("job-1", "tg:gor")
    steering.enqueue("job-1", "stop")
    assert len(steering.take("job-1")) == 1
    assert steering.take("job-1") == []


def test_messages_keep_their_order():
    steering.register_turn("job-1", "tg:gor")
    for t in ("first", "second", "third"):
        steering.enqueue("job-1", t)
    assert [m.text for m in steering.take("job-1")] == [
        "first", "second", "third"]


def test_a_later_message_still_gets_through():
    steering.register_turn("job-1", "tg:gor")
    steering.enqueue("job-1", "one")
    steering.take("job-1")
    steering.enqueue("job-1", "two")
    assert [m.text for m in steering.take("job-1")] == ["two"]


def test_empty_text_is_not_a_steer():
    steering.register_turn("job-1", "tg:gor")
    assert steering.enqueue("job-1", "   ") is False


def test_the_backlog_is_bounded():
    """A burst must not grow the tool result without limit."""
    steering.register_turn("job-1", "tg:gor")
    accepted = [steering.enqueue("job-1", f"m{i}")
                for i in range(steering.MAX_PENDING_PER_JOB + 4)]
    assert accepted.count(True) == steering.MAX_PENDING_PER_JOB
    assert accepted[-1] is False, (
        "a refusal must be visible so the caller starts a normal turn "
        "instead of dropping the message")


# ── nothing is silently dropped ─────────────────────────────────────

def test_a_message_the_turn_never_read_is_handed_back():
    """The user was told it went to the running task. If that turn ended
    without reading it, it is owed a turn of its own."""
    steering.register_turn("job-1", "tg:gor")
    steering.enqueue("job-1", "unread", speaker_id="tg:gor")
    left = steering.close_turn("job-1")
    assert [m.text for m in left] == ["unread"]
    assert [m.text for m in steering.pop_orphans("tg:gor")] == ["unread"]


def test_what_the_turn_did_read_is_not_handed_back():
    """Re-dispatching a delivered steer would run it twice."""
    steering.register_turn("job-1", "tg:gor")
    steering.enqueue("job-1", "seen", speaker_id="tg:gor")
    steering.take("job-1")
    assert steering.close_turn("job-1") == []
    assert steering.pop_orphans("tg:gor") == []


def test_orphans_are_collected_once():
    steering.register_turn("job-1", "tg:gor")
    steering.enqueue("job-1", "unread", speaker_id="tg:gor")
    steering.close_turn("job-1")
    assert len(steering.pop_orphans("tg:gor")) == 1
    assert steering.pop_orphans("tg:gor") == []


def test_orphans_are_bounded():
    for i in range(steering.MAX_ORPHANS_PER_SPEAKER + 5):
        steering.register_turn(f"job-{i}", "tg:gor")
        steering.enqueue(f"job-{i}", f"m{i}", speaker_id="tg:gor")
        steering.close_turn(f"job-{i}")
    kept = steering.pop_orphans("tg:gor")
    assert len(kept) == steering.MAX_ORPHANS_PER_SPEAKER
    assert kept[-1].text.endswith(str(steering.MAX_ORPHANS_PER_SPEAKER + 4)), \
        "the newest are the ones still worth answering"


# ── what the model is shown ─────────────────────────────────────────

def test_the_marker_quotes_the_user_verbatim():
    m = [steering.SteeringMessage(text="no, the second case")]
    assert "no, the second case" in steering.render_marker(m)


def test_the_marker_says_no_second_turn_is_coming():
    """Without this the model defers, assuming another turn will handle
    it — which is exactly the behaviour being removed."""
    out = steering.render_marker(
        [steering.SteeringMessage(text="x")]).lower()
    assert "no second turn" in out


def test_the_marker_does_not_decide_what_the_message_is():
    """It must offer the readings, not assert one. Asserting 'this is a
    correction' is the classification this module refuses to make."""
    out = steering.render_marker([steering.SteeringMessage(text="x")])
    low = out.lower()
    for reading in ("a correction", "extra information", "a separate request"):
        assert reading in low
    assert "the user has corrected you" not in low


def test_the_marker_forbids_carrying_on_regardless():
    out = steering.render_marker([steering.SteeringMessage(text="x")]).lower()
    assert "silently" in out and "acknowledge" in out


def test_no_marker_when_nothing_arrived():
    """The common case, on every tool result of every turn."""
    assert steering.render_marker([]) == ""


# ── wiring ──────────────────────────────────────────────────────────

def test_the_turn_registers_and_releases_itself():
    """A leak here would make every later message park against a dead job."""
    import inspect
    from backend import job_runner
    src = inspect.getsource(job_runner.run_tracked)
    assert "register_turn" in src
    assert "close_turn" in src


def test_the_agent_loop_reads_the_queue():
    import inspect
    from backend import unified_agent
    src = inspect.getsource(unified_agent)
    assert "_steer.take(" in src, "the marker never reaches the model"


def test_the_channel_parks_instead_of_opening_a_second_job():
    import inspect
    from backend import channels
    src = inspect.getsource(channels)
    assert "active_job_for" in src and "pop_orphans" in src


# ── the steer that lands after the last tool call ───────────────────

def test_a_steer_after_the_final_tool_call_still_turns_the_turn_around():
    """Measured on the first live test. The turn made two tool calls, the
    correction arrived after the second, there was never a third, and the
    answer the user had just withdrawn was delivered in full.

    The tool-result injection point cannot cover this by construction:
    nothing else is going to be appended to. The correction round is the
    last place a turn that has stopped calling tools can be turned around.
    """
    from backend.unified_agent import _decide_self_correction
    steering.register_turn("job-9", "webui:default")
    steering.enqueue("job-9", "stop, not the file list -- give me the total")
    tag, corrective = _decide_self_correction(
        task="list the ten biggest files",
        answer="| 1 | builtin_tools.py | 206803 |",
        turn_tools=["terminal_exec"],
        job_id="job-9",
    )
    assert tag == "user-steer"
    assert "not the file list" in corrective
    assert "do not send that draft unchanged" in corrective.lower()


def test_the_steer_outranks_a_policy_that_disables_correction(monkeypatch):
    """A turn policy may switch the structural gates off. The user speaking
    is not one of those gates and must not be switched off with them."""
    import backend.unified_agent as ua
    from backend.unified_agent import _decide_self_correction

    class _Policy:
        enforce_action_progress = False

    monkeypatch.setattr("backend.turn_policy.current_policy",
                        lambda: _Policy())
    steering.register_turn("job-10", "webui:default")
    steering.enqueue("job-10", "actually, do the other thing")
    tag, corrective = _decide_self_correction(
        task="t", answer="a", turn_tools=[], job_id="job-10")
    assert tag == "user-steer"
    assert "do the other thing" in corrective


def test_a_steer_is_consumed_by_the_correction_round():
    """Otherwise the second round re-fires on the same message and the turn
    loops on a correction it has already taken."""
    from backend.unified_agent import _decide_self_correction
    steering.register_turn("job-11", "webui:default")
    steering.enqueue("job-11", "change of plan")
    _decide_self_correction(task="t", answer="a", turn_tools=[],
                            job_id="job-11")
    assert steering.has_pending("job-11") is False


def test_no_steer_means_the_normal_gates_still_decide():
    """The new branch must be invisible when nobody wrote anything."""
    from backend.unified_agent import _decide_self_correction
    steering.register_turn("job-12", "webui:default")
    tag, _ = _decide_self_correction(
        task="what is 2+2", answer="4", turn_tools=[], job_id="job-12")
    assert tag != "user-steer"
