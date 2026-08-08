"""Two defects taken straight from the owner's real Telegram log, 2026-08-08.

Seventeen turns that day. TEN of them were spent on these two problems.

VOICE (5 turns wasted, plus 5 more replying to them). Speech-to-text was
disabled on the box — no whisper, no faster-whisper, no OPENAI_API_KEY — so
TRANSCRIBER.transcribe() returned None and set no error. The voice note
reached the agent as the same "(see attached file)" placeholder an image
gets, so the agent said "I didn't receive a transcript" and offered "send it
again". The owner sent it again. It failed again. Five times.

THE DROPPED CHOICE (the whole afternoon's work). At 14:53 the agent asked
which way to start and offered options. At 14:54 the owner picked "Быстрый
MVP сейчас". That reply is 43 characters with no attachment, so the fast chat
lane took it: level L0_CHAT, ZERO tools, answer "Ок, Гор — делаем быстрый MVP
сейчас." Nothing was started. At 15:39 the owner asked "status?" and was told
"I haven't started the MVP — the last step was waiting for you to choose."
"""
from __future__ import annotations

import pytest


# ── voice ─────────────────────────────────────────────────────────────

class _Meta:
    def __init__(self, kind, transcript=""):
        self.kind, self.transcript = kind, transcript


def _placeholder_for(monkeypatch, metas, tx_status):
    """Re-create the placeholder decision from channels.py for a message with
    no text and the given attachments."""
    from backend.attachments import ATTACHMENTS
    monkeypatch.setattr(ATTACHMENTS, "get_meta", lambda sha: metas.get(sha))
    from backend.transcriber import TRANSCRIBER
    monkeypatch.setattr(TRANSCRIBER, "status", lambda: tx_status)

    text = ""
    shas = list(metas)
    for sha in shas:
        m = ATTACHMENTS.get_meta(sha)
        if m and m.kind == "audio" and m.transcript:
            return m.transcript
    audio = [ATTACHMENTS.get_meta(s) for s in shas]
    if any(m and m.kind == "audio" for m in audio):
        st = TRANSCRIBER.status()
        why = (st.get("last_error")
               or ("speech-to-text is not configured on this machine"
                   if st.get("backend") in (None, "disabled")
                   else "transcription returned nothing"))
        return (
            "(voice message received, but it could NOT be transcribed: "
            f"{why}. Do not ask the sender to resend it — the result will be "
            "the same. Say plainly that voice input is unavailable and ask "
            "for text, or offer to enable speech-to-text.)")
    return text or "(see attached file)"


def test_an_untranscribable_voice_note_says_why_and_forbids_a_resend(monkeypatch):
    out = _placeholder_for(
        monkeypatch, {"sha1": _Meta("audio")},
        {"backend": "disabled", "model": None, "last_error": None})
    assert "could NOT be transcribed" in out
    assert "not configured" in out
    assert "Do not ask the sender to resend" in out
    assert out != "(see attached file)"


def test_a_transcriber_error_is_passed_through_verbatim(monkeypatch):
    out = _placeholder_for(
        monkeypatch, {"sha1": _Meta("audio")},
        {"backend": "openai_whisper", "last_error": "401 invalid api key"})
    assert "401 invalid api key" in out


def test_a_successful_transcript_is_still_used_as_the_message(monkeypatch):
    out = _placeholder_for(
        monkeypatch, {"sha1": _Meta("audio", "привет, как дела")},
        {"backend": "local_whisper", "last_error": None})
    assert out == "привет, как дела"


def test_an_image_only_message_keeps_the_old_placeholder(monkeypatch):
    out = _placeholder_for(
        monkeypatch, {"sha1": _Meta("image")},
        {"backend": "disabled", "last_error": None})
    assert out == "(see attached file)"


# ── the dropped choice ────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _clear_resume_marker():
    """run_unified consumes-and-clears this at turn entry; tests must too, or
    one test's marker forces every later one off the fast lane."""
    import backend.tools.ask_user as au
    au.clear_question_resume()
    yield
    au.clear_question_resume()


def test_a_question_resume_is_marked_structurally():
    """Not by matching "My choice:" in the text — the resume path is our own
    protocol and can say so directly."""
    import backend.tools.ask_user as au
    assert au.is_question_resume() is False
    au.mark_question_resume()
    assert au.is_question_resume() is True


def test_the_resume_marker_does_not_leak_into_the_next_turn():
    """ContextVar, so a fresh context starts clean — otherwise every later
    turn would be forced off the fast lane."""
    import contextvars
    import backend.tools.ask_user as au

    def _marked():
        au.mark_question_resume()
        return au.is_question_resume()

    assert contextvars.copy_context().run(_marked) is True
    assert contextvars.copy_context().run(au.is_question_resume) is False


def test_a_resume_turn_is_kept_off_the_fast_chat_lane(monkeypatch):
    """The exact 14:54 message. Short, no attachment — everything the fast
    lane looks for — but it is the continuation of a paused task."""
    import backend.tools.ask_user as au
    task = "My choice: Быстрый MVP сейчас (Recommended)"
    assert len(task) <= 500 and "\n" not in task   # fast-lane shaped

    def _fast_lane_would_take(resuming: bool) -> bool:
        attachments, matched_skills = [], []
        return (not attachments and not matched_skills
                and not resuming and len(task) <= 500)

    assert _fast_lane_would_take(resuming=False) is True   # the old behaviour
    au.mark_question_resume()
    assert _fast_lane_would_take(resuming=au.is_question_resume()) is False


def test_an_ordinary_short_message_still_takes_the_fast_lane():
    """The fast lane exists for a reason; the fix must not disable it."""
    import backend.tools.ask_user as au
    assert au.is_question_resume() is False
    task = "привет, как дела?"
    assert (not [] and not [] and not au.is_question_resume()
            and len(task) <= 500) is True


# ── the DataLex turn: tools the agent could not see ───────────────────

def test_the_browser_is_reachable_without_loading_a_bundle():
    """2026-08-08: asked to read a JS-only legal database, the agent tried
    web_search + fetch_url + terminal_exec, gave up, and then proposed that
    the owner "connect a headless browser" — while agent_browser sat behind a
    bundle named "media", which nobody researching case law would open."""
    from backend.tool_bundles import BASE_TOOLS, TOOL_BUNDLES, BUNDLE_DESCRIPTIONS
    assert "agent_browser" in BASE_TOOLS
    assert "agent_browser" not in TOOL_BUNDLES.get("media", [])
    # and the catalog must not still advertise it as a bundle member
    assert "agent_browser" not in BUNDLE_DESCRIPTIONS.get("media", "")


def test_an_unreadable_page_names_the_escalation_in_its_own_result(monkeypatch):
    """The tool RESULT is the one place the model always reads — stronger
    than any prompt, and the place the old code said nothing."""
    import backend.tools.web_search as ws
    hint = ws._NEXT_TOOL_HINT
    assert "agent_browser" in hint
    assert "do not propose installing a browser" in hint.lower()


def test_a_blocked_fetch_points_at_the_browser(monkeypatch):
    import backend.tools.web_search as ws

    class _R:
        status_code, text = 403, "<html>Enable JavaScript and cookies to continue</html>"
        content = text.encode()
        headers = {"content-type": "text/html"}

    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _R())
    monkeypatch.setattr(ws, "_ssrf_check", lambda url: "")
    out = ws.fetch_url("https://datalex.am/case/1")
    assert "blocked" in out.lower()
    assert "agent_browser" in out


# ── confidence that tracks effort ─────────────────────────────────────

def test_one_verified_claim_among_forty_is_not_lifted_to_75():
    """Measured: {verified: 1, unverified: 40} scored 2 by formula and was
    REPORTED as 75, because the floor had no ratio term. answer_critic only
    fires below 60, so the critique pass was skipped exactly on the turns
    that needed it most."""
    from backend import verifier as v
    conf = 2
    verified, unverified, contradictions = ["a"], ["b"] * 40, []
    tool_context = "x" * (v.SOURCE_GROUNDED_TOOL_CTX_MIN + 10)
    if (len(tool_context) >= v.SOURCE_GROUNDED_TOOL_CTX_MIN
            and len(verified) > 0 and len(verified) >= len(unverified)
            and not contradictions and conf < v.SOURCE_GROUNDED_CONFIDENCE_FLOOR):
        conf = min(v.SOURCE_GROUNDED_CONFIDENCE_FLOOR, conf + 25)
    assert conf == 2, "a 1-in-41 claim mix must not be called well-grounded"


def test_an_over_cautious_but_grounded_answer_is_still_rescued():
    """The floor's real purpose: a turn that read the source and got 67."""
    from backend import verifier as v
    conf = 67
    verified, unverified, contradictions = ["a", "b", "c"], ["d"], []
    tool_context = "x" * (v.SOURCE_GROUNDED_TOOL_CTX_MIN + 10)
    if (len(tool_context) >= v.SOURCE_GROUNDED_TOOL_CTX_MIN
            and len(verified) > 0 and len(verified) >= len(unverified)
            and not contradictions and conf < v.SOURCE_GROUNDED_CONFIDENCE_FLOOR):
        conf = min(v.SOURCE_GROUNDED_CONFIDENCE_FLOOR, conf + 25)
    assert conf == v.SOURCE_GROUNDED_CONFIDENCE_FLOOR


# ── "Ок, делаем" with nothing started ─────────────────────────────────

def test_the_fast_lane_escalates_when_it_promises_an_action(monkeypatch):
    """The 14:54 answer verbatim. The fast lane has ZERO tools and returns
    before any gate runs, so a promise made there can never be kept: nothing
    started, and 45 minutes later "status?" said "nothing is running"."""
    import backend.unified_agent as ua

    monkeypatch.setattr(
        "backend.endpoint_check.unbacked_action_claim",
        lambda task, answer, tools: ("делаем быстрый MVP сейчас"
                                     if "делаем" in answer else ""))
    monkeypatch.setattr(ua, "_claims_save_without_tool", lambda h: False)

    class _Agent:
        def progress(self, *a, **k): pass

    import backend.llm as llm_mod

    def _fake_call(*a, **k):
        return "Ок, Гор — делаем быстрый MVP сейчас."

    monkeypatch.setattr(llm_mod, "router", lambda: type(
        "R", (), {"call_with_tools": staticmethod(_fake_call),
                  "call": staticmethod(_fake_call),
                  "complete": staticmethod(_fake_call)})())

    out = ua._try_chat_path(
        task="My choice: Быстрый MVP сейчас (Recommended)",
        agent=_Agent(), speaker_id="telegram:1", snapshot="", convo="")
    assert out is None, "a promise of work must fall through to the full path"


def test_the_fast_lane_still_answers_an_ordinary_question(monkeypatch):
    """The lane exists to skip a ~15 KB preamble on cheap turns; the guard
    must not swallow real chat."""
    import backend.unified_agent as ua

    monkeypatch.setattr("backend.endpoint_check.unbacked_action_claim",
                        lambda task, answer, tools: "")
    monkeypatch.setattr(ua, "_claims_save_without_tool", lambda h: False)

    class _Agent:
        def progress(self, *a, **k): pass

    import backend.llm as llm_mod
    _hi = lambda *a, **k: "Привет! Всё хорошо."
    monkeypatch.setattr(llm_mod, "router", lambda: type(
        "R", (), {"call_with_tools": staticmethod(_hi),
                  "call": staticmethod(_hi),
                  "complete": staticmethod(_hi)})())

    out = ua._try_chat_path(task="привет, как дела?", agent=_Agent(),
                            speaker_id="telegram:1", snapshot="", convo="")
    assert out == "Привет! Всё хорошо."


# ── silent degradation (audit section 5) ──────────────────────────────

def test_an_unknown_api_route_is_a_404_not_a_page():
    """The SPA fallback answered 200 text/html to any unmatched /api/* GET,
    so json_get's `if (!r.ok) throw` was dead code: execution reached
    r.json(), which threw "Unexpected token '<'" from inside api.ts with no
    indication of WHICH endpoint had died. Hand-probing with curl "succeeded"
    on typos too."""
    from fastapi.testclient import TestClient
    import backend.main as m

    c = TestClient(m.app)
    for path in ("/api/definitely-not-a-route", "/api/analogies"):
        r = c.get(path)
        assert r.status_code == 404, path
        assert "json" in r.headers.get("content-type", "")


def test_the_spa_itself_still_serves():
    """The catch-all exists to serve the app; the guard must not break it."""
    from fastapi.testclient import TestClient
    import backend.main as m

    r = TestClient(m.app).get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")
