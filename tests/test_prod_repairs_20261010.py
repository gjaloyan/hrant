"""Repairs from the 2026-10-10 review of the agent's real conversations.

Each block pins one defect that was measured on prod, not imagined:

1. Dead credential took the whole agent down. The ChatGPT login behind the
   Codex provider was invalidated on 2026-10-05; every call came back 401,
   auth was not a fallback class, so the enabled and funded OpenRouter entry
   second in the chain was never tried. Nothing was logged where the agent
   could read it, and nobody was told.
2. A revoked token was never refreshed, because refresh only looked at `exp`.
3. A question turn produced three question cards: the plan-incomplete gate
   re-prompted a turn that was paused on `ask_user`, and the second card
   superseded the first — the one the user was actually shown.
4. Photos vanished after a question: the resumed turn had no attachments,
   and the model was never told the ids its tools needed.
5. The model's private citation syntax leaked into answers.
6. Every Telegram exchange was written to sessions.json twice; the fast lane
   wrote none.
"""
from __future__ import annotations

import json
import sys
import types

import pytest

from backend import llm as _llm
from backend.llm_error_classify import LLMError, _is_auth_failure, _should_fallback


CODEX_INVALIDATED = (
    'Codex Responses API 401 (openai_codex/gpt-5.6-terra): {\n  "error": {\n'
    '    "message": "Your authentication token has been invalidated. Please '
    'try signing in again.",\n    "type": "invalid_request_error",\n'
    '    "code": "token_invalidated",\n    "param": null\n  },\n  "status": 401\n}'
)
CODEX_EXPIRED = (
    'Codex Responses API 401 (openai_codex/gpt-5.6-terra): {"error": '
    '{"message": "Your authentication token has expired. Please try '
    'refreshing it.", "code": "token_expired"}}'
)


# ── 1. a dead credential falls back instead of killing the turn ──────

@pytest.mark.parametrize("msg", [
    CODEX_INVALIDATED,
    CODEX_EXPIRED,
    "OpenRouter API 401 (model='z-ai/glm-5.2'): invalid_request_error: No auth credentials found",
    "Anthropic API 403 (model='x'): permission_error: forbidden",
    "Codex auth: Codex not logged in — run `codex login` to create ~/.codex/auth.json",
    "No API key for OpenAI-compatible provider (model=gpt)",
])
def test_dead_credentials_fall_back(msg):
    err = LLMError(msg)
    assert _is_auth_failure(err)
    assert _should_fallback(err)


@pytest.mark.parametrize("msg", [
    "OpenRouter API 400 (model='x'): invalid_request_error: context length exceeded",
    "LLM returned non-JSON: found 401 matching rows",
    "JSON parse error: Expecting value",
])
def test_ordinary_errors_are_not_auth(msg):
    assert not _is_auth_failure(LLMError(msg))


def test_the_fallback_reason_names_the_login():
    reason = _llm._short_fallback_reason(LLMError(CODEX_INVALIDATED))
    assert "re-authenticate" in reason


class _Prov:
    def __init__(self, pid, *, err=None, answer="ok", calls=None, tool=None):
        self.id = pid
        self.name = pid
        self.model = "m"
        self._err = err
        self._answer = answer
        self._calls = calls if calls is not None else []
        self._tool = tool

    def call(self, task_type, system, user, **kw):
        self._calls.append(self.id)
        if self._err:
            raise LLMError(self._err)
        return self._answer

    def call_with_tools(self, task_type, system, user, tools, execute_tool, **kw):
        self._calls.append(self.id)
        if self._tool:
            execute_tool(self._tool, {})
        if self._err:
            raise LLMError(self._err)
        return self._answer


@pytest.fixture
def error_log(tmp_path, monkeypatch):
    from backend import provider_error_log as pel
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "provider_errors.jsonl")
    alerts = []
    monkeypatch.setattr(pel, "alert_owner_once",
                        lambda key, text, **kw: alerts.append((key, text)) or True)
    return pel, alerts


def test_a_revoked_primary_hands_the_turn_to_the_next_provider(
        monkeypatch, error_log):
    pel, alerts = error_log
    calls = []
    chain = [_Prov("openai-codex-sub", err=CODEX_INVALIDATED, calls=calls),
             _Prov("openrouter", answer="answered by openrouter", calls=calls)]
    monkeypatch.setattr(_llm, "_active_provider_chain", lambda tt: chain)

    out = _llm.DualModelRouter().call(_llm.TaskType.COMPLEX_SOLVING, "s", "u")

    assert out == "answered by openrouter"
    assert calls == ["openai-codex-sub", "openrouter"]
    rows = pel.read_all()
    assert rows and rows[0]["context"]["category"] == "auth"
    assert alerts and alerts[0][0] == "auth:openai-codex-sub"
    assert "codex login" in alerts[0][1]


def test_a_terminal_chain_failure_is_recorded_for_the_next_turn(
        monkeypatch, error_log):
    """`recent_unresolved` feeds the UNRESOLVED AGENT-SIDE FAILURES block.
    It was only written from the legacy tail that never runs, so the live
    chain path left it empty for two months."""
    pel, _alerts = error_log
    chain = [_Prov("only", err="OpenRouter API 400 (model='x'): bad request")]
    monkeypatch.setattr(_llm, "_active_provider_chain", lambda tt: chain)

    with pytest.raises(LLMError):
        _llm.DualModelRouter().call(_llm.TaskType.COMPLEX_SOLVING, "s", "u")

    assert pel.recent_unresolved(within_hours=1)


def test_a_chain_that_runs_out_of_providers_is_recorded(monkeypatch, error_log):
    """Every provider failed with a retryable class (quota) — no single
    failure was terminal on its own, the exhausted chain is."""
    pel, _alerts = error_log
    chain = [_Prov("a", err="Codex subscription quota exhausted (usage_limit_reached)"),
             _Prov("b", err="OpenRouter API 429 (model='x'): rate limit exceeded")]
    monkeypatch.setattr(_llm, "_active_provider_chain", lambda tt: chain)

    with pytest.raises(LLMError):
        _llm.DualModelRouter().call(_llm.TaskType.COMPLEX_SOLVING, "s", "u")

    assert pel.recent_unresolved(within_hours=1)


def test_auth_failover_still_refuses_to_repeat_a_write(monkeypatch, error_log):
    calls, effects = [], []

    def _exec(name, args, *a, **k):
        effects.append(name)
        return "ok", False

    chain = [_Prov("primary", err=CODEX_EXPIRED, calls=calls, tool="save_user_fact"),
             _Prov("secondary", calls=calls, tool="save_user_fact")]
    monkeypatch.setattr(_llm, "_active_provider_chain", lambda tt: chain)

    with pytest.raises(LLMError):
        _llm.DualModelRouter().call_with_tools(
            _llm.TaskType.COMPLEX_SOLVING, "s", "u",
            tools=[{"name": "save_user_fact"}], execute_tool=_exec,
        )
    assert calls == ["primary"]
    assert effects == ["save_user_fact"]


# ── owner alert ───────────────────────────────────────────────────

def test_the_owner_alert_is_sent_once_per_window(tmp_path, monkeypatch):
    from backend import provider_error_log as pel
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "provider_errors.jsonl")
    monkeypatch.setattr(pel, "_primary_owner_chat_id", lambda: 42)
    sent = []
    fake = types.SimpleNamespace(CHANNELS=types.SimpleNamespace(
        deliver_dm=lambda chat_id, text: sent.append((chat_id, text)) or True))
    monkeypatch.setitem(sys.modules, "backend.channels", fake)

    assert pel.alert_owner_once("auth:x", "login dead") is True
    assert pel.alert_owner_once("auth:x", "login dead") is False
    assert pel.alert_owner_once("auth:y", "other provider") is True
    assert [c for c, _ in sent] == [42, 42]


def test_no_alert_outside_the_running_server(tmp_path, monkeypatch):
    """A CLI command or a test must never page the owner."""
    from backend import provider_error_log as pel
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "provider_errors.jsonl")
    monkeypatch.delitem(sys.modules, "backend.channels", raising=False)
    assert pel.alert_owner_once("auth:x", "login dead") is False


def test_an_undelivered_alert_is_retried_later(tmp_path, monkeypatch):
    from backend import provider_error_log as pel
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "provider_errors.jsonl")
    monkeypatch.setattr(pel, "_primary_owner_chat_id", lambda: 42)
    fake = types.SimpleNamespace(CHANNELS=types.SimpleNamespace(
        deliver_dm=lambda chat_id, text: False))
    monkeypatch.setitem(sys.modules, "backend.channels", fake)
    assert pel.alert_owner_once("auth:x", "t") is False
    assert not (tmp_path / "provider_alerts.json").exists()


# ── 2. a revoked token is refreshed on the 401, not on paper ─────────

def _auth_file(tmp_path, *, refresh="r-old"):
    p = tmp_path / "auth.json"
    p.write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {
        "access_token": "a-old", "refresh_token": refresh, "id_token": "i-old",
        "account_id": "acc"}}), encoding="utf-8")
    return p


def test_force_refresh_ignores_exp_and_stores_the_new_tokens(tmp_path, monkeypatch):
    from backend.providers import CodexAuthManager
    m = CodexAuthManager(path=_auth_file(tmp_path))
    monkeypatch.setattr(m, "_refresh", lambda rt: {
        "access_token": "a-new", "refresh_token": "r-new"})

    access, account = m.force_refresh()

    assert (access, account) == ("a-new", "acc")
    stored = json.loads((tmp_path / "auth.json").read_text(encoding="utf-8"))
    assert stored["tokens"]["access_token"] == "a-new"
    assert stored["tokens"]["refresh_token"] == "r-new"
    assert stored["tokens"]["id_token"] == "i-old"


def test_force_refresh_without_a_refresh_token_says_log_in(tmp_path):
    from backend.providers import CodexAuthManager
    m = CodexAuthManager(path=_auth_file(tmp_path, refresh=""))
    with pytest.raises(RuntimeError, match="codex login"):
        m.force_refresh()


class _Resp:
    def __init__(self, status, *, lines=(), body=b""):
        self.status_code = status
        self.request = None
        self._lines = list(lines)
        self._body = body

    def iter_bytes(self):
        yield self._body

    def iter_lines(self):
        yield from self._lines


class _CM:
    def __init__(self, resp):
        self.resp = resp

    def __enter__(self):
        return self.resp

    def __exit__(self, *a):
        return False


_OK_LINES = [
    'data: {"type": "response.output_text.delta", "delta": "hello"}',
    'data: {"type": "response.completed", "response": {"usage": {}}}',
]


@pytest.fixture
def codex(monkeypatch):
    import backend.providers as _prov
    auth = types.SimpleNamespace(refreshed=0, fail=None)

    def _get():
        return ("tok-%d" % auth.refreshed, "acc")

    def _force():
        auth.attempts += 1
        if auth.fail:
            raise RuntimeError(auth.fail)
        auth.refreshed += 1
        return _get()

    auth.attempts = 0
    monkeypatch.setattr(_prov, "CODEX_AUTH", types.SimpleNamespace(
        get_access_token=_get, force_refresh=_force))
    monkeypatch.setattr(_llm.CodexLLM, "_refresh_failure", None)
    monkeypatch.setattr(_llm.time, "sleep", lambda s: None)
    sent = []

    def _install(responses):
        it = iter(responses)

        def _stream(method, url, json=None, headers=None, timeout=None):
            sent.append({"auth": headers.get("Authorization"),
                         "session": headers.get("session_id"),
                         "payload": dict(json)})
            return _CM(next(it))
        monkeypatch.setattr(_llm.httpx, "stream", _stream)
        return sent

    llm = _llm.CodexLLM({"model": "gpt-x"})
    return llm, auth, _install


def test_a_401_refreshes_the_token_and_retries(codex):
    llm, auth, install = codex
    sent = install([_Resp(401, body=CODEX_INVALIDATED.encode()),
                    _Resp(200, lines=_OK_LINES)])

    text, _items, _usage = llm._stream_and_aggregate({"model": "gpt-x"})

    assert text == "hello"
    assert auth.refreshed == 1
    assert [s["auth"] for s in sent] == ["Bearer tok-0", "Bearer tok-1"]


def test_a_refused_refresh_says_log_in_and_falls_back(codex):
    llm, auth, install = codex
    auth.fail = "Codex token refresh failed (401): refresh_token_reused"
    install([_Resp(401, body=CODEX_INVALIDATED.encode())])

    with pytest.raises(LLMError) as err:
        llm._stream_and_aggregate({"model": "gpt-x"})

    assert "codex login" in str(err.value)
    assert _should_fallback(err.value)


def test_a_dead_login_is_not_re_refreshed_on_every_call(codex):
    """Each judge call of a turn would otherwise pay a refresh round trip
    before falling back."""
    llm, auth, install = codex
    auth.fail = "refresh_token_reused"
    install([_Resp(401, body=CODEX_INVALIDATED.encode()),
             _Resp(401, body=CODEX_INVALIDATED.encode())])
    for _ in range(2):
        with pytest.raises(LLMError):
            llm._stream_and_aggregate({"model": "gpt-x"})
    assert auth.attempts == 1


# ── prompt cache key ──────────────────────────────────────────────

def test_requests_with_the_same_instructions_share_a_cache_key(monkeypatch):
    monkeypatch.setattr(_llm.CodexLLM, "_cache_key_supported", True)
    llm = _llm.CodexLLM({"model": "gpt-x"})
    a = llm._build_payload("SOUL " * 100 + "turn one", [], None, None, None)
    b = llm._build_payload("SOUL " * 100 + "turn one", [], None, None, None)
    c = llm._build_payload("a different judge prompt", [], None, None, None)
    assert a["prompt_cache_key"] == b["prompt_cache_key"]
    assert a["prompt_cache_key"] != c["prompt_cache_key"]


def test_a_rejected_cache_key_is_dropped_not_fatal(codex, monkeypatch):
    monkeypatch.setattr(_llm.CodexLLM, "_cache_key_supported", True)
    llm, _auth, install = codex
    sent = install([
        _Resp(400, body=b'{"detail": "Unsupported parameter: prompt_cache_key"}'),
        _Resp(200, lines=_OK_LINES),
    ])
    payload = llm._build_payload("instructions", [], None, None, None)

    text, _i, _u = llm._stream_and_aggregate(payload)

    assert text == "hello"
    assert "prompt_cache_key" in sent[0]["payload"]
    assert "prompt_cache_key" not in sent[1]["payload"]
    assert sent[0]["session"] and sent[1]["session"] is None
    assert _llm.CodexLLM._cache_key_supported is False


# ── 3 + 4. questions keep their files and are asked once ─────────────

@pytest.fixture
def questions(tmp_path, monkeypatch):
    from backend.tools import ask_user as aq
    monkeypatch.setattr(aq.STORE, "_root_override", tmp_path / "questions")
    aq.set_turn_attachments([])
    yield aq
    aq.set_turn_attachments([])


def test_a_question_remembers_the_turns_files(questions):
    aq = questions
    aq.set_turn_attachments(["sha-photo-1", "sha-photo-2"])
    q = aq.create_question(question="Which zone?",
                           options=[{"label": "Kitchen"}, {"label": "Bedroom"}],
                           asker_speaker_id="telegram:1")
    assert aq.STORE.get(q.question_id).attachments == ["sha-photo-1", "sha-photo-2"]
    assert "attached to this message again" in aq._resume_message(q, "Kitchen", "opt-0")


def test_a_question_without_files_says_nothing_about_them(questions):
    aq = questions
    q = aq.create_question(question="Which?", options=[{"label": "A"}, {"label": "B"}])
    assert q.attachments == []
    assert "attached" not in aq._resume_message(q, "A", "opt-0")


def test_questions_stored_before_the_field_still_load(questions):
    aq = questions
    q = aq.PendingQuestion.from_dict({
        "question_id": "q-old", "asked_at": 0.0, "question": "x", "why": "",
        "header": "", "options": [], "multi_select": False,
        "default_option_id": "", "asker_speaker_id": "", "asker_chat_id": None,
        "channel": "telegram"})
    assert q.attachments == []


def test_the_webui_resume_passes_the_files_back(questions, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.models import AgentAnswer, VerificationResult
    aq = questions
    monkeypatch.setattr("backend.api.chat.require_owner_for_writes", lambda *a, **k: None)
    monkeypatch.setattr("backend.api.chat.check_chat_rate", lambda *a, **k: None)
    aq.set_turn_attachments(["sha-room"])
    q = aq.create_question(question="Zone?", options=[
        {"label": "A", "id": "a"}, {"label": "B", "id": "b"}],
        asker_speaker_id="webui:default", channel="webui")
    seen = {}

    def _run(agent, task, project=None, attachments=None, **kw):
        seen["attachments"] = attachments
        seen["task"] = task
        return AgentAnswer(answer="ok", verification=VerificationResult(confidence=85)), "j"

    monkeypatch.setattr("backend.api.chat.run_tracked", _run)
    r = TestClient(app).post("/api/chat/answer-question",
                             json={"question_id": q.question_id, "choice": "a"})
    assert r.status_code == 200
    assert seen["attachments"] == ["sha-room"]


def test_the_telegram_resume_passes_the_files_back():
    """The button-tap runner lives in a closure registered with the bot, so
    pin the wiring in its source."""
    import inspect
    from backend.tools import ask_user as aq
    src = inspect.getsource(aq._register_telegram_callback)
    assert 'getattr(marked, "attachments"' in src


def test_a_paused_turn_is_not_re_prompted(monkeypatch):
    """With plan steps pending, the gate re-prompts — unless the turn ended
    on a question the user is looking at."""
    import backend.unified_agent as ua
    from backend.tools import plan_scratchpad
    monkeypatch.setattr(plan_scratchpad, "unfinished_steps",
                        lambda: [(2, "render the photos"), (3, "send them")])

    tag, corrective = ua._decide_self_correction(
        task="design the room", answer="❓ Which zone?", turn_tools=["ask_user"],
        awaiting_user=True)
    assert (tag, corrective) == ("", "")

    tag, corrective = ua._decide_self_correction(
        task="design the room", answer="done", turn_tools=["ask_user"],
        awaiting_user=False)
    assert tag.startswith("plan-incomplete")


def test_the_correction_loop_tells_the_gate_about_the_question():
    import inspect
    import backend.unified_agent as ua
    src = inspect.getsource(ua.run_unified)
    assert 'awaiting_user=bool(_pending_question_id["v"])' in src
    assert "QUESTION_ALREADY_PENDING" in src


def test_the_model_is_told_what_the_attachments_are_called():
    class _M:
        def __init__(self, kind, sha, filename=""):
            self.kind, self.sha256, self.filename = kind, sha, filename
            self.mime_type = "image/jpeg"
            self.transcript = "hello"

    manifest = _llm._attachment_manifest([
        (_M("image", "abc123", "room.jpg"), b"x"),
        (_M("audio", "voice1"), b"y"),
    ])
    assert "sha256=abc123" in manifest
    assert "room.jpg" in manifest
    assert "voice1" not in manifest
    assert _llm._attachment_manifest([(_M("audio", "v"), b"")]) == ""


def test_codex_content_carries_the_manifest(monkeypatch):
    class _M:
        kind, sha256, filename, mime_type, transcript = (
            "image", "abc123", "", "image/jpeg", "")
    monkeypatch.setattr(_llm, "_resolve_attachments", lambda refs: [(_M(), b"\x89PNG")])
    blocks = _llm.CodexLLM._build_user_content_blocks("design it", ["abc123"])
    texts = [b.get("text", "") for b in blocks if b["type"] == "input_text"]
    assert any("sha256=abc123" in t for t in texts)
    assert blocks[-1]["text"] == "design it"


# ── 5. citation markers ───────────────────────────────────────────

def test_private_citation_markers_are_stripped():
    import backend.unified_agent as ua
    raw = ("Прибор называется **влагомер зерна**. "
           "\ue200cite\ue202turn0search0\ue202turn0search3\ue201")
    assert ua._strip_citation_markers(raw) == "Прибор называется **влагомер зерна**."
    assert ua._strip_citation_markers("plain text") == "plain text"
    assert ua._strip_citation_markers("") == ""


# ── 6. one session row per exchange ───────────────────────────────

def test_the_telegram_bridge_no_longer_writes_its_own_session_row():
    import backend.channels as ch
    assert not hasattr(ch, "_log_channel_message")


def test_the_fast_lane_writes_a_session_row(monkeypatch):
    import backend.unified_agent as ua
    from backend.sessions import SESSIONS
    monkeypatch.setattr(ua, "_try_chat_path", lambda **kw: "FAST_ANSWER_7")
    added = []
    monkeypatch.setattr(SESSIONS, "add_turn",
                        lambda turn, **kw: added.append((turn, kw)))

    class _Agent:
        def __init__(self):
            self._trace, self._llm_calls, self._last_turn_id = [], [], ""

        def progress(self, *a, **kw):
            pass

        def _record_llm_call(self, *a, **kw):
            pass

        def _get_token_usage(self):
            return None

    ua.run_unified(agent=_Agent(), task="hi", project=None, attachments=None,
                   channel="telegram", speaker_id="telegram:77",
                   session_key="telegram:hrant:77:77")

    assert len(added) == 1
    turn, kw = added[0]
    assert turn["answer"] == "FAST_ANSWER_7"
    assert kw["session_key"] == "telegram:hrant:77:77"


def test_the_cache_key_travels_as_a_session_header(codex, monkeypatch):
    """Measured on prod: the body field alone did not move the hit rate;
    the matching session_id header did."""
    monkeypatch.setattr(_llm.CodexLLM, "_cache_key_supported", True)
    llm, _auth, install = codex
    sent = install([_Resp(200, lines=_OK_LINES), _Resp(200, lines=_OK_LINES)])
    for _ in range(2):
        llm._stream_and_aggregate(llm._build_payload("same instructions", [], None, None, None))
    assert sent[0]["session"] and sent[0]["session"] == sent[1]["session"]


# ── 7. reply language: Russian by default, the user chooses ──────────

def _preamble(tmp_path, monkeypatch, *, default, pin=""):
    from pathlib import Path
    from backend.config import CONFIG
    from backend.identity import IdentityManager
    monkeypatch.setitem(CONFIG._data, "response_language", None)
    monkeypatch.setitem(CONFIG._data, "default_response_language", default)
    im = IdentityManager(base_dir=Path(tmp_path))
    if pin:
        im.user_path.parent.mkdir(parents=True, exist_ok=True)
        im.user_path.write_text(
            "# User Profile\n\n## Language\n- " + pin + "\n", encoding="utf-8")
    return im.preamble()


def test_unchosen_language_defaults_to_russian(tmp_path, monkeypatch):
    pre = _preamble(tmp_path, monkeypatch, default="ru")
    assert "Write your final user-facing answer in Russian" in pre
    assert "explicitly asks for another language" in pre
    assert "SAME language the user wrote" not in pre


def test_a_users_own_choice_beats_the_default(tmp_path, monkeypatch):
    pre = _preamble(tmp_path, monkeypatch, default="ru",
                    pin="Respond to the user in Armenian.")
    assert "Respond to the user in Armenian." in pre
    assert "the default here" not in pre


def test_mirror_is_still_the_shipped_default(tmp_path, monkeypatch):
    pre = _preamble(tmp_path, monkeypatch, default=None)
    assert "SAME language the user wrote" in pre


def test_no_rule_in_the_prompt_orders_mirroring_any_more():
    """The rules block and the fast lane used to say "mirror the user's
    language" on their own, contradicting a pinned or default language."""
    import backend.prompt_modules as pm
    import backend.unified_agent as ua
    import inspect
    assert "Mirror the user's language" not in inspect.getsource(pm)
    assert "Match the user's language" not in inspect.getsource(ua)
