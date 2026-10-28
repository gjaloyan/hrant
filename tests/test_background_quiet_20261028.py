"""What reached the owner from background work on 2026-10-27/28, and should not have.

Read from the threads and turn artifacts of 2026-10-28:

* 15:15 — a PEPE check on the fallback model (the subscription quota was
  spent) wrote its whole check and put [SILENT] under it; the owner got both.
* 21:02 — the evening review arrived with its Markdown unrendered and an
  English "ⓘ not verified" line; 05:47 — a PEPE report with "[TURN GATE] NOT
  DONE" under it (tests in test_background_runs_20261027.py).
* 04:46, 05:47 — reports at night, with sound.
* 11:02 — two 20 MB parts of an archive timed out downloading; the agent told
  the owner twice that he had not sent them.
* 11:17 — the 3D-skill install finished; the confirmation it had promised
  went into a session log, never to the phone, as for every job started from
  Telegram since August.
* 08:00 — the night report said the evening review "went out truncated": the
  review had read it cut at 900 characters, unmarked.
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace as NS

import pytest

from backend import scheduled_runs as bg


# ── [SILENT], read the way Hermes reads it ───────────────────────────

@pytest.mark.parametrize("answer", [
    "[SILENT]",
    "[SILENT] nothing new since the last check",
    "Checked Binance and DexScreener: no confirmed signal.\n\n[SILENT]",
    "**[SILENT]**",
    "SILENT",
    "NO_REPLY",
    "",
])
def test_silence_is_recognised(answer):
    assert bg.parse_outcome(answer)[1] is True


def test_a_marker_inside_a_sentence_is_content():
    text = "Normally this would be [SILENT], but PEPE fell 12% in an hour."
    out, silent, _ = bg.parse_outcome(text)
    assert not silent and out == text


# ── no pressure to act, no proof owed ────────────────────────────────

def test_a_background_turn_is_not_pushed_to_act():
    from backend import tool_registry as tr, turn_contract as tc, turn_policy as tp
    reading = NS(advances=False)

    def banners() -> list[str]:
        tr.reset_per_turn_call_cache()
        return [b for b in (tr._update_nudge_state_and_maybe_banner(reading)
                            for _ in range(3 * tr.NUDGE_THRESHOLD)) if b]

    tok = tp.begin_turn()
    try:
        assert banners(), "a chat turn that only reads is nudged"
    finally:
        tp.reset_turn(tok)
    tok = tp.begin_turn(background=True)
    try:
        assert tp.current_policy().mode == "background"
        assert banners() == [], "a run that reports is not told to ask_user"
    finally:
        tp.reset_turn(tok)
    ctok = tc.begin_turn(active=False)
    try:
        assert tc.note_mutation() == "" and tc.render_user_block() == ""
    finally:
        tc.reset_turn(ctok)


# ── rendered like a chat answer, files attached, quiet at night ──────

def _telegram(monkeypatch):
    from backend import channels
    sent = []

    class FakeBotApi:
        async def send_message(self, chat_id, text, **kw):
            sent.append(("message", text, kw))

        async def send_document(self, chat_id, document, **kw):
            sent.append(("document", document.name, kw))

    b = channels.TelegramBot("token", "telegram")
    b._running, b._app, b._loop = True, NS(bot=FakeBotApi()), object()
    monkeypatch.setattr(channels.asyncio, "run_coroutine_threadsafe",
                        lambda coro, loop: asyncio.run(coro))
    return b, sent


def test_an_agent_report_is_rendered_and_its_files_attached(monkeypatch, tmp_path):
    model = tmp_path / "topper.3mf"
    model.write_bytes(b"3mf")
    b, sent = _telegram(monkeypatch)
    assert b.send_text(f"**Готово**: модель.\nMEDIA:{model}", chat_id=1,
                       markdown=True, silent=True)
    kinds = [s[0] for s in sent]
    assert kinds == ["document", "message"]
    _, text, kw = sent[1]
    assert "<b>Готово</b>" in text and "MEDIA:" not in text
    assert kw["parse_mode"] == "HTML"
    assert all(s[2].get("disable_notification") is True for s in sent), "no sound"


def test_plain_sends_stay_plain(monkeypatch):
    b, sent = _telegram(monkeypatch)
    assert b.send_text("Позвонить маме *сегодня*", chat_id=1)
    assert sent == [("message", "Позвонить маме *сегодня*", {})]


def test_a_report_at_night_arrives_without_sound(monkeypatch):
    from backend import follow_up
    from backend import scheduled_messages as sm
    import backend.agent as agent_mod
    monkeypatch.setattr(agent_mod.Agent, "run",
                        lambda self, task, **kw: NS(answer="PEPE −9%"))
    seen = {}
    monkeypatch.setattr(sm, "send_to_speaker",
                        lambda t, text, label="", **kw: seen.update(kw) or (True, ""))
    monkeypatch.setattr(follow_up, "in_quiet_hours", lambda *_a, **_k: True)
    sm.run_agent_task({"id": "x", "target_speaker": "telegram:1", "text": "Check PEPE."})
    assert seen == {"markdown": True, "silent": True}


# ── a file that did not arrive is named, not denied ──────────────────

def test_a_failed_download_is_named_in_the_turn():
    from backend.channels import with_failed_downloads
    text = with_failed_downloads(
        "(see attached file)",
        ["document '3dprint.zip.001' (20.0 MB): Timed out"])
    assert "3dprint.zip.001" in text and "did not send it" in text
    assert text.endswith("(see attached file)")
    assert with_failed_downloads("hi", []) == "hi"


def test_the_handler_reports_failed_downloads():
    import inspect
    from backend import channels
    src = inspect.getsource(channels)
    assert "text = with_failed_downloads(text, failed_media)" in src
    assert 'entry["failed"].extend(failed_media)' in src, "albums too"
    assert "read_timeout=300" in src


# ── a finished job reports back where it was started ────────────────

@pytest.fixture
def jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    from backend.tools import background_jobs as _bg
    monkeypatch.setattr(_bg.STORE, "_root_override", tmp_path / "jobs")
    return _bg


def test_a_finished_job_reports_to_the_chat_it_came_from(jobs, monkeypatch):
    from backend import follow_up, job_supervisor as _jsup
    from backend import scheduled_messages as sm
    from backend.builtin_tools import _complete_supervisor_handler
    job = jobs.start_job(command="true", label="install print3d-designer",
                         original_speaker_id="telegram:1")
    assert job.original_chat_id is None, "how every Telegram job was started"
    deadline = time.time() + 2.0
    while time.time() < deadline and (jobs.STORE.get(job.job_id) or job).status == "running":
        time.sleep(0.05)
    calls = []
    monkeypatch.setattr(sm, "send_to_speaker",
                        lambda t, text, label="", **kw: calls.append((t, text, kw)) or (True, ""))
    monkeypatch.setattr(follow_up, "in_quiet_hours", lambda *_a, **_k: False)
    token = _jsup.set_active_supervisor_job(job.job_id)
    try:
        body = json.loads(_complete_supervisor_handler(
            decision="done", final_message="Skill installed, self-test 8/8.",
            reason="ok"))
    finally:
        _jsup.reset_active_supervisor_job(token)
    assert body["ok"] is True
    assert calls == [("telegram:1", "Skill installed, self-test 8/8.",
                      {"markdown": True, "silent": False})]


# ── the night review sees where a message was cut ────────────────────

def test_the_review_transcript_marks_a_cut(tmp_path):
    from backend.consolidation.review import day_transcript
    from backend.threads import ThreadStore
    store = ThreadStore(root=tmp_path / "t")
    store.append("telegram:hrant:1:1", user="[scheduled task: evening review]",
                 answer="x" * 2593, speaker_id="telegram:1")
    text, _ = day_transcript("2000-01-01", store=store)
    assert "1693 more characters, not shown here" in text


# ── the user's condition, not a stricter one ─────────────────────────

def test_a_task_keeps_the_users_condition():
    from backend.builtin_tools import register_builtin_tools
    from backend.tool_registry import get_registry
    register_builtin_tools()
    d = " ".join(get_registry().tools["schedule_task"].description.split())
    assert "do not make it stricter than they said" in d


# ── owner, 2026-10-28: no "ⓘ not verified" where nothing changed ──────

@pytest.mark.parametrize("code,owes", [
    ("print(17 * 23)", False),
    ("data = open('report.txt').read()", False),
    ("with open(p, 'rb') as f: head = f.read(64)", False),
    ("import shutil; print(shutil.disk_usage('/'))", False),
    ("open('out.txt', 'w').write('x')", True),
    ("Path('model.json').open('wb')", True),
    ("Path('notes.md').write_text('hi')", True),
    ("import os; os.remove('old.stl')", True),
    ("subprocess.run(['orca-slicer', 'model.3mf'])", True),
])
def test_run_python_owes_a_proof_only_for_a_change(code, owes):
    from backend.builtin_tools import register_builtin_tools
    from backend.tool_registry import get_registry
    register_builtin_tools()
    sem = get_registry().tools["run_python"].resolve_semantics({"code": code})
    assert sem.requires_proof is owes


def test_a_waiver_on_a_turn_that_changed_nothing_shows_nothing():
    from backend import turn_contract as tc
    tok = tc.begin_turn()
    try:
        tc.waive_proof("only computed a product")
        assert tc.render_user_block() == ""
        tc.note_mutation()
        tc.waive_proof("could not restart the service to check it")
        assert "not verified" in tc.render_user_block(), "kept where something changed"
    finally:
        tc.reset_turn(tok)
