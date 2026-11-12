"""General-purpose agent fixes (2026-11-12) — no hardcoded domain crutches.

Covers:
1. `web_search` recency + publication dates (`_searxng` + `_google_news_rss`)
2. `fetch_url` fast-path for JSON API payloads (no trafilatura error or 45s
   headless browser launch on short JSON responses)
3. Same-turn `save_user_fact` -> `recall_facts` visibility + `threads`
   conversation search in `recall_facts`
4. Scheduled task run status (`runs_so_far`, `last_run`, `last_run_result`)
   and `CHECKS:` completeness instruction in `scheduled_runs.run_prompt`
5. TTS/STT error logging to `provider_error_log` + proactive self-repair
   guidance in `UNRESOLVED AGENT-SIDE FAILURES`
"""
from __future__ import annotations

from datetime import datetime, timezone
from email.utils import format_datetime
import json
from types import SimpleNamespace as NS

import pytest


# ── 1. web_search: publishedDate & recency via Google News RSS ────────

def test_searxng_preserves_published_date_in_snippet(monkeypatch):
    from backend.tools import web_search as ws

    class _Resp:
        status_code = 200
        text = "{}"

        def json(self):
            return {
                "results": [
                    {
                        "url": "https://example.com/a",
                        "title": "Dated article",
                        "content": "Market moved on CPI release.",
                        "publishedDate": "2026-11-12T10:30:00+00:00",
                    },
                ]
            }

    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _Resp())
    hits = ws._searxng("cpi", 5)
    assert len(hits) == 1
    assert hits[0].snippet.startswith("[2026-11-12 10:30]")
    assert "Market moved on CPI release." in hits[0].snippet


def test_web_search_recency_supplements_undated_searxng_with_dated_rss(monkeypatch):
    from backend.tools import web_search as ws

    now_rfc = format_datetime(datetime.now(timezone.utc))
    rss_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>Fed holds rates steady - Reuters</title>
        <link>https://reuters.example/fed</link>
        <pubDate>{now_rfc}</pubDate>
        <source>Reuters</source>
        <description>Federal Reserve kept benchmark rates unchanged.</description>
      </item>
    </channel></rss>"""

    def _fake_get(url, params=None, **kw):
        if "news.google.com" in url:
            assert "when:1d" in (params or {}).get("q", "")
            return NS(status_code=200, text=rss_xml)
        return NS(
            status_code=200,
            text="{}",
            json=lambda: {
                "results": [
                    {
                        "url": "https://static.example/page",
                        "title": "Undated general page",
                        "content": "Evergreen overview.",
                    }
                ]
            },
        )

    monkeypatch.setattr(ws.httpx, "get", _fake_get)
    out = ws.web_search_detailed("Fed rate decision", max_results=5, recency="day")
    urls = [r.url for r in out["results"]]
    assert urls == ["https://reuters.example/fed", "https://static.example/page"]
    assert "Reuters" in out["results"][0].snippet
    assert out["results"][0].title == "Fed holds rates steady"


# ── 2. fetch_url: JSON payload fast-path ──────────────────────────────

def test_fetch_url_returns_short_json_without_trafilatura_or_headless_browser(monkeypatch):
    from backend.tools import web_search as ws

    payload = '{"symbol":"PEPEUSDT","lastPrice":"0.00000371","priceChangePercent":"-2.10"}'

    class _Resp:
        status_code = 200
        text = payload
        headers = {"Content-Type": "application/json; charset=utf-8"}

    monkeypatch.setattr(ws, "_ssrf_check", lambda url: "")
    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(
        ws, "_extract_main_content",
        lambda *a, **k: pytest.fail("trafilatura must not run on JSON"),
    )
    monkeypatch.setattr(
        ws, "_render_with_browser",
        lambda *a, **k: pytest.fail("headless browser must not run on JSON"),
    )
    out = ws.fetch_url("https://api.example.com/ticker")
    assert out == payload


# ── 3. save_user_fact -> recall_facts + conversation thread search ────

def test_save_user_fact_is_immediately_recalled_in_same_turn(tmp_path, monkeypatch):
    from backend import builtin_tools as bt
    from backend import fact_search as fs
    from backend import identity as id_mod
    from backend import roles

    monkeypatch.setattr(fs, "_facts_path", lambda: tmp_path / "memory_facts.jsonl")
    monkeypatch.setattr(fs, "_fact_store_path", lambda: tmp_path / "fact_embeddings.json")
    fs._new_store_for_test()
    monkeypatch.setattr(
        fs.EMBEDDER, "status",
        lambda: {"backend": "disabled", "model": None, "dim": 0},
    )
    monkeypatch.setattr(id_mod.IDENTITY, "add_user_fact", lambda *a, **k: None)

    tok = roles.set_current_speaker("webui:default")
    try:
        saved = json.loads(bt._save_user_fact_handler(
            "about_user", "Verification token ZEBRA-7741 for recall check",
        ))
        assert saved["ok"] is True
        recalled = json.loads(bt._recall_facts_handler(query="ZEBRA-7741"))
    finally:
        roles.reset_current_speaker(tok)

    assert any("ZEBRA-7741" in m["fact"] for m in recalled["matches"])
    assert recalled["matches"][0]["grounding"] == "user_stated"


def test_recall_facts_searches_conversation_threads_with_speaker_isolation(tmp_path, monkeypatch):
    from backend import builtin_tools as bt
    from backend import fact_search as fs
    from backend import roles
    from backend import threads

    store = threads.ThreadStore(root=tmp_path / "threads")
    store.append(
        "telegram:hrant:1:1",
        user="Мой sudo пароль на сервере — SudoPass9912.",
        answer="Понял, принял к сведению.",
        speaker_id="telegram:1",
        ts="2026-10-15 12:00:00",
    )
    store.append(
        "telegram:hrant:2:2",
        user="Other user's secret token.",
        answer="Noted.",
        speaker_id="telegram:2",
        ts="2026-10-15 12:05:00",
    )
    monkeypatch.setattr(threads, "THREADS", store)
    monkeypatch.setattr(fs, "count_facts", lambda: 0)
    monkeypatch.setattr(fs, "search_facts", lambda *a, **k: [])
    monkeypatch.setattr(fs, "search_user_stated_facts", lambda *a, **k: [])

    # Non-owner only sees their own conversation thread.
    monkeypatch.setattr(roles, "is_owner", lambda sid=None: False)
    tok = roles.set_current_speaker("telegram:1")
    try:
        out = json.loads(bt._recall_facts_handler(query="sudo пароль"))
    finally:
        roles.reset_current_speaker(tok)

    assert "conversation_matches" in out
    assert len(out["conversation_matches"]) == 1
    assert "SudoPass9912" in out["conversation_matches"][0]["user"]

    # Another non-owner cannot see telegram:1's thread.
    tok = roles.set_current_speaker("telegram:2")
    try:
        out2 = json.loads(bt._recall_facts_handler(query="sudo пароль"))
    finally:
        roles.reset_current_speaker(tok)
    assert "conversation_matches" not in out2


# ── 4. Scheduled run completeness prompt & execution status ───────────

def test_scheduled_run_prompt_requires_tool_checks_and_forbids_not_checked():
    from backend import scheduled_runs as bg

    p = bg.run_prompt("Check BTC and news.")
    assert "CHECKS:" in p
    assert "not checked" in p


def test_scheduled_task_records_and_lists_run_history(tmp_path, monkeypatch):
    from datetime import timedelta
    from backend import builtin_tools as bt
    from backend import roles
    from backend import scheduled_messages as sm
    import backend.agent as agent_mod

    from backend.config import CONFIG
    monkeypatch.setitem(
        CONFIG._data, "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    monkeypatch.setattr(
        agent_mod.Agent, "run",
        lambda self, task, **kw: NS(answer="[SILENT]"),
    )
    monkeypatch.setattr(sm, "send_to_speaker", lambda *a, **k: (True, ""))

    due = (datetime.now(timezone.utc) - timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    sm.schedule(
        target_speaker="webui:default",
        text="Check system status.",
        due_at=due,
        requested_by="webui:default",
        kind="agent_task",
        repeat="30m",
        meta={"name": "status watch"},
    )
    sm.deliver_due()

    tok = roles.set_current_speaker("webui:default")
    try:
        listed = json.loads(bt._list_scheduled_handler())
    finally:
        roles.reset_current_speaker(tok)

    item = listed["reminders"][0]
    assert item["runs_so_far"] == 1
    assert item["last_run_result"] == "nothing to report"
    assert item["last_run"]


# ── 5. Subsystem error surfacing & proactive self-repair prompt ───────

def test_tts_and_stt_failures_record_to_provider_error_log(tmp_path, monkeypatch):
    from backend import provider_error_log as pel
    from backend import transcriber
    from backend import tts

    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "errors.jsonl")
    tts._report_failure("edge_tts", RuntimeError("403 Invalid response status"))

    tx = transcriber.Transcriber()
    tx._backend = "faster_whisper"
    tx._model = "medium"
    monkeypatch.setattr(tx, "_tx_faster_whisper", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("model crashed")))
    assert tx.transcribe(b"123") is None

    rows = pel.recent_unresolved(within_hours=1)
    assert len(rows) == 2
    providers = {r["provider"] for r in rows}
    assert "tts:edge_tts" in providers
    assert "stt:faster_whisper" in providers


def test_unresolved_failures_prompt_instructs_self_repair():
    import inspect
    from backend import unified_agent

    src = inspect.getsource(unified_agent.run_unified)
    assert "Self-repair first (when talking to the owner):" in src
