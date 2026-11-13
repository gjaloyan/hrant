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


def test_search_threads_matches_cross_language_and_inflected_queries(tmp_path):
    from backend import threads

    store = threads.ThreadStore(root=tmp_path / "threads")
    store.append(
        "webui:hrant:webui:default:1",
        user="Вчера настроил локальный PostgreSQL на порту 5433, пароль от тестовой базы: PgPass_KhorVirap_42.",
        answer="Принял.",
        speaker_id="webui:default",
        ts="2026-11-12 10:00:00",
    )

    # English query with 4 tokens where only the technical identifier matches Russian thread
    hits_en = threads.search_threads(
        "PostgreSQL test database password", store=store,
    )
    assert len(hits_en) == 1
    assert "PgPass_KhorVirap_42" in hits_en[0]["user"]

    # Inflected Russian query ("пароля тестовой")
    hits_ru = threads.search_threads("пароля тестовой базы", store=store)
    assert len(hits_ru) == 1
    assert "PgPass_KhorVirap_42" in hits_ru[0]["user"]


def test_rearmed_scheduled_task_exposes_series_and_cancels_by_series_id(tmp_path, monkeypatch):
    from datetime import timedelta
    from backend import builtin_tools as bt
    from backend import roles
    from backend import scheduled_messages as sm
    from backend.config import CONFIG
    import backend.agent as agent_mod

    monkeypatch.setitem(
        CONFIG._data, "knowledge",
        {**CONFIG._data["knowledge"], "base_dir": str(tmp_path)},
    )
    monkeypatch.setattr(
        agent_mod.Agent, "run",
        lambda self, task, **kw: NS(answer="[SILENT]"),
    )
    monkeypatch.setattr(sm, "send_to_speaker", lambda *a, **k: (True, ""))
    monkeypatch.setattr(roles, "is_owner", lambda sid=None: True)

    due = (datetime.now(timezone.utc) - timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    initial = sm.schedule(
        target_speaker="webui:default",
        text="Monitor BTC.",
        due_at=due,
        requested_by="webui:default",
        kind="agent_task",
        repeat="30m",
        meta={"name": "BTC watch"},
    )
    orig_id = initial["id"]
    sm.deliver_due()

    tok = roles.set_current_speaker("webui:default")
    try:
        listed = json.loads(bt._list_scheduled_handler())
        item = listed["reminders"][0]
        assert item["id"] != orig_id
        assert item["series"] == orig_id

        # Calling get_background_job with the scheduled task id returns a helpful hint
        bg_err = json.loads(bt._get_background_job_handler(orig_id))
        assert bg_err["ok"] is False
        assert "list_scheduled" in bg_err.get("hint", "")

        # Cancelling by the original series id cancels the re-armed pending row
        cancelled = json.loads(bt._cancel_scheduled_handler(orig_id))
        assert cancelled["ok"] is True
        after = json.loads(bt._list_scheduled_handler())
        assert after["count"] == 0
    finally:
        roles.reset_current_speaker(tok)


def test_openai_compatible_call_with_tools_bumps_max_tokens_on_reasoning_truncation(monkeypatch):
    from backend.llm import OpenAICompatibleLLM

    llm = OpenAICompatibleLLM({
        "base_url": "https://api.example.com/v1",
        "api_key": "test-key",
        "model": "reasoning-model",
    })
    seen_max_tokens: list[int] = []

    def _fake_post(payload, **kw):
        seen_max_tokens.append(payload.get("max_tokens"))
        if len(seen_max_tokens) == 1:
            return {
                "choices": [{
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning": "long chain of thought...",
                    },
                }],
            }
        return {
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Final answer after token bump.",
                },
            }],
        }

    monkeypatch.setattr(llm, "_post", _fake_post)
    out = llm.complete_with_tools(
        system="sys",
        user="hi",
        tools=[{"name": "noop", "description": "noop", "input_schema": {"type": "object", "properties": {}}}],
        execute_tool=lambda name, args: "{}",
        max_tokens=2000,
    )
    assert out == "Final answer after token bump."
    assert seen_max_tokens == [2000, 6000]


def test_subagent_receives_current_date_and_disables_action_progress_nudge_for_researcher(monkeypatch):
    from backend import turn_policy as tp
    from backend.subagents import dispatch

    seen: dict = {}

    class _FakeRouter:
        def call_with_tools(self, task_type, system, user, **kw):
            seen["system"] = system
            seen["enforce_action_progress"] = tp.current_policy().enforce_action_progress
            return "Verified report."

    monkeypatch.setattr(dispatch, "router", lambda: _FakeRouter())

    parent_tok = tp.begin_turn()
    try:
        res = dispatch.run_subagent("researcher", "Verify recent news.", require_owner=False)
        assert res.ok is True
        assert "Current date and time:" in seen["system"]
        assert "NOT future-dated" in seen["system"]
        assert seen["enforce_action_progress"] is False
        # Parent policy restored afterwards
        assert tp.current_policy().enforce_action_progress is True
    finally:
        tp.reset_turn(parent_tok)


def test_searxng_filters_cjk_spam_and_wikipedia_homepage_on_non_cjk_queries(monkeypatch):
    from backend.tools import web_search as ws

    class _Resp:
        status_code = 200
        text = "{}"

        def json(self):
            return {
                "results": [
                    {
                        "url": "https://www.zhihu.com/question/556839389",
                        "title": "SpaceX 为什么不上市？ - 知乎",
                        "content": "spam",
                    },
                    {
                        "url": "https://www.wikipedia.org/",
                        "title": "Wikipedia",
                        "content": "Wikipedia is a free online encyclopedia.",
                    },
                    {
                        "url": "https://www.cnbc.com/2026/06/12/spacex-ipo.html",
                        "title": "SpaceX debuts on Nasdaq",
                        "content": "SpaceX shares opened on Friday.",
                    },
                ]
            }

    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _Resp())
    hits = ws._searxng("SpaceX IPO Nasdaq", 5)
    assert [h.url for h in hits] == ["https://www.cnbc.com/2026/06/12/spacex-ipo.html"]


def test_google_news_rss_retries_with_compacted_query_when_stuffed_query_returns_empty(monkeypatch):
    from backend.tools import web_search as ws

    now_rfc = format_datetime(datetime.now(timezone.utc))
    rss_empty = """<?xml version="1.0" encoding="UTF-8"?><rss version="2.0"><channel></channel></rss>"""
    rss_hit = f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel>
      <item>
        <title>Anthropic files S-1 for IPO - Reuters</title>
        <link>https://reuters.example/anthropic-ipo</link>
        <pubDate>{now_rfc}</pubDate>
        <source>Reuters</source>
        <description>Anthropic filed for an initial public offering.</description>
      </item>
    </channel></rss>"""

    queries_seen: list[str] = []

    def _fake_get(url, params=None, **kw):
        q = (params or {}).get("q", "")
        queries_seen.append(q)
        if len(queries_seen) == 1:
            return NS(status_code=200, text=rss_empty)
        return NS(status_code=200, text=rss_hit)

    monkeypatch.setattr(ws.httpx, "get", _fake_get)
    hits = ws._google_news_rss(
        "Anthropic IPO filing November 12 2026 Reuters news latest today",
        max_results=5,
        recency="month",
    )
    assert len(queries_seen) == 2
    assert queries_seen[1] == "Anthropic IPO filing Reuters when:30d"
    assert len(hits) == 1
    assert hits[0].title == "Anthropic files S-1 for IPO"


def test_tts_russian_failure_and_timeout_record_and_queue_self_repair(tmp_path, monkeypatch):
    from backend import channels, provider_error_log as pel, roles, scheduled_messages, tts

    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "provider_errors.jsonl")
    monkeypatch.setattr(roles, "primary_owner", lambda: "telegram:1")

    s = tts.Synthesizer()
    s._backend = "edge_tts"
    s._voice = "en-US-AriaNeural"
    s._voice_ru = "ru-RU-SvetlanaNeural"

    def _boom_403(*a, **k):
        raise RuntimeError("403, message='Invalid response status'")

    monkeypatch.setattr(s, "_tx_edge_tts", _boom_403)
    # Russian text on edge_tts uses ru-RU-SvetlanaNeural, so a 403 must NOT
    # be misclassified as "no installed voice can speak Cyrillic".
    assert s.synthesize("Привет, проверка голосовой связи") is None
    assert "no installed voice" not in (s._last_error or "")
    unresolved = pel.recent_unresolved(within_hours=24)
    assert len(unresolved) == 1
    assert unresolved[0]["provider"] == "tts:edge_tts"
    assert unresolved[0]["status_code"] == 403

    # Post-turn failure queues a single one-shot TTS self-repair agent_task.
    channels._queue_tts_self_repair("telegram:1", s._last_error or "")
    channels._queue_tts_self_repair("telegram:1", s._last_error or "")
    pending = [
        r for r in scheduled_messages.list_pending()
        if (r.get("meta") or {}).get("tag") == "auto-repair:tts"
    ]
    assert len(pending) == 1
    assert pending[0]["kind"] == "agent_task"

    # Acknowledging by category/prefix ("tts") resolves the provider error.
    assert pel.acknowledge("tts", resolution="upgraded edge-tts and verified synth") is True
    assert pel.recent_unresolved(within_hours=24) == []


def test_nightly_review_sees_system_errors_and_runs_autonomous_repairs(tmp_path, monkeypatch):
    from backend import learned_rules as lr, provider_error_log as pel, roles, scheduled_messages, threads
    from backend.consolidation import config as ccfg, review
    from backend.threads import ThreadStore

    data_dir = tmp_path / "data"
    (data_dir / "knowledge").mkdir(parents=True)
    monkeypatch.setenv("HRANT_DATA_DIR", str(data_dir))
    store = ThreadStore(root=tmp_path / "threads")
    monkeypatch.setattr(threads, "THREADS", store)
    monkeypatch.setattr(lr, "_path", lambda: tmp_path / "rules.json")
    monkeypatch.setattr(ccfg, "user_md_path", lambda: tmp_path / "missing.md")
    monkeypatch.setattr(ccfg, "profile_path_for_speaker", lambda s: tmp_path / "missing2.md")
    monkeypatch.setattr(roles, "primary_owner", lambda: "telegram:1")
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "provider_errors.jsonl")

    # Seed an already-escalated rule that is broken again tonight.
    rule_text = "In diagnoses and crypto reviews, cover agreed risks and news."
    (tmp_path / "rules.json").write_text(json.dumps({
        "rules": [{
            "text": rule_text,
            "added": "2026-10-30",
            "escalated": "2026-11-01",
            "violations": [
                {"night": "2026-10-31", "what": "omitted news"},
                {"night": "2026-11-01", "what": "omitted macro and news"},
            ],
        }],
        "history": [],
    }), encoding="utf-8")

    # Seed a 2,200-char crypto report whose tail (news/conclusions) lies past 900 chars.
    long_report = ("📊 PEPE/USDT\n" + ("Технический блок. " * 60)
                   + "\n📰 Новости и макро: не проверялись в этом запуске.")
    assert len(long_report) > 1100
    store.append("tg:1", speaker_id="telegram:1", user="проверь PEPE", answer=long_report)

    # Seed an unresolved TTS provider error and recurring tool errors in error_log.jsonl.
    pel.log_provider_error(
        provider="tts:edge_tts", model="", status_code=403,
        message="voice replies are failing (403 from the speech service)",
        context={"category": "tts"},
    )
    err_log = data_dir / "knowledge" / "error_log.jsonl"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    err_log.write_text(
        "\n".join([
            json.dumps({
                "ts": now_str,
                "source": "tool",
                "service": "fetch_url",
                "message": "HTTP 400 Bad Request",
                "tool_args": {"url": "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=PEPEUSDT"},
            }),
            json.dumps({
                "ts": now_str,
                "source": "tool",
                "service": "fetch_url",
                "message": "HTTP 400 Bad Request",
                "tool_args": {"url": "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=PEPEUSDT"},
            }),
        ]) + "\n",
        encoding="utf-8",
    )

    seen_review_user: dict = {}
    seen_repair_prompt: dict = {}

    def _fake_call_json(system, user, *, max_tokens):
        seen_review_user["user"] = user
        return {
            "summary": "PEPE report skipped news again and TTS failed.",
            "issues": [{
                "kind": "ignored_preference",
                "what": "Skipped news and macro check in PEPE review.",
                "evidence": "Новости и макро: не проверялись",
                "breaks_rule": 1,
            }],
            "lessons": [],
            "report": "В обзоре PEPE снова были пропущены новости, а также падал TTS.",
        }

    def _fake_repair_agent(prompt: str, owner: str) -> str:
        seen_repair_prompt["prompt"] = prompt
        seen_repair_prompt["owner"] = owner
        return (
            "- Обновлён скилл crypto-asset-analysis: 1000PEPEUSDT вместо PEPEUSDT и обязательный поиск новостей.\n"
            "- Проверен и восстановлен TTS (edge-tts)."
        )

    monkeypatch.setattr(review, "_call_json", _fake_call_json)
    monkeypatch.setattr(review, "_run_repair_agent", _fake_repair_agent)
    scheduled_texts: list[str] = []
    monkeypatch.setattr(scheduled_messages, "list_pending", lambda: [])
    monkeypatch.setattr(scheduled_messages, "schedule", lambda **kw: scheduled_texts.append(kw["text"]))

    out = review.run_review()
    # 1. Full report (>900 chars) and system/tool errors reached the reviewer.
    assert "Новости и макро: не проверялись" in seen_review_user["user"]
    assert "tts:edge_tts" in seen_review_user["user"]
    assert "fapi.binance.com" in seen_review_user["user"]
    # 2. Repair agent ran with broken rule, provider errors, and recurring tool errors.
    assert seen_repair_prompt["owner"] == "telegram:1"
    assert rule_text in seen_repair_prompt["prompt"]
    assert "fapi.binance.com" in seen_repair_prompt["prompt"]
    assert len(out["repairs"]) == 2
    # 3. Morning report includes 🔧 repair lines.
    assert scheduled_texts
    assert "🔧 Обновлён скилл crypto-asset-analysis" in scheduled_texts[0]


def test_propose_with_diff_excerpts_middle_of_large_file_and_multi_module(tmp_path, monkeypatch):
    from backend import self_modifier as sm

    backend_dir = tmp_path / "backend"
    backend_dir.mkdir(parents=True)
    # Create a 90,000-char file where target function is at char ~45,000 (middle of file)
    padding_top = "# top padding line\n" * 2400
    target_fn = (
        "def _recall_facts_handler(query: str = '', n: int = 8) -> str:\n"
        "    matches = ['old_behavior']\n"
        "    return str(matches)\n"
    )
    padding_bottom = "# bottom padding line\n" * 2400
    large_code = padding_top + target_fn + padding_bottom
    assert len(large_code) > 80000
    (backend_dir / "big_mod.py").write_text(large_code, encoding="utf-8")
    (backend_dir / "helper_mod.py").write_text(
        "def search_user_stated_facts(query: str):\n    return []\n",
        encoding="utf-8",
    )

    mod_instance = sm.SelfModifier(path=tmp_path / "proposals.json")
    mod_instance._backend_dir = backend_dir
    monkeypatch.setattr(sm, "SELF_MODIFIER", mod_instance)
    monkeypatch.setattr(sm, "_fire_proposal_created", lambda p: None)

    captured_prompt: dict = {}

    class _FakeRouter:
        def call_json(self, task_type, system_prompt, user_prompt, **kw):
            captured_prompt["user"] = user_prompt
            return {
                "title": "Fix recall_facts and helper",
                "description": "Include confirmed facts",
                "changes": [
                    {
                        "module": "backend/big_mod.py",
                        "old_code": "    matches = ['old_behavior']",
                        "new_code": "    matches = ['new_behavior']",
                    },
                    {
                        "module": "backend/helper_mod.py",
                        "old_code": "    return []",
                        "new_code": "    return ['confirmed']",
                    },
                ],
                "impact": "Recalls confirmed facts reliably",
                "risk": "low",
            }

    monkeypatch.setattr(sm, "router", lambda: _FakeRouter())

    prop = sm.propose_with_diff(
        description="Fix _recall_facts_handler and search_user_stated_facts",
        module="backend/big_mod.py,backend/helper_mod.py",
        rationale="Both _recall_facts_handler and search_user_stated_facts need updating",
        requester="telegram:1",
    )
    assert prop is not None
    assert prop.has_diff() is True
    assert "def _recall_facts_handler" in captured_prompt["user"]
    assert "def search_user_stated_facts" in captured_prompt["user"]
    diff_html = sm._render_diff_for_telegram(prop)
    assert "new_behavior" in diff_html
    assert "confirmed" in diff_html






def test_cancel_during_agent_task_prevents_mark_sent_and_rearm(tmp_path, monkeypatch):
    """If a repeating agent_task is cancelled while run_agent_task is still in
    flight, deliver_due must not overwrite status='cancelled' back to 'sent'
    and _rearm must not spawn a duplicate occurrence in that series."""
    from backend import scheduled_messages as sm

    ledger = tmp_path / "scheduled_messages.jsonl"
    monkeypatch.setattr(sm, "_path", lambda: ledger)

    row = sm.schedule(
        target_speaker="telegram:848732236",
        text="check market",
        due_at="2020-01-01T00:00:00Z",
        requested_by="telegram:848732236",
        kind="agent_task",
        repeat="30m",
    )

    observed = {}

    def _fake_run(r):
        # Task must already be marked delivering while in flight, and cancelling
        # by id or series must succeed cleanly mid-run.
        in_flight = [x for x in sm._read_all() if x["id"] == r["id"]]
        observed["status_in_flight"] = in_flight[0]["status"]
        observed["cancel_ok"] = sm.cancel(r["id"], by_speaker="telegram:848732236")
        return ("sent", "some report")

    monkeypatch.setattr(sm, "run_agent_task", _fake_run)

    summary = sm.deliver_due()
    assert observed["status_in_flight"] == "delivering"
    assert observed["cancel_ok"] is True
    assert row["id"] not in summary["sent"]
    assert not summary.get("rearmed")

    rows = sm._read_all()
    assert len(rows) == 1
    assert rows[0]["id"] == row["id"]
    assert rows[0]["status"] == "cancelled"

    # Explicit late status mutations and direct _rearm must stay no-ops.
    sm.mark_sent(row["id"])
    sm.mark_failed(row["id"], "late error")
    sm._rearm(row, {}, outcome="sent", last_report="late")
    rows_after = sm._read_all()
    assert len(rows_after) == 1
    assert rows_after[0]["status"] == "cancelled"


def test_list_scheduled_preserves_long_instructions(tmp_path, monkeypatch):
    """list_scheduled must expose full task instructions up to 2000 chars so
    follow-up edits/audits see all constraints instead of a 160-char prefix."""
    import json
    from backend import builtin_tools as bt
    from backend import scheduled_messages as sm
    from backend import roles, settings

    ledger = tmp_path / "scheduled_messages.jsonl"
    monkeypatch.setattr(sm, "_path", lambda: ledger)
    monkeypatch.setattr(roles, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles, "is_owner", lambda sid: True)
    monkeypatch.setattr(settings, "user_timezone", lambda: "UTC")

    long_650 = ("Macro liquidity ETF flows geopolitics SEC EDGAR. " * 15)[:650]
    long_2500 = "X" * 2500

    r1 = sm.schedule(
        target_speaker="telegram:848732236",
        text=long_650,
        due_at="2099-01-01T00:00:00Z",
        requested_by="webui:default",
        kind="agent_task",
        repeat="daily",
    )
    r2 = sm.schedule(
        target_speaker="telegram:848732236",
        text=long_2500,
        due_at="2099-01-02T00:00:00Z",
        requested_by="webui:default",
        kind="agent_task",
    )

    out = json.loads(bt._list_scheduled_handler(horizon_days=100000, scope="all"))
    by_id = {r["id"]: r for r in out["reminders"]}
    assert by_id[r1["id"]]["text"] == long_650
    assert by_id[r1["id"]]["text_truncated"] is False
    assert len(by_id[r2["id"]]["text"]) == 2000
    assert by_id[r2["id"]]["text_truncated"] is True


def test_load_skill_refreshes_after_state_change_and_disk_edit(tmp_path, monkeypatch):
    """State-changing tools (e.g. run_python updating SKILL.md) must invalidate the
    per-turn dedup cache, and load_skill must re-read SKILL.md from disk."""
    from backend import builtin_tools as bt
    from backend import skills as skills_mod
    from backend import tool_registry as tr

    sk_dir = tmp_path / "demo-skill"
    sk_dir.mkdir(parents=True)
    sk_md = sk_dir / "SKILL.md"
    sk_md.write_text(
        "---\nname: demo-skill\ndescription: Demo skill\n---\nVERSION_ONE_BODY\n",
        encoding="utf-8",
    )
    sk = skills_mod._parse_skill_md(sk_md)
    assert sk is not None

    monkeypatch.setattr(skills_mod.SKILLS, "skills", [sk])

    reg = tr.ToolRegistry()
    reg.register_func(
        name="load_skill",
        description="load skill",
        input_schema={"type": "object", "properties": {"name": {"type": "string"}}},
        handler=bt._load_skill_handler,
    )

    def _fake_run_python(code: str = "") -> str:
        sk_md.write_text(
            "---\nname: demo-skill\ndescription: Demo skill\n---\nVERSION_TWO_BODY\n",
            encoding="utf-8",
        )
        return '{"ok": true}'

    reg.register_func(
        name="run_python",
        description="run python",
        input_schema={"type": "object", "properties": {"code": {"type": "string"}}},
        handler=_fake_run_python,
    )

    tr.reset_per_turn_call_cache()
    out1, err1 = reg.execute("load_skill", {"name": "demo-skill"})
    assert err1 is False
    assert "VERSION_ONE_BODY" in out1

    _, err_py = reg.execute("run_python", {"code": "update skill"})
    assert err_py is False

    out2, err2 = reg.execute("load_skill", {"name": "demo-skill"})
    assert err2 is False
    assert "DUPLICATE CALL" not in out2
    assert "VERSION_TWO_BODY" in out2


def test_recall_finds_unembedded_extracted_and_cyrillic_decimal_facts(tmp_path, monkeypatch):
    """search_user_stated_facts and _recall_facts_handler must find unembedded
    MemoryExtractor facts (with Cyrillic inflection & comma-decimal numbers)
    even when semantic search_facts already saturates limit, and _append_log
    must embed immediately when an embedder is active."""
    import json
    from backend import builtin_tools as bt
    from backend import fact_search as fs
    from backend import memory_extractor as me
    from backend import roles

    facts_file = tmp_path / "memory_facts.jsonl"
    store_file = tmp_path / "fact_embeddings.json"
    monkeypatch.setattr(fs, "_facts_path", lambda: facts_file)
    monkeypatch.setattr(fs, "_fact_store_path", lambda: store_file)
    fs._new_store_for_test()

    # 1. With embedder disabled, _append_log still writes the fact unembedded
    monkeypatch.setattr(
        fs.EMBEDDER, "status",
        lambda: {"backend": "disabled", "dim": 0, "model": ""},
    )
    extractor = me.MemoryExtractor(log_path=facts_file)
    extractor._append_log(me.MemoryFact(
        summary="Пользователь полностью вышел из позиции PEPE по цене $0.0000050.",
        triples=[],
        tags=["pepe"],
        category="event",
        source_turn="yes",
        speaker_id="telegram:848732236",
    ))

    # Query uses inflected Cyrillic ("позиция" vs "позиции") and comma decimal ("0,0000050")
    lex_hits = fs.search_user_stated_facts("PEPE позиция $0,0000050", limit=5)
    assert any("0.0000050" in h["summary"] for h in lex_hits)

    # Even if semantic search_facts returns 8 older facts (filling n=8),
    # _recall_facts_handler must still invoke lexical fallback and return the new fact.
    older_semantic = [
        {
            "summary": f"Older PEPE fact #{i}",
            "category": "event",
            "score": 0.72 - 0.01 * i,
            "ts": "2026-11-01 00:00:00",
            "speaker_id": "telegram:848732236",
            "grounding": "inferred",
        }
        for i in range(8)
    ]
    monkeypatch.setattr(fs, "search_facts", lambda *a, **k: older_semantic)
    monkeypatch.setattr(roles, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles, "is_owner", lambda sid: True)

    recalled = json.loads(bt._recall_facts_handler(query="PEPE позиция $0,0000050"))
    recalled_texts = [m["fact"] for m in recalled["matches"]]
    assert any("0.0000050" in txt for txt in recalled_texts)

    # 2. When embedder is enabled, _append_log embeds immediately into fact_embeddings.json
    sub_facts = tmp_path / "sub_facts.jsonl"
    sub_store = tmp_path / "sub_store.json"
    monkeypatch.setattr(fs, "_facts_path", lambda: sub_facts)
    monkeypatch.setattr(fs, "_fact_store_path", lambda: sub_store)
    fs._new_store_for_test()
    monkeypatch.setattr(
        fs.EMBEDDER, "status",
        lambda: {"backend": "test", "dim": 4, "model": "test-embed"},
    )
    monkeypatch.setattr(fs.EMBEDDER, "embed", lambda text: [0.5, 0.5, 0.5, 0.5])

    extractor2 = me.MemoryExtractor(log_path=sub_facts)
    summary = "Entered ETH swing position at 3100."
    extractor2._append_log(me.MemoryFact(
        summary=summary, triples=[], tags=["eth"], category="event", source_turn="t1",
    ))
    assert fs.get_store().has(fs.fact_id(summary)) is True


def test_terminal_exec_accepts_cwd(tmp_path, monkeypatch):
    """terminal_exec must accept an optional cwd argument in schema and forward
    it to run_terminal without unexpected keyword TypeError."""
    import json
    from backend import builtin_tools as bt
    from backend import roles
    from backend.tool_registry import get_registry
    from backend.tools import terminal_exec as tex

    bt.register_builtin_tools()
    tool = get_registry().tools.get("terminal_exec")
    assert tool is not None
    assert "cwd" in tool.input_schema["properties"]

    monkeypatch.setattr(roles, "current_speaker", lambda: "webui:default")
    monkeypatch.setattr(roles, "is_owner", lambda sid: True)

    seen = {}

    def _fake_run(command, *, cwd=None, timeout_seconds=60):
        seen["command"] = command
        seen["cwd"] = cwd
        seen["timeout"] = timeout_seconds
        return tex.TerminalResult(
            ok=True, command=command, exit_code=0, stdout=str(cwd),
            stderr="", truncated=False, elapsed_ms=5, error="",
        )

    monkeypatch.setattr(bt, "run_terminal", _fake_run)
    out = json.loads(bt._terminal_exec_handler(command="pwd", cwd=str(tmp_path)))
    assert out["ok"] is True
    assert seen["cwd"] == str(tmp_path)


def test_split_telegram_html_never_cuts_through_tags():
    """Long answers were sliced at fixed offsets, cutting <i>…</i> pairs;
    Telegram rejected the chunk and the fallback showed raw tags."""
    import re
    from backend.tg_format import (
        markdown_to_telegram_html, split_telegram_html, telegram_html_to_plain,
    )

    para = (
        "🏦 *Макро и ликвидность.* По данным [CNBC](https://www.cnbc.com/x), "
        "инфляция выросла на **0,4%** & ожидания <ФРС> сдвинулись. "
    )
    md = "\n\n".join(para * 3 for _ in range(12)) + "\n\n" + ("*" + "слово " * 900 + "*")
    html = markdown_to_telegram_html(md)
    assert len(html) > 8000

    chunks = split_telegram_html(html, 4000)
    assert len(chunks) >= 3
    tag_re = re.compile(r"<(/?)([a-zA-Z]+)[^>]*>")
    for c in chunks:
        assert len(c) <= 4000
        # every chunk is balanced on its own
        stack = []
        for m in tag_re.finditer(c):
            if m.group(1):
                assert stack and stack[-1] == m.group(2), c[:200]
                stack.pop()
            else:
                stack.append(m.group(2))
        assert stack == []
        # no tag or entity cut in half at chunk edges
        assert not re.search(r"<[^>]*$", c)
        assert not re.search(r"&[#a-zA-Z0-9]*$", c)
    # nothing lost apart from re-opened/closed tags
    plain_all = "".join(telegram_html_to_plain(c) for c in chunks)
    assert plain_all.count("Макро и ликвидность") == 36

    plain = telegram_html_to_plain('<b>x</b> <a href="https://a.b/c">CNBC</a> &amp; &lt;y&gt;')
    assert plain == "x CNBC (https://a.b/c) & <y>"


def test_edge_tts_retries_a_stalled_attempt_and_segments_long_text(monkeypatch):
    """A stalled edge-tts stream used to eat the whole 30s budget, so the
    retry never ran and long replies always lost their voice bubble."""
    import asyncio
    import sys
    import types
    from backend import tts

    calls = []

    class _FakeCommunicate:
        def __init__(self, text, voice, rate="+0%", **kw):
            self.text = text
            calls.append((text, kw))

        async def stream(self):
            # first attempt of the first segment stalls "forever"
            if len(calls) == 1:
                await asyncio.sleep(3600)
            yield {"type": "audio", "data": b"A" * len(self.text)}

    monkeypatch.setitem(sys.modules, "edge_tts", types.SimpleNamespace(Communicate=_FakeCommunicate))
    monkeypatch.setattr(tts, "_EDGE_ATTEMPT_BASE_S", 0.2)
    monkeypatch.setattr(tts, "_EDGE_CHARS_PER_S", 1e9)
    monkeypatch.setattr(tts, "load_config", lambda: {})

    s = tts.Synthesizer()
    s._backend = "edge_tts"
    s._voice = "en-US-AriaNeural"
    s._voice_ru = "ru-RU-SvetlanaNeural"

    text = ("Это предложение про рынок и ликвидность. " * 40).strip()
    audio = s.synthesize(text)
    assert audio, s._last_error
    assert s._last_error is None
    segs = tts._split_for_speech(text, tts._EDGE_SEGMENT_CHARS)
    assert len(segs) >= 3 and all(len(x) <= tts._EDGE_SEGMENT_CHARS for x in segs)
    # one stalled attempt + one call per segment
    assert len(calls) == len(segs) + 1
    assert calls[0][1].get("receive_timeout") == tts._EDGE_RECEIVE_TIMEOUT
    assert tts._edge_total_budget(text) > 30


def test_tts_self_repair_prompt_flags_recurring_failures(tmp_path, monkeypatch):
    from backend import channels, provider_error_log as pel, roles, scheduled_messages

    monkeypatch.setenv("HRANT_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(pel, "_log_path", lambda: tmp_path / "provider_errors.jsonl")
    monkeypatch.setattr(roles, "primary_owner", lambda: "telegram:1")
    monkeypatch.setattr(scheduled_messages, "_path", lambda: tmp_path / "scheduled_messages.jsonl")
    import time as _time
    now = _time.time()
    for offset in (3 * 3600, 60):  # two separate incidents (dedup window is 5 min)
        monkeypatch.setattr(pel, "_now", lambda o=offset: now - o)
        pel.log_provider_error(provider="tts:edge_tts", model="", status_code=504,
                               message="edge_tts timed out", context={"category": "tts"})
    monkeypatch.setattr(pel, "_now", lambda: now)

    channels._queue_tts_self_repair("telegram:1", "edge_tts timed out after 30s", 1000)
    rows = [r for r in scheduled_messages.list_pending()
            if (r.get("meta") or {}).get("tag") == "auto-repair:tts"]
    assert len(rows) == 1
    text = rows[0]["text"]
    assert "NOT a one-off" in text
    assert "1000 characters" in text
    assert "propose_self_modification" in text and "terminal_exec" in text
    assert "run_command" not in text and "propose_code_change" not in text
