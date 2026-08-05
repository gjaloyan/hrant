"""Web reading fixes for the Jul-15 incident ("web_search is not effective —
you should read pages like a human").

Measured on the prod box 2026-08-05:
  * the bare "Mozilla/5.0" User-Agent this module sent got HTTP 403 from
    reddit.com and en.wikipedia.org; a real Chrome UA got 200;
  * duckduckgo.com/html intermittently answers HTTP 202 with a CAPTCHA page,
    which raise_for_status() ignores -> 0 parsed -> "[no results]", read by the
    model as "the web has nothing";
  * lite.duckduckgo.com answered 200 with parseable results in the same second;
  * GeckoTerminal yields ~380 extractable chars raw and ~10 000 after a
    headless render.
"""
from __future__ import annotations

import json

from backend.tools import web_search as ws


# ── User-Agent ────────────────────────────────────────────────────────
def test_browser_ua_is_not_the_blocked_bare_string():
    assert ws.BROWSER_UA != "Mozilla/5.0"
    assert "Chrome/" in ws.BROWSER_UA
    assert ws._BROWSER_HEADERS["User-Agent"] == ws.BROWSER_UA


# ── anti-bot / unreadable detection ───────────────────────────────────
def test_detects_captcha_and_block_pages():
    assert ws.looks_like_bot_wall(
        "Please complete the following challenge: Select all squares "
        "containing a duck") is True
    assert ws.looks_like_bot_wall(
        "You've been blocked by network security") is True
    assert ws.looks_like_bot_wall("<p>Bitcoin is a cryptocurrency</p>") is False


def test_js_shell_is_unreadable_but_article_is_not():
    shell = "<html><head><script>var a=1</script></head><body><div id='root'></div></body></html>"
    assert ws.looks_unreadable(200, shell, None) is True
    article = "<html><body><p>" + ("word " * 400) + "</p></body></html>"
    assert ws.looks_unreadable(200, article, "x" * 600) is False


def test_bot_wall_does_not_trigger_a_pointless_render():
    # Rendering cannot beat a challenge page; escalating would burn ~10s.
    wall = "<html><body>Verify you are human</body></html>"
    assert ws.looks_unreadable(200, wall, None) is False


def test_http_error_counts_as_unreadable():
    assert ws.looks_unreadable(403, "<html>nope</html>", None) is True


# ── DDG parsing ───────────────────────────────────────────────────────
def test_bold_title_row_is_not_dropped_and_entities_decode():
    html = (
        '<a rel="nofollow" class="result__a" '
        'href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fex.com">'
        "Python&#x27;s <b>asyncio</b></a>"
        '<a class="result__snippet" href="#">Snip &amp; text</a>'
    )
    out = ws.parse_ddg_html(html, 5)
    assert len(out) == 1
    assert out[0].title == "Python's asyncio"     # <b> row survived, entity decoded
    assert out[0].url == "https://ex.com"          # redirect wrapper unwrapped
    assert out[0].snippet == "Snip & text"


def test_attribute_order_change_still_parses():
    html = ('<a href="https://a.example" class="result__a" rel="nofollow">A</a>')
    assert ws.parse_ddg_html(html, 5)[0].url == "https://a.example"


def test_missing_snippet_does_not_steal_the_next_result_text():
    html = (
        '<a class="result__a" href="https://one.example">One</a>'
        '<a class="result__a" href="https://two.example">Two</a>'
        '<a class="result__snippet" href="#">belongs to two</a>'
    )
    out = ws.parse_ddg_html(html, 5)
    assert [r.title for r in out] == ["One", "Two"]
    assert out[0].snippet == ""                    # not stolen from the next row
    assert out[1].snippet == "belongs to two"


def test_lite_endpoint_parser():
    html = '<a class="result-link" href="https://lite.example">Lite hit</a>'
    out = ws.parse_ddg_lite(html, 5)
    assert out[0].url == "https://lite.example" and out[0].title == "Lite hit"


# ── diagnostics instead of a silent empty list ────────────────────────
def test_empty_search_reports_why_per_provider(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    class _Resp:
        status_code = 202
        text = "Select all squares containing a duck"

    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(ws.httpx, "post", lambda *a, **k: _Resp())

    detail = ws.web_search_detailed("anything", 5)
    assert detail["results"] == []
    providers = {a["provider"] for a in detail["attempts"]}
    assert {"tavily", "ddg_html", "ddg_lite"} <= providers
    assert any("anti-bot" in (a["reason"] or "") for a in detail["attempts"])
    assert "NOT evidence" in detail["note"]


def test_tool_layer_surfaces_diagnostics_not_bare_no_results(monkeypatch):
    import backend.builtin_tools as bt

    monkeypatch.setattr(
        bt, "web_search_detailed",
        lambda q, max_results=5: {
            "results": [],
            "attempts": [{"provider": "ddg_html", "status": 202, "bytes": 14159,
                          "parsed": 0, "reason": "anti-bot challenge page"}],
            "note": "Every search provider failed — this is NOT evidence...",
        },
    )
    monkeypatch.setattr(bt.WEB_CACHE, "get", lambda *a, **k: None)

    out = bt._web_search_handler("anything")
    assert out.strip().startswith("{")           # structured, not "[no results]"
    payload = json.loads(out)
    assert payload["results"] == []
    assert payload["attempts"][0]["reason"] == "anti-bot challenge page"


# ── fetch_url behaviour ───────────────────────────────────────────────
def test_fetch_url_keeps_the_body_on_a_block(monkeypatch):
    class _Resp:
        status_code = 403
        text = "<html><body>You've been blocked by network security</body></html>"

    monkeypatch.setattr(ws, "_ssrf_check", lambda url: "")
    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _Resp())
    out = ws.fetch_url("https://example.com/x")
    assert "blocked" in out.lower()
    assert "ACCESS failure" in out                # tells the model how to read it
    assert "blocked by network security" in out   # body preserved, not discarded


def test_fetch_url_escalates_to_browser_on_js_shell(monkeypatch):
    class _Resp:
        status_code = 200
        text = "<html><head><script>x</script></head><body><div id='root'></div></body></html>"

    monkeypatch.setattr(ws, "_ssrf_check", lambda url: "")
    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(
        ws, "_render_with_browser",
        lambda url, timeout_s=45: "<html><body><p>" + ("real content " * 80) + "</p></body></html>",
    )
    out = ws.fetch_url("https://spa.example")
    assert out.startswith("[rendered via headless browser]")
    assert "real content" in out


def test_fetch_url_does_not_render_a_healthy_page(monkeypatch):
    class _Resp:
        status_code = 200
        text = "<html><body><p>" + ("plain article text " * 60) + "</p></body></html>"

    called = {"n": 0}

    def _render(url, timeout_s=45):
        called["n"] += 1
        return "<html><body>should not be used</body></html>"

    monkeypatch.setattr(ws, "_ssrf_check", lambda url: "")
    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _Resp())
    monkeypatch.setattr(ws, "_render_with_browser", _render)
    out = ws.fetch_url("https://blog.example")
    assert called["n"] == 0                       # fast path stayed fast
    assert "plain article text" in out
