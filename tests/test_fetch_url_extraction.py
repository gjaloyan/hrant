"""fetch_url main-content extraction (2026-06-13).

The old fetch_url ran every page through a crude regex strip that
kept nav/footer/ad boilerplate, diluting signal and burning tokens.
trafilatura extracts the readable article body; we fall back to the
regex strip when it's absent or finds no main content.
"""
from __future__ import annotations

import backend.tools.web_search as ws


_PAGE = """<!doctype html><html><head><title>T</title>
<style>.x{color:red}</style><script>var a=1;</script></head>
<body>
<nav>HOME PRODUCTS PRICING LOGIN — cookie banner accept all</nav>
<header>Site-wide promo: SALE 50% OFF subscribe now!!!</header>
<article><h1>The Real Article</h1>
<p>This is the genuine body paragraph that the reader wants,
with the actual information about the topic.</p>
<p>A second substantive paragraph continuing the real content.</p>
</article>
<footer>(c) 2026 · privacy · terms · 200 links of boilerplate</footer>
</body></html>"""


def test_regex_strip_keeps_everything():
    out = ws._regex_strip_html(_PAGE)
    # The crude path keeps boilerplate AND body — proves the contrast.
    assert "The Real Article" in out
    assert "cookie banner" in out
    assert "var a=1" not in out  # scripts dropped
    assert "color:red" not in out  # styles dropped


def test_main_extraction_drops_boilerplate_when_available():
    main = ws._extract_main_content(_PAGE, "https://example.com/post")
    if main is None:
        import pytest
        pytest.skip("trafilatura not installed in this env")
    assert "genuine body paragraph" in main
    assert "second substantive paragraph" in main
    # Boilerplate gone — the whole point.
    assert "cookie banner" not in main
    assert "SALE 50% OFF" not in main
    assert "200 links of boilerplate" not in main


def test_fetch_url_falls_back_when_no_main_content(monkeypatch):
    """A page with no article body (e.g. JSON/search results) ->
    extraction returns None -> regex strip is used, not an empty
    answer."""
    monkeypatch.setattr(ws, "_ssrf_check", lambda u: None)
    monkeypatch.setattr(ws, "_extract_main_content", lambda html, url: None)

    class _Resp:
        status_code = 200
        text = "<body><div>just a tiny page</div></body>"
        def raise_for_status(self): pass

    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _Resp())
    out = ws.fetch_url("https://example.com")
    assert "just a tiny page" in out


def test_fetch_url_uses_main_when_available(monkeypatch):
    monkeypatch.setattr(ws, "_ssrf_check", lambda u: None)
    monkeypatch.setattr(
        ws, "_extract_main_content",
        lambda html, url: "CLEAN ARTICLE BODY ONLY",
    )

    class _Resp:
        status_code = 200
        text = "<body><nav>noise</nav><article>x</article></body>"
        def raise_for_status(self): pass

    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _Resp())
    out = ws.fetch_url("https://example.com")
    assert out == "CLEAN ARTICLE BODY ONLY"
    assert "noise" not in out


def test_fetch_url_respects_ssrf_and_max_chars(monkeypatch):
    monkeypatch.setattr(ws, "_ssrf_check", lambda u: "blocked: localhost")
    assert "fetch refused" in ws.fetch_url("http://localhost")

    monkeypatch.setattr(ws, "_ssrf_check", lambda u: None)
    monkeypatch.setattr(ws, "_extract_main_content", lambda h, u: "A" * 999)

    class _Resp:
        status_code = 200
        text = "<body>x</body>"
        def raise_for_status(self): pass

    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: _Resp())
    assert len(ws.fetch_url("https://example.com", max_chars=100)) == 100
