"""SearXNG provider tests.

The three scenario tests (success-preferred, transport failure, empty results)
were written by the agent alongside its own `_searxng` patch. They were
extended on 2026-08-06 after mutation testing showed 5 of 6 deliberate
breakages slipping through the original suite — including dropping the
`content` fallback, which would have blanked every snippet in production while
the suite stayed green.

The root cause of that blind spot: the original payloads used a `snippet` key.
A real SearXNG result has no such key (verified against the live instance: the
keys are category/content/engine/engines/img_src/parsed_url/positions/priority/
publishedDate/score/template/thumbnail/title/url). Every payload below is
shaped like the real thing.
"""
import httpx
import pytest

from backend.tools import web_search as ws

SEARX_DEFAULT = "http://127.0.0.1:8888/search"


class FakeResponse:
    def __init__(self, status_code=200, text="", data=None):
        self.status_code = status_code
        self.text = text
        self._data = data if data is not None else {}

    def json(self):
        return self._data


def _result(url, title="t", content="c", **extra):
    """A result entry shaped like real SearXNG output."""
    return {"url": url, "title": title, "content": content,
            "engine": "duckduckgo", "score": 1.0, "category": "general",
            **extra}


@pytest.fixture(autouse=True)
def _no_tavily(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SEARXNG_URL", raising=False)


def _stub_duckduckgo(monkeypatch, counter=None):
    def fake_duckduckgo(query, max_results, attempts=None):
        if counter is not None:
            counter["count"] += 1
        if attempts is not None:
            attempts.append({"provider": "ddg_test", "status": 200,
                             "bytes": 1, "parsed": 1, "reason": ""})
        return [ws.WebResult("DDG", "https://duck.example", "fallback")]

    monkeypatch.setattr(ws, "_duckduckgo", fake_duckduckgo)


# ── the agent's original scenarios ────────────────────────────────────

def test_searxng_success_is_preferred_and_duckduckgo_not_called(monkeypatch):
    duck = {"count": 0}
    body = '{"results":[{"url":"https://example.com"}]}'

    def fake_get(url, params=None, headers=None, timeout=None, **kwargs):
        assert url == SEARX_DEFAULT
        assert params == {"q": "local query", "format": "json"}
        assert headers == {"Accept": "application/json",
                           "User-Agent": ws.BROWSER_UA}
        return FakeResponse(200, body, {"results": [
            _result("https://example.com", title="SearX hit",
                    content="from content"),
            {"title": "No URL", "content": "ignored"},
        ]})

    monkeypatch.setattr(ws.httpx, "get", fake_get)
    _stub_duckduckgo(monkeypatch, duck)

    out = ws.web_search_detailed("local query", max_results=1)

    assert out["note"] == "searxng"
    assert [r.url for r in out["results"]] == ["https://example.com"]
    assert duck["count"] == 0
    assert out["attempts"] == [{
        "provider": "searxng", "status": 200, "bytes": len(body),
        "parsed": 1, "url": SEARX_DEFAULT, "reason": "",
    }]


def test_searxng_failure_falls_back_to_duckduckgo_with_reason(monkeypatch):
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.local/")

    def fake_get(*args, **kwargs):
        raise RuntimeError("searx down")

    monkeypatch.setattr(ws.httpx, "get", fake_get)
    _stub_duckduckgo(monkeypatch)

    out = ws.web_search_detailed("local query", max_results=3)

    assert [r.url for r in out["results"]] == ["https://duck.example"]
    attempt = out["attempts"][0]
    assert attempt["provider"] == "searxng"
    assert attempt["url"] == "http://searxng.local/search"
    assert attempt["status"] is None
    assert attempt["bytes"] == 0
    assert attempt["parsed"] == 0
    assert "RuntimeError" in attempt["reason"]
    assert "searx down" in attempt["reason"]


def test_searxng_empty_results_falls_back_to_duckduckgo_with_reason(monkeypatch):
    monkeypatch.setattr(ws.httpx, "get",
                        lambda *a, **k: FakeResponse(200, '{"results":[]}',
                                                     {"results": []}))
    _stub_duckduckgo(monkeypatch)

    out = ws.web_search_detailed("local query", max_results=3)

    assert [r.url for r in out["results"]] == ["https://duck.example"]
    attempt = out["attempts"][0]
    assert attempt["provider"] == "searxng"
    assert attempt["status"] == 200
    assert attempt["bytes"] == len('{"results":[]}')
    assert attempt["parsed"] == 0
    assert attempt["reason"] == "empty results"


# ── gaps the mutation run exposed ─────────────────────────────────────

def test_snippet_comes_from_the_content_key(monkeypatch):
    """The ONLY key real SearXNG ships. Dropping the `content` fallback used
    to leave every snippet empty with the whole suite still green."""
    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: FakeResponse(
        200, "{}", {"results": [_result("https://example.com",
                                        title="T", content="the real snippet")]}))

    hits = ws._searxng("q", 5)

    assert len(hits) == 1
    assert hits[0].snippet == "the real snippet"
    assert hits[0].title == "T"


def test_max_results_is_honoured(monkeypatch):
    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: FakeResponse(
        200, "{}", {"results": [_result(f"https://e{i}.com") for i in range(6)]}))

    hits = ws._searxng("q", 2)

    assert [h.url for h in hits] == ["https://e0.com", "https://e1.com"]


def test_entries_without_a_url_are_skipped_not_returned_blank(monkeypatch):
    """The url-less entry sits in the middle, away from the truncation
    boundary, so `break` cannot mask a missing skip."""
    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: FakeResponse(
        200, "{}", {"results": [
            _result("https://first.com"),
            {"title": "no url at all", "content": "x"},
            _result("https://third.com"),
        ]}))

    hits = ws._searxng("q", 10)

    assert [h.url for h in hits] == ["https://first.com", "https://third.com"]


@pytest.mark.parametrize("status", [400, 403, 429, 500, 503])
def test_error_statuses_are_reported_not_parsed(monkeypatch, status):
    """A 429 body is an error page, not results. Parsing it used to be a
    surviving mutant because no test covered any 4xx."""
    monkeypatch.setattr(ws.httpx, "get", lambda *a, **k: FakeResponse(
        status, "rate limited", {"results": [_result("https://leaked.com")]}))

    attempts = []
    hits = ws._searxng("q", 5, attempts=attempts)

    assert hits == []
    assert attempts[0]["reason"] == f"http {status}"
    assert attempts[0]["status"] == status


# ── regressions fixed 2026-08-06 ──────────────────────────────────────

def test_connect_timeout_is_short_so_a_dead_instance_cannot_stall_search(
        monkeypatch):
    """Measured on prod: a flat 15s timeout cost 15.02s on EVERY web_search
    when the host DROPped the packet. Connect must give up fast; the read
    budget stays generous because SearXNG proxies slow upstream engines."""
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None, **kwargs):
        seen["timeout"] = timeout
        return FakeResponse(200, "{}", {"results": []})

    monkeypatch.setattr(ws.httpx, "get", fake_get)
    ws._searxng("q", 5)

    timeout = seen["timeout"]
    assert isinstance(timeout, httpx.Timeout)
    assert timeout.connect == 2.0
    assert timeout.read == 10.0


def test_redirects_are_followed(monkeypatch):
    """httpx defaults to follow_redirects=False (unlike requests). Behind a
    reverse proxy or an http->https hop the 3xx body fails to parse and the
    provider silently reports 'no results'."""
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None, **kwargs):
        seen.update(kwargs)
        return FakeResponse(200, "{}", {"results": []})

    monkeypatch.setattr(ws.httpx, "get", fake_get)
    ws._searxng("q", 5)

    assert seen.get("follow_redirects") is True


def test_exhaustion_note_names_the_searxng_failure(monkeypatch):
    """When the self-hosted primary is what broke, the model must be told to
    check it — not sent off to buy a Tavily key."""
    monkeypatch.setenv("SEARXNG_URL", "http://100.64.0.5:8888")

    def fake_get(*a, **k):
        raise httpx.ConnectTimeout("timed out")

    monkeypatch.setattr(ws.httpx, "get", fake_get)
    monkeypatch.setattr(ws, "_duckduckgo",
                        lambda q, n, attempts=None: [])

    out = ws.web_search_detailed("q", max_results=3)

    assert out["results"] == []
    assert "SearXNG" in out["note"]
    assert "http://100.64.0.5:8888/search" in out["note"]
    assert "ConnectTimeout" in out["note"]
