from backend.tools import web_search as ws


class FakeResponse:
    def __init__(self, status_code=200, text="", data=None):
        self.status_code = status_code
        self.text = text
        self._data = data if data is not None else {}

    def json(self):
        return self._data


def test_searxng_success_is_preferred_and_duckduckgo_not_called(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("SEARXNG_URL", raising=False)
    duck_calls = {"count": 0}

    def fake_get(url, params=None, headers=None, timeout=None, **kwargs):
        assert url == "http://127.0.0.1:8888/search"
        assert params == {"q": "local query", "format": "json"}
        assert headers == {"Accept": "application/json", "User-Agent": ws.BROWSER_UA}
        assert timeout == 15.0
        return FakeResponse(
            200,
            '{"results":[{"title":"SearX hit","url":"https://example.com","snippet":"from snippet"}]}',
            {"results": [
                {"title": "SearX hit", "url": "https://example.com", "snippet": "from snippet"},
                {"title": "No URL", "snippet": "ignored"},
            ]},
        )

    def fake_duckduckgo(query, max_results, attempts=None):
        duck_calls["count"] += 1
        return [ws.WebResult("DDG", "https://duck.example", "")]

    monkeypatch.setattr(ws.httpx, "get", fake_get)
    monkeypatch.setattr(ws, "_duckduckgo", fake_duckduckgo)

    out = ws.web_search_detailed("local query", max_results=1)

    assert out["note"] == "searxng"
    assert [r.url for r in out["results"]] == ["https://example.com"]
    assert duck_calls["count"] == 0
    assert out["attempts"] == [{
        "provider": "searxng",
        "status": 200,
        "bytes": len('{"results":[{"title":"SearX hit","url":"https://example.com","snippet":"from snippet"}]}'),
        "parsed": 1,
        "url": "http://127.0.0.1:8888/search",
        "reason": "",
    }]


def test_searxng_failure_falls_back_to_duckduckgo_with_reason(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setenv("SEARXNG_URL", "http://searxng.local/")

    def fake_get(*args, **kwargs):
        raise RuntimeError("searx down")

    def fake_duckduckgo(query, max_results, attempts=None):
        if attempts is not None:
            attempts.append({"provider": "ddg_test", "status": 200,
                             "bytes": 1, "parsed": 1, "reason": ""})
        return [ws.WebResult("DDG", "https://duck.example", "fallback")]

    monkeypatch.setattr(ws.httpx, "get", fake_get)
    monkeypatch.setattr(ws, "_duckduckgo", fake_duckduckgo)

    out = ws.web_search_detailed("local query", max_results=3)

    assert [r.url for r in out["results"]] == ["https://duck.example"]
    searx_attempt = out["attempts"][0]
    assert searx_attempt["provider"] == "searxng"
    assert searx_attempt["url"] == "http://searxng.local/search"
    assert searx_attempt["status"] is None
    assert searx_attempt["bytes"] == 0
    assert searx_attempt["parsed"] == 0
    assert "RuntimeError" in searx_attempt["reason"]
    assert "searx down" in searx_attempt["reason"]


def test_searxng_empty_results_falls_back_to_duckduckgo_with_reason(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    def fake_get(*args, **kwargs):
        return FakeResponse(200, '{"results":[]}', {"results": []})

    def fake_duckduckgo(query, max_results, attempts=None):
        if attempts is not None:
            attempts.append({"provider": "ddg_test", "status": 200,
                             "bytes": 1, "parsed": 1, "reason": ""})
        return [ws.WebResult("DDG", "https://duck.example", "fallback")]

    monkeypatch.setattr(ws.httpx, "get", fake_get)
    monkeypatch.setattr(ws, "_duckduckgo", fake_duckduckgo)

    out = ws.web_search_detailed("local query", max_results=3)

    assert [r.url for r in out["results"]] == ["https://duck.example"]
    searx_attempt = out["attempts"][0]
    assert searx_attempt["provider"] == "searxng"
    assert searx_attempt["status"] == 200
    assert searx_attempt["bytes"] == len('{"results":[]}')
    assert searx_attempt["parsed"] == 0
    assert searx_attempt["reason"] == "empty results"
