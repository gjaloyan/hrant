"""Search could always ask for recent results. It never did.

`_searxng` sent `{"q": ..., "format": "json"}` and nothing else, while
the instance honours `time_range` — verified live 2026-09-04 on the same
query: no filter gave 67 results led by documentation pages, `month` gave
2026 resource lists, `day` gave repositories published that day.

This is the one thing worth taking from `last30days-skill` outright. Its
other machinery — entity resolution into handles and subreddits,
engagement ranking, cross-platform clustering — needs per-platform API
keys and a great deal of code. A time window needs one parameter.

Who decides stays the model: `recency` is an argument on the tool, not a
rule about which questions deserve fresh results.
"""
from unittest.mock import patch

from backend.tools import web_search as ws


class _Resp:
    status_code = 200
    content = b"{}"

    def json(self):
        return {"results": []}


def _params_sent(**kwargs):
    seen = {}

    def _get(url, params=None, headers=None, timeout=None, **kw):
        seen.update(params or {})
        return _Resp()

    with patch.object(ws.httpx, "get", _get):
        ws._searxng("тест", 5, **kwargs)
    return seen


def test_no_recency_asked_means_no_filter_sent():
    """Unchanged behaviour when the model does not ask for it."""
    assert "time_range" not in _params_sent()


def test_a_recency_window_reaches_searxng():
    assert _params_sent(recency="month")["time_range"] == "month"


def test_only_windows_searxng_understands_are_forwarded():
    """SearXNG takes day/week/month/year. Anything else is dropped rather
    than passed through to be silently ignored or to error."""
    for bad in ("hour", "last 30 days", "30d", "", None):
        assert "time_range" not in _params_sent(recency=bad)


def test_the_handler_passes_the_window_through():
    """The tool argument has to reach the provider, or it is decoration."""
    seen = {}

    def _detailed(query, max_results=5, recency=None):
        seen["recency"] = recency
        return {"results": [], "attempts": [], "note": ""}

    from backend import builtin_tools as bt
    with patch.object(bt, "web_search_detailed", _detailed):
        bt._web_search_handler("что нового", max_results=3, recency="week")
    assert seen["recency"] == "week"


def test_the_cache_does_not_serve_a_stale_window():
    """Same query, different window, different answer — the cache key has
    to include it or "this week" returns whatever "any time" cached."""
    from backend import builtin_tools as bt
    calls = []

    def _detailed(query, max_results=5, recency=None):
        calls.append(recency)
        return {"results": [], "attempts": [], "note": ""}

    with patch.object(bt, "web_search_detailed", _detailed):
        bt._web_search_handler("одинаковый запрос", recency=None)
        bt._web_search_handler("одинаковый запрос", recency="day")
    assert calls == [None, "day"]
