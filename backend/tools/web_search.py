"""Веб-поиск: Tavily (если есть ключ), иначе заглушка через DuckDuckGo HTML."""
from __future__ import annotations
import os
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import httpx


@dataclass
class WebResult:
    title: str
    url: str
    snippet: str


def _tavily(query: str, max_results: int) -> list[WebResult]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    try:
        r = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=30.0,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    return [
        WebResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("content", ""),
        )
        for item in data.get("results", [])
    ]


def _unwrap_ddg_url(url: str) -> str:
    """DDG HTML-страница отдаёт ссылки вида //duckduckgo.com/l/?uddg=<encoded>.

    Это редирект-обёртка. Мы хотим конечный URL, иначе последующий fetch_url
    упирается в DDG (rate-limit, лишний хоп, поломанные source-чекеры).
    """
    if not url:
        return url
    # Бывает с протокол-relative: //duckduckgo.com/l/?uddg=...
    if url.startswith("//"):
        url = "https:" + url
    try:
        parsed = urlparse(url)
    except Exception:
        return url
    if "duckduckgo.com" not in parsed.netloc:
        return url
    if not parsed.path.startswith("/l/"):
        return url
    qs = parse_qs(parsed.query)
    target = qs.get("uddg", [""])[0]
    if not target:
        return url
    return unquote(target)


def _duckduckgo(query: str, max_results: int) -> list[WebResult]:
    """Резервный вариант: HTML-парсинг DDG (без ключей)."""
    try:
        r = httpx.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20.0,
            follow_redirects=True,
        )
        r.raise_for_status()
    except Exception:
        return []
    html = r.text
    pattern = re.compile(
        r'<a rel="nofollow" class="result__a" href="([^"]+)">([^<]+)</a>.*?'
        r'<a class="result__snippet"[^>]*>(.*?)</a>',
        re.S,
    )
    out: list[WebResult] = []
    for m in pattern.finditer(html):
        raw_url = re.sub(r"<.*?>", "", m.group(1))
        url = _unwrap_ddg_url(raw_url)
        title = re.sub(r"<.*?>", "", m.group(2)).strip()
        snippet = re.sub(r"<.*?>", "", m.group(3)).strip()
        out.append(WebResult(title=title, url=url, snippet=snippet))
        if len(out) >= max_results:
            break
    return out


def web_search(query: str, max_results: int = 5) -> list[WebResult]:
    results = _tavily(query, max_results)
    if results:
        return results
    return _duckduckgo(query, max_results)


def fetch_url(url: str, max_chars: int = 8000) -> str:
    try:
        r = httpx.get(url, timeout=30.0, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        return f"[fetch error: {e}]"
    text = re.sub(r"<script.*?</script>", " ", r.text, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]
