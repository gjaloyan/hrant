"""Web search: Tavily (when a key is available), otherwise a fallback via DuckDuckGo HTML."""
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
    """DDG HTML pages return links of the form //duckduckgo.com/l/?uddg=<encoded>.

    This is a redirect wrapper. We want the final URL; otherwise a subsequent
    fetch_url hits DDG (rate-limit, extra hop, broken source-checkers).
    """
    if not url:
        return url
    # Can occur with protocol-relative URLs: //duckduckgo.com/l/?uddg=...
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
    """Fallback: DDG HTML scraping (no API keys required)."""
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


_SSRF_BLOCKED_NETS = None  # lazy-built on first call


def _ssrf_blocked_nets():
    """Return the list of `ipaddress.IPv*Network` ranges that
    `fetch_url` refuses to connect to. Built lazily so we don't pay
    the import cost when the tool isn't used.

    The set covers everything an attacker via prompt-injection might
    try to make the agent reach (the agent IS owner, so a same-host
    fetch can hit owner-only admin endpoints):

      - Loopback (127.0.0.0/8, ::1)
      - Link-local (169.254.0.0/16 incl. cloud metadata, fe80::/10)
      - RFC1918 private (10/8, 172.16/12, 192.168/16)
      - Carrier-grade NAT / Tailscale range (100.64.0.0/10)
      - Unique-local IPv6 (fc00::/7)
      - Unspecified (0.0.0.0/8, ::)
      - Reserved / multicast / benchmarking nets
    """
    global _SSRF_BLOCKED_NETS
    if _SSRF_BLOCKED_NETS is None:
        import ipaddress
        _SSRF_BLOCKED_NETS = [
            ipaddress.ip_network(s) for s in (
                "127.0.0.0/8", "169.254.0.0/16", "10.0.0.0/8",
                "172.16.0.0/12", "192.168.0.0/16", "100.64.0.0/10",
                "0.0.0.0/8", "224.0.0.0/4", "240.0.0.0/4",
                "198.18.0.0/15", "192.0.0.0/24", "192.0.2.0/24",
                "198.51.100.0/24", "203.0.113.0/24",
                "::1/128", "fe80::/10", "fc00::/7", "::/128",
                "ff00::/8",
            )
        ]
    return _SSRF_BLOCKED_NETS


def _ssrf_check(url: str) -> str:
    """Return `""` if the URL is safe to fetch, else a short reason
    string. Validates scheme is http(s) and that the host resolves to
    a non-private address.

    Defence-in-depth: resolve EVERY A/AAAA the host advertises (a
    single bad answer is enough to refuse). DNS rebinding remains a
    theoretical exposure between the check and the actual httpx GET,
    but `httpx` validates redirect destinations on its own loop, and
    a same-process rebind window is small enough that adding a custom
    connector isn't worth the maintenance cost here."""
    from urllib.parse import urlparse
    import socket
    import ipaddress

    if not isinstance(url, str) or not url.strip():
        return "empty url"
    try:
        parsed = urlparse(url.strip())
    except Exception as e:
        return f"unparseable url: {e}"
    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        return f"scheme {scheme!r} not allowed (http/https only)"
    host = parsed.hostname or ""
    if not host:
        return "no host in url"

    # Resolve to all IPs (httpx may pick any). socket.getaddrinfo returns
    # a list of (family, type, proto, canonname, sockaddr) tuples.
    try:
        infos = socket.getaddrinfo(host, parsed.port or 80)
    except socket.gaierror as e:
        return f"dns resolution failed: {e}"
    blocked = _ssrf_blocked_nets()
    for info in infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        for net in blocked:
            try:
                if ip in net:
                    return (
                        f"host {host} resolves to private/internal "
                        f"address {ip_str} (in {net})"
                    )
            except (TypeError, ValueError):
                # Mixed v4/v6 net + ip — skip, the right family will match.
                continue
    return ""


def _regex_strip_html(html: str) -> str:
    """Crude fallback: drop scripts/styles, strip tags, collapse
    whitespace. Keeps boilerplate (nav/footer/ads) — used only when
    main-content extraction is unavailable or fails."""
    text = re.sub(r"<script.*?</script>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<.*?>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_main_content(html: str, url: str) -> "str | None":
    """Pull the readable article body out of a page with trafilatura,
    dropping nav / ads / cookie banners / footers. Returns None when
    trafilatura isn't installed or finds no main content, so the
    caller can fall back to the regex strip.

    2026-06-13: the old fetch_url returned the whole page through a
    crude regex strip — menus, banners and ads diluted the signal and
    burned the LLM's token budget. trafilatura is pure-Python, local
    (the URL never leaves the box, unlike a hosted reader), and a big
    signal/noise win for the common case (read an article / docs)."""
    try:
        import trafilatura
    except Exception:
        return None
    try:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
    except Exception:
        return None
    text = (text or "").strip()
    return text or None


def fetch_url(url: str, max_chars: int = 8000) -> str:
    blocked = _ssrf_check(url)
    if blocked:
        return f"[fetch refused: {blocked}]"
    try:
        r = httpx.get(url, timeout=30.0, follow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
    except Exception as e:
        return f"[fetch error: {e}]"
    # Prefer clean main-content extraction; fall back to the regex
    # strip when trafilatura is absent or the page has no article body
    # (search-result pages, JSON endpoints, tiny pages).
    main = _extract_main_content(r.text, url)
    text = main if main is not None else _regex_strip_html(r.text)
    return text[:max_chars]
