"""Web search: Tavily (when a key is available), otherwise a fallback via DuckDuckGo HTML."""
from __future__ import annotations
import html as _html
import logging
import os
import re
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlparse

import httpx

log = logging.getLogger(__name__)

# A REAL browser User-Agent. The bare "Mozilla/5.0" this module used to send is
# on every anti-bot blocklist: measured 2026-08-05 from the prod box,
# reddit.com and en.wikipedia.org answer 403 to it and 200 to the string below.
# That one header was half of the "the agent cannot read the web" complaint.
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)
_BROWSER_HEADERS = {
    "User-Agent": BROWSER_UA,
    "Accept": ("text/html,application/xhtml+xml,application/xml;q=0.9,"
               "image/avif,image/webp,*/*;q=0.8"),
    "Accept-Language": "en-US,en;q=0.9,ru;q=0.8",
}


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


_BOT_WALL_MARKERS: tuple[str, ...] = (
    "prove your humanity", "you've been blocked by network security",
    "enable javascript and cookies to continue", "checking your browser",
    "verify you are human", "select all squares containing",
    "unusual traffic from your computer", "access denied",
    "cf-browser-verification", "captcha",
)


def looks_like_bot_wall(text: str) -> bool:
    """True when a 2xx body is actually an anti-bot challenge, not content.

    DuckDuckGo answers HTTP 202 with a CAPTCHA page under load — measured
    live 2026-08-05: 14 KB of "Select all squares containing a duck".
    `raise_for_status()` does not raise on 202, the result regex matched
    nothing, and the model was told "[no results]", i.e. "the web has
    nothing" — the exact Jul-15 failure. Detect it and say so instead."""
    low = (text or "")[:4000].lower()
    return any(m in low for m in _BOT_WALL_MARKERS)


def visible_text_ratio(html: str) -> float:
    """Fraction of the payload that is human-visible text. A JS shell is
    almost all markup and script: measured GeckoTerminal raw at ~0.005
    (3305 visible chars of 8441 bytes was reddit's shell), real articles
    sit well above 0.05."""
    if not html:
        return 0.0
    stripped = _regex_strip_html(html)
    return len(stripped) / max(len(html), 1)


def looks_unreadable(status: int, html: str, extracted: "str | None") -> bool:
    """True when the cheap HTTP path produced nothing a human could read,
    so the caller should pay for a headless render.

    Calibrated on live measurements (2026-08-05): a JS-only shell yields a
    tiny extraction and a near-zero visible ratio; a real article yields
    either a healthy extraction or plenty of visible text."""
    if status >= 400:
        return True
    if looks_like_bot_wall(html):
        return False  # a render will not beat a bot wall — report it honestly
    if extracted and len(extracted.strip()) >= 400:
        return False
    return len(_regex_strip_html(html or "")) < 600


def _searxng(query: str, max_results: int,
             attempts: list | None = None) -> list[WebResult]:
    """Self-hosted SearXNG — the preferred source: local, key-free, and it
    aggregates several engines at once.

    Deployed on this box 2026-08-05 (docker, port 8888). It matters because the
    engines this box can reach directly are exactly the ones that block us:
    SearXNG reports duckduckgo CAPTCHA and brave rate-limited while still
    returning 20+ results through the engines that do answer. Disabled by
    simply not setting SEARXNG_URL / not running the container — the attempt is
    recorded either way so an empty result set can explain itself."""
    base = (os.getenv("SEARXNG_URL") or "http://127.0.0.1:8888").rstrip("/")
    rec = {"provider": "searxng", "status": None, "bytes": 0, "parsed": 0,
           "reason": ""}
    try:
        r = httpx.get(f"{base}/search",
                      params={"q": query, "format": "json"},
                      headers=_BROWSER_HEADERS, timeout=20.0,
                      follow_redirects=True)
    except Exception as e:
        rec["reason"] = f"not reachable: {type(e).__name__}"
        if attempts is not None:
            attempts.append(rec)
        return []
    rec["status"] = r.status_code
    rec["bytes"] = len(r.text or "")
    if r.status_code >= 400:
        # A 403 here usually means `formats: [html, json]` is missing from
        # settings.yml — JSON output is off by default in SearXNG.
        rec["reason"] = (f"http {r.status_code}"
                         + (" (is `formats: [html, json]` set in settings.yml?)"
                            if r.status_code == 403 else ""))
        if attempts is not None:
            attempts.append(rec)
        return []
    try:
        data = r.json()
    except Exception:
        rec["reason"] = "response was not JSON"
        if attempts is not None:
            attempts.append(rec)
        return []
    out = [
        WebResult(title=(item.get("title") or "").strip(),
                  url=(item.get("url") or "").strip(),
                  snippet=(item.get("content") or "").strip())
        for item in (data.get("results") or [])
        if item.get("url")
    ][:max_results]
    rec["parsed"] = len(out)
    if not out:
        rec["reason"] = "0 results"
        dead = data.get("unresponsive_engines") or []
        if dead:
            rec["reason"] += f"; engines down: {dead[:3]}"
    if attempts is not None:
        attempts.append(rec)
    return out


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


def _clean(fragment: str) -> str:
    """Tag-strip + HTML-entity decode. Without the decode, titles reached
    the model as `Python&#x27;s asyncio` (measured live 2026-08-05)."""
    return _html.unescape(re.sub(r"<.*?>", "", fragment or "")).strip()


def parse_ddg_html(html: str, max_results: int) -> list[WebResult]:
    """Parse the classic DDG HTML result page.

    Attribute-order independent, and the title body accepts markup (`<b>`
    highlighting used to make the old `[^<]+` title pattern DROP whole rows).
    Snippet is optional so a result without one is still returned."""
    out: list[WebResult] = []
    # Match the whole anchor, then pull class/href out of the attribute blob —
    # a positional pattern breaks the moment DDG reorders attributes.
    anchor = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.S | re.I)
    href_re = re.compile(r'href="([^"]+)"', re.I)
    snippet_re = re.compile(
        r'<a\b[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>', re.S | re.I)

    def _is_result_anchor(attrs: str) -> bool:
        return "result__a" in attrs

    for m in anchor.finditer(html):
        attrs = m.group(1)
        if not _is_result_anchor(attrs):
            continue
        href = href_re.search(attrs)
        if not href:
            continue
        url = _unwrap_ddg_url(_clean(href.group(1)))
        title = _clean(m.group(2))
        # Look for a snippet only inside the window before the NEXT result,
        # so a missing snippet cannot steal the following result's text.
        # Window = up to the NEXT RESULT anchor (not just any anchor: the
        # snippet itself is an anchor). A missing snippet must not let this
        # row absorb the following result's text.
        end = len(html)
        for nxt in anchor.finditer(html, m.end()):
            if _is_result_anchor(nxt.group(1)):
                end = nxt.start()
                break
        sm = snippet_re.search(html[m.end():end])
        out.append(WebResult(title=title, url=url,
                             snippet=_clean(sm.group(1)) if sm else ""))
        if len(out) >= max_results:
            break
    return out


def parse_ddg_lite(html: str, max_results: int) -> list[WebResult]:
    """Parse the lite.duckduckgo.com table layout — a different endpoint
    that stayed HTTP 200 with 10 parseable results in the same second the
    main HTML endpoint served a CAPTCHA (measured live 2026-08-05)."""
    out: list[WebResult] = []
    for m in re.finditer(
        r'<a[^>]*class="[^"]*result-link[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
        html, re.S | re.I,
    ):
        out.append(WebResult(title=_clean(m.group(2)),
                             url=_unwrap_ddg_url(_clean(m.group(1))),
                             snippet=""))
        if len(out) >= max_results:
            break
    if not out:  # older/plain lite markup: bare links in the results table
        for m in re.finditer(r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>',
                             html, re.S | re.I):
            url = _unwrap_ddg_url(_clean(m.group(1)))
            if "duckduckgo.com" in url:
                continue
            out.append(WebResult(title=_clean(m.group(2)), url=url, snippet=""))
            if len(out) >= max_results:
                break
    return out


def _duckduckgo(query: str, max_results: int,
                attempts: list | None = None) -> list[WebResult]:
    """DDG search over two independent endpoints (no API key required).

    Every attempt records provider / status / bytes / parsed / reason into
    `attempts` so an empty result set can say WHY — pre-fix all six failure
    paths returned a bare `[]` and the model was told "[no results]", which
    reads as "the web has nothing" (the Jul-15 incident)."""
    endpoints = (
        ("ddg_html", "GET", "https://duckduckgo.com/html/", parse_ddg_html),
        ("ddg_lite", "POST", "https://lite.duckduckgo.com/lite/", parse_ddg_lite),
    )
    for name, method, url, parser in endpoints:
        rec = {"provider": name, "status": None, "bytes": 0, "parsed": 0,
               "reason": ""}
        try:
            if method == "POST":
                r = httpx.post(url, data={"q": query}, headers=_BROWSER_HEADERS,
                               timeout=20.0, follow_redirects=True)
            else:
                r = httpx.get(url, params={"q": query}, headers=_BROWSER_HEADERS,
                              timeout=20.0, follow_redirects=True)
        except Exception as e:
            rec["reason"] = f"{type(e).__name__}: {e}"[:160]
            if attempts is not None:
                attempts.append(rec)
            continue
        rec["status"] = r.status_code
        rec["bytes"] = len(r.text or "")
        if r.status_code >= 400:
            rec["reason"] = f"http {r.status_code}"
        elif looks_like_bot_wall(r.text):
            # HTTP 202 + CAPTCHA is the classic here; raise_for_status() is
            # blind to it, so it used to masquerade as "no results".
            rec["reason"] = "anti-bot challenge page (not a result page)"
        else:
            hits = parser(r.text, max_results)
            rec["parsed"] = len(hits)
            if hits:
                if attempts is not None:
                    attempts.append(rec)
                return hits
            rec["reason"] = "parsed 0 results (markup may have changed)"
        if attempts is not None:
            attempts.append(rec)
    return []


def web_search_detailed(query: str, max_results: int = 5) -> dict:
    """Search with per-attempt diagnostics. Returns
    {results, attempts, note} — the tool layer surfaces `attempts` to the
    model when `results` is empty so "the tool is blocked" is never again
    reported as "the web has nothing"."""
    attempts: list[dict] = []
    # Self-hosted first: no key, no rate limit, several engines behind one call.
    hits = _searxng(query, max_results, attempts=attempts)
    if hits:
        return {"results": hits, "attempts": attempts, "note": ""}
    if os.getenv("TAVILY_API_KEY"):
        hits = _tavily(query, max_results)
        attempts.append({"provider": "tavily", "status": None,
                         "bytes": 0, "parsed": len(hits),
                         "reason": "" if hits else "no results / request failed"})
        if hits:
            return {"results": hits, "attempts": attempts, "note": ""}
    else:
        attempts.append({"provider": "tavily", "status": None, "bytes": 0,
                         "parsed": 0, "reason": "TAVILY_API_KEY not set"})
    hits = _duckduckgo(query, max_results, attempts=attempts)
    note = ""
    if not hits:
        blocked = [a for a in attempts if "anti-bot" in (a.get("reason") or "")]
        note = (
            "Every search provider failed — this is NOT evidence that nothing "
            "exists. " + ("A provider served an anti-bot challenge; retry, "
                          "rephrase, or fetch a known URL directly. "
                          if blocked else "")
            + "Set TAVILY_API_KEY for a reliable API path."
        )
        log.warning("web_search exhausted providers for %r: %s", query, attempts)
    return {"results": hits, "attempts": attempts, "note": note}


def web_search(query: str, max_results: int = 5) -> list[WebResult]:
    """Back-compatible entry point: results only."""
    return web_search_detailed(query, max_results)["results"]


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


def _render_with_browser(url: str, timeout_s: int = 45) -> "str | None":
    """Render a URL in headless Chrome and return the DOM, or None.

    The cheap HTTP path cannot see content that JavaScript writes:
    GeckoTerminal serves ~380 extractable chars raw and ~10 000 after a
    render (measured 2026-08-05). Never raises — the caller falls back."""
    import shutil
    import subprocess
    browser = (shutil.which("google-chrome") or shutil.which("chromium")
               or shutil.which("chromium-browser"))
    if not browser:
        return None
    try:
        proc = subprocess.run(
            [browser, "--headless", "--disable-gpu", "--no-sandbox",
             "--virtual-time-budget=9000", f"--user-agent={BROWSER_UA}",
             "--dump-dom", url],
            capture_output=True, text=True, timeout=timeout_s,
            errors="replace",
        )
    except Exception as e:
        log.debug("headless render failed for %s: %s", url, e)
        return None
    dom = proc.stdout or ""
    return dom or None


def fetch_url(url: str, max_chars: int = 8000) -> str:
    blocked = _ssrf_check(url)
    if blocked:
        return f"[fetch refused: {blocked}]"
    status = 0
    body = ""
    try:
        r = httpx.get(url, timeout=30.0, follow_redirects=True,
                      headers=_BROWSER_HEADERS)
        status, body = r.status_code, r.text
    except Exception as e:
        return f"[fetch error: {e}]"

    # Prefer clean main-content extraction; fall back to the regex strip when
    # trafilatura is absent or the page has no article body (search-result
    # pages, JSON endpoints, tiny pages).
    main = _extract_main_content(body, url) if status < 400 else None
    text = main if main is not None else _regex_strip_html(body)

    # A bot wall is a fact about ACCESS, not about the topic. Say so, and keep
    # the body: pre-fix raise_for_status() discarded it and the model saw only
    # "[fetch error: 403]" with no idea whether to retry, use another route, or
    # conclude the page is gone.
    if status >= 400 or looks_like_bot_wall(body):
        head = text[: min(max_chars, 1200)]
        return (
            f"[blocked: HTTP {status}{' + anti-bot challenge' if looks_like_bot_wall(body) else ''}] "
            "This is an ACCESS failure, not evidence the information does not "
            "exist. Try another source, an official API, or an authenticated "
            f"route.\n--- page said ---\n{head}"
        )

    # JS-only shell -> pay for a real browser once.
    if looks_unreadable(status, body, main):
        dom = _render_with_browser(url)
        if dom:
            rendered_main = _extract_main_content(dom, url)
            rendered = (rendered_main if rendered_main is not None
                        else _regex_strip_html(dom))
            if looks_like_bot_wall(dom):
                return (
                    "[blocked: anti-bot challenge survived a headless render] "
                    "This site refuses automated access; it is not evidence "
                    "the information does not exist.\n--- page said ---\n"
                    + rendered[:1200]
                )
            if len(rendered.strip()) > len(text.strip()):
                return ("[rendered via headless browser]\n"
                        + rendered[:max_chars])
    return text[:max_chars]
