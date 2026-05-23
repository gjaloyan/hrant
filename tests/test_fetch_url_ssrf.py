"""Tests for the 2026-05-23 SSRF guard in `fetch_url` (audit Critical #4).

Pre-fix: `httpx.get(url, follow_redirects=True)` would happily hit
loopback / RFC1918 / Tailnet / link-local addresses. The agent IS
owner, so a same-host `fetch_url("http://127.0.0.1:3333/api/...")`
bypasses every owner gate on its own admin surface.

Post-fix: `_ssrf_check` rejects non-http(s) schemes and resolves
every A/AAAA, refusing if any answer is in a blocked range.
"""
from __future__ import annotations

import socket
from unittest.mock import patch

import pytest


# ─── _ssrf_check: scheme + host validation ────────────────────────


def test_ssrf_refuses_non_http_scheme():
    from backend.tools.web_search import _ssrf_check
    assert "scheme" in _ssrf_check("file:///etc/passwd")
    assert "scheme" in _ssrf_check("ftp://example.com")
    assert "scheme" in _ssrf_check("gopher://x")
    assert "scheme" in _ssrf_check("javascript:alert(1)")


def test_ssrf_refuses_empty_or_unparseable():
    from backend.tools.web_search import _ssrf_check
    assert _ssrf_check("")
    assert _ssrf_check("   ")


def test_ssrf_refuses_loopback_v4():
    from backend.tools.web_search import _ssrf_check
    # Direct IP — no DNS round-trip needed; resolves to itself.
    msg = _ssrf_check("http://127.0.0.1:3333/api/health")
    assert msg
    assert "127.0.0.1" in msg
    assert "in" in msg


def test_ssrf_refuses_rfc1918():
    from backend.tools.web_search import _ssrf_check
    for url in (
        "http://10.0.0.1/", "http://172.16.5.4/",
        "http://192.168.1.1/admin",
    ):
        msg = _ssrf_check(url)
        assert msg, f"{url} should be refused"


def test_ssrf_refuses_tailscale_cgn_range():
    """Tailnet is 100.64.0.0/10. Prod box is on 100.124.210.21 — a
    prompt-injection asking fetch_url("http://100.x.x.x/...") could
    hit a peer admin port. Block the whole CGN range."""
    from backend.tools.web_search import _ssrf_check
    msg = _ssrf_check("http://100.124.210.21:8000/api/internal")
    assert msg
    assert "100.124.210.21" in msg


def test_ssrf_refuses_link_local_metadata():
    """169.254.169.254 is cloud metadata (AWS/GCP/etc). If we ever
    deploy in a cloud, this prevents IAM-token theft via injection."""
    from backend.tools.web_search import _ssrf_check
    msg = _ssrf_check("http://169.254.169.254/latest/meta-data/")
    assert msg


def test_ssrf_refuses_ipv6_loopback():
    from backend.tools.web_search import _ssrf_check
    msg = _ssrf_check("http://[::1]:3333/")
    assert msg


def test_ssrf_allows_public_dns_resolved_host(monkeypatch):
    """`example.com` resolves to a public address; should pass."""
    from backend.tools.web_search import _ssrf_check
    # Stub getaddrinfo to a public-looking IP so the test doesn't need
    # actual DNS.
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port: [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", port)),
        ],
    )
    assert _ssrf_check("https://example.com/") == ""


def test_ssrf_refuses_when_one_answer_is_private(monkeypatch):
    """DNS rebind protection: if a host resolves to ANY private IP,
    refuse — even if other answers look public."""
    from backend.tools.web_search import _ssrf_check
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda host, port: [
            (socket.AF_INET, socket.SOCK_STREAM, 0, "",
             ("93.184.216.34", port)),
            (socket.AF_INET, socket.SOCK_STREAM, 0, "",
             ("127.0.0.1", port)),
        ],
    )
    assert _ssrf_check("https://evil.example.com/") != ""


# ─── fetch_url integration ────────────────────────────────────────


def test_fetch_url_refuses_loopback_with_safe_error():
    """fetch_url returns a `[fetch refused: ...]` string instead of
    raising — the agent's LLM consumes the result without crashing."""
    from backend.tools.web_search import fetch_url
    out = fetch_url("http://127.0.0.1:3333/api/health")
    assert out.startswith("[fetch refused:")
    assert "127.0.0.1" in out


def test_fetch_url_refuses_non_http_scheme():
    from backend.tools.web_search import fetch_url
    out = fetch_url("file:///etc/passwd")
    assert out.startswith("[fetch refused:")


def test_fetch_url_does_not_call_httpx_on_refusal():
    """Performance + safety: when SSRF check refuses, httpx.get must
    NOT be invoked. Pin via mock."""
    from backend.tools import web_search
    called = {"n": 0}

    def _boom(*a, **kw):
        called["n"] += 1
        raise AssertionError("httpx.get should not be called on refusal")

    with patch.object(web_search.httpx, "get", _boom):
        out = web_search.fetch_url("http://127.0.0.1/")
    assert called["n"] == 0
    assert out.startswith("[fetch refused:")
