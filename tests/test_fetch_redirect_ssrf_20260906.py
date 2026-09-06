"""A public page could redirect the fetcher onto the loopback interface.

From the GPT-6 Astra audit, 2026-09-05. Their reproduction:

    visited: https://audit-public.example/start
          -> http://127.0.0.1:3333/api/identity
    returned_internal_content: true

`_ssrf_check` validated the URL the caller asked for and nothing else;
`httpx.get(..., follow_redirects=True)` then went wherever the page
pointed. The docstring claimed "httpx validates redirect destinations on
its own loop" — it does not. httpx follows redirects; it has no notion
of which addresses are private.

This matters even bound to 127.0.0.1: the agent reads attacker-chosen
pages all day, and one 302 turns that into a read of the agent's own
API.
"""
from unittest.mock import patch

import pytest

from backend.tools import web_search as ws


class _Resp:
    def __init__(self, status, *, location=None, text="body"):
        self.status_code = status
        self.text = text
        self.headers = {"Location": location} if location else {}
        self.content = text.encode()


def _resolve(host, *a, **k):
    """`audit-public.example` is a public host; everything else resolves
    as itself. Keeps `_ssrf_check` REAL — it is the thing under test for
    every hop — without a DNS round trip in a unit test."""
    import socket
    if host == "audit-public.example":
        return [(socket.AF_INET, None, None, "", ("93.184.216.34", 0))]
    return [(socket.AF_INET, None, None, "", (host, 0))]


def _fetch_with(responses):
    """Serve `responses` in order, recording every URL requested."""
    visited = []

    def _get(url, **kw):
        visited.append(url)
        assert kw.get("follow_redirects") is not True, (
            "redirects must be followed hop by hop, not by httpx"
        )
        return responses[len(visited) - 1]

    with patch.object(ws.httpx, "get", _get),          patch("socket.getaddrinfo", _resolve):
        out = ws.fetch_url("https://audit-public.example/start")
    return out, visited


def test_a_redirect_onto_loopback_is_refused():
    out, visited = _fetch_with([
        _Resp(302, location="http://127.0.0.1:3333/api/identity"),
    ])
    assert visited == ["https://audit-public.example/start"], (
        "the private hop must never be requested"
    )
    assert "refused" in out.lower() or "blocked" in out.lower()
    assert "127.0.0.1" in out


def test_a_redirect_to_a_private_network_is_refused():
    out, visited = _fetch_with([
        _Resp(301, location="http://192.168.18.58:8080/admin"),
    ])
    assert len(visited) == 1
    assert "refused" in out.lower() or "blocked" in out.lower()


def test_an_ordinary_public_redirect_still_works():
    out, visited = _fetch_with([
        _Resp(302, location="https://audit-public.example/final"),
        _Resp(200, text="<html><body>the real article body</body></html>"),
    ])
    assert visited == [
        "https://audit-public.example/start",
        "https://audit-public.example/final",
    ]
    assert "refused" not in out.lower()


def test_a_relative_redirect_is_resolved_against_the_current_url():
    out, visited = _fetch_with([
        _Resp(302, location="/moved"),
        _Resp(200, text="<html><body>arrived</body></html>"),
    ])
    assert visited[1] == "https://audit-public.example/moved"


def test_a_redirect_loop_ends_rather_than_spinning():
    responses = [_Resp(302, location="https://audit-public.example/start")] * 30
    out, visited = _fetch_with(responses)
    assert len(visited) <= ws._MAX_REDIRECTS + 1
    assert "redirect" in out.lower()
