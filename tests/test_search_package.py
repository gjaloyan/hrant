"""Tests for G4 — search_package.

Pinned behaviour:
  - The function rejects empty / whitespace-containing names
    and unsupported managers without making a network call.
  - PyPI / crates.io / npm responses get normalised into a
    uniform result shape so the LLM doesn't have to know
    registry-specific JSON layouts.
  - Network failures (any httpx exception) and 4xx responses
    surface as ok=False with a clear error — never an
    uncaught exception.
  - The builtin_tools handler returns JSON the tool layer
    can serialise.
  - search_package tool is registered.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest


# ─── input validation ───────────────────────────────────────────────


def test_search_package_rejects_empty_name():
    from backend.tools.search_package import search_package
    out = search_package("")
    assert out["ok"] is False
    assert "required" in out["error"]


def test_search_package_rejects_whitespace_name():
    from backend.tools.search_package import search_package
    out = search_package("foo bar")
    assert out["ok"] is False
    assert "whitespace" in out["error"]


def test_search_package_rejects_unsupported_manager():
    from backend.tools.search_package import search_package
    out = search_package("foo", manager="rubygems")
    assert out["ok"] is False
    assert "unsupported manager" in out["error"]


# ─── registry calls — mock httpx so we don't hit the network ────────


def _fake_response(status_code: int, payload: dict):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = payload
    return r


def _patch_httpx_get(monkeypatch, response):
    from backend.tools import search_package as mod

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, headers=None):
            return response

    monkeypatch.setattr(mod.httpx, "Client", FakeClient)


def test_pypi_hit_returns_normalised_shape(monkeypatch):
    from backend.tools.search_package import search_package
    payload = {
        "info": {
            "name": "Pillow",
            "version": "11.0.0",
            "summary": "Python Imaging Library (Fork)",
            "home_page": "https://python-pillow.org",
            "project_urls": {"Documentation": "https://pillow.readthedocs.io"},
            "license": "HPND",
            "author": "Jeffrey A. Clark",
            "requires_python": ">=3.9",
        },
        "releases": {"10.0.0": [], "11.0.0": [{"yanked": False}]},
    }
    _patch_httpx_get(monkeypatch, _fake_response(200, payload))
    out = search_package("Pillow", manager="pip")
    assert out["ok"] is True
    assert out["exists"] is True
    assert out["name"] == "Pillow"
    assert out["latest_version"] == "11.0.0"
    assert "Python Imaging" in out["summary"]
    assert out["homepage"] == "https://python-pillow.org"
    assert out["license"] == "HPND"
    assert out["canonical_install"] == "pip install Pillow"
    assert "pypi.org/project/Pillow" in out["registry_url"]


def test_pypi_404_returns_ok_false(monkeypatch):
    from backend.tools.search_package import search_package
    _patch_httpx_get(monkeypatch, _fake_response(404, {}))
    out = search_package("does-not-exist-pkg-xyz", manager="pip")
    assert out["ok"] is False
    assert out["exists"] is False
    assert "not found" in out["error"].lower()


def test_pypi_alias_pypi_works(monkeypatch):
    """'pypi' should alias to 'pip'."""
    from backend.tools.search_package import search_package
    _patch_httpx_get(monkeypatch, _fake_response(200, {"info": {"name": "x", "version": "1"}, "releases": {}}))
    out = search_package("x", manager="pypi")
    assert out["ok"] is True
    assert out["manager"] == "pip"


def test_crates_hit_returns_normalised_shape(monkeypatch):
    from backend.tools.search_package import search_package
    payload = {
        "crate": {
            "id": "ripgrep",
            "newest_version": "14.1.1",
            "description": "Fast recursive search tool.",
            "homepage": "https://github.com/BurntSushi/ripgrep",
            "documentation": "https://docs.rs/ripgrep",
            "repository": "https://github.com/BurntSushi/ripgrep",
        },
        "versions": [
            {"license": "MIT OR Unlicense", "yanked": False},
            {"license": "MIT", "yanked": False},
        ],
    }
    _patch_httpx_get(monkeypatch, _fake_response(200, payload))
    out = search_package("ripgrep", manager="cargo")
    assert out["ok"] is True
    assert out["exists"] is True
    assert out["name"] == "ripgrep"
    assert out["latest_version"] == "14.1.1"
    assert "Fast recursive" in out["summary"]
    assert out["license"] == "MIT OR Unlicense"
    assert out["release_count"] == 2
    assert out["canonical_install"] == "cargo install ripgrep"


def test_npm_hit_returns_normalised_shape(monkeypatch):
    from backend.tools.search_package import search_package
    payload = {
        "name": "axios",
        "dist-tags": {"latest": "1.6.0"},
        "description": "Promise based HTTP client for the browser and node.js",
        "homepage": "https://axios-http.com",
        "license": "MIT",
        "repository": {"url": "git+https://github.com/axios/axios.git"},
        "bugs": {"url": "https://github.com/axios/axios/issues"},
        "versions": {
            "1.5.0": {"author": {"name": "Matt Zabriskie"}},
            "1.6.0": {"author": {"name": "Matt Zabriskie"}, "license": "MIT"},
        },
    }
    _patch_httpx_get(monkeypatch, _fake_response(200, payload))
    out = search_package("axios", manager="npm")
    assert out["ok"] is True
    assert out["exists"] is True
    assert out["name"] == "axios"
    assert out["latest_version"] == "1.6.0"
    assert "Promise based HTTP" in out["summary"]
    assert out["license"] == "MIT"
    assert out["author"] == "Matt Zabriskie"
    assert "axios/axios" in out["project_urls"]["repository"]
    assert out["canonical_install"] == "npm install axios"


def test_network_error_returns_ok_false(monkeypatch):
    """Any httpx exception (timeout, dns, refused) must surface as
    ok=False — never propagate."""
    from backend.tools import search_package as mod

    class BoomClient:
        def __init__(self, *a, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get(self, url, headers=None):
            raise mod.httpx.ConnectError("DNS resolution failed")

    monkeypatch.setattr(mod.httpx, "Client", BoomClient)
    out = mod.search_package("foo", manager="pip")
    assert out["ok"] is False
    assert out["exists"] is False


# ─── builtin_tools handler ───────────────────────────────────────────


def test_handler_returns_json(monkeypatch):
    from backend import builtin_tools
    _patch_httpx_get(monkeypatch, _fake_response(200, {"info": {"name": "x", "version": "1.0"}, "releases": {}}))
    out = builtin_tools._search_package_handler("x", "pip")
    data = json.loads(out)
    assert data["ok"] is True
    assert data["latest_version"] == "1.0"


def test_handler_handles_internal_error(monkeypatch):
    """If something in search_package itself raises, the handler
    must still produce JSON with ok=False, not crash the tool loop."""
    from backend import builtin_tools

    def kaboom(*a, **kw):
        raise RuntimeError("simulated bug")

    monkeypatch.setattr(builtin_tools, "_search_package", kaboom)
    out = builtin_tools._search_package_handler("x", "pip")
    data = json.loads(out)
    assert data["ok"] is False
    assert "search error" in data["error"]


def test_search_package_tool_is_registered():
    from backend import builtin_tools
    from backend.tool_registry import get_registry
    builtin_tools.register_builtin_tools()
    assert "search_package" in get_registry().tools


# ─── universal_resolver mentions search_package ─────────────────────


def test_universal_resolver_step_4_mentions_search_package():
    from backend import skills
    skills.SKILLS._loaded = False
    skills.SKILLS.skills = []
    skills.SKILLS.ensure_loaded()
    sk = skills.SKILLS.get("universal_resolver")
    assert sk is not None
    body = sk.body or ""
    assert "search_package" in body
    # Should call out that registry > blog.
    assert "registry" in body.lower()
