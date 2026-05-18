"""Query authoritative package registries (PyPI / crates.io / npm)
for "is this library real, who maintains it, what's the latest
version, where's the homepage".

Used by the universal_resolver Research step (4) instead of
hand-rolled `web_search` + blog scraping. The registry APIs give
you the canonical info — official URL, maintainer, latest version,
license — without the LLM having to interpret a blog post.

Registry endpoints are public + read-only + don't need auth:

  - PyPI:     https://pypi.org/pypi/<name>/json
  - crates:   https://crates.io/api/v1/crates/<name>
  - npm:      https://registry.npmjs.org/<name>

Each returns JSON we summarise into a uniform shape — the agent
sees one consistent structure regardless of manager.

For G4 we cover those three. If/when the resolver needs to ask
about RubyGems or Maven, drop another _query_<manager> function
in and register it in the dispatch table.
"""
from __future__ import annotations

import logging
from typing import Optional

import httpx

log = logging.getLogger(__name__)


# Conservative timeout — these are well-known endpoints, but the
# resolver shouldn't stall a turn waiting on a slow CDN.
DEFAULT_TIMEOUT = 10.0


_USER_AGENT = "hrant-agent (search_package; +https://github.com/gjaloyan/hrant)"


def _http_get(url: str, *, timeout: float) -> Optional[dict]:
    """Fetch a JSON registry response. Returns None on any failure
    (network, 4xx, parse) so the caller can synthesise an error
    payload uniformly."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as c:
            r = c.get(url, headers={"User-Agent": _USER_AGENT, "Accept": "application/json"})
        if r.status_code >= 400:
            return None
        return r.json()
    except Exception as e:
        log.debug("search_package http error for %s: %s", url, e)
        return None


def _query_pypi(name: str, *, timeout: float) -> dict:
    """PyPI: https://pypi.org/pypi/<name>/json"""
    data = _http_get(f"https://pypi.org/pypi/{name}/json", timeout=timeout)
    if data is None:
        return {
            "ok": False,
            "manager": "pip",
            "name": name,
            "exists": False,
            "error": "not found on PyPI (or registry unreachable)",
        }
    info = data.get("info") or {}
    releases = data.get("releases") or {}
    return {
        "ok": True,
        "manager": "pip",
        "name": info.get("name") or name,
        "exists": True,
        "latest_version": info.get("version") or "",
        "summary": (info.get("summary") or "").strip()[:300],
        "homepage": info.get("home_page") or info.get("project_url") or "",
        "project_urls": info.get("project_urls") or {},
        "license": info.get("license") or "",
        "author": info.get("author") or info.get("maintainer") or "",
        "requires_python": info.get("requires_python") or "",
        "release_count": len(releases),
        "yanked": False,
        "canonical_install": f"pip install {name}",
        "registry_url": f"https://pypi.org/project/{name}/",
    }


def _query_crates(name: str, *, timeout: float) -> dict:
    """crates.io: https://crates.io/api/v1/crates/<name>"""
    data = _http_get(f"https://crates.io/api/v1/crates/{name}", timeout=timeout)
    if data is None:
        return {
            "ok": False,
            "manager": "cargo",
            "name": name,
            "exists": False,
            "error": "not found on crates.io (or registry unreachable)",
        }
    crate = data.get("crate") or {}
    versions = data.get("versions") or []
    return {
        "ok": True,
        "manager": "cargo",
        "name": crate.get("id") or name,
        "exists": True,
        "latest_version": crate.get("newest_version") or crate.get("max_version") or "",
        "summary": (crate.get("description") or "").strip()[:300],
        "homepage": crate.get("homepage") or "",
        "project_urls": {
            "documentation": crate.get("documentation") or "",
            "repository": crate.get("repository") or "",
        },
        "license": (versions[0].get("license") if versions else "") or "",
        "author": "",  # crates.io doesn't surface this on the crate object
        "release_count": len(versions),
        "yanked": (versions[0].get("yanked", False) if versions else False),
        "canonical_install": f"cargo install {name}",
        "registry_url": f"https://crates.io/crates/{name}",
    }


def _query_npm(name: str, *, timeout: float) -> dict:
    """npm: https://registry.npmjs.org/<name>"""
    data = _http_get(f"https://registry.npmjs.org/{name}", timeout=timeout)
    if data is None:
        return {
            "ok": False,
            "manager": "npm",
            "name": name,
            "exists": False,
            "error": "not found on npm (or registry unreachable)",
        }
    dist_tags = data.get("dist-tags") or {}
    latest = dist_tags.get("latest") or ""
    versions = data.get("versions") or {}
    latest_meta = (versions.get(latest) or {}) if latest else {}
    return {
        "ok": True,
        "manager": "npm",
        "name": data.get("name") or name,
        "exists": True,
        "latest_version": latest,
        "summary": (data.get("description") or "").strip()[:300],
        "homepage": data.get("homepage") or "",
        "project_urls": {
            "repository": (data.get("repository") or {}).get("url", "") if isinstance(data.get("repository"), dict) else "",
            "bugs": (data.get("bugs") or {}).get("url", "") if isinstance(data.get("bugs"), dict) else "",
        },
        "license": (latest_meta.get("license") or data.get("license") or ""),
        "author": (latest_meta.get("author") or {}).get("name", "") if isinstance(latest_meta.get("author"), dict) else (str(latest_meta.get("author") or "")),
        "release_count": len(versions),
        "yanked": False,
        "canonical_install": f"npm install {name}",
        "registry_url": f"https://www.npmjs.com/package/{name}",
    }


_DISPATCH = {
    "pip": _query_pypi,
    "pypi": _query_pypi,
    "cargo": _query_crates,
    "crates": _query_crates,
    "npm": _query_npm,
}


def search_package(
    name: str,
    manager: str = "pip",
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Look up `name` in the registry for `manager`.

    Returns a dict with `ok` (registry call succeeded), `exists`
    (package found), `latest_version`, `summary`, `homepage`,
    `license`, `author`, `release_count`, `canonical_install`,
    `registry_url`. On miss / network failure, `ok=False` and
    `error` carries a short reason — never raises.

    Args:
        name: package identifier (e.g. "pillow", "ripgrep", "axios").
        manager: 'pip' (alias 'pypi'), 'cargo' (alias 'crates'),
            'npm'. Other values return an error dict.
        timeout: per-call HTTP timeout in seconds.
    """
    if not name or not isinstance(name, str):
        return {"ok": False, "exists": False, "error": "name required"}
    name = name.strip()
    if not name:
        return {"ok": False, "exists": False, "error": "name required"}
    if any(ch.isspace() for ch in name):
        return {
            "ok": False, "exists": False,
            "error": "package name has whitespace — pass a single identifier",
        }
    mgr = (manager or "pip").strip().lower()
    fn = _DISPATCH.get(mgr)
    if fn is None:
        return {
            "ok": False,
            "exists": False,
            "manager": mgr,
            "error": (
                f"unsupported manager {mgr!r}; supported: "
                + ", ".join(sorted(set(_DISPATCH.keys())))
            ),
        }
    return fn(name, timeout=timeout)
