"""Multi-provider failover — when the active LLM is unreachable,
rate-limited, or down, fall through a user-configured chain of
(provider, model) pairs.

The pattern openrouter.ai uses for paid users; we apply it locally
so a personal agent isn't held hostage by a single provider's 429.

Wiring:
  - Router.call() and Router.call_with_tools() in backend/llm.py
    build an `attempts` list — primary active LLM first, then each
    enabled chain entry — and hand it to `try_call(attempts)`.
  - `try_call` walks the list, classifies any `LLMError` raised:
      * retryable (rate_limit / server_error / timeout / auth / connection)
        → continue to next attempt
      * non-retryable (bad_request / content_policy / context_length)
        → raise immediately; trying another provider won't help if
        the prompt itself is the problem.
  - Each attempt — success or failure — appends one entry to the
    active Job's `attempts[]` list via the ContextVar bridge below.
    The WebUI's Jobs-tab "PROVIDER ATTEMPTS" panel reads from there.

Config storage:
  ~/.hrant/data/knowledge/failover_config.json
    {
      "enabled": false,
      "chain": [
        {"provider_id": "openai-default", "model": "gpt-4o"},
        {"provider_id": "anthropic-default", "model": "claude-3-5-sonnet"}
      ],
      "retry_on": ["rate_limit", "server_error", "timeout", "auth_error", "connection"],
      "max_attempts": 4
    }

The pinned active model is ALWAYS tried first — the `chain` is what
runs AFTER the primary fails. A chain entry that matches the active
model is skipped (we just tried it).
"""
from __future__ import annotations

import json
import logging
import re
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


# ─── Error classification ─────────────────────────────────────────────


# Categories we know how to label. Names chosen to match what a user
# would actually see in WebUI badges ("rate limit", "auth error", ...).
CATEGORIES: tuple[str, ...] = (
    "rate_limit",
    "server_error",
    "timeout",
    "auth_error",
    "connection",
    "bad_request",
    "content_policy",
    "context_length",
    "unknown",
)


# Default retry policy: everything that's not "the prompt itself is
# broken" gets a second chance on a different provider. Auth errors
# are included because a user's expired Anthropic key shouldn't
# silently block them when OpenAI is configured and working.
DEFAULT_RETRY_ON: tuple[str, ...] = (
    "rate_limit",
    "server_error",
    "timeout",
    "auth_error",
    "connection",
)


# Pattern → category. Order matters: more specific patterns first.
# All match against `str(exception).lower()` so case differences in
# provider error messages don't matter.
_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b429\b|rate.?limit|too many requests"), "rate_limit"),
    (re.compile(r"\b5\d\d\b|server error|internal error|service unavailable|bad gateway"), "server_error"),
    (re.compile(r"timeout|timed out|read.?timeout|connect.?timeout"), "timeout"),
    (re.compile(r"\b401\b|\b403\b|invalid.?api.?key|unauthor|forbidden|authentication"), "auth_error"),
    (re.compile(r"connection|network|name or service not known|getaddrinfo|connect.?refused"), "connection"),
    (re.compile(r"\b400\b|bad request|invalid.?request|malformed"), "bad_request"),
    (re.compile(r"content.?policy|content_policy|safety|moderation"), "content_policy"),
    (re.compile(r"context.?length|max.?tokens|prompt is too long|too many tokens"), "context_length"),
)


def classify(exc: Exception) -> str:
    """Map an exception to one of CATEGORIES. Best-effort string
    match against `str(exc)` — provider error messages embed the
    HTTP status code or a textual hint that's enough to route."""
    msg = str(exc).lower()
    for pattern, category in _PATTERNS:
        if pattern.search(msg):
            return category
    return "unknown"


def should_retry(category: str, retry_on: Optional[list[str]] = None) -> bool:
    """True iff the category is in the retry set. Unknown errors are
    NOT retried by default — they could be programming bugs, and we
    don't want to silently mask them by trying a different provider."""
    retry_set = set(retry_on or DEFAULT_RETRY_ON)
    return category in retry_set


# ─── Config storage ──────────────────────────────────────────────────


def _config_path() -> Path:
    """Lazy resolution so tests that monkeypatch HRANT_DATA_DIR
    after import still see the override."""
    from . import paths
    return paths.knowledge_dir() / "failover_config.json"


def default_config() -> dict:
    """Fresh-install defaults. enabled=False until the user opts in
    — automatic provider switching is a behaviour change, so it
    should be explicit, not surprising."""
    return {
        "enabled": False,
        "chain": [],
        "retry_on": list(DEFAULT_RETRY_ON),
        "max_attempts": 4,
    }


def load_config() -> dict:
    p = _config_path()
    if not p.exists():
        return default_config()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("failover_config.json unreadable (%s); using defaults", e)
        return default_config()
    # Fill missing fields with defaults — forward-compatible if the
    # schema grows (e.g. per-entry timeout overrides).
    base = default_config()
    base.update(data)
    return base


def save_config(cfg: dict) -> dict:
    """Validate then persist. Strips unknown fields from chain
    entries so a malformed PUT can't poison the file."""
    out = default_config()
    out["enabled"] = bool(cfg.get("enabled", False))
    out["max_attempts"] = max(1, min(10, int(cfg.get("max_attempts", 4))))
    retry_on = cfg.get("retry_on") or DEFAULT_RETRY_ON
    out["retry_on"] = [c for c in retry_on if c in CATEGORIES]
    chain = cfg.get("chain") or []
    cleaned: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for entry in chain:
        if not isinstance(entry, dict):
            continue
        pid = str(entry.get("provider_id", "")).strip()
        model = str(entry.get("model", "")).strip()
        if not pid or not model:
            continue
        key = (pid, model)
        if key in seen:
            continue  # dedupe — chain of [A, A, B] becomes [A, B]
        seen.add(key)
        cleaned.append({"provider_id": pid, "model": model})
    out["chain"] = cleaned

    p = _config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


# ─── ContextVar bridge to the active Job ─────────────────────────────


# Set by `job_runner.run_tracked` before calling `agent.run`, cleared
# afterwards. Allows the deeply-nested LLM call inside `Router.call`
# / `Router.call_with_tools` to know which Job to append attempts to
# without threading the id through every call signature.
_current_job_id: ContextVar[Optional[str]] = ContextVar("hrant_current_job_id", default=None)


def set_current_job_id(job_id: Optional[str]) -> Any:
    return _current_job_id.set(job_id)


def reset_current_job_id(token: Any) -> None:
    _current_job_id.reset(token)


def get_current_job_id() -> Optional[str]:
    return _current_job_id.get()


def record_attempt(
    *,
    provider_id: str,
    model: str,
    ok: bool,
    error: Optional[str] = None,
    category: Optional[str] = None,
    elapsed_ms: Optional[int] = None,
) -> None:
    """Append one entry to the active Job's `attempts[]`. No-op when
    no Job is active (CLI test runs, autonomic ticks). Best-effort:
    a logging failure must NOT bubble up and kill the LLM call."""
    jid = get_current_job_id()
    if not jid:
        return
    try:
        from . import jobs as _jobs
        _jobs.JOBS.add_attempt(jid, {
            "provider_id": provider_id,
            "model": model,
            "ok": ok,
            "error": error,
            "category": category,
            "elapsed_ms": elapsed_ms,
            "started_at": time.time(),
        })
    except Exception as e:
        log.warning("failover.record_attempt(%s) failed: %s", jid, e)


# ─── The retry loop ──────────────────────────────────────────────────


# (provider_id, model, call_fn) where call_fn takes no args and
# either returns the result or raises LLMError. Built by the caller.
AttemptSpec = tuple[str, str, Callable[[], Any]]


def try_call(
    attempts: list[AttemptSpec],
    *,
    retry_on: Optional[list[str]] = None,
    max_attempts: Optional[int] = None,
) -> Any:
    """Walk the attempts list, returning the first successful result.

    Each call_fn is invoked at most once. After each call we append a
    Job.attempts entry (via the ContextVar bridge). On retryable
    error, move to the next entry. On non-retryable error, raise
    immediately — no point trying another provider if the prompt
    itself is the problem.

    `retry_on` defaults to DEFAULT_RETRY_ON when unspecified.
    `max_attempts` caps the walk; further entries are ignored even
    if the chain configures more. Default = len(attempts).
    """
    if not attempts:
        from .llm import LLMError
        raise LLMError("failover: no attempts configured")

    cap = max_attempts if max_attempts is not None else len(attempts)
    cap = max(1, min(cap, len(attempts)))

    last_error: Optional[Exception] = None
    for i, (provider_id, model, fn) in enumerate(attempts[:cap]):
        t0 = time.monotonic()
        try:
            result = fn()
        except Exception as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            category = classify(e)
            record_attempt(
                provider_id=provider_id, model=model, ok=False,
                error=str(e)[:500], category=category, elapsed_ms=elapsed_ms,
            )
            last_error = e
            if not should_retry(category, retry_on):
                # Stop — trying another provider on a bad-request /
                # content-policy error would just produce the same
                # failure and waste API quota.
                log.info(
                    "failover: %s/%s failed with %s (non-retryable); "
                    "stopping chain",
                    provider_id, model, category,
                )
                raise
            log.info(
                "failover: %s/%s failed with %s; trying next entry "
                "(%d of %d)",
                provider_id, model, category, i + 1, cap,
            )
            continue
        # Success path.
        elapsed_ms = int((time.monotonic() - t0) * 1000)
        record_attempt(
            provider_id=provider_id, model=model, ok=True,
            elapsed_ms=elapsed_ms,
        )
        if i > 0:
            log.info(
                "failover: succeeded on attempt %d (%s/%s) after %d "
                "failure(s)",
                i + 1, provider_id, model, i,
            )
        return result

    # All attempts exhausted — re-raise the last error so callers
    # see the most-recent failure rather than the first one (which
    # may already be fixed by the time the user retries).
    if last_error:
        raise last_error
    from .llm import LLMError
    raise LLMError("failover: all attempts exhausted with no error")


# ─── Helper: resolve a chain entry into an LLM config ────────────────


def resolve_entry_cfg(provider_id: str, model: str) -> Optional[dict]:
    """Build a dict suitable for `llm.create_llm(cfg)` from a chain
    entry. Returns None if the provider is missing or disabled —
    the caller should skip such entries silently.

    Mirrors `providers.ActiveModelManager.resolve_llm_config` but
    takes the (provider_id, model) explicitly instead of reading
    from the pinned active selection.
    """
    from .providers import get_api_key, get_provider
    provider = get_provider(provider_id)
    if not provider:
        return None
    if not provider.get("enabled", True):
        return None
    cfg = {
        "provider": provider.get("type", ""),
        "model": model or provider.get("default_model", ""),
        "api_key": get_api_key(provider),
        "api_key_env": provider.get("api_key_env", ""),
        "base_url": provider.get("base_url", ""),
        "max_tokens": provider.get("max_tokens", 2000),
        "temperature": provider.get("temperature", 0.3),
        "auth_type": provider.get("auth_type", "api_key"),
        "provider_id": provider_id,
    }
    if provider.get("auth_type") == "oauth":
        cfg["oauth"] = provider.get("oauth") or {}
    return cfg
