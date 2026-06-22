"""Dual-model router: Claude Sonnet (brain) + Qwen 7B (apprentice).

Provider-selection logic on each call:
  1. TaskType ∈ MODEL_A_TASKS → candidate = A (Claude)
  2. TaskType ∈ MODEL_B_TASKS → candidate = B (Qwen)
  3. verification + always_use_model_a=true → hard A (override)
  4. shift_schedule (auto_shift_after_finetune) → some A-tasks randomly go to B
  5. daily_api_budget_usd exceeded → fallback to B
  6. Claude API unavailable → fallback to B (if fallback_to_local=true)

State (per-day counters and total calls) is persisted in knowledge/router_state.json.
"""
from __future__ import annotations
import contextvars
import json
import logging
import os
import random
import threading
import time
from datetime import date, datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import httpx

from .config import CONFIG


# Module-level logger. Audit 2026-06-10 (N1): the previous code only
# defined `log` / `_log` inside individual functions, so call sites
# like `Anthropic.call_with_tools`, `Router.call`, etc. that referenced
# bare `log.warning(...)` would NameError at runtime. The function-scope
# loggers in this file (post_with_retry, _call_with_safety_failover)
# stay for their existing patterns, but new code should reach for this
# module-level instance.
log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


# Substrings that mark a provider-side safety refusal. Conservative
# narrow match — we want to catch Codex Responses API
# 'content flagged for cybersecurity risk' AND Anthropic safety
# refusals, NOT generic 5xx / rate-limit / timeout errors. The match
# is on the error message, NOT user input — this is content-of-error
# matching, not user-text keyword routing.
_SAFETY_REFUSAL_MARKERS: tuple[str, ...] = (
    "flagged",
    "content_policy",
    "content policy",
    "cybersecurity_risk",
    "cybersecurity risk",
    "safety",
)


def _is_safety_refusal(err: "LLMError") -> bool:
    """True iff `err`'s message contains a provider-side safety-refusal
    marker (case-insensitive substring match).
    """
    msg = (str(err) or "").lower()
    return any(marker in msg for marker in _SAFETY_REFUSAL_MARKERS)


_QUOTA_EXHAUSTED_MARKERS: tuple[str, ...] = (
    "usage_limit_reached",
    "usage limit",
    "quota exhausted",
    "quota_exhausted",
    "rate_limit_exceeded",
    "rate limit exceeded",
    "too many requests",
    "429",
)


def _is_quota_exhausted(err: "LLMError") -> bool:
    """True iff `err`'s message indicates provider-side usage/quota
    exhaustion or rate limiting. Narrow substring match — we want
    Codex 'usage_limit_reached', OpenAI/Anthropic '429 too many
    requests', and similar, NOT generic 5xx / connection errors."""
    msg = (str(err) or "").lower()
    return any(marker in msg for marker in _QUOTA_EXHAUSTED_MARKERS)


_PROVIDER_MALFORMED_MARKERS: tuple[str, ...] = (
    "provider returned error",   # OpenRouter's wrapper when an upstream faults
    "unexpected end of data",    # truncated provider response
    "unexpected end of json",    # variant of the above
)


def _is_provider_malformed(err: "LLMError") -> bool:
    """True iff the provider returned a MALFORMED/TRUNCATED response rather than
    a valid completion or a legitimate client error.

    This is a provider-side fault — the model didn't finish streaming, or the
    upstream gateway wrapped an internal failure — so the router should retry on
    the next provider, exactly like a safety/quota refusal. Surfaced 2026-06-19:
    `nex-n2-pro:free` truncated a large generation at ~12 KB, OpenRouter wrapped
    it as a 400 'Provider returned error / unexpected end of data', and the turn
    hard-crashed instead of degrading to a working provider.

    Deliberately narrow: it does NOT match a bare 5xx or timeout (post_with_retry
    already HTTP-retries those; a persistent 5xx is surfaced, not masked across
    providers), nor a legitimate client-side bad request (falling back would just
    fail again and hide the real payload bug)."""
    msg = (str(err) or "").lower()
    return any(marker in msg for marker in _PROVIDER_MALFORMED_MARKERS)


def _should_fallback(err: "LLMError") -> bool:
    """The router fallback engages on a safety refusal, a quota/rate-limit
    failure, OR a malformed/truncated provider response. All three mean 'this
    provider can't serve this call right now; try the next one.'"""
    return (
        _is_safety_refusal(err)
        or _is_quota_exhausted(err)
        or _is_provider_malformed(err)
    )


class _ProviderChainAdapter:
    """Thin adapter wrapping a provider record from
    `backend.providers.get_providers()` so it can be walked by
    `_run_with_safety_fallback`.

    Built lazily — the underlying `BaseLLM` (via `create_llm`) is
    not instantiated until the chain actually reaches this entry.
    That keeps cold-path import-time cost zero AND avoids paying
    for OAuth handshakes / API-key probes on entries that the
    primary preempts.
    """
    __slots__ = ("_prov", "_task_type", "_llm", "id", "name", "_model_override")

    def __init__(self, prov: dict, task_type=None, model_override: str = ""):
        self._prov = prov
        self._task_type = task_type
        self._llm: "BaseLLM | None" = None
        self.id = prov.get("id", "")
        self.name = prov.get("name") or prov.get("id") or "?"
        # When the user pinned a specific model on THIS provider, the chain
        # forces it here instead of falling back to the provider's default.
        self._model_override = model_override or ""

    @property
    def model(self) -> str:
        return self._model_override or self._prov.get("default_model", "") or (
            (self._prov.get("models") or [""])[0]
        )

    def _build(self) -> "BaseLLM":
        if self._llm is not None:
            return self._llm
        # Reuse failover.resolve_entry_cfg — same translation from a
        # providers.json record to a create_llm-shaped cfg used by
        # the existing per-provider failover chain. Single source of
        # truth for that mapping.
        from . import failover as _fo
        pid = self._prov.get("id", "")
        model = self.model
        cfg = _fo.resolve_entry_cfg(pid, model)
        if cfg is None:
            raise LLMError(
                f"provider {pid!r} is missing/disabled — cannot build LLM"
            )
        try:
            self._llm = create_llm(cfg)
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(
                f"create_llm({cfg.get('provider')}/{cfg.get('model')}): {e}"
            ) from e
        return self._llm

    def call(self, task_type, system, user, **kw):
        llm = self._build()
        tt = task_type.value if hasattr(task_type, "value") else (task_type or "")
        return llm.complete(
            system, user,
            max_tokens=kw.get("max_tokens"),
            temperature=kw.get("temperature"),
            attachments=kw.get("attachments"),
            _task_type=tt,
        )

    def call_with_tools(self, task_type, system, user, tools, execute_tool, **kw):
        llm = self._build()
        tt = task_type.value if hasattr(task_type, "value") else (task_type or "")
        if not _supports_tools(
            llm, tools, tools_provider=kw.get("tools_provider")
        ):
            raise LLMError(
                f"chain entry {self.id} does not support tools"
            )
        return llm.complete_with_tools(
            system, user, tools, execute_tool,
            max_tokens=kw.get("max_tokens"),
            temperature=kw.get("temperature"),
            max_iterations=kw.get("max_iterations", 6),
            on_tool_call=kw.get("on_tool_call"),
            attachments=kw.get("attachments"),
            tools_provider=kw.get("tools_provider"),
            _task_type=tt,
        )

    def call_json(self, task_type, system, user, **kw):
        raw = self.call(
            task_type,
            system + "\n\nReply with ONLY valid JSON, no markdown wrappers.",
            user,
            **kw,
        )
        return _parse_json_response(raw)


def _active_provider_chain(task_type=None):
    """Return the ordered list of enabled provider adapters the
    router walks via `_run_with_safety_fallback`. Each entry exposes
    `.call(task_type, system, user, **kw)`, `.call_with_tools(...)`,
    and `.call_json(...)`.

    In production this delegates to `backend.providers.get_providers()`
    and filters by `enabled`. The first entry is the primary; the
    rest are fallbacks tried in order when a provider raises a
    safety- or quota-shaped LLMError.

    Tests monkeypatch this function to inject pre-built fake providers
    — the seam pattern lets them bypass the real provider registry.
    """
    try:
        from . import providers as _p
        records = _p.get_providers()
    except Exception:
        return []
    enabled = [p for p in records
               if isinstance(p, dict) and p.get("enabled", True)]
    # The active-model selection (the UI selector / ACTIVE_MODEL) is the chain
    # PRIMARY: its provider, forced to the SELECTED model. The remaining enabled
    # providers follow as fallbacks. Without this the selector was silently
    # overridden by each provider's `default_model` (the chain ignored it),
    # so picking a model in the UI never changed which model actually ran.
    pinned_id = pinned_model = ""
    try:
        from .providers import ACTIVE_MODEL
        _pin = ACTIVE_MODEL.get() or {}
        pinned_id = _pin.get("provider_id") or ""
        pinned_model = _pin.get("model") or ""
    except Exception:
        pass
    out: list[_ProviderChainAdapter] = []
    if pinned_id and pinned_model:
        head = next((p for p in enabled if p.get("id") == pinned_id), None)
        if head is not None:
            out.append(_ProviderChainAdapter(
                head, task_type=task_type, model_override=pinned_model))
            enabled = [p for p in enabled if p.get("id") != pinned_id]
    for prov in enabled:
        out.append(_ProviderChainAdapter(prov, task_type=task_type))
    return out


# Per-turn capture of WHY the router fell back off the selected provider, so
# the turn can honestly tell the user "selected X unavailable (reason)". Reset
# at turn start by run_unified; read at turn finalization.
_TURN_FALLBACK_REASON: "contextvars.ContextVar[str]" = contextvars.ContextVar(
    "turn_fallback_reason", default="",
)


def reset_turn_fallback_reason() -> None:
    _TURN_FALLBACK_REASON.set("")


def turn_fallback_reason() -> str:
    return _TURN_FALLBACK_REASON.get()


def _short_fallback_reason(err: "LLMError") -> str:
    s = str(err)
    low = s.lower()
    if "usage_limit_reached" in low or "quota" in low:
        return "quota exhausted"
    if "429" in low or "rate" in low:
        return "rate limited"
    if _is_provider_malformed(err):
        return "provider error (malformed/truncated response)"
    return s[:80].strip()


def _run_with_safety_fallback(chain, method_name, task_type, system, user, *args, **kw):
    """Iterate `chain` and call `provider.<method_name>(task_type,
    system, user, *args, **kw)` on each. If a provider raises an
    LLMError that `_is_safety_refusal` recognises, log a warning and
    advance to the next provider. Any non-safety LLMError propagates
    immediately. If every provider safety-refuses, the last refusal
    is re-raised. Empty chain -> LLMError("no providers configured").
    """
    _log = logging.getLogger("llm")
    last_err: LLMError | None = None
    for prov in chain:
        try:
            method = getattr(prov, method_name)
            return method(task_type, system, user, *args, **kw)
        except LLMError as e:
            if _should_fallback(e):
                _log.warning(
                    "router: provider %s returned a retryable failure (%s); "
                    "falling back to next. detail=%s",
                    getattr(prov, "name", "?"), _short_fallback_reason(e), e,
                )
                last_err = e
                try:
                    _TURN_FALLBACK_REASON.set(_short_fallback_reason(e))
                except Exception:
                    pass
                continue
            raise
    if last_err is not None:
        raise last_err
    raise LLMError("no providers configured")


# ---------- Token usage tracking ----------
class CallRecord:
    """Single LLM API call record."""
    __slots__ = (
        "ts", "task_type", "model", "provider",
        "input_tokens", "output_tokens", "total_tokens",
        "cache_read_tokens", "cache_creation_tokens",
        "cost_usd", "duration_ms", "prompt_preview",
    )

    def __init__(
        self,
        task_type: str = "",
        model: str = "",
        provider: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
        prompt_preview: str = "",
    ):
        self.ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.task_type = task_type
        self.model = model
        self.provider = provider
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = input_tokens + output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.cache_creation_tokens = cache_creation_tokens
        self.cost_usd = cost_usd
        self.duration_ms = duration_ms
        self.prompt_preview = prompt_preview[:300]

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "task_type": self.task_type,
            "model": self.model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "duration_ms": self.duration_ms,
            "prompt_preview": self.prompt_preview,
        }


def _utc_today() -> str:
    """Canonical day key for the daily token counter: the UTC calendar
    day. `record()` and the no-tz `stats_today()` view both use this so
    the counter's day and the queried day always agree. Previously the
    counter was keyed by `date.today()` (server-LOCAL date) while
    `stats_today` compared against a UTC day — near the UTC midnight
    boundary they disagreed and the daily view returned spurious zeros.
    A `tz=`-qualified `stats_today` still reports the operator's local
    calendar day (and zeros when it straddles the UTC counter day, as
    documented)."""
    return datetime.now(timezone.utc).date().isoformat()


class TokenTracker:
    """Tracks token usage across all LLM calls."""

    # Pricing per 1M tokens — loaded from providers module
    @staticmethod
    def _load_pricing() -> dict:
        try:
            from .providers import KNOWN_PRICING
            return KNOWN_PRICING
        except Exception:
            return {}

    DEFAULT_PRICING = {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_create": 3.75}

    def __init__(self, max_log: int = 500):
        self._lock = threading.Lock()
        self._log: list[CallRecord] = []
        self._max_log = max_log
        self._traces: list[dict] = []
        # Running totals for current request (reset per agent.run())
        self._request_input = 0
        self._request_output = 0
        self._request_cache_read = 0
        self._request_cache_create = 0
        self._request_cost = 0.0
        self._request_calls = 0
        # Per-request call list — preserved separately from the global log
        # so `request_breakdown()` can answer "which stage burned the
        # tokens on THIS turn" without scanning history. Cleared on
        # `reset_request()`. Bounded by the same max_log to keep memory
        # bounded if someone forgets to reset between requests.
        self._request_calls_log: list[CallRecord] = []
        # Global totals
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0
        self._total_calls = 0
        # Audit follow-up — daily counters (token-first telemetry).
        # router_state.json keeps `api_cost_today` based on a fixed
        # per-call estimate; the codex audit flagged that as
        # cost-first thinking. These counters are TOKEN-first +
        # cost as a derived secondary. UI reads `stats_today()`
        # via the new /api/tokens/today endpoint.
        #
        # Audit follow-up #2: counters PERSIST to disk so a service
        # restart doesn't lie about "today" usage. The file lives at
        # `~/.hrant/data/knowledge/tokens_today.json` and is touched
        # on every record() call (cheap: ~one tiny JSON write per
        # API call, ≪ 1 ms). Auto-restored on __init__.
        self._today_date = _utc_today()
        self._today_input = 0
        self._today_output = 0
        self._today_cost = 0.0
        self._today_calls = 0
        self._today_path = self._daily_counter_path()
        self._load_today_from_disk()

    def record(
        self,
        task_type: str,
        model: str,
        provider: str,
        usage: dict,
        duration_ms: int = 0,
        prompt_preview: str = "",
    ) -> CallRecord:
        """Record a single API call's token usage."""
        input_tok = usage.get("input_tokens", 0)
        output_tok = usage.get("output_tokens", 0)
        cache_read = 0
        cache_create = 0
        # Anthropic nests cache info
        cache_info = usage.get("cache_creation_input_tokens", 0)
        cache_read_info = usage.get("cache_read_input_tokens", 0)
        if cache_info:
            cache_create = cache_info
        if cache_read_info:
            cache_read = cache_read_info

        pricing = self._load_pricing().get(model, self.DEFAULT_PRICING)
        cost = (
            (input_tok / 1_000_000) * pricing["input"]
            + (output_tok / 1_000_000) * pricing["output"]
            + (cache_read / 1_000_000) * pricing.get("cache_read", 0.3)
            + (cache_create / 1_000_000) * pricing.get("cache_create", 3.75)
        )

        rec = CallRecord(
            task_type=task_type,
            model=model,
            provider=provider,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cache_read_tokens=cache_read,
            cache_creation_tokens=cache_create,
            cost_usd=cost,
            duration_ms=duration_ms,
            prompt_preview=prompt_preview,
        )

        with self._lock:
            self._log.append(rec)
            if len(self._log) > self._max_log:
                self._log = self._log[-self._max_log:]
            self._request_calls_log.append(rec)
            if len(self._request_calls_log) > self._max_log:
                self._request_calls_log = self._request_calls_log[-self._max_log:]
            self._request_input += input_tok
            self._request_output += output_tok
            self._request_cache_read += cache_read
            self._request_cache_create += cache_create
            self._request_cost += cost
            self._request_calls += 1
            self._total_input += input_tok
            self._total_output += output_tok
            self._total_cost += cost
            self._total_calls += 1
            # Audit follow-up — daily counters. Reset on date change
            # (UTC). The first call after midnight zeroes the buckets
            # so /api/tokens/today reflects the new day cleanly.
            today = _utc_today()
            if today != self._today_date:
                self._today_date = today
                self._today_input = 0
                self._today_output = 0
                self._today_cost = 0.0
                self._today_calls = 0
            self._today_input += input_tok
            self._today_output += output_tok
            self._today_cost += cost
            self._today_calls += 1
            # Persist to disk so /api/tokens/today survives service
            # restarts. Best-effort — disk failure must not break
            # the record path.
            self._flush_today_to_disk()

        return rec

    def save_request_trace(self, question: str, trace: list[dict], usage: dict) -> None:
        """Save a completed request's thinking trace for Usage page."""
        with self._lock:
            self._traces.append({
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "question": question[:200],
                "trace": trace,
                "usage": usage,
            })
            if len(self._traces) > 100:
                self._traces = self._traces[-100:]

    def recent_traces(self, limit: int = 20) -> list[dict]:
        """Return recent request traces."""
        with self._lock:
            return list(reversed(self._traces[-limit:]))

    def last_record(self) -> dict | None:
        """Snapshot of the most recently recorded `CallRecord` as a
        dict. Used by Agent._record_llm_call to attribute model name
        to a per-call dev capture without plumbing it through every
        LLM class signature. Returns None when nothing has been
        recorded yet (cold start)."""
        with self._lock:
            if not self._log:
                return None
            return self._log[-1].to_dict()

    def request_calls_count(self) -> int:
        """Current length of the per-request call log. Snapshot this
        BEFORE a tool-loop call, then call `request_calls_since(index)`
        AFTER to retrieve just the records produced by that call. Used
        by unified_agent to emit per-iteration LLM telemetry into the
        turn artifact (2026-05-21 audit follow-up — previously the
        whole tool loop collapsed into one `_unified` aggregate)."""
        with self._lock:
            return len(self._request_calls_log)

    def request_calls_since(self, start_index: int) -> list[dict]:
        """Return `CallRecord` dicts for entries added to the request
        log after `start_index`. Clamped to the current log length so
        a caller passing a stale index gets an empty list, not an
        IndexError."""
        with self._lock:
            if start_index < 0:
                start_index = 0
            log = self._request_calls_log
            if start_index >= len(log):
                return []
            return [rec.to_dict() for rec in log[start_index:]]

    def reset_request(self) -> None:
        """Reset per-request counters (called at start of agent.run())."""
        with self._lock:
            self._request_input = 0
            self._request_output = 0
            self._request_cache_read = 0
            self._request_cache_create = 0
            self._request_cost = 0.0
            self._request_calls = 0
            self._request_calls_log = []

    def request_breakdown(self) -> dict:
        """Per-stage attribution for the current request.

        Group calls by `task_type`. Stage = the prefix before the first
        colon (`solve:tool_iter_0` → "solve"); subtask = the full
        task_type. Both views are returned so callers can:
          - show a coarse "where did the tokens go" stat (stages)
          - drill into a tool-loop iteration when debugging (subtasks)

        This is the diagnostic baseline for any future structural
        optimisation — once we can see which stage owns the bill, the
        next change isn't a guess.
        """
        with self._lock:
            calls = list(self._request_calls_log)

        def _empty() -> dict:
            return {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "cost_usd": 0.0,
                "duration_ms": 0,
            }

        stages: dict[str, dict] = {}
        subtasks: dict[str, dict] = {}
        for rec in calls:
            full = rec.task_type or "(unknown)"
            stage = full.split(":", 1)[0] or "(unknown)"
            for bucket in (stages.setdefault(stage, _empty()),
                           subtasks.setdefault(full, _empty())):
                bucket["calls"] += 1
                bucket["input_tokens"] += rec.input_tokens
                bucket["output_tokens"] += rec.output_tokens
                bucket["cache_read_tokens"] += rec.cache_read_tokens
                bucket["cache_creation_tokens"] += rec.cache_creation_tokens
                bucket["cost_usd"] += rec.cost_usd
                bucket["duration_ms"] += rec.duration_ms

        # Round costs and add total_tokens convenience field, sort by
        # input_tokens descending so the heaviest stage is up top.
        def _finalize(d: dict) -> dict:
            for v in d.values():
                v["total_tokens"] = v["input_tokens"] + v["output_tokens"]
                v["cost_usd"] = round(v["cost_usd"], 6)
            return dict(sorted(
                d.items(), key=lambda kv: kv[1]["input_tokens"], reverse=True,
            ))

        return {
            "stages": _finalize(stages),
            "subtasks": _finalize(subtasks),
        }

    def request_usage(self) -> dict:
        """Get token usage for the current request."""
        with self._lock:
            return {
                "input_tokens": self._request_input,
                "output_tokens": self._request_output,
                "total_tokens": self._request_input + self._request_output,
                "cache_read_tokens": self._request_cache_read,
                "cache_creation_tokens": self._request_cache_create,
                "cost_usd": round(self._request_cost, 6),
                "llm_calls": self._request_calls,
            }

    def recent_calls(self, limit: int = 50) -> list[dict]:
        """Return recent call records."""
        with self._lock:
            calls = self._log[-limit:]
        return [r.to_dict() for r in reversed(calls)]

    # --- Daily-counter persistence (audit follow-up) ------------------

    @staticmethod
    def _daily_counter_path():
        """Path to the on-disk daily counter file. Lives under the
        user data_dir so it survives `hrant update` (engine repo gets
        wiped on update) but doesn't follow per-process state into
        random tmp dirs. Fails gracefully when data_dir isn't yet
        resolvable (early init in tests)."""
        from pathlib import Path
        try:
            from .paths import knowledge_dir
            return knowledge_dir() / "tokens_today.json"
        except Exception:
            # Pre-init: return an in-memory-only sentinel so the load
            # is a no-op. Tests that don't have data_dir wired stay
            # working. tempfile.gettempdir() is cross-platform —
            # the hardcoded "/tmp" predecessor didn't exist on Windows.
            import tempfile
            return Path(tempfile.gettempdir()) / "_hrant_tokens_today_devstub.json"

    def _load_today_from_disk(self) -> None:
        """Restore today's counters from disk on tracker init.
        If the file's date doesn't match today, the counters start
        fresh (file gets rewritten on first record)."""
        import json
        try:
            if not self._today_path.exists():
                return
            raw = json.loads(self._today_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(raw, dict):
            return
        # Only restore if the file is for the SAME date we're booting
        # under. Stale file gets overwritten on the next record().
        if raw.get("date") == self._today_date:
            self._today_input = int(raw.get("input_tokens") or 0)
            self._today_output = int(raw.get("output_tokens") or 0)
            self._today_cost = float(raw.get("cost_usd") or 0.0)
            self._today_calls = int(raw.get("llm_calls") or 0)

    def _flush_today_to_disk(self) -> None:
        """Write the current daily counters to disk. Best-effort —
        a write failure (disk full, race) doesn't break the record
        path. Atomic via .tmp rename."""
        import json
        try:
            p = self._today_path
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(p.suffix + ".tmp")
            payload = {
                "date": self._today_date,
                "input_tokens": self._today_input,
                "output_tokens": self._today_output,
                "cost_usd": round(self._today_cost, 6),
                "llm_calls": self._today_calls,
            }
            tmp.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            tmp.replace(p)
        except Exception:
            pass

    def stats_today(self, *, tz: str | None = None) -> dict:
        """Token-first daily aggregate.

        On date change (UTC), buckets auto-reset on the next
        `record()` call. The on-disk daily-counter file
        (`tokens_today.json`) survives service restarts so the
        "today" view stays honest after a redeploy.

        Audit follow-up — timezone parameter. Server lives in UTC
        but operators may be in a different zone (e.g. Asia/Yerevan
        is UTC+4 so at 02:00 local it's still 22:00 UTC the previous
        day). Pass `tz="Asia/Yerevan"` and the returned `date` field
        reflects the local calendar day. If the local-day boundary
        is on the OTHER side of the UTC boundary, the counters are
        zero (we don't have historical counter snapshots to retro-
        fit). Defaults to no-tz which returns UTC's view, same as
        the original behaviour.
        """
        with self._lock:
            # Determine "today" under the requested tz, if any.
            if tz:
                try:
                    import datetime as _dt
                    try:
                        from zoneinfo import ZoneInfo
                        local_now = _dt.datetime.now(ZoneInfo(tz))
                    except Exception:
                        # zoneinfo missing or invalid tz — fall back
                        # to UTC silently.
                        local_now = _dt.datetime.now(_dt.timezone.utc)
                    today_view = local_now.date().isoformat()
                except Exception:
                    today_view = _utc_today()
            else:
                today_view = _utc_today()
            in_ = self._today_input
            out = self._today_output
            cost = self._today_cost
            calls = self._today_calls
            counter_date = self._today_date
            if today_view != counter_date:
                # Counter holds a different date than the requested
                # tz-local one. We don't lie about totals — return
                # zeros under the requested date and surface the
                # mismatch via the `counter_date` field so the UI
                # can show "since UTC midnight" if it cares.
                return {
                    "date": today_view,
                    "counter_date": counter_date,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "input_output_ratio": 0.0,
                    "cost_usd": 0.0,
                    "llm_calls": 0,
                }
            ratio = round(in_ / max(1, out), 2) if out > 0 else float(in_)
            return {
                "date": counter_date,
                "input_tokens": in_,
                "output_tokens": out,
                "total_tokens": in_ + out,
                "input_output_ratio": ratio,
                "cost_usd": round(cost, 6),
                "llm_calls": calls,
            }

    def stats(self) -> dict:
        """Overall statistics."""
        with self._lock:
            by_task: dict[str, dict] = {}
            by_model: dict[str, dict] = {}
            for r in self._log:
                # By task type
                if r.task_type not in by_task:
                    by_task[r.task_type] = {"calls": 0, "input": 0, "output": 0, "cost": 0.0}
                by_task[r.task_type]["calls"] += 1
                by_task[r.task_type]["input"] += r.input_tokens
                by_task[r.task_type]["output"] += r.output_tokens
                by_task[r.task_type]["cost"] += r.cost_usd
                # By model
                if r.model not in by_model:
                    by_model[r.model] = {"calls": 0, "input": 0, "output": 0, "cost": 0.0}
                by_model[r.model]["calls"] += 1
                by_model[r.model]["input"] += r.input_tokens
                by_model[r.model]["output"] += r.output_tokens
                by_model[r.model]["cost"] += r.cost_usd

            # Round costs
            for d in by_task.values():
                d["cost"] = round(d["cost"], 4)
            for d in by_model.values():
                d["cost"] = round(d["cost"], 4)

            return {
                "total_calls": self._total_calls,
                "total_input_tokens": self._total_input,
                "total_output_tokens": self._total_output,
                "total_cost_usd": round(self._total_cost, 4),
                "by_task_type": by_task,
                "by_model": by_model,
            }


TOKENS = TokenTracker()


# ---------- TaskType ----------
class TaskType(Enum):
    TASK_ANALYSIS = "task_analysis"
    LEARNING = "learning"
    COMPLEX_SOLVING = "complex_solving"
    VERIFICATION = "verification"
    NOTE_CREATION = "note_creation"
    SIMPLE_LOOKUP = "simple_lookup"
    KEYWORD_EXTRACTION = "keyword_extraction"
    NOTE_SEARCH = "note_search"
    QUICK_ANSWER = "quick_answer"
    CLASSIFICATION = "classification"
    # Post-turn meta-task: the LLM reviews its own work and decides
    # whether to write a reusable skill. Separated from complex_solving
    # so telemetry can attribute the cost cleanly and so the reasoning
    # router can dial it down to `medium` (the default is `high` for
    # complex_solving, which is overkill for reflection). Audit
    # follow-up 2026-05-21 — previously masqueraded as complex_solving.
    SKILL_REFLECTION = "skill_reflection"
    # Post-verifier critic-revise pass (AGI roadmap 2026-06-11): the
    # answer gets one evidence-grounded revision when the verifier
    # found contradictions / unsupported claims. Separate member so
    # telemetry attributes the cost and the reasoning router's
    # existing "self_critic": "high" entry finally has a consumer.
    SELF_CRITIC = "self_critic"


MODEL_A_TASKS = {
    TaskType.TASK_ANALYSIS,
    TaskType.LEARNING,
    TaskType.COMPLEX_SOLVING,
    TaskType.VERIFICATION,
    TaskType.NOTE_CREATION,
}

MODEL_B_TASKS = {
    TaskType.SIMPLE_LOOKUP,
    TaskType.KEYWORD_EXTRACTION,
    TaskType.NOTE_SEARCH,
    TaskType.QUICK_ANSWER,
    TaskType.CLASSIFICATION,
}


# ---------- base JSON parser ----------
def _parse_json_response(raw: str) -> dict:
    # A provider can hand back None / "" (empty completion, blanked by a
    # safety filter, or a failed call that returned nothing). Guard before
    # .strip() so JSON callers (call_json) get a clean LLMError instead of a
    # cryptic "'NoneType' object has no attribute 'strip'" — which surfaced as
    # repeated self_study / self_reflection lever failures (2026-06-20 logs).
    if not raw or not raw.strip():
        raise LLMError("LLM returned an empty response")
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"LLM returned non-JSON: {raw[:200]}")
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"JSON parse error: {e}\n{raw[:300]}") from e


# ---------- providers ----------
def _resolve_attachments(refs: list[str] | None) -> list[tuple]:
    """Resolve a list of sha256 ids to (meta, bytes) tuples.

    Returns only entries that actually exist on disk; missing ids are
    silently dropped (they'd be a UX bug in the caller, not the LLM's
    problem). Used by vision-capable LLM classes to inline image bytes
    or pull voice transcripts.
    """
    from .attachments import ATTACHMENTS
    out: list[tuple] = []
    for sha in refs or []:
        if not sha:
            continue
        meta = ATTACHMENTS.get_meta(sha)
        if not meta:
            continue
        data = ATTACHMENTS.get_bytes(sha)
        if data is None:
            continue
        out.append((meta, data))
    return out


# Caps applied to tool results BEFORE they re-enter `messages` for the
# next iteration of a tool-use loop. Without this, a 60k `read_file`
# of agent.py travels with every subsequent turn — the cumulative
# growth dominates the input-token bill on review tasks. The agent's
# verifier-side `tool_outputs` cap is separate; that one feeds the
# verifier prompt, not the next solver iteration.
_TOOL_LLM_RESULT_CAPS: dict[str, int] = {
    "calc": 1000,
    "web_search": 4000,
    "fetch_url": 8000,
    # Round 11: reverted 12k -> 16k. Tighter cap from Round 10 was
    # starving the model of context (verifier confidence dropped
    # noticeably on self-analysis turns — answers cited line ranges
    # they hadn't actually seen). The real fix for the 278k blowup
    # is the curated forced-synthesis payload below — caps are a
    # blunt instrument and they were costing more in answer quality
    # than they were saving in input tokens.
    "read_file": 16000,
    "view_file": 16000,
    "read_note": 8000,
    "run_python": 12000,
    "list_files": 4000,
    "glob": 4000,
    "grep": 4000,
    "search": 4000,
}
_TOOL_LLM_DEFAULT_CAP = 6000
_TOOL_LLM_ERROR_CAP = 2000


def _tool_loop_input_budget_exceeded() -> bool:
    """Has the current request already burned through the per-loop
    input-token budget? Used by every `complete_with_tools` to break
    a long-running tool-use chain before it spirals further. Single
    source of truth so a future tweak (lower cap, separate caps per
    provider) only changes one place."""
    try:
        used = TOKENS.request_usage().get("input_tokens", 0)
    except Exception:
        return False
    cap = int(CONFIG.router.get("tool_loop_input_budget", 200000) or 0)
    return cap > 0 and used >= cap


def _compact_tool_result_for_llm(
    name: str, result: str, *, is_error: bool = False
) -> str:
    """Truncate a tool result before re-feeding it to the LLM in the
    next tool-loop iteration. Errors get a tighter cap because they
    rarely contain new information past the first message + a stack
    snippet. Successful results use a per-tool cap (read_file gets
    16k, calc 1k, etc.).

    The truncation marker tells the model the body was cut so it
    doesn't try to "read the rest" by re-calling the same tool with
    different arguments — that was a real failure mode before
    `read_file` got start_line/end_line in Round 3.
    """
    if not result:
        return result
    cap = _TOOL_LLM_ERROR_CAP if is_error else _TOOL_LLM_RESULT_CAPS.get(
        name, _TOOL_LLM_DEFAULT_CAP
    )
    if len(result) <= cap:
        return result
    return (
        result[:cap]
        + f"\n…[+{len(result) - cap} chars truncated before re-feeding "
        f"to the LLM. Use `read_file(start_line=…, end_line=…)` for "
        f"specific regions.]"
    )


# --- Curated forced-synthesis payload ---------------------------------------
#
# Round 11. Every `complete_with_tools` runs a final tool-less call when
# `max_iterations` is hit so the user gets a real answer instead of the
# last tool-call preamble. Before this round, that synthesis call inherited
# the WHOLE accumulated `messages` array — every assistant tool_use, every
# tool_result. On long self-reviews this dominated the input bill (a
# 14-iteration loop carried 14 × ~12k tool_results into the final synth).
#
# We replace `messages` for the synth call with one curated user turn:
#   - the original task verbatim
#   - a one-line digest per tool call (name + short args + first line of
#     result + "(+N more)") — gives the model a ledger without the bodies
#   - the model's own running narration (assistant text from prior turns)
#   - a "now answer, no tools" directive
#
# Tool result bodies never reach the synth payload. The model has been
# reasoning about them across iterations; the digest is enough for it to
# write the final answer it was already drafting.
_SYNTH_DIGEST_CHAR_CAP = 3000
_SYNTH_NARRATION_CHAR_CAP = 4000


def _summarize_tool_call_for_synth(
    name: str, args, result: str, *, is_error: bool = False
) -> str:
    """One-line digest of a tool call for the curated synthesis prompt."""
    try:
        if isinstance(args, str):
            args_str = args
        else:
            args_str = json.dumps(args, ensure_ascii=False)
    except Exception:
        args_str = str(args)
    if len(args_str) > 200:
        args_str = args_str[:200] + "…}"
    result_str = (result or "").strip()
    if not result_str:
        head = "(empty result)"
        more = 0
    else:
        lines = result_str.splitlines()
        head = lines[0]
        if len(head) > 280:
            head = head[:280] + "…"
        more = len(lines) - 1
    err_marker = " [ERR]" if is_error else ""
    suffix = f" (+{more} more lines)" if more > 0 else ""
    return f"- {name}({args_str}){err_marker} → {head}{suffix}"


def _build_synth_user_text(
    original_user: str,
    digest_lines: list[str],
    narration_chunks: list[str],
    *,
    digest_cap: int = _SYNTH_DIGEST_CHAR_CAP,
    narration_cap: int = _SYNTH_NARRATION_CHAR_CAP,
) -> str:
    """Compose the single user message sent to the forced synthesis call.
    Caps prevent a runaway loop's digest from itself blowing out the synth
    input — same reason we cap individual tool results."""
    parts = [original_user.rstrip()]
    if digest_lines:
        digest_text = "\n".join(digest_lines)
        if len(digest_text) > digest_cap:
            digest_text = digest_text[:digest_cap] + "\n… (digest truncated)"
        parts.append(
            "\n\n## Work you already did this turn — tool calls + actions "
            "(do NOT repeat these calls)\n"
            + digest_text
        )
    if narration_chunks:
        narration = "\n\n".join(narration_chunks)
        if len(narration) > narration_cap:
            narration = narration[:narration_cap] + "\n… (narration truncated)"
        parts.append("\n\n## My running analysis from prior turns\n" + narration)
    parts.append(
        "\n\nWrite the FINAL answer now from the work above. Tools are disabled "
        "for this turn. The summary above is what YOU already did this turn — if "
        "it shows you completed actions (wrote a file, started a server, set a "
        "reminder, created a tracker, applied a patch), report plainly what you "
        "accomplished and where. Do NOT disclaim or cast doubt on actions the "
        "summary shows you took — recognize your own success and close the loop. "
        "Only flag as unfinished what genuinely was not done; don't guess about "
        "facts you never gathered."
    )
    return "".join(parts)


def _flatten_anthropic_text(content) -> str:
    """Anthropic content can be a string or a list of blocks. Return only
    the text portion (text + input_text); ignore image / tool_use blocks."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content) if content is not None else ""
    out: list[str] = []
    for b in content:
        if isinstance(b, dict) and b.get("type") in ("text", "input_text"):
            out.append(b.get("text", "") or "")
    return "".join(out)


def _extract_synth_inputs_anthropic(
    messages: list[dict],
) -> tuple[str, list[str], list[str]]:
    """Walk Anthropic-format tool-loop messages and return
    (original_user_text, digest_lines, narration_chunks)."""
    if not messages:
        return ("", [], [])
    original_user = _flatten_anthropic_text(messages[0].get("content", ""))
    digest_lines: list[str] = []
    narration_chunks: list[str] = []
    pending: dict[str, dict] = {}
    for msg in messages[1:]:
        role = msg.get("role")
        content = msg.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list):
            continue
        if role == "assistant":
            for b in content:
                if not isinstance(b, dict):
                    continue
                btype = b.get("type")
                if btype == "text":
                    txt = (b.get("text") or "").strip()
                    if txt:
                        narration_chunks.append(txt)
                elif btype == "tool_use":
                    pending[b.get("id", "")] = {
                        "name": b.get("name", ""),
                        "args": b.get("input") or {},
                    }
        elif role == "user":
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_result":
                    info = pending.pop(b.get("tool_use_id", ""), {})
                    raw = b.get("content", "")
                    if isinstance(raw, list):
                        result_text = "".join(
                            x.get("text", "") for x in raw
                            if isinstance(x, dict) and x.get("type") == "text"
                        )
                    else:
                        result_text = str(raw or "")
                    digest_lines.append(_summarize_tool_call_for_synth(
                        info.get("name", "?"),
                        info.get("args", {}),
                        result_text,
                        is_error=bool(b.get("is_error")),
                    ))
    return (original_user, digest_lines, narration_chunks)


def _curate_synth_messages_anthropic(messages: list[dict]) -> list[dict]:
    """Anthropic / Bedrock: replace accumulated tool-loop messages with
    one curated user turn. Caller still passes `system` separately."""
    original, digest, narration = _extract_synth_inputs_anthropic(messages)
    return [{"role": "user", "content": _build_synth_user_text(
        original, digest, narration,
    )}]


def _extract_synth_inputs_openai(
    messages: list[dict],
) -> tuple[str, list[str], list[str]]:
    """OpenAI-compat: assistant has `tool_calls`, tool role has
    `tool_call_id` + content."""
    if not messages:
        return ("", [], [])
    original_user = ""
    pending: dict[str, dict] = {}
    digest_lines: list[str] = []
    narration_chunks: list[str] = []
    user_seen = False
    for m in messages:
        role = m.get("role")
        if role == "user" and not user_seen:
            content = m.get("content", "")
            if isinstance(content, list):
                original_user = "".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") in ("text", "input_text")
                )
            else:
                original_user = str(content or "")
            user_seen = True
            continue
        if not user_seen:
            continue
        if role == "assistant":
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                narration_chunks.append(content.strip())
            for tc in m.get("tool_calls") or []:
                func = tc.get("function", {}) or {}
                pending[tc.get("id", "")] = {
                    "name": func.get("name", ""),
                    "args": func.get("arguments", "{}"),
                }
        elif role == "tool":
            info = pending.pop(m.get("tool_call_id", ""), {})
            args_raw = info.get("args", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {"_raw": args_raw}
            content = m.get("content", "")
            if isinstance(content, list):
                result_text = "".join(
                    x.get("text", "") for x in content
                    if isinstance(x, dict)
                )
            else:
                result_text = str(content or "")
            digest_lines.append(_summarize_tool_call_for_synth(
                info.get("name", "?"), args, result_text,
            ))
    return (original_user, digest_lines, narration_chunks)


def _curate_synth_messages_openai(messages: list[dict]) -> list[dict]:
    """OpenAI-compat: keep leading system message(s), replace the rest
    with one curated user turn."""
    if not messages:
        return messages
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            out.append(m)
        else:
            break
    original, digest, narration = _extract_synth_inputs_openai(messages)
    out.append({"role": "user", "content": _build_synth_user_text(
        original, digest, narration,
    )})
    return out


def _extract_synth_inputs_codex(
    input_items: list[dict],
) -> tuple[str, list[str], list[str]]:
    """Codex Responses API uses `input_items` with `message`,
    `function_call`, `function_call_output`, and reasoning items.
    Reasoning items are tied to the live tool-use chain — useless for
    a tool-less synth call, drop them."""
    if not input_items:
        return ("", [], [])
    original_user = ""
    pending: dict[str, dict] = {}
    digest_lines: list[str] = []
    narration_chunks: list[str] = []
    user_seen = False
    for item in input_items:
        itype = item.get("type")
        if itype == "message":
            role = item.get("role")
            content = item.get("content", [])
            text = ""
            if isinstance(content, list):
                text = "".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") in (
                        "input_text", "output_text", "text"
                    )
                )
            elif isinstance(content, str):
                text = content
            if role == "user" and not user_seen:
                original_user = text
                user_seen = True
            elif role == "assistant" and text.strip():
                narration_chunks.append(text.strip())
        elif itype == "function_call":
            cid = item.get("call_id") or item.get("id", "")
            pending[cid] = {
                "name": item.get("name", ""),
                "args": item.get("arguments", "{}"),
            }
        elif itype == "function_call_output":
            info = pending.pop(item.get("call_id", ""), {})
            args_raw = info.get("args", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {"_raw": args_raw}
            digest_lines.append(_summarize_tool_call_for_synth(
                info.get("name", "?"), args, item.get("output", "") or "",
            ))
    return (original_user, digest_lines, narration_chunks)


def _curate_synth_input_items_codex(input_items: list[dict]) -> list[dict]:
    """Codex Responses API: replace input_items with one curated user
    message. Drops all function_call / function_call_output / reasoning
    items — none are needed for a tool-less synth."""
    original, digest, narration = _extract_synth_inputs_codex(input_items)
    return [{
        "type": "message",
        "role": "user",
        "content": [{
            "type": "input_text",
            "text": _build_synth_user_text(original, digest, narration),
        }],
    }]


def _extract_synth_inputs_cohere(
    messages: list[dict],
) -> tuple[str, list[str], list[str]]:
    """Cohere v2: assistant has `tool_plan` plus `tool_calls`; tool role
    has `tool_call_id` and content as `[{"type":"text","text":...}]`."""
    if not messages:
        return ("", [], [])
    original_user = ""
    pending: dict[str, dict] = {}
    digest_lines: list[str] = []
    narration_chunks: list[str] = []
    user_seen = False
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user" and not user_seen:
            original_user = str(m.get("content", "") or "")
            user_seen = True
            continue
        if not user_seen:
            continue
        if role == "assistant":
            plan = (m.get("tool_plan") or "").strip()
            if plan:
                narration_chunks.append(plan)
            content = m.get("content")
            if isinstance(content, str) and content.strip():
                narration_chunks.append(content.strip())
            for tc in m.get("tool_calls") or []:
                func = tc.get("function", {}) or {}
                pending[tc.get("id", "")] = {
                    "name": func.get("name", ""),
                    "args": func.get("arguments", "{}"),
                }
        elif role == "tool":
            info = pending.pop(m.get("tool_call_id", ""), {})
            args_raw = info.get("args", "{}")
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {"_raw": args_raw}
            content = m.get("content", "")
            if isinstance(content, list):
                result_text = "".join(
                    x.get("text", "") for x in content
                    if isinstance(x, dict) and x.get("type") == "text"
                )
            else:
                result_text = str(content or "")
            digest_lines.append(_summarize_tool_call_for_synth(
                info.get("name", "?"), args, result_text,
            ))
    return (original_user, digest_lines, narration_chunks)


def _curate_synth_messages_cohere(messages: list[dict]) -> list[dict]:
    if not messages:
        return messages
    out: list[dict] = []
    for m in messages:
        if m.get("role") == "system":
            out.append(m)
        else:
            break
    original, digest, narration = _extract_synth_inputs_cohere(messages)
    out.append({"role": "user", "content": _build_synth_user_text(
        original, digest, narration,
    )})
    return out


# ─── HTTP retry helper (audit #22) ─────────────────────────────────────
#
# Every provider client (Anthropic, OpenAI-compatible, Codex, Cohere,
# Google, Bedrock, Copilot, Ollama — 8 in total) does roughly the
# same thing in its `_post`:
#   1. POST to a URL with headers + JSON payload
#   2. On 4xx/5xx/connection error, retry with exponential backoff
#   3. On non-retryable status, raise LLMError with a parsed message
# Pre-fix that loop was duplicated ~8× with subtle bugs in each
# copy. The helper below extracts the common shape; provider
# clients can call it (or keep their own `_post` for back-compat).
# Migrating one client at a time is safer than a big-bang rewrite.

# Default set of statuses worth retrying. Providers can override.
_DEFAULT_RETRYABLE_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 529})


def post_with_retry(
    url: str,
    *,
    payload: dict,
    headers: dict,
    provider_name: str,
    model: str = "",
    timeout: float = 120.0,
    max_retries: int = 5,
    retryable_statuses: frozenset[int] = _DEFAULT_RETRYABLE_STATUSES,
    parse_error: Optional[Callable[[httpx.HTTPStatusError], str]] = None,
) -> dict:
    """POST `payload` to `url` with retry on transient errors. On
    final failure raises `LLMError(provider/model: detail)`.

    `parse_error` lets the caller customise how a non-retryable
    response gets formatted (Anthropic / OpenAI / Cohere each
    have different error JSON shapes). Default extracts `error.message`
    from the response JSON, falling back to the raw body.
    """
    log = logging.getLogger("llm")
    last_error: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
            # OpenRouter (and some OpenAI-compat gateways) can return HTTP
            # 200 with an {"error": {...}} body and NO choices when the
            # upstream model rate-limits / errors under burst. Treat that
            # like a transient failure: retry with backoff, then raise
            # cleanly. Never hand a choices-less body back to the caller —
            # it would KeyError deep in the tool loop and crash the turn.
            if isinstance(data, dict) and data.get("error") and not data.get("choices"):
                _err = data.get("error")
                _msg = _err.get("message") if isinstance(_err, dict) else str(_err)
                if attempt < max_retries:
                    wait = min(2 ** attempt * 2, 60.0)
                    log.warning(
                        "%s returned a 200 error-body (%s), retry %d/%d in %.1fs...",
                        provider_name, _msg, attempt + 1, max_retries, wait,
                    )
                    time.sleep(wait)
                    continue
                _tag = f"(model={model!r})" if model else ""
                raise LLMError(
                    f"{provider_name} API 200-error {_tag}: {_msg}".strip()
                )
            return data
        except httpx.HTTPStatusError as e:
            last_error = e
            status = e.response.status_code
            if status in retryable_statuses and attempt < max_retries:
                retry_after = e.response.headers.get("retry-after")
                if retry_after:
                    try:
                        wait = min(float(retry_after), 60.0)
                    except ValueError:
                        wait = min(2 ** attempt * 2, 60.0)
                else:
                    wait = min(2 ** attempt * 2, 60.0)
                log.warning(
                    "%s API %s, retry %d/%d in %.1fs...",
                    provider_name, status, attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
                continue
            # Non-retryable status or retries exhausted — format detail.
            if parse_error is not None:
                detail = parse_error(e)
            else:
                body = (e.response.text or "").strip()
                try:
                    err = e.response.json().get("error", {})
                    detail = f"{err.get('type', '?')}: {err.get('message', body)}"
                except Exception:
                    detail = body or str(e)
            tag = f"(model={model!r})" if model else ""
            raise LLMError(
                f"{provider_name} API {status} {tag}: {detail}".strip()
            ) from e
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            last_error = e
            if attempt < max_retries:
                wait = min(2 ** attempt * 2, 60.0)
                log.warning(
                    "%s connection error: %s, retry %d/%d in %.1fs...",
                    provider_name, e, attempt + 1, max_retries, wait,
                )
                time.sleep(wait)
                continue
            raise LLMError(
                f"{provider_name} API error after {max_retries} retries: {e}"
            ) from e
    raise LLMError(
        f"{provider_name} API failed after {max_retries} retries: {last_error}"
    )


class BaseLLM:
    def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        attachments: list[str] | None = None,
    ) -> str:
        """Subclasses override. `attachments` is a list of sha256 ids
        resolvable via backend.attachments.ATTACHMENTS — vision-capable
        backends inline image bytes; others ignore (text-only)."""
        raise NotImplementedError


class AnthropicLLM(BaseLLM):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        key_env = cfg.get("api_key_env", "ANTHROPIC_API_KEY")
        self.api_key = os.getenv(key_env)
        if not self.api_key:
            raise LLMError(f"{key_env} is not set in the environment/.env")
        self.model = cfg["model"]
        self.default_max = cfg.get("max_tokens", 2000)
        self.default_temp = cfg.get("temperature", 0.3)
        self.url = "https://api.anthropic.com/v1/messages"

    # ----- low-level POST -----
    def _post(self, payload: dict, *, _max_retries: int = 5) -> dict:
        """POST to Anthropic API with automatic retry on transient errors.

        Audit #22: delegates the retry loop to `post_with_retry` so
        the shared logic (exponential backoff, Retry-After handling,
        retryable-status set) lives in one place instead of being
        copy-pasted across the 8 provider clients."""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        return post_with_retry(
            self.url,
            payload=payload,
            headers=headers,
            provider_name="Anthropic",
            model=self.model,
            max_retries=_max_retries,
        )

    @staticmethod
    def _build_user_content(user: str, attachments: list[str] | None) -> "str | list[dict]":
        """Anthropic content array with images inlined as base64.

        For audio attachments we use the transcript (Anthropic doesn't
        accept raw audio); fallback to placeholder if no transcript yet.
        For video: lazily ffmpeg-extract N sampled frames + audio
        transcript via `video_processor.preprocess_video`, then inline
        the frames as images and the transcript as text — see
        backend/tools/video_processor.py for the cache semantics.
        """
        resolved = _resolve_attachments(attachments)
        if not resolved:
            return user
        import base64 as _b64
        blocks: list[dict] = []

        def _emit_image_block(image_meta, image_bytes: bytes) -> None:
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": image_meta.mime_type,
                    "data": _b64.b64encode(image_bytes).decode("ascii"),
                },
            })

        for meta, data in resolved:
            if meta.kind == "image":
                _emit_image_block(meta, data)
            elif meta.kind == "audio":
                if meta.transcript:
                    blocks.append({"type": "text", "text": f"[voice transcript]\n{meta.transcript}"})
                else:
                    blocks.append({"type": "text", "text": "[voice attachment, transcript unavailable]"})
            elif meta.kind == "video":
                from .tools.video_processor import preprocess_video
                from .attachments import ATTACHMENTS as _ATT
                pre = preprocess_video(meta.sha256)
                blocks.append({
                    "type": "text",
                    "text": (
                        f"[video {meta.filename or meta.sha256[:12]}: "
                        f"{pre.duration_seconds:.1f}s, "
                        f"{len(pre.frame_shas)} sampled frames "
                        f"(evenly along the timeline)]"
                    ),
                })
                for fsha in pre.frame_shas:
                    fmeta = _ATT.get_meta(fsha)
                    fdata = _ATT.get_bytes(fsha)
                    if fmeta and fdata:
                        _emit_image_block(fmeta, fdata)
                if pre.audio_transcript:
                    blocks.append({
                        "type": "text",
                        "text": f"[video audio transcript]\n{pre.audio_transcript}",
                    })
            else:
                blocks.append({"type": "text", "text": f"[file attachment: {meta.filename or meta.sha256[:12]}]"})
        if user:
            blocks.append({"type": "text", "text": user})
        return blocks

    def complete(self, system, user, *, max_tokens=None, temperature=None,
                 attachments=None, _task_type: str = ""):
        content = self._build_user_content(user, attachments)
        payload = {
            "model": self.model,
            "max_tokens": max_tokens or self.default_max,
            "temperature": temperature if temperature is not None else self.default_temp,
            "system": system,
            "messages": [{"role": "user", "content": content}],
        }
        t0 = time.time()
        data = self._post(payload)
        duration_ms = int((time.time() - t0) * 1000)
        # Record token usage
        usage = data.get("usage", {})
        if usage:
            TOKENS.record(
                task_type=_task_type,
                model=self.model,
                provider="anthropic",
                usage=usage,
                duration_ms=duration_ms,
                prompt_preview=user[:300],
            )
        try:
            for block in data.get("content", []):
                if block.get("type") == "text":
                    return block.get("text", "")
            return data["content"][0]["text"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected response format: {data}") from e

    # ---------- tool-use loop ----------
    def complete_with_tools(
        self,
        system: str,
        user: str,
        tools: list[dict],
        execute_tool: "ToolExecutor",
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_iterations: int = 6,
        on_tool_call: "ToolCallCB | None" = None,
        attachments: list[str] | None = None,
        _task_type: str = "",
        tools_provider=None,
    ) -> str:
        """Multi-turn tool-use loop.

        Steps:
          1. Send messages + tools.
          2. If the LLM returns stop_reason=tool_use, execute all
             tool_use blocks and put tool_result back into messages.
          3. Repeat until stop_reason=end_turn (or max_iterations).

        Returns the final assistant text (concatenated text blocks).
        Degrades safely when tools is empty or the model returns text
        immediately. `attachments` (sha256 list) attach images on
        the first user turn.

        Phase 2 (2026-05-23): `tools_provider`, when set, is called
        before each iteration to re-derive the tool schema. Lets
        `load_tool_bundle` take effect in the next iteration of the
        same turn. When None, falls back to the static `tools` arg.
        """
        first_content = self._build_user_content(user, attachments)
        messages: list[dict] = [{"role": "user", "content": first_content}]
        final_text = ""
        for _iter in range(max_iterations):
            # Per-loop input budget guard. The first iteration always
            # runs (we need to call the LLM at least once); subsequent
            # ones break out if we've already burned through the cap
            # so the forced synthesis below can wrap up cheaply.
            if _iter > 0 and _tool_loop_input_budget_exceeded():
                break
            # Phase 2: per-iteration tool list — bundle aware when
            # tools_provider supplied.
            if tools_provider is not None:
                try:
                    current_tools = tools_provider()
                except Exception as e:
                    log.warning(
                        "tools_provider raised, falling back to "
                        "static tools list: %s", e,
                    )
                    current_tools = tools
            else:
                current_tools = tools
            # Prompt caching: wrap the system prompt in a content-block list
            # and mark the last tool with cache_control so the
            # system+tools prefix is cached across tool-loop iterations.
            # Only costs cache_creation on the first iteration; all
            # subsequent iterations pay the much-cheaper cache_read rate.
            cached_system = _build_cached_system_blocks(system)
            cached_tools = _apply_cache_control_to_tools(current_tools) if current_tools else current_tools
            payload = {
                "model": self.model,
                "max_tokens": max_tokens or self.default_max,
                "temperature": temperature if temperature is not None else self.default_temp,
                "system": cached_system,
                "messages": messages,
            }
            if cached_tools:
                payload["tools"] = cached_tools
            t0 = time.time()
            data = self._post(payload)
            duration_ms = int((time.time() - t0) * 1000)
            # Record token usage for each iteration
            usage = data.get("usage", {})
            if usage:
                TOKENS.record(
                    task_type=f"{_task_type}:tool_iter_{_iter}",
                    model=self.model,
                    provider="anthropic",
                    usage=usage,
                    duration_ms=duration_ms,
                    prompt_preview=user[:300] if _iter == 0 else f"(tool iteration {_iter})",
                )

            content_blocks = data.get("content", [])
            stop_reason = data.get("stop_reason", "end_turn")

            # Collect text and tool_use blocks
            text_parts: list[str] = []
            tool_uses: list[dict] = []
            for block in content_blocks:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_uses.append(block)

            # Only capture text when this turn is NOT a tool-use turn.
            # Otherwise the text is preamble narration ("Now I will check
            # the source...") that travels alongside a tool_use block —
            # if we let it overwrite final_text and then hit max_iterations,
            # we'd return the preamble as the user-facing answer.
            if text_parts and not tool_uses:
                final_text = "\n".join(p for p in text_parts if p).strip()

            if stop_reason != "tool_use" or not tool_uses:
                return final_text

            # Put the assistant reply (with all blocks) and tool_result into messages
            messages.append({"role": "assistant", "content": content_blocks})
            tool_results = []
            for tu in tool_uses:
                name = tu.get("name", "")
                args = tu.get("input", {}) or {}
                tool_use_id = tu.get("id", "")
                result_text, is_error = execute_tool(name, args)
                if on_tool_call:
                    try:
                        on_tool_call(name, args, result_text, is_error)
                    except Exception:
                        pass
                # Truncate before sending back to LLM (see
                # `_compact_tool_result_for_llm` for the per-tool caps).
                # The on_tool_call callback above gets the FULL body so
                # the agent's verifier-side / dev capture buffers stay
                # accurate — only the next-iteration LLM input is cut.
                llm_result = _compact_tool_result_for_llm(
                    name, result_text, is_error=is_error,
                )
                result_block: dict = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": llm_result,
                }
                if is_error:
                    result_block["is_error"] = True
                tool_results.append(result_block)
            messages.append({"role": "user", "content": tool_results})

        # Hit max_iterations with the model still wanting to call tools.
        # Force one final tool-less synthesis call so the user gets a
        # real answer instead of the last preamble. The synthesis budget
        # is generous — review-style tasks (`analyze your code`) ran into
        # the regular default and ended mid-sentence. 6000 fits the
        # longest reviews we've seen and stays well under model caps.
        synth_max = max(
            max_tokens or self.default_max,
            int(CONFIG.router.get("tool_synth_max_tokens", 4000)),
        )
        synth_payload = {
            "model": self.model,
            "max_tokens": synth_max,
            "temperature": temperature if temperature is not None else self.default_temp,
            "system": _build_cached_system_blocks(system),
            "messages": _curate_synth_messages_anthropic(messages),
        }
        try:
            t0 = time.time()
            data = self._post(synth_payload)
            duration_ms = int((time.time() - t0) * 1000)
            usage = data.get("usage", {})
            if usage:
                TOKENS.record(
                    task_type=f"{_task_type}:tool_synth",
                    model=self.model,
                    provider="anthropic",
                    usage=usage,
                    duration_ms=duration_ms,
                    prompt_preview="(forced final synthesis)",
                )
            synth_text = "\n".join(
                b.get("text", "") for b in data.get("content", [])
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if synth_text:
                return synth_text
        except Exception:
            pass
        return final_text or "[max tool-use iterations reached]"


# ----- type aliases for the tool loop -----
from typing import Callable as _Callable, Tuple as _Tuple

ToolExecutor = _Callable[[str, dict], _Tuple[str, bool]]
ToolCallCB = _Callable[[str, dict, str, bool], None]


# ---------- prompt-caching helpers ----------

def _is_anthropic_model(model: str) -> bool:
    """Return True when the model string indicates an Anthropic/Claude model.

    Covers:
      - Native AnthropicLLM models (e.g. "claude-3-5-sonnet-20241022")
      - OpenRouter Anthropic models (e.g. "anthropic/claude-sonnet-4-5")
      - Bedrock Anthropic cross-region prefixes (e.g. "us.anthropic.claude-…")
    """
    m = model.lower()
    return "claude" in m or m.startswith("anthropic/") or m.startswith("anthropic.")


def _build_cached_system_blocks(system: str) -> list[dict]:
    """Convert a plain system string to an Anthropic content-block list with
    a cache breakpoint on the last (and only) block.

    Anthropic docs (2024-10): set cache_control on a content block to mark
    the boundary up to which the prefix should be cached.  We put it on the
    single text block that comprises the system prompt so the ENTIRE system
    prompt is cached.
    """
    return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]


def _apply_cache_control_to_tools(tools: list[dict]) -> list[dict]:
    """Return a shallow copy of the tools list with cache_control added to the
    last entry.

    Anthropic docs (2024-10): placing cache_control on the last tool in the
    tools array marks the system+tools prefix as a cache boundary, so every
    subsequent iteration that sends the same system+tools pays cache-read cost
    instead of input-token cost.
    """
    if not tools:
        return tools
    out = list(tools)          # shallow copy — do not mutate caller's list
    last = dict(out[-1])       # copy the last tool
    last["cache_control"] = {"type": "ephemeral"}
    out[-1] = last
    return out


def _openai_usage_to_record(usage: dict) -> dict:
    """Normalize an OpenAI-compatible `usage` block into the dict
    TokenTracker.record expects, surfacing prompt-cache info that
    OpenRouter/OpenAI nest under `prompt_tokens_details`.

    OpenAI semantics differ from Anthropic: `prompt_tokens` INCLUDES the
    cached + cache-write tokens, while TokenTracker expects `input_tokens`
    to be the NON-cached portion with cache read/write counted separately.
    Splitting them keeps the cost accurate and makes the cache savings
    visible (cached prompt tokens bill ~10x cheaper)."""
    prompt = usage.get("prompt_tokens", 0) or 0
    ptd = usage.get("prompt_tokens_details") or {}
    cached = ptd.get("cached_tokens", 0) or 0
    cache_write = ptd.get("cache_write_tokens", 0) or 0
    return {
        "input_tokens": max(0, prompt - cached - cache_write),
        "output_tokens": usage.get("completion_tokens", 0) or 0,
        "cache_read_input_tokens": cached,
        "cache_creation_input_tokens": cache_write,
    }


def _apply_openrouter_anthropic_cache(messages: list[dict], system: str) -> list[dict]:
    """Return a NEW messages list where the system turn's content is a
    parts-array with cache_control.

    OpenRouter passes Anthropic prompt-caching when the system message content
    is structured as a content-parts array with cache_control on the last
    element (mirrors the native Anthropic format, routed transparently).
    Reference: https://openrouter.ai/docs/prompt-caching#anthropic-models

    Only called when the model is in the Anthropic family (gated by caller).
    Does NOT mutate the input list.
    """
    if not messages:
        return messages
    first = messages[0]
    if first.get("role") != "system":
        return messages
    new_system_msg = {
        "role": "system",
        "content": [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ],
    }
    return [new_system_msg] + messages[1:]


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI-compatible provider — works with OpenAI, Groq, Together,
    DeepSeek, Mistral, OpenRouter, and any OpenAI-compatible endpoint."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg["model"]
        self.auth_type = cfg.get("auth_type", "api_key")
        self.provider_id = cfg.get("provider_id", "")
        self.api_key = cfg.get("api_key") or os.getenv(cfg.get("api_key_env", ""), "")
        if self.auth_type == "api_key" and not self.api_key:
            raise LLMError(f"No API key for OpenAI-compatible provider (model={self.model})")
        base = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.url = f"{base}/chat/completions"
        self.default_max = cfg.get("max_tokens", 2000)
        self.default_temp = cfg.get("temperature", 0.3)
        self.provider_name = cfg.get("provider_name", "openai")

    def _get_auth_headers(self) -> dict[str, str]:
        if self.auth_type == "oauth" and self.provider_id:
            from .providers import resolve_auth_header, get_provider
            provider = get_provider(self.provider_id)
            if provider:
                return resolve_auth_header(provider)
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def _post(self, payload: dict, *, _max_retries: int = 5) -> dict:
        """Audit #22: delegates the retry loop to `post_with_retry`.
        Custom error formatter preserves the pre-fix message shape
        `OpenAI API {status} ({provider_name}/{model}): {body[:500]}`
        so log greps and existing test assertions still hit."""
        auth = self._get_auth_headers()
        headers = {**auth, "Content-Type": "application/json"}

        def _parse_openai_error(e: httpx.HTTPStatusError) -> str:
            return f"({self.provider_name}/{self.model}): {(e.response.text or '')[:500]}"

        return post_with_retry(
            self.url,
            payload=payload,
            headers=headers,
            provider_name="OpenAI",
            model=self.model,
            max_retries=_max_retries,
            parse_error=_parse_openai_error,
        )

    @staticmethod
    def _build_user_content(user: str, attachments: list[str] | None) -> "str | list[dict]":
        """OpenAI chat-completions content array with image_url data URIs.

        Audio attachments fall back to their transcript (text); raw audio
        in chat-completions isn't broadly supported across compat providers.
        Video: sampled frames + transcript via video_processor.
        """
        resolved = _resolve_attachments(attachments)
        if not resolved:
            return user
        import base64 as _b64
        blocks: list[dict] = []

        def _emit_image_block(image_meta, image_bytes: bytes) -> None:
            b64 = _b64.b64encode(image_bytes).decode("ascii")
            blocks.append({
                "type": "image_url",
                "image_url": {"url": f"data:{image_meta.mime_type};base64,{b64}"},
            })

        for meta, data in resolved:
            if meta.kind == "image":
                _emit_image_block(meta, data)
            elif meta.kind == "audio":
                if meta.transcript:
                    blocks.append({"type": "text", "text": f"[voice transcript]\n{meta.transcript}"})
                else:
                    blocks.append({"type": "text", "text": "[voice attachment, transcript unavailable]"})
            elif meta.kind == "video":
                from .tools.video_processor import preprocess_video
                from .attachments import ATTACHMENTS as _ATT
                pre = preprocess_video(meta.sha256)
                blocks.append({
                    "type": "text",
                    "text": (
                        f"[video {meta.filename or meta.sha256[:12]}: "
                        f"{pre.duration_seconds:.1f}s, "
                        f"{len(pre.frame_shas)} sampled frames "
                        f"(evenly along the timeline)]"
                    ),
                })
                for fsha in pre.frame_shas:
                    fmeta = _ATT.get_meta(fsha)
                    fdata = _ATT.get_bytes(fsha)
                    if fmeta and fdata:
                        _emit_image_block(fmeta, fdata)
                if pre.audio_transcript:
                    blocks.append({
                        "type": "text",
                        "text": f"[video audio transcript]\n{pre.audio_transcript}",
                    })
            else:
                blocks.append({"type": "text", "text": f"[file attachment: {meta.filename or meta.sha256[:12]}]"})
        if user:
            blocks.append({"type": "text", "text": user})
        return blocks

    def complete(self, system, user, *, max_tokens=None, temperature=None,
                 attachments=None, _task_type: str = ""):
        user_content = self._build_user_content(user, attachments)
        payload = {
            "model": self.model,
            "max_tokens": max_tokens or self.default_max,
            "temperature": temperature if temperature is not None else self.default_temp,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        }
        t0 = time.time()
        data = self._post(payload)
        duration_ms = int((time.time() - t0) * 1000)
        usage = data.get("usage", {})
        if usage:
            TOKENS.record(
                task_type=_task_type,
                model=self.model,
                provider=self.provider_name,
                usage=_openai_usage_to_record(usage),
                duration_ms=duration_ms,
                prompt_preview=user[:300],
            )
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected response from {self.provider_name}: {data}") from e

    def complete_with_tools(
        self, system, user, tools, execute_tool,
        *, max_tokens=None, temperature=None,
        max_iterations=6, on_tool_call=None, attachments=None, _task_type="",
        tools_provider=None,
    ) -> str:
        """OpenAI-style tool-use loop. `attachments` (sha256 list) attach
        images on the first user turn via image_url data URIs.

        Phase 2 (2026-05-23): `tools_provider` callable rebuilds the
        tool schema before each iteration (bundle-aware)."""

        def _to_oai_tools(tool_list):
            """Convert Anthropic tool format to OpenAI format if needed."""
            out = []
            for t in (tool_list or []):
                if "function" in t:
                    out.append(t)
                else:
                    out.append({
                        "type": "function",
                        "function": {
                            "name": t["name"],
                            "description": t.get("description", ""),
                            "parameters": t.get("input_schema", {}),
                        },
                    })
            return out

        static_oai_tools = _to_oai_tools(tools) if tools_provider is None else None

        first_user = self._build_user_content(user, attachments)
        # Prompt caching for Anthropic models via OpenRouter: structure the
        # system message content as a parts-array with cache_control so
        # OpenRouter forwards the cache breakpoint to Anthropic.
        # OpenRouter docs: https://openrouter.ai/docs/prompt-caching#anthropic-models
        # For non-Anthropic models (OpenAI, DeepSeek, etc.) the plain string
        # is left unchanged — those providers auto-cache a stable prefix.
        if _is_anthropic_model(self.model):
            system_content: "str | list[dict]" = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        else:
            system_content = system
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": first_user},
        ]
        final_text = ""
        for _iter in range(max_iterations):
            if _iter > 0 and _tool_loop_input_budget_exceeded():
                break
            # Phase 2: bundle-aware per-iteration tool list.
            if tools_provider is not None:
                try:
                    oai_tools = _to_oai_tools(tools_provider())
                except Exception as e:
                    log.warning(
                        "tools_provider raised, falling back to static: %s", e,
                    )
                    oai_tools = _to_oai_tools(tools)
            else:
                oai_tools = static_oai_tools or []
            payload = {
                "model": self.model,
                "max_tokens": max_tokens or self.default_max,
                "temperature": temperature if temperature is not None else self.default_temp,
                "messages": messages,
            }
            if oai_tools:
                payload["tools"] = oai_tools
            t0 = time.time()
            data = self._post(payload)
            duration_ms = int((time.time() - t0) * 1000)
            usage = data.get("usage", {})
            if usage:
                TOKENS.record(
                    task_type=f"{_task_type}:tool_iter_{_iter}",
                    model=self.model,
                    provider=self.provider_name,
                    usage=_openai_usage_to_record(usage),
                    duration_ms=duration_ms,
                    prompt_preview=user[:300] if _iter == 0 else f"(tool iteration {_iter})",
                )

            try:
                choice = data["choices"][0]
            except (KeyError, IndexError) as e:
                # Defense-in-depth: post_with_retry already turns a 200
                # error-body into an LLMError, but never crash the tool
                # loop on an unexpected shape — raise cleanly instead.
                raise LLMError(
                    f"Unexpected response from {self.provider_name}: {data}"
                ) from e
            msg = choice["message"]
            finish_reason = choice.get("finish_reason", "stop")

            tool_calls = msg.get("tool_calls", [])
            # Only capture text when this turn is NOT a tool-call turn —
            # otherwise the text is preamble narration that should not
            # leak out as the final answer if we hit max_iterations.
            if msg.get("content") and not tool_calls:
                final_text = msg["content"]

            if finish_reason != "tool_calls" and not tool_calls:
                return final_text

            messages.append(msg)
            for tc in tool_calls:
                func = tc["function"]
                name = func["name"]
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                result_text, is_error = execute_tool(name, args)
                if on_tool_call:
                    try:
                        on_tool_call(name, args, result_text, is_error)
                    except Exception:
                        pass
                # See _compact_tool_result_for_llm — full body still
                # reaches the agent's callback above for verifier and
                # dev capture; only next-iteration LLM input is cut.
                llm_result = _compact_tool_result_for_llm(
                    name, result_text, is_error=is_error,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": llm_result,
                })

        # Forced tool-less synthesis at the cap. Generous budget so
        # review-style answers don't truncate (see Anthropic synth note).
        synth_max = max(
            max_tokens or self.default_max,
            int(CONFIG.router.get("tool_synth_max_tokens", 4000)),
        )
        synth_payload = {
            "model": self.model,
            "max_tokens": synth_max,
            "temperature": temperature if temperature is not None else self.default_temp,
            "messages": _curate_synth_messages_openai(messages),
        }
        try:
            t0 = time.time()
            data = self._post(synth_payload)
            duration_ms = int((time.time() - t0) * 1000)
            usage = data.get("usage", {})
            if usage:
                TOKENS.record(
                    task_type=f"{_task_type}:tool_synth",
                    model=self.model,
                    provider=self.provider_name,
                    # Same cache-aware split the other OpenAI-compat
                    # call sites use (2026-06-12) — raw prompt_tokens
                    # include the cached portion.
                    usage=_openai_usage_to_record(usage),
                    duration_ms=duration_ms,
                    prompt_preview="(forced final synthesis)",
                )
            synth_text = (data["choices"][0]["message"].get("content") or "").strip()
            if synth_text:
                return synth_text
        except Exception:
            pass
        return final_text or "[max tool-use iterations reached]"


class CodexLLM(BaseLLM):
    """OpenAI Responses API via ChatGPT subscription auth (~/.codex/auth.json).

    Differs from OpenAICompatibleLLM:
      - Endpoint is `{base_url}/responses` (not /chat/completions).
      - Server requires `stream: true` — we consume SSE and aggregate events.
      - Request body uses `instructions` + `input` (not `messages`).
      - Auth headers include ChatGPT-Account-ID alongside Bearer.
      - Tools are flat objects: {type, name, description, parameters} (not nested under `function`).
      - Tool results re-enter the `input` array as `function_call_output` items.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg["model"]
        self.provider_id = cfg.get("provider_id", "")
        base = cfg.get("base_url") or "https://chatgpt.com/backend-api/codex"
        self.url = f"{base.rstrip('/')}/responses"
        self.default_max = cfg.get("max_tokens", 2000)
        self.default_temp = cfg.get("temperature", 0.3)
        self.provider_name = cfg.get("provider_name", "openai_codex")

    def _auth_headers(self) -> dict[str, str]:
        from .providers import resolve_auth_header, get_provider
        if self.provider_id:
            provider = get_provider(self.provider_id)
            if provider:
                return resolve_auth_header(provider)
        # Fallback: build directly from CodexAuthManager
        from .providers import CODEX_AUTH
        try:
            access, account_id = CODEX_AUTH.get_access_token()
        except RuntimeError as e:
            raise LLMError(f"Codex auth: {e}") from e
        h = {"Authorization": f"Bearer {access}"}
        if account_id:
            h["ChatGPT-Account-ID"] = account_id
        return h

    def _stream_and_aggregate(
        self, payload: dict, *, _max_retries: int = 3
    ) -> tuple[str, list[dict], dict]:
        """Stream the Responses API call and aggregate.

        Returns (final_text, output_items, usage). final_text is the concatenated
        output_text deltas. output_items is the list of completed items (messages,
        function_calls) extracted from response.output_item.done events. usage
        comes from the response.completed event.
        """
        # Note: 429 from chatgpt.com/backend-api/codex is usually `usage_limit_reached`
        # (subscription quota exhausted), not transient rate-limiting — don't retry.
        RETRYABLE = {500, 502, 503, 529}
        last_error: Exception | None = None
        payload = dict(payload)
        payload["stream"] = True  # server requires stream=true

        for attempt in range(_max_retries + 1):
            headers = {**self._auth_headers(), "Content-Type": "application/json", "Accept": "text/event-stream"}
            try:
                with httpx.stream(
                    "POST", self.url, json=payload, headers=headers, timeout=180.0
                ) as r:
                    if r.status_code in RETRYABLE and attempt < _max_retries:
                        last_error = httpx.HTTPStatusError(
                            f"{r.status_code}", request=r.request, response=r
                        )
                        time.sleep(min(2 ** attempt * 2, 60.0))
                        continue
                    if r.status_code >= 400:
                        body = b"".join(r.iter_bytes())[:500].decode("utf-8", errors="replace")
                        if r.status_code == 429 and "usage_limit_reached" in body:
                            raise LLMError(
                                f"Codex subscription quota exhausted ({self.provider_name}/{self.model}). "
                                f"Wait for reset or switch to a different provider. Detail: {body}"
                            )
                        raise LLMError(
                            f"Codex Responses API {r.status_code} ({self.provider_name}/{self.model}): {body}"
                        )
                    return _consume_responses_sse(r.iter_lines())
            except LLMError:
                raise
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < _max_retries:
                    time.sleep(min(2 ** attempt * 2, 60.0))
                    continue
                raise LLMError(
                    f"Codex Responses API error after {_max_retries} retries: {e}"
                ) from e
        raise LLMError(f"Codex Responses API failed: {last_error}")

    @staticmethod
    def _function_calls(output: list[dict]) -> list[dict]:
        return [item for item in output or [] if item.get("type") == "function_call"]

    def _record_usage(self, usage: dict, _task_type: str, prompt_preview: str, duration_ms: int) -> None:
        if not usage:
            return
        # Responses API reports `input_tokens` INCLUSIVE of cached
        # tokens, with the cached subset under
        # `input_tokens_details.cached_tokens`. Pre-fix (2026-06-12)
        # we recorded the raw number as fresh input — the 208k-token
        # reminder turn showed in=202k when 193k of it was cache reads
        # billed at ~10%. Split so display and cost are honest.
        raw_in = usage.get("input_tokens", 0) or 0
        details = usage.get("input_tokens_details") or {}
        cached = details.get("cached_tokens", 0) or 0
        TOKENS.record(
            task_type=_task_type,
            model=self.model,
            provider=self.provider_name,
            usage={
                "input_tokens": max(0, raw_in - cached),
                "output_tokens": usage.get("output_tokens", 0),
                "cache_read_input_tokens": cached,
            },
            duration_ms=duration_ms,
            prompt_preview=prompt_preview,
        )

    def _build_payload(
        self, system: str, input_items: list[dict], tools: list[dict] | None,
        max_tokens: int | None, temperature: float | None,
        task_type: str = "",
    ) -> dict:
        # ChatGPT-subscription tier (chatgpt.com/backend-api/codex) rejects
        # `max_output_tokens` ("Unsupported parameter") and likely `temperature`.
        # We accept the args to keep BaseLLM.complete() signature-compatible
        # but drop them — the server applies its own defaults per model.
        _ = max_tokens, temperature
        payload: dict = {
            "model": self.model,
            "instructions": system,
            "input": input_items,
            # ChatGPT-subscription tier rejects store=true with HTTP 400
            # ("Store must be set to false"). With store=false we MUST strip
            # server-assigned `id` fields from re-fed output items (see
            # _sanitize_for_input) and ride on encrypted_content instead.
            "store": False,
            # Carry encrypted reasoning inside each item so we don't lose
            # chain-of-thought when re-feeding (matches Codex CLI's pattern).
            "include": ["reasoning.encrypted_content"],
        }
        # Hybrid reasoning routing (May 2026 follow-up). When the caller
        # supplies a task_type, look up the configured effort level
        # (low/medium/high) in `reasoning_routing` and attach it. The
        # default (`medium`) matches the Codex CLI behaviour and the
        # cached `default_reasoning_level` for gpt-5.x. Higher levels
        # cost more tokens but produce better planning + judgment on
        # supervisor / complex-solving turns.
        try:
            from . import reasoning_routing as _rr
            level = _rr.level_for(task_type)
            if level and level != "none":
                payload["reasoning"] = {"effort": level}
        except Exception as e:
            log.debug("reasoning_routing lookup failed: %s", e)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
            payload["parallel_tool_calls"] = True
        return payload

    @staticmethod
    def _sanitize_for_input(items: list[dict]) -> list[dict]:
        """Strip server-assigned `id` fields before re-feeding output items
        as input on the next turn.

        With store=true the IDs would resolve, but stripping them is harmless
        and protects against accidental store=false flips. `call_id` is kept
        because it's the tool↔result correlator, not a server lookup key.
        """
        cleaned: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            stripped = {k: v for k, v in item.items() if k != "id"}
            cleaned.append(stripped)
        return cleaned

    @staticmethod
    def _build_user_content_blocks(user: str, attachments: list[str] | None) -> list[dict]:
        """Responses API `content` array: input_image for images, text otherwise.

        Audio attachments are inlined as transcript text — Responses API
        accepts `input_audio` only via Realtime API, not Codex chat path.
        Video: sampled frames + transcript via video_processor.
        """
        resolved = _resolve_attachments(attachments)
        blocks: list[dict] = []
        if resolved:
            import base64 as _b64

            def _emit_image_block(image_meta, image_bytes: bytes) -> None:
                b64 = _b64.b64encode(image_bytes).decode("ascii")
                blocks.append({
                    "type": "input_image",
                    "image_url": f"data:{image_meta.mime_type};base64,{b64}",
                })

            for meta, data in resolved:
                if meta.kind == "image":
                    _emit_image_block(meta, data)
                elif meta.kind == "audio":
                    if meta.transcript:
                        blocks.append({"type": "input_text", "text": f"[voice transcript]\n{meta.transcript}"})
                    else:
                        blocks.append({"type": "input_text", "text": "[voice attachment, transcript unavailable]"})
                elif meta.kind == "video":
                    from .tools.video_processor import preprocess_video
                    from .attachments import ATTACHMENTS as _ATT
                    pre = preprocess_video(meta.sha256)
                    blocks.append({
                        "type": "input_text",
                        "text": (
                            f"[video {meta.filename or meta.sha256[:12]}: "
                            f"{pre.duration_seconds:.1f}s, "
                            f"{len(pre.frame_shas)} sampled frames "
                            f"(evenly along the timeline)]"
                        ),
                    })
                    for fsha in pre.frame_shas:
                        fmeta = _ATT.get_meta(fsha)
                        fdata = _ATT.get_bytes(fsha)
                        if fmeta and fdata:
                            _emit_image_block(fmeta, fdata)
                    if pre.audio_transcript:
                        blocks.append({
                            "type": "input_text",
                            "text": f"[video audio transcript]\n{pre.audio_transcript}",
                        })
                else:
                    blocks.append({"type": "input_text", "text": f"[file attachment: {meta.filename or meta.sha256[:12]}]"})
        if user:
            blocks.append({"type": "input_text", "text": user})
        return blocks

    def complete(self, system, user, *, max_tokens=None, temperature=None,
                 attachments=None, _task_type: str = ""):
        input_items = [
            {
                "type": "message",
                "role": "user",
                "content": self._build_user_content_blocks(user, attachments),
            }
        ]
        payload = self._build_payload(
            system, input_items, None, max_tokens, temperature,
            task_type=_task_type,
        )
        t0 = time.time()
        text, _items, usage = self._stream_and_aggregate(payload)
        duration_ms = int((time.time() - t0) * 1000)
        self._record_usage(usage, _task_type, user[:300], duration_ms)
        if not text:
            raise LLMError("Codex Responses API returned no text")
        return text

    def complete_with_tools(
        self, system, user, tools, execute_tool,
        *, max_tokens=None, temperature=None,
        max_iterations=6, on_tool_call=None, attachments=None, _task_type: str = "",
        tools_provider=None,
    ) -> str:
        # Convert tools to Responses API flat format. Accept either:
        #   {"name", "description", "input_schema"}                  (Anthropic-ish)
        #   {"type":"function","function":{"name","description","parameters"}}  (Chat Completions)
        def _to_flat_tools(tool_list):
            out: list[dict] = []
            for t in tool_list or []:
                if t.get("type") == "function" and isinstance(t.get("function"), dict):
                    fn = t["function"]
                    out.append({
                        "type": "function",
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                        "strict": False,
                    })
                else:
                    out.append({
                        "type": "function",
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema") or t.get("parameters") or {},
                        "strict": False,
                    })
            return out

        static_flat_tools = _to_flat_tools(tools) if tools_provider is None else None

        input_items: list[dict] = [
            {
                "type": "message",
                "role": "user",
                "content": self._build_user_content_blocks(user, attachments),
            }
        ]

        final_text = ""
        for _iter in range(max_iterations):
            if _iter > 0 and _tool_loop_input_budget_exceeded():
                break
            # Phase 2: bundle-aware per-iteration tool list.
            if tools_provider is not None:
                try:
                    flat_tools = _to_flat_tools(tools_provider())
                except Exception as e:
                    log.warning(
                        "tools_provider raised, falling back to static: %s", e,
                    )
                    flat_tools = _to_flat_tools(tools)
            else:
                flat_tools = static_flat_tools or []
            payload = self._build_payload(
                system, input_items, flat_tools or None, max_tokens, temperature,
                task_type=_task_type,
            )
            t0 = time.time()
            text, items, usage = self._stream_and_aggregate(payload)
            duration_ms = int((time.time() - t0) * 1000)
            self._record_usage(
                usage,
                f"{_task_type}:tool_iter_{_iter}",
                user[:300] if _iter == 0 else f"(tool iteration {_iter})",
                duration_ms,
            )
            calls = self._function_calls(items)
            # Don't capture preamble text from a turn that also has function calls.
            if text and not calls:
                final_text = text

            if not calls:
                return final_text

            # Re-feed every output item back so the model sees its own state,
            # then add tool results. Strip server-assigned `id` fields so the
            # API doesn't try to look them up in its persistent store.
            input_items.extend(self._sanitize_for_input(items))
            for fc in calls:
                name = fc.get("name", "")
                call_id = fc.get("call_id") or fc.get("id", "")
                try:
                    args = json.loads(fc.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                result_text, is_error = execute_tool(name, args)
                if on_tool_call:
                    try:
                        on_tool_call(name, args, result_text, is_error)
                    except Exception:
                        pass
                # See _compact_tool_result_for_llm — caps the body
                # before re-feeding to next Responses-API turn.
                llm_result = _compact_tool_result_for_llm(
                    name, result_text, is_error=is_error,
                )
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": llm_result,
                })

        # Forced tool-less synthesis at the cap — call again with no tools.
        # Generous budget for review-style answers (see Anthropic synth note).
        synth_max = max(
            max_tokens or self.default_max,
            int(CONFIG.router.get("tool_synth_max_tokens", 4000)),
        )
        synth_payload = self._build_payload(
            system, _curate_synth_input_items_codex(input_items),
            None, synth_max, temperature,
            task_type=f"{_task_type}:tool_synth",
        )
        try:
            t0 = time.time()
            synth_text, _items, usage = self._stream_and_aggregate(synth_payload)
            duration_ms = int((time.time() - t0) * 1000)
            self._record_usage(
                usage, f"{_task_type}:tool_synth",
                "(forced final synthesis)", duration_ms,
            )
            if synth_text:
                return synth_text
        except Exception:
            pass
        return final_text or "[max tool-use iterations reached]"


class BedrockLLM(BaseLLM):
    """AWS Bedrock invoke_model — v0 supports Anthropic Claude on Bedrock only.

    Bedrock isn't a single HTTP API; each model family on Bedrock has its
    own request body shape (Anthropic, Llama, Cohere on Bedrock, Titan,
    Mistral on Bedrock). For v0 we focus on the most common case —
    Anthropic Claude — using the same Messages API shape as native Anthropic
    but wrapped in a Bedrock invoke.

    boto3 is a soft dependency. If it's not installed we raise an LLMError
    with a clear `pip install boto3` instruction at construction time.

    Auth: requires `aws.access_key_id`, `aws.secret_access_key`, `aws.region`
    in the provider config. We don't probe the AWS env / instance metadata
    fallback because the use case is "user explicitly configured creds in
    the UI" — env-based auth would silently leak whatever's in the dev shell.
    """

    def __init__(self, cfg: dict):
        try:
            import boto3  # noqa: F401
        except ImportError as e:
            raise LLMError(
                "AWS Bedrock provider requires boto3. Install it with: "
                ".venv/Scripts/python.exe -m pip install boto3"
            ) from e

        self.cfg = cfg
        self.model = cfg["model"]
        if not self.model.startswith(("anthropic.", "us.anthropic.", "eu.anthropic.", "apac.anthropic.")):
            raise LLMError(
                f"Bedrock provider v0 only supports anthropic.* models "
                f"(got '{self.model}'). Other Bedrock families need per-family "
                f"request shaping — file an issue if you need them."
            )
        aws = cfg.get("aws") or {}
        self.access_key_id = aws.get("access_key_id") or os.getenv("AWS_ACCESS_KEY_ID", "")
        self.secret_access_key = aws.get("secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY", "")
        self.region = aws.get("region") or os.getenv("AWS_REGION", "us-east-1")
        if not self.access_key_id or not self.secret_access_key:
            raise LLMError(
                "Bedrock provider needs aws.access_key_id and aws.secret_access_key"
            )
        self.default_max = cfg.get("max_tokens", 2000)
        self.default_temp = cfg.get("temperature", 0.3)
        self.provider_name = cfg.get("provider_name", "aws_bedrock")
        self._client = None

    def _bedrock_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
            )
        return self._client

    def _invoke(self, payload: dict) -> dict:
        client = self._bedrock_client()
        try:
            t0 = time.time()
            response = client.invoke_model(
                modelId=self.model,
                body=json.dumps(payload).encode("utf-8"),
                contentType="application/json",
                accept="application/json",
            )
            duration_ms = int((time.time() - t0) * 1000)
        except Exception as e:
            raise LLMError(f"Bedrock invoke_model failed ({self.provider_name}/{self.model}): {e}") from e
        body_bytes = response["body"].read()
        try:
            data = json.loads(body_bytes)
        except Exception as e:
            raise LLMError(f"Bedrock returned non-JSON body: {e}") from e
        data["__duration_ms"] = duration_ms
        return data

    def _record_usage(self, data: dict, _task_type: str, prompt_preview: str, duration_ms: int) -> None:
        usage = data.get("usage") or {}
        if not usage:
            return
        TOKENS.record(
            task_type=_task_type,
            model=self.model,
            provider=self.provider_name,
            usage={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
            duration_ms=duration_ms,
            prompt_preview=prompt_preview,
        )

    def complete(self, system, user, *, max_tokens=None, temperature=None,
                 attachments=None, _task_type: str = ""):
        # Vision not supported here — attachments accepted for API
        # compatibility with the BaseLLM signature, then ignored.
        _ = attachments
        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.default_max,
            "temperature": temperature if temperature is not None else self.default_temp,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        data = self._invoke(payload)
        self._record_usage(data, _task_type, user[:300], data.get("__duration_ms", 0))
        # Anthropic-on-Bedrock returns content array
        content = data.get("content") or []
        text = ""
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text += block.get("text", "")
        if not text:
            raise LLMError(f"Bedrock returned no text content: {str(data)[:300]}")
        return text

    def complete_with_tools(
        self, system, user, tools, execute_tool,
        *, max_tokens=None, temperature=None,
        max_iterations=6, on_tool_call=None, attachments=None, _task_type: str = "",
        tools_provider=None,
    ) -> str:
        _ = attachments  # vision not supported on this backend
        # Convert tools to Anthropic shape: {name, description, input_schema}
        def _to_anthropic_tools(tool_list):
            out = []
            for t in tool_list or []:
                if "input_schema" in t:
                    out.append(t)
                elif t.get("type") == "function" and isinstance(t.get("function"), dict):
                    fn = t["function"]
                    out.append({
                        "name": fn.get("name", ""),
                        "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {}),
                    })
            return out

        static_anthropic_tools = _to_anthropic_tools(tools) if tools_provider is None else None

        messages: list[dict] = [{"role": "user", "content": user}]
        final_text = ""
        for _iter in range(max_iterations):
            if _iter > 0 and _tool_loop_input_budget_exceeded():
                break
            # Phase 2: bundle-aware per-iteration tool list.
            if tools_provider is not None:
                try:
                    anthropic_tools = _to_anthropic_tools(tools_provider())
                except Exception as e:
                    log.warning(
                        "tools_provider raised, falling back to static: %s", e,
                    )
                    anthropic_tools = _to_anthropic_tools(tools)
            else:
                anthropic_tools = static_anthropic_tools or []
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens or self.default_max,
                "temperature": temperature if temperature is not None else self.default_temp,
                "system": system,
                "messages": messages,
            }
            if anthropic_tools:
                payload["tools"] = anthropic_tools
            data = self._invoke(payload)
            self._record_usage(
                data,
                f"{_task_type}:tool_iter_{_iter}",
                user[:300] if _iter == 0 else f"(tool iteration {_iter})",
                data.get("__duration_ms", 0),
            )
            content = data.get("content") or []
            text_now = "".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )
            tool_uses = [b for b in content if isinstance(b, dict) and b.get("type") == "tool_use"]

            # Don't capture preamble text from a turn that also has tool_use.
            if text_now and not tool_uses:
                final_text = text_now

            if not tool_uses:
                return final_text

            messages.append({"role": "assistant", "content": content})
            tool_results = []
            for tu in tool_uses:
                name = tu.get("name", "")
                args = tu.get("input") or {}
                tool_id = tu.get("id", "")
                result_text, is_error = execute_tool(name, args)
                if on_tool_call:
                    try:
                        on_tool_call(name, args, result_text, is_error)
                    except Exception:
                        pass
                # See _compact_tool_result_for_llm — caps the body
                # before re-feeding to next Bedrock turn.
                llm_result = _compact_tool_result_for_llm(
                    name, result_text, is_error=is_error,
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": llm_result,
                    **({"is_error": True} if is_error else {}),
                })
            messages.append({"role": "user", "content": tool_results})

        # Forced tool-less synthesis at the cap (see Anthropic synth note).
        synth_max = max(
            max_tokens or self.default_max,
            int(CONFIG.router.get("tool_synth_max_tokens", 4000)),
        )
        synth_payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": synth_max,
            "temperature": temperature if temperature is not None else self.default_temp,
            "system": system,
            "messages": _curate_synth_messages_anthropic(messages),
        }
        try:
            data = self._invoke(synth_payload)
            self._record_usage(
                data,
                f"{_task_type}:tool_synth",
                "(forced final synthesis)",
                data.get("__duration_ms", 0),
            )
            synth_text = "".join(
                b.get("text", "") for b in (data.get("content") or [])
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if synth_text:
                return synth_text
        except Exception:
            pass
        return final_text or "[max tool-use iterations reached]"


class CopilotLLM(OpenAICompatibleLLM):
    """GitHub Copilot subscription via api.githubcopilot.com.

    The Copilot chat endpoint is OpenAI-shaped (`/chat/completions` with
    `messages`, returns `choices[0].message.content`), but it requires a
    short-lived Bearer obtained by exchanging the persistent oauth_token
    (`gho_...`) — handled by CopilotAuthManager — and several editor-
    identification headers without which the request is rejected.

    We extend OpenAICompatibleLLM and only override `_get_auth_headers()`
    so the rest of the request path (retry, tool-use loop, response parsing)
    is shared with every other OpenAI-compatible provider.
    """

    DEFAULT_BASE_URL = "https://api.githubcopilot.com"
    EDITOR_VERSION = "vscode/1.95.0"
    EDITOR_PLUGIN_VERSION = "copilot-chat/0.20.0"
    USER_AGENT = "GitHubCopilotChat/0.20.0"
    INTEGRATION_ID = "vscode-chat"

    def __init__(self, cfg: dict):
        cfg_copy = dict(cfg)
        cfg_copy.setdefault("base_url", self.DEFAULT_BASE_URL)
        # Skip OpenAICompatibleLLM's "no API key" check — auth is via
        # CopilotAuthManager, not a static api_key.
        cfg_copy.setdefault("auth_type", "copilot_subscription")
        cfg_copy.setdefault("api_key", "x")  # placeholder so init doesn't raise
        super().__init__(cfg_copy)
        self.api_key = ""  # never used; auth is via _get_auth_headers
        self.provider_name = cfg.get("provider_name", "github_copilot")

    def _get_auth_headers(self) -> dict[str, str]:
        from .providers import COPILOT_AUTH
        try:
            bearer, _endpoints = COPILOT_AUTH.get_bearer()
        except RuntimeError as e:
            raise LLMError(f"Copilot auth: {e}") from e
        return {
            "Authorization": f"Bearer {bearer}",
            "Editor-Version": self.EDITOR_VERSION,
            "Editor-Plugin-Version": self.EDITOR_PLUGIN_VERSION,
            "User-Agent": self.USER_AGENT,
            "Copilot-Integration-Id": self.INTEGRATION_ID,
            "OpenAI-Intent": "conversation-panel",
        }


class CohereLLM(BaseLLM):
    """Cohere v2 Chat API (`POST /v2/chat`).

    Differs from OpenAICompatibleLLM:
      - Endpoint is `/v2/chat`, not `/chat/completions`.
      - Response shape: `message.content` is an array of `{type, text}`
        blocks, not a single string under `choices[0].message.content`.
      - Tool calls live on `message.tool_calls` (mostly OpenAI-shaped).
      - Tool results re-enter as `{role: "tool", tool_call_id, content}`,
        same as OpenAI tool-use loop.
      - Usage is reported under `usage.tokens` (not `usage.prompt_tokens`).
    """

    DEFAULT_BASE_URL = "https://api.cohere.com/v2"

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg["model"]
        self.api_key = cfg.get("api_key") or os.getenv(cfg.get("api_key_env", "COHERE_API_KEY"), "")
        if not self.api_key:
            raise LLMError(f"No API key for Cohere (model={self.model})")
        base = (cfg.get("base_url") or self.DEFAULT_BASE_URL).rstrip("/")
        self.url = f"{base}/chat"
        self.default_max = cfg.get("max_tokens", 2000)
        self.default_temp = cfg.get("temperature", 0.3)
        self.provider_name = cfg.get("provider_name", "cohere")

    def _post(self, payload: dict, *, _max_retries: int = 3) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        RETRYABLE = {429, 500, 502, 503, 529}
        last_error: Exception | None = None
        for attempt in range(_max_retries + 1):
            try:
                r = httpx.post(self.url, json=payload, headers=headers, timeout=120.0)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code in RETRYABLE and attempt < _max_retries:
                    time.sleep(min(2 ** attempt * 2, 60.0))
                    continue
                body = e.response.text[:500]
                raise LLMError(
                    f"Cohere API {e.response.status_code} ({self.provider_name}/{self.model}): {body}"
                ) from e
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < _max_retries:
                    time.sleep(min(2 ** attempt * 2, 60.0))
                    continue
                raise LLMError(f"Cohere API error after {_max_retries} retries: {e}") from e
        raise LLMError(f"Cohere API failed: {last_error}")

    @staticmethod
    def _extract_text(message: dict) -> str:
        content = message.get("content") or []
        chunks = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
                chunks.append(c["text"])
        return "".join(chunks)

    def _record_usage(self, data: dict, _task_type: str, prompt_preview: str, duration_ms: int) -> None:
        usage = ((data.get("usage") or {}).get("tokens") or {})
        if not usage:
            return
        TOKENS.record(
            task_type=_task_type,
            model=self.model,
            provider=self.provider_name,
            usage={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            },
            duration_ms=duration_ms,
            prompt_preview=prompt_preview,
        )

    def complete(self, system, user, *, max_tokens=None, temperature=None,
                 attachments=None, _task_type: str = ""):
        # Vision not supported here — attachments accepted for API
        # compatibility with the BaseLLM signature, then ignored.
        _ = attachments
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens or self.default_max,
            "temperature": temperature if temperature is not None else self.default_temp,
            "stream": False,
        }
        t0 = time.time()
        data = self._post(payload)
        duration_ms = int((time.time() - t0) * 1000)
        self._record_usage(data, _task_type, user[:300], duration_ms)
        text = self._extract_text(data.get("message") or {})
        if not text:
            raise LLMError(f"Cohere returned no text: {str(data)[:300]}")
        return text

    def complete_with_tools(
        self, system, user, tools, execute_tool,
        *, max_tokens=None, temperature=None,
        max_iterations=6, on_tool_call=None, attachments=None, _task_type: str = "",
        tools_provider=None,
    ) -> str:
        _ = attachments  # vision not supported on this backend
        # Cohere v2 accepts OpenAI-style {type:"function", function:{...}} tools.
        def _to_coh_tools(tool_list):
            out: list[dict] = []
            for t in tool_list or []:
                if t.get("type") == "function" and isinstance(t.get("function"), dict):
                    out.append(t)
                else:
                    out.append({
                        "type": "function",
                        "function": {
                            "name": t.get("name", ""),
                            "description": t.get("description", ""),
                            "parameters": t.get("input_schema") or t.get("parameters") or {},
                        },
                    })
            return out

        static_coh_tools = _to_coh_tools(tools) if tools_provider is None else None

        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        final_text = ""
        for _iter in range(max_iterations):
            if _iter > 0 and _tool_loop_input_budget_exceeded():
                break
            # Phase 2: bundle-aware per-iteration tool list.
            if tools_provider is not None:
                try:
                    coh_tools = _to_coh_tools(tools_provider())
                except Exception as e:
                    log.warning(
                        "tools_provider raised, falling back to static: %s", e,
                    )
                    coh_tools = _to_coh_tools(tools)
            else:
                coh_tools = static_coh_tools or []
            payload = {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens or self.default_max,
                "temperature": temperature if temperature is not None else self.default_temp,
                "stream": False,
            }
            if coh_tools:
                payload["tools"] = coh_tools
                payload["tool_choice"] = "auto"
            t0 = time.time()
            data = self._post(payload)
            duration_ms = int((time.time() - t0) * 1000)
            self._record_usage(
                data,
                f"{_task_type}:tool_iter_{_iter}",
                user[:300] if _iter == 0 else f"(tool iteration {_iter})",
                duration_ms,
            )
            msg = data.get("message") or {}
            text_now = self._extract_text(msg)
            tool_calls = msg.get("tool_calls") or []
            # Don't capture preamble text from a tool-call turn.
            if text_now and not tool_calls:
                final_text = text_now
            if not tool_calls:
                return final_text
            # Append assistant message + tool results, loop.
            messages.append({
                "role": "assistant",
                "tool_plan": msg.get("tool_plan", ""),
                "tool_calls": tool_calls,
            })
            for tc in tool_calls:
                fn = tc.get("function", {}) or {}
                name = fn.get("name", "")
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}
                result_text, is_error = execute_tool(name, args)
                if on_tool_call:
                    try:
                        on_tool_call(name, args, result_text, is_error)
                    except Exception:
                        pass
                # See _compact_tool_result_for_llm — caps body
                # before re-feeding to next Cohere turn.
                llm_result = _compact_tool_result_for_llm(
                    name, result_text, is_error=is_error,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": [{"type": "text", "text": llm_result}],
                })
        # Forced tool-less synthesis at the cap (see Anthropic synth note).
        synth_max = max(
            max_tokens or self.default_max,
            int(CONFIG.router.get("tool_synth_max_tokens", 4000)),
        )
        synth_payload = {
            "model": self.model,
            "messages": _curate_synth_messages_cohere(messages),
            "max_tokens": synth_max,
            "temperature": temperature if temperature is not None else self.default_temp,
            "stream": False,
        }
        try:
            t0 = time.time()
            data = self._post(synth_payload)
            duration_ms = int((time.time() - t0) * 1000)
            self._record_usage(
                data, f"{_task_type}:tool_synth",
                "(forced final synthesis)", duration_ms,
            )
            synth_text = self._extract_text(data.get("message") or {})
            if synth_text:
                return synth_text
        except Exception:
            pass
        return final_text or "[max tool-use iterations reached]"


def _consume_responses_sse(line_iter) -> tuple[str, list[dict], dict]:
    """Aggregate an SSE stream from the Responses API.

    Recognised event types (per openai/codex codex-rs/codex-api/src/sse/responses.rs):
      response.output_text.delta   → append .delta to text buffer
      response.output_item.done    → final form of message/function_call → collect
      response.completed           → response.usage and end-of-turn
      response.failed              → error
    """
    text_chunks: list[str] = []
    output_items: list[dict] = []
    usage: dict = {}
    last_error: str = ""

    for raw in line_iter:
        if raw is None:
            continue
        line = raw.strip() if isinstance(raw, str) else raw.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":") or line.startswith("event:"):
            continue
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if not data_str or data_str == "[DONE]":
            continue
        try:
            evt = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        kind = evt.get("type") or evt.get("kind") or ""
        if kind == "response.output_text.delta":
            d = evt.get("delta")
            if isinstance(d, str):
                text_chunks.append(d)
        elif kind == "response.refusal.delta":
            # Refusal bug fix (2026-06-12, found via conversational
            # audit): when the model declines (e.g. "write a script to
            # dump browser passwords"), the Responses API streams the
            # decline as `response.refusal.delta` — NOT output_text —
            # and the final content part has type "refusal" with the
            # text under a "refusal" key. The old parser read only
            # output_text, so EVERY refusal came back as an empty
            # answer: the agent correctly declined but said nothing,
            # violating the soul's "refuse AND explain". Capture it.
            d = evt.get("delta")
            if isinstance(d, str):
                text_chunks.append(d)
        elif kind == "response.output_item.done":
            item = evt.get("item")
            if isinstance(item, dict):
                output_items.append(item)
        elif kind == "response.completed":
            resp = evt.get("response") or {}
            usage = resp.get("usage") or {}
            # Fallback: if delta events were not emitted, take text from final output.
            if not text_chunks:
                for it in resp.get("output") or []:
                    if it.get("type") == "message":
                        for c in it.get("content") or []:
                            if not isinstance(c, dict):
                                continue
                            ctype = c.get("type")
                            if ctype == "output_text" and c.get("text"):
                                text_chunks.append(c["text"])
                            elif ctype == "refusal" and c.get("refusal"):
                                text_chunks.append(c["refusal"])
        elif kind in ("response.failed", "response.incomplete"):
            err = evt.get("response") or {}
            err_obj = err.get("error") if isinstance(err, dict) else None
            last_error = (err_obj or {}).get("message") if isinstance(err_obj, dict) else ""
            if not last_error:
                last_error = f"{kind} (no details)"
            raise LLMError(f"Codex Responses API stream error: {last_error}")

    return "".join(text_chunks), output_items, usage


class GoogleLLM(BaseLLM):
    """Google Gemini provider via REST API."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg["model"]
        self.auth_type = cfg.get("auth_type", "api_key")
        self.provider_id = cfg.get("provider_id", "")
        self.api_key = cfg.get("api_key") or os.getenv(cfg.get("api_key_env", "GOOGLE_API_KEY"), "")
        if self.auth_type == "api_key" and not self.api_key:
            raise LLMError(f"No API key for Google provider (model={self.model})")
        base = cfg.get("base_url", "https://generativelanguage.googleapis.com/v1beta").rstrip("/")
        self.base_url = base
        self.default_max = cfg.get("max_tokens", 2000)
        self.default_temp = cfg.get("temperature", 0.3)

    def complete(self, system, user, *, max_tokens=None, temperature=None,
                 attachments=None, _task_type: str = ""):
        # Vision not supported here — attachments accepted for API
        # compatibility with the BaseLLM signature, then ignored.
        _ = attachments
        # For OAuth: use Bearer header instead of query param
        if self.auth_type == "oauth" and self.provider_id:
            from .providers import OAUTH_TOKENS
            token = OAUTH_TOKENS.get_valid_token(self.provider_id)
            if token:
                url = f"{self.base_url}/models/{self.model}:generateContent"
                extra_headers = {"Authorization": f"Bearer {token}"}
            else:
                raise LLMError("No valid OAuth token for Google provider")
        else:
            url = f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}"
            extra_headers = {}
        payload = {
            "contents": [{"parts": [{"text": user}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {
                "maxOutputTokens": max_tokens or self.default_max,
                "temperature": temperature if temperature is not None else self.default_temp,
            },
        }
        try:
            t0 = time.time()
            r = httpx.post(url, json=payload, headers=extra_headers or None, timeout=120.0)
            r.raise_for_status()
            duration_ms = int((time.time() - t0) * 1000)
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Google API {e.response.status_code}: {e.response.text[:500]}") from e
        except httpx.HTTPError as e:
            raise LLMError(f"Google API error: {e}") from e

        data = r.json()
        usage_meta = data.get("usageMetadata", {})
        if usage_meta:
            TOKENS.record(
                task_type=_task_type,
                model=self.model,
                provider="google",
                usage={
                    "input_tokens": usage_meta.get("promptTokenCount", 0),
                    "output_tokens": usage_meta.get("candidatesTokenCount", 0),
                },
                duration_ms=duration_ms,
                prompt_preview=user[:300],
            )
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise LLMError(f"Unexpected Google response: {data}") from e


class OllamaLLM(BaseLLM):
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.model = cfg["model"]
        self.base_url = cfg.get("base_url", "http://localhost:11434").rstrip("/")
        self.url = f"{self.base_url}/api/chat"
        self.default_max = cfg.get("max_tokens", 2000)
        self.default_temp = cfg.get("temperature", 0.3)

    def complete(self, system, user, *, max_tokens=None, temperature=None,
                 attachments=None, _task_type: str = ""):
        # Vision not supported here — attachments accepted for API
        # compatibility with the BaseLLM signature, then ignored.
        _ = attachments
        payload = {
            "model": self.model,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self.default_temp,
                "num_predict": max_tokens or self.default_max,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            t0 = time.time()
            r = httpx.post(self.url, json=payload, timeout=300.0)
            if r.status_code == 404:
                raise LLMError(
                    f"Ollama model '{self.model}' not found. "
                    f"Pull it first: ollama pull {self.model}"
                )
            r.raise_for_status()
            duration_ms = int((time.time() - t0) * 1000)
        except LLMError:
            raise
        except httpx.HTTPError as e:
            raise LLMError(f"Ollama error: {e}") from e
        data = r.json()
        input_tok = data.get("prompt_eval_count", 0)
        output_tok = data.get("eval_count", 0)
        if input_tok or output_tok:
            TOKENS.record(
                task_type=_task_type,
                model=self.model,
                provider="ollama",
                usage={"input_tokens": input_tok, "output_tokens": output_tok},
                duration_ms=duration_ms,
                prompt_preview=user[:300],
            )
        return data["message"]["content"]


# ---------- LLM factory ----------
def create_llm(cfg: dict) -> BaseLLM:
    """Create an LLM instance from a provider config dict.

    cfg must have 'provider' (type) and 'model' keys.
    Supported providers: anthropic, openai, openai_codex, google, groq,
    deepseek, mistral, openai_compatible, ollama.
    """
    provider = cfg.get("provider", "")
    if provider == "anthropic":
        return AnthropicLLM(cfg)
    elif provider == "ollama":
        return OllamaLLM(cfg)
    elif provider == "google":
        return GoogleLLM(cfg)
    elif provider == "openai_codex":
        cfg_copy = dict(cfg)
        cfg_copy.setdefault("provider_name", "openai_codex")
        cfg_copy.setdefault("base_url", "https://chatgpt.com/backend-api/codex")
        return CodexLLM(cfg_copy)
    elif provider == "cohere":
        cfg_copy = dict(cfg)
        cfg_copy.setdefault("provider_name", "cohere")
        cfg_copy.setdefault("base_url", "https://api.cohere.com/v2")
        return CohereLLM(cfg_copy)
    elif provider == "github_copilot":
        cfg_copy = dict(cfg)
        cfg_copy.setdefault("provider_name", "github_copilot")
        cfg_copy.setdefault("base_url", "https://api.githubcopilot.com")
        return CopilotLLM(cfg_copy)
    elif provider == "aws_bedrock":
        cfg_copy = dict(cfg)
        cfg_copy.setdefault("provider_name", "aws_bedrock")
        return BedrockLLM(cfg_copy)
    elif provider in (
        # OpenAI-compatible providers — same wire format, only base_url differs.
        "openai", "groq", "deepseek", "mistral", "openai_compatible", "together", "openrouter",
        "qwen", "xai", "perplexity", "moonshot", "minimax", "huggingface", "lmstudio", "vllm",
    ):
        cfg_copy = dict(cfg)
        cfg_copy.setdefault("provider_name", provider)
        # Set default base_url for known providers
        base_urls = {
            "openai": "https://api.openai.com/v1",
            "groq": "https://api.groq.com/openai/v1",
            "deepseek": "https://api.deepseek.com/v1",
            "mistral": "https://api.mistral.ai/v1",
            "together": "https://api.together.xyz/v1",
            "openrouter": "https://openrouter.ai/api/v1",
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "xai": "https://api.x.ai/v1",
            "perplexity": "https://api.perplexity.ai",
            "moonshot": "https://api.moonshot.cn/v1",
            "minimax": "https://api.minimaxi.chat/v1",
            "huggingface": "https://api-inference.huggingface.co/v1",
            "lmstudio": "http://localhost:1234/v1",
            "vllm": "http://localhost:8000/v1",
        }
        if not cfg_copy.get("base_url") and provider in base_urls:
            cfg_copy["base_url"] = base_urls[provider]
        return OpenAICompatibleLLM(cfg_copy)
    else:
        raise LLMError(f"Unknown provider type: {provider}")


def _supports_tools(
    llm: "BaseLLM",
    tools: list[dict] | None,
    *,
    tools_provider=None,
) -> bool:
    """Check that an LLM concretely overrides `complete_with_tools`.

    Some provider classes (GoogleLLM, OllamaLLM) extend BaseLLM but
    never implement the tool-use loop, and `BaseLLM` itself doesn't
    define one. Without this guard, calling `complete_with_tools` on
    them raises AttributeError mid-request — the caller can't tell
    whether it was a code bug or a routing decision. The qualname
    check rejects `BaseLLM.complete_with_tools` if it ever gets added
    later as a stub; only real subclass overrides count as supporting.

    Phase 2: when `tools_provider` is supplied, the caller intends
    to ship a dynamic per-iteration tool list — treat this as
    "has tools" even if the static `tools` arg is empty.
    """
    if not tools and tools_provider is None:
        return False
    fn = getattr(llm, "complete_with_tools", None)
    if fn is None:
        return False
    qual = getattr(fn, "__qualname__", "")
    return qual.split(".")[0] != "BaseLLM"


# ---------- DualModelRouter ----------
class DualModelRouter:
    def __init__(self, state_path: Path | None = None):
        self.cfg_a = CONFIG.model_a
        self.cfg_b = CONFIG.model_b      # may be None (cloud_only)
        self.cfg_router = CONFIG.router
        self.cfg_verification = CONFIG.verification
        self.mode = CONFIG.mode

        self._model_a: AnthropicLLM | None = None
        self._model_b: OllamaLLM | None = None
        self._active_llm: BaseLLM | None = None
        self._active_cfg_hash: str = ""  # to detect changes
        # Per-task model routing (2026-06-11): "provider_id:model" →
        # constructed LLM, so a routed classifier call doesn't pay
        # client setup every time. Bounded by the routing table size
        # (a handful of entries).
        self._routed_llms: dict[str, BaseLLM] = {}
        self._api_cache: tuple[float, bool] = (0.0, True)
        # If model_b is disabled (cloud_only) — mark it unavailable right away
        self._ollama_cache: tuple[float, bool] = (9e18, self.cfg_b is not None)

        if state_path is None:
            from .knowledge_manager import KM
            state_path = KM.base / "router_state.json"
        self.state_path = state_path
        self.state = self._load_state()

    # ---- per-task model routing (2026-06-11) ----
    def _get_routed_llm(self, task_type) -> "tuple[BaseLLM, str, str] | None":
        """Resolve a per-task model override from model_routing.json.

        Returns (llm, provider_id, model) when this task type is
        routed to a specific model, None otherwise. Never raises —
        an unresolvable route logs and falls back to the active pin.
        """
        try:
            from .model_routing import route_for
            tt = task_type.value if hasattr(task_type, "value") else str(task_type)
            route = route_for(tt)
        except Exception:
            return None
        if not route:
            return None
        provider_id, model = route
        key = f"{provider_id}:{model}"
        llm = self._routed_llms.get(key)
        if llm is None:
            try:
                from .failover import resolve_entry_cfg
                cfg = resolve_entry_cfg(provider_id, model)
                if cfg is None:
                    log.warning(
                        "model routing: %s -> %s/%s not resolvable "
                        "(provider missing/disabled); using active model",
                        task_type, provider_id, model,
                    )
                    return None
                llm = create_llm(cfg)
            except Exception as e:
                log.warning(
                    "model routing: create_llm(%s/%s) failed (%s); "
                    "using active model", provider_id, model, e,
                )
                return None
            if len(self._routed_llms) > 8:
                self._routed_llms.clear()
            self._routed_llms[key] = llm
        return llm, provider_id, model

    # ---- active model (runtime switchable) ----
    def _get_active_llm(self) -> BaseLLM | None:
        """Return an LLM for the user-selected active model, or None to use default A/B routing."""
        from .providers import ACTIVE_MODEL
        cfg = ACTIVE_MODEL.resolve_llm_config()
        if cfg is None:
            self._active_llm = None
            self._active_cfg_hash = ""
            return None
        # Check if config changed since last time
        cfg_hash = f"{cfg.get('provider_id')}:{cfg.get('model')}"
        if cfg_hash != self._active_cfg_hash or self._active_llm is None:
            self._active_llm = create_llm(cfg)
            self._active_cfg_hash = cfg_hash
        return self._active_llm

    # ---- lazy providers ----
    @property
    def model_a(self) -> AnthropicLLM:
        if self._model_a is None:
            self._model_a = AnthropicLLM(self.cfg_a)
        return self._model_a

    @property
    def model_b(self) -> OllamaLLM:
        if self.cfg_b is None:
            raise LLMError(
                f"Model B is disabled in mode: {self.mode}. "
                "Switch mode in config.yaml to local_full / cloud_finetune / local_cpu."
            )
        if self._model_b is None:
            self._model_b = OllamaLLM(self.cfg_b)
        return self._model_b

    # ---- state persistence ----
    def _load_state(self) -> dict:
        today = date.today().isoformat()
        default = {
            "date": today,
            "api_calls_today": 0,
            "api_cost_today": 0.0,
            "model_b_calls_today": 0,
            "total_a_calls": 0,
            "total_b_calls": 0,
            # When the user pins a specific model (Codex / Cohere /
            # Copilot / OpenAI-compatible / a non-default Anthropic),
            # the call doesn't go through the A/B picker — count it
            # separately so dashboards don't muddle pinned-model usage
            # into model-A totals.
            "active_model_calls_today": 0,
            "total_active_model_calls": 0,
            # provider_id:model -> lifetime call count. Lets the
            # WebUI break down "where did today's calls go" without
            # parsing every TokenTracker record.
            "active_model_breakdown": {},
            "last_reason": "",
        }
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return default
        if raw.get("date") != today:
            raw["date"] = today
            raw["api_calls_today"] = 0
            raw["api_cost_today"] = 0.0
            raw["model_b_calls_today"] = 0
            raw["active_model_calls_today"] = 0
        for k, v in default.items():
            raw.setdefault(k, v)
        return raw

    def _save_state(self) -> None:
        try:
            # C3: atomic .tmp + rename so a crash during the budget
            # snapshot doesn't truncate failover.json — that file
            # tracks today's api_calls/cost and the active_model
            # selection; a torn write resets the budget mid-day.
            from .paths import write_atomic_json
            write_atomic_json(self.state_path, self.state)
        except Exception:
            pass

    def _capture_terminal_llm_error(
        self, exc: Exception, *, active=None,
    ) -> None:
        """Persist a structured record of an unrecoverable LLM call.

        Called from the terminal `raise primary_err` paths in
        `call()` / `call_with_tools()` — the points where the
        failover chain has been exhausted and the agent will see
        an LLMError. The error_log is what `run_unified()` reads
        on the NEXT user-facing turn to self-surface the failure
        and let the agent explain + propose a fix.

        Best-effort: a failure in this hook never crashes the
        outer call — the user still gets the original LLMError.
        """
        try:
            from .provider_error_log import (
                classify_llm_error,
                log_provider_error,
            )
            info = classify_llm_error(str(exc))
            provider = "unknown"
            model = "unknown"
            if active is not None:
                provider = (
                    getattr(active, "provider_name", None)
                    or type(active).__name__
                )
                model = getattr(active, "model", None) or "unknown"
            log_provider_error(
                provider=str(provider),
                model=str(model),
                status_code=int(info.get("status_code") or 0),
                message=str(info.get("message") or "")[:300],
                context={
                    "category": info.get("category"),
                    "active_cfg_hash": self._active_cfg_hash,
                },
            )
        except Exception:
            # Never let logging crash an already-failing path.
            pass

    def _track_active_model_call(
        self,
        *,
        provider_id: str | None = None,
        model: str | None = None,
    ) -> None:
        """Bump counters for a call that went through the user-pinned
        model branch. Kept separate from the A/B totals so dashboards
        and `stats()` consumers can tell pinned usage from auto-routed
        usage. The previous implementation lumped pinned calls under
        `total_a_calls` regardless of which provider the model lived
        on (Codex / Cohere / Copilot / Bedrock / OpenAI-compatible),
        which the agent's own self-review correctly flagged as muddled.

        Also bumps `api_cost_today` by the same per-call estimate the
        regular A path uses. Without this, the daily-budget gate
        `api_cost_today >= budget` never fires on pinned-only days
        and a runaway pinned model could spend past the cap silently.
        Real per-token cost is still tracked by TokenTracker; this
        is the router-side budget tally.

        When failover answers via a chain entry (not the pinned
        primary), `provider_id` / `model` override the cached
        `_active_cfg_hash` so the breakdown attributes the call to
        the provider that actually billed, not the one we tried
        first.
        """
        self.state["api_calls_today"] += 1
        self.state["api_cost_today"] += float(
            self.cfg_router.get("estimated_cost_per_call_usd", 0.01)
        )
        self.state["active_model_calls_today"] = (
            int(self.state.get("active_model_calls_today", 0)) + 1
        )
        self.state["total_active_model_calls"] = (
            int(self.state.get("total_active_model_calls", 0)) + 1
        )
        breakdown = self.state.setdefault("active_model_breakdown", {})
        if provider_id and model:
            key = f"{provider_id}:{model}"
        else:
            key = self._active_cfg_hash or "unknown"
        breakdown[key] = int(breakdown.get(key, 0)) + 1

    def _call_with_failover_chain(
        self,
        *,
        primary_fn: Callable[[], str],
        fallback_factory: Callable[[dict], Callable[[], str]],
    ) -> tuple[str, str, str, bool]:
        """Shared failover wrapper for the active-model branches of
        `call` and `call_with_tools`.

        Builds the `attempts` list (primary first, then any enabled
        chain entries that don't duplicate the primary or point at a
        missing/disabled provider) and hands it to `failover.try_call`.

        Returns `(result, used_provider_id, used_model, via_failover)`:
          - `via_failover=True` iff a non-primary chain entry actually
            delivered the result — callers use that to switch the
            usage-breakdown key.
          - `via_failover=False` for the off / chain-empty / primary-won
            paths so legacy attribution (via `_active_cfg_hash`) keeps
            working unchanged.

        When failover is disabled OR the chain is empty, runs the
        primary directly without the failover overhead — same
        behaviour as before Phase 15B.
        """
        from . import failover as _fo
        from .providers import ACTIVE_MODEL as _AM
        # Defensive `or {}`: in real code `_get_active_llm` only
        # returns truthy when `resolve_llm_config` was non-None
        # earlier, but the LLM gets cached on `self._active_llm`
        # while the config dict is not. Tests routinely stub
        # `_get_active_llm` without stubbing `resolve_llm_config`,
        # and a config flip between the two calls (provider got
        # disabled mid-session) is also legitimate. "?" attribution
        # is preferable to a TypeError.
        active_cfg = _AM.resolve_llm_config() or {}
        primary_id: str = active_cfg.get("provider_id") or "?"
        primary_model: str = active_cfg.get("model") or "?"

        cfg = _fo.load_config()
        if not cfg.get("enabled"):
            return primary_fn(), primary_id, primary_model, False

        attempts: list = [(primary_id, primary_model, primary_fn)]
        for entry in cfg.get("chain", []):
            pid = entry.get("provider_id", "")
            em = entry.get("model", "")
            if (pid, em) == (primary_id, primary_model):
                continue  # already tried as primary
            entry_cfg = _fo.resolve_entry_cfg(pid, em)
            if entry_cfg is None:
                continue  # provider gone or disabled
            attempts.append((pid, em, fallback_factory(entry_cfg)))

        # No chain entries actually built (all duplicates / disabled).
        # Skip failover entirely so attribution stays clean.
        if len(attempts) == 1:
            return primary_fn(), primary_id, primary_model, False

        winner: list[str] = [primary_id, primary_model]
        via_failover_flag = [False]

        def _on_success(used_id: str, used_model: str) -> None:
            winner[0] = used_id
            winner[1] = used_model
            if (used_id, used_model) != (primary_id, primary_model):
                via_failover_flag[0] = True

        result = _fo.try_call(
            attempts,
            retry_on=cfg.get("retry_on"),
            max_attempts=cfg.get("max_attempts"),
            on_success=_on_success,
        )
        return result, winner[0], winner[1], via_failover_flag[0]

    def stats(self) -> dict:
        out = dict(self.state)
        out["mode"] = self.mode
        out["model_a_id"] = self.cfg_a.get("model")
        out["model_b_id"] = self.cfg_b.get("model") if self.cfg_b else None
        out["budget_usd"] = self.cfg_router.get("daily_api_budget_usd", 0.0)
        out["model_a_available"] = self._api_available()
        out["model_b_available"] = self._ollama_available()
        # Active model info
        from .providers import ACTIVE_MODEL
        active = ACTIVE_MODEL.get()
        if active:
            out["active_model"] = active
        return out

    # ---- helpers ----
    def _api_available(self) -> bool:
        ttl = self.cfg_router.get("api_ping_cache_seconds", 60)
        ts, ok = self._api_cache
        if time.time() - ts < ttl:
            return ok
        try:
            httpx.head("https://api.anthropic.com", timeout=3.0)
            ok = True
        except Exception:
            ok = False
        self._api_cache = (time.time(), ok)
        return ok

    def _ollama_available(self) -> bool:
        """Check that the local Ollama is running and responding."""
        # In cloud_only mode the local model is disabled
        if self.cfg_b is None:
            return False
        ttl = self.cfg_router.get("api_ping_cache_seconds", 60)
        ts, ok = self._ollama_cache
        if time.time() - ts < ttl:
            return ok
        try:
            base = self.cfg_b.get("base_url", "http://localhost:11434").rstrip("/")
            httpx.get(f"{base}/api/tags", timeout=2.0)
            ok = True
        except Exception:
            ok = False
        self._ollama_cache = (time.time(), ok)
        return ok

    def _current_shift_pct_b(self) -> float:
        """Return the % of A-tasks that should go to B (based on the current Qwen version)."""
        if not self.cfg_router.get("auto_shift_after_finetune", False):
            return 0.0
        schedule = self.cfg_router.get("shift_schedule") or {}
        if not schedule:
            return 0.0
        try:
            from .model_versions import VERSIONS
            cur = VERSIONS.current()
            tag = cur.tag if cur else "v0"
        except Exception:
            tag = "v0"

        def _num(t: str) -> int:
            return int(t[1:]) if t.startswith("v") and t[1:].isdigit() else 0

        cur_n = _num(tag)
        best: str | None = None
        for key in schedule.keys():
            if _num(key) <= cur_n and (best is None or _num(key) > _num(best)):
                best = key
        if best is None:
            return 0.0
        return float(schedule[best].get("model_b_pct", 0))

    def _pick(self, task_type: TaskType) -> tuple[str, str]:
        """Return (provider, reason)."""
        # 1) hard override: verification → A
        if (
            task_type == TaskType.VERIFICATION
            and self.cfg_verification.get("always_use_model_a", True)
        ):
            choice, reason = "a", "verification: forced A"
        else:
            # 2) base on task set
            if task_type in MODEL_A_TASKS:
                choice, reason = "a", f"{task_type.value}: default A"
            elif task_type in MODEL_B_TASKS:
                choice, reason = "b", f"{task_type.value}: default B"
            else:
                choice, reason = "a", f"{task_type.value}: unknown → A"

            # 3) shift_schedule — some A-tasks go to B (only if B is available)
            if choice == "a":
                pct_b = self._current_shift_pct_b()
                if (
                    pct_b > 0
                    and self._ollama_available()
                    and random.random() * 100 < pct_b
                ):
                    choice = "b"
                    reason = f"shift_schedule: {pct_b:.0f}% → B"

            # 4) budget — fallback only if Ollama is up
            if choice == "a":
                budget = self.cfg_router.get("daily_api_budget_usd", 0.0) or 0.0
                if budget > 0 and self.state["api_cost_today"] >= budget:
                    if (
                        self.cfg_router.get("fallback_to_local", True)
                        and self._ollama_available()
                    ):
                        choice = "b"
                        reason = (
                            f"over budget "
                            f"(${self.state['api_cost_today']:.2f} ≥ ${budget}) → B"
                        )
                    else:
                        reason += " (over budget, no B available)"

        # 5) API availability
        if choice == "a" and not self._api_available():
            if (
                self.cfg_router.get("fallback_to_local", True)
                and self._ollama_available()
            ):
                choice = "b"
                reason = "Claude API unavailable → fallback B"
            else:
                raise LLMError(
                    "Claude API unavailable, Qwen/Ollama also unavailable "
                    "(or fallback_to_local=false)"
                )

        # 6) if B was chosen but Ollama is down — escalate to A (if A is available)
        if choice == "b" and not self._ollama_available():
            if self._api_available():
                choice = "a"
                reason = f"{reason} → Ollama down, escalate A"
            else:
                raise LLMError("Both models unavailable (Claude API and Ollama)")

        return choice, reason

    # ---- public calls ----
    def call(
        self,
        task_type: TaskType,
        system: str,
        user: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        attachments: list[str] | None = None,
    ) -> str:
        # Per-task model routing (2026-06-11): cheap task types
        # (classification judges, keyword extraction, quick answers)
        # can run on a cheap model while the pinned active model keeps
        # the heavy turns. Checked BEFORE the safety-fallback chain —
        # routing is an explicit per-task owner choice and must win
        # over generic chain dispatch (on prod the chain is non-empty,
        # so a post-chain hook would never fire — caught in the
        # 2026-06-11 live verify). Direct call, no failover ceremony;
        # on ANY failure we fall through to the normal chain/active
        # path so a routed-model outage can never break a turn.
        _routed = self._get_routed_llm(task_type)
        if _routed is not None:
            _r_llm, _r_pid, _r_model = _routed
            self.state["last_reason"] = f"task-routed: {_r_pid}/{_r_model}"
            try:
                out = _r_llm.complete(
                    system, user, max_tokens=max_tokens,
                    temperature=temperature, attachments=attachments,
                    _task_type=task_type.value,
                )
                self._track_active_model_call(
                    provider_id=_r_pid, model=_r_model,
                )
                self._save_state()
                return out
            except LLMError as e:
                log.warning(
                    "task-routed %s -> %s/%s failed (%s); falling back "
                    "to active model",
                    task_type.value, _r_pid, _r_model,
                    str(e)[:200],
                )

        # Safety-refusal fallback chain (Task 5). When
        # `_active_provider_chain` yields a non-empty ordered list
        # (tests monkeypatch this seam), iterate it and skip past any
        # provider that raises a safety-shaped LLMError. Production
        # default returns [], so the existing A/B + failover-chain
        # orchestration below runs unchanged.
        chain = _active_provider_chain(task_type)
        if chain:
            return _run_with_safety_fallback(
                chain, "call", task_type, system, user,
                max_tokens=max_tokens, temperature=temperature,
                attachments=attachments,
            )

        # Check if user selected a specific model
        active = self._get_active_llm()
        if active is not None:
            tt = task_type.value
            self.state["last_reason"] = f"active model: {self._active_cfg_hash}"

            def _primary_call() -> str:
                return active.complete(
                    system, user, max_tokens=max_tokens, temperature=temperature,
                    attachments=attachments, _task_type=tt,
                )

            def _make_fallback(entry_cfg: dict):
                def _fb(c=entry_cfg) -> str:
                    # Build the fallback LLM lazily so a chain entry
                    # that's never reached (primary succeeded) doesn't
                    # pay for httpx client setup / OAuth handshake.
                    # create_llm exceptions are wrapped as LLMError so
                    # `failover.classify` can route them through the
                    # retry policy — without this, a missing-key
                    # entry stops the whole chain (classify returns
                    # "unknown" for bare KeyError/ValueError).
                    try:
                        llm = create_llm(c)
                    except LLMError:
                        raise
                    except Exception as e:
                        raise LLMError(
                            f"create_llm({c.get('provider_id')}/"
                            f"{c.get('model')}): {e}"
                        ) from e
                    return llm.complete(
                        system, user, max_tokens=max_tokens,
                        temperature=temperature, attachments=attachments,
                        _task_type=tt,
                    )
                return _fb

            out, used_id, used_model, via_failover = self._call_with_failover_chain(
                primary_fn=_primary_call,
                fallback_factory=_make_fallback,
            )
            # #2 fix: attribute the call to the provider that actually
            # answered, BUT only override the breakdown key when
            # failover delivered via a chain entry. If the primary
            # won (or failover was off entirely), fall through to the
            # legacy `_active_cfg_hash`-based attribution that the
            # existing test suite relies on.
            if via_failover:
                self._track_active_model_call(
                    provider_id=used_id, model=used_model,
                )
            else:
                self._track_active_model_call()
            self._save_state()
            return out

        choice, reason = self._pick(task_type)
        self.state["last_reason"] = reason
        tt = task_type.value
        # Audit #13: when failover is enabled but no active model is
        # pinned, the default A/B path should ALSO use the chain
        # (after the legacy A→B fallback fires). Pre-fix the chain
        # only applied to pinned-model turns, which surprised users
        # who configured a failover chain expecting it to always
        # work. Strategy: try the picked model first; on LLMError,
        # try B (if A failed) per legacy behaviour, then walk the
        # failover chain as a final tier. The chain is best-effort
        # — if it's disabled or empty, behaviour matches pre-fix.
        from . import failover as _fo

        def _ab_primary():
            if choice == "a":
                out_local = self.model_a.complete(
                    system, user, max_tokens=max_tokens, temperature=temperature,
                    attachments=attachments, _task_type=tt,
                )
                self.state["api_calls_today"] += 1
                self.state["api_cost_today"] += float(
                    self.cfg_router.get("estimated_cost_per_call_usd", 0.01)
                )
                self.state["total_a_calls"] += 1
                return out_local
            else:
                out_local = self.model_b.complete(
                    system, user, max_tokens=max_tokens, temperature=temperature,
                    _task_type=tt,
                )
                self.state["model_b_calls_today"] += 1
                self.state["total_b_calls"] += 1
                return out_local

        try:
            out = _ab_primary()
            self._save_state()
            return out
        except LLMError as primary_err:
            # Tier 1: legacy A→B fallback (the only fallback before
            # Phase 15B). Preserved for back-compat — Qwen-via-Ollama
            # is the documented "free local model" path.
            if (
                choice == "a"
                and self.cfg_router.get("fallback_to_local", True)
                and self._ollama_available()
            ):
                try:
                    out = self.model_b.complete(
                        system, user, max_tokens=max_tokens, temperature=temperature,
                        _task_type=tt,
                    )
                    self.state["model_b_calls_today"] += 1
                    self.state["total_b_calls"] += 1
                    self.state["last_reason"] = "A runtime error → fallback B"
                    self._save_state()
                    return out
                except LLMError:
                    pass
            # Tier 2 (audit #13): walk the user-configured failover
            # chain. Each entry is a real provider+model registered
            # via WebUI / `hrant failover add`. Failure to load the
            # chain config or build a client falls through to the
            # original re-raise.
            try:
                cfg = _fo.load_config()
                if cfg.get("enabled") and (cfg.get("chain") or []):
                    attempts: list = []
                    for entry in cfg.get("chain") or []:
                        pid = entry.get("provider_id", "")
                        em = entry.get("model", "")
                        entry_cfg = _fo.resolve_entry_cfg(pid, em)
                        if entry_cfg is None:
                            continue

                        def _fb(c=entry_cfg) -> str:
                            try:
                                llm = create_llm(c)
                            except LLMError:
                                raise
                            except Exception as e:
                                raise LLMError(
                                    f"create_llm({c.get('provider_id')}/"
                                    f"{c.get('model')}): {e}"
                                ) from e
                            return llm.complete(
                                system, user, max_tokens=max_tokens,
                                temperature=temperature, attachments=attachments,
                                _task_type=tt,
                            )

                        attempts.append((pid, em, _fb))
                    if attempts:
                        out = _fo.try_call(
                            attempts,
                            retry_on=cfg.get("retry_on"),
                            max_attempts=cfg.get("max_attempts"),
                        )
                        self.state["last_reason"] = (
                            f"{reason} → A/B failed → failover chain"
                        )
                        self._save_state()
                        return out
            except LLMError:
                # Chain exhausted too — fall through to the original
                # raise so the caller sees a coherent error.
                pass
            except Exception as e:
                log.warning("failover chain in A/B path crashed: %s", e)
            self._api_cache = (time.time(), False)
            # Audit follow-up 2026-05-28: persist a self-aware error
            # record so the next user-facing turn can surface this
            # failure with diagnosis + suggested fix.
            self._capture_terminal_llm_error(primary_err, active=active)
            raise primary_err

    def call_json(self, task_type: TaskType, system: str, user: str, **kw) -> dict:
        # Per-task model routing (2026-06-11): when this task type is
        # routed, delegate straight to call() whose routed-first
        # branch handles it (and falls back to chain/active on
        # failure). Without this short-circuit, a non-empty
        # production chain would dispatch call_json to the chain
        # adapters and the routing table would never apply.
        if self._get_routed_llm(task_type) is not None:
            raw = self.call(
                task_type,
                system + "\n\nReply with ONLY valid JSON, no markdown wrappers.",
                user,
                **kw,
            )
            return _parse_json_response(raw)
        # Safety-refusal fallback chain (Task 5). See `call()` for rationale.
        chain = _active_provider_chain(task_type)
        if chain:
            return _run_with_safety_fallback(
                chain, "call_json", task_type, system, user, **kw,
            )
        raw = self.call(
            task_type,
            system + "\n\nReply with ONLY valid JSON, no markdown wrappers.",
            user,
            **kw,
        )
        return _parse_json_response(raw)

    def call_with_tools(
        self,
        task_type: TaskType,
        system: str,
        user: str,
        tools: list[dict],
        execute_tool: ToolExecutor,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_iterations: int = 6,
        on_tool_call: ToolCallCB | None = None,
        attachments: list[str] | None = None,
        tools_provider=None,
        model_override: "tuple[str, str] | None" = None,
    ) -> str:
        # Cascade test-mode (2026-06-12): `model_override` runs THIS
        # tool loop on an explicit (provider_id, model) instead of the
        # active pin — the cascade's small-tier attempt. Explicit by
        # contract: unresolvable override raises LLMError rather than
        # silently substituting the active model (the caller owns the
        # escalation decision). Bypasses the safety chain — the caller
        # IS the fallback orchestrator here.
        if model_override is not None:
            ov_pid, ov_model = model_override
            key = f"{ov_pid}:{ov_model}"
            ov_llm = self._routed_llms.get(key)
            if ov_llm is None:
                from .failover import resolve_entry_cfg
                cfg = resolve_entry_cfg(ov_pid, ov_model)
                if cfg is None:
                    raise LLMError(
                        f"model_override {ov_pid}/{ov_model} not "
                        f"resolvable (provider missing/disabled)"
                    )
                ov_llm = create_llm(cfg)
                if len(self._routed_llms) > 8:
                    self._routed_llms.clear()
                self._routed_llms[key] = ov_llm
            if not _supports_tools(ov_llm, tools, tools_provider=tools_provider):
                raise LLMError(
                    f"model_override {ov_pid}/{ov_model} does not "
                    f"support tool use"
                )
            self.state["last_reason"] = f"override: {key}"
            out = ov_llm.complete_with_tools(
                system, user, tools, execute_tool,
                max_tokens=max_tokens,
                temperature=temperature,
                max_iterations=max_iterations,
                on_tool_call=on_tool_call,
                attachments=attachments,
                _task_type=task_type.value,
                tools_provider=tools_provider,
            )
            self._track_active_model_call(provider_id=ov_pid, model=ov_model)
            self._save_state()
            return out
        # Safety-refusal fallback chain (Task 5). See `call()` for rationale.
        chain = _active_provider_chain(task_type)
        if chain:
            return _run_with_safety_fallback(
                chain, "call_with_tools", task_type, system, user,
                tools, execute_tool,
                max_tokens=max_tokens, temperature=temperature,
                max_iterations=max_iterations, on_tool_call=on_tool_call,
                attachments=attachments, tools_provider=tools_provider,
            )
        """Tool-use loop via selected model.

        If an active model is set and supports tools, use it. Otherwise
        falls back to A/B routing — and **B keeps its tool-use loop**
        when its LLM class implements `complete_with_tools` (fix #4 from
        the self-review). Only models that lack tool support fall through
        to plain `call()`.

        Phase 2: `tools_provider` (callable returning the current tool
        list) lets unified_agent rebuild the schema each iteration
        for bundle-aware loadouts. When supplied, treats "has tools"
        as true even if static `tools` is empty.
        """
        tt = task_type.value

        # Check active model first
        active = self._get_active_llm()
        if active is not None and (tools or tools_provider is not None):
            self.state["last_reason"] = f"active model: {self._active_cfg_hash}"
            if not _supports_tools(active, tools, tools_provider=tools_provider):
                # Pinned providers like GoogleLLM and OllamaLLM extend
                # BaseLLM but never override `complete_with_tools` —
                # the previous code unconditionally called it and got
                # `AttributeError`. Surface a clear LLMError instead so
                # the operator sees WHY their pinned model can't run a
                # tool task. Falling back to default A/B silently would
                # also be wrong: the user explicitly pinned a model
                # for a reason, swapping it out without a signal hides
                # bugs.
                raise LLMError(
                    f"Active model {self._active_cfg_hash!r} does not "
                    f"support tool use. Switch to a tool-capable provider "
                    f"(Anthropic / OpenAI / Codex / Cohere / Bedrock) or "
                    f"clear the pinned model for this task."
                )
            def _primary_call() -> str:
                return active.complete_with_tools(
                    system, user, tools, execute_tool,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_iterations=max_iterations,
                    on_tool_call=on_tool_call,
                    attachments=attachments,
                    _task_type=tt,
                    tools_provider=tools_provider,
                )

            def _make_fallback(entry_cfg: dict):
                def _fb(c=entry_cfg) -> str:
                    # Lazy build per #16 — probe LLM only when the
                    # chain actually reaches this entry. Tool-support
                    # check happens BEFORE the actual call so we don't
                    # consume an API attempt on a non-tool model.
                    try:
                        llm = create_llm(c)
                    except LLMError:
                        raise
                    except Exception as e:
                        raise LLMError(
                            f"create_llm({c.get('provider_id')}/"
                            f"{c.get('model')}): {e}"
                        ) from e
                    if not _supports_tools(llm, tools, tools_provider=tools_provider):
                        raise LLMError(
                            f"chain entry {c.get('provider_id')}/"
                            f"{c.get('model')} does not support tools"
                        )
                    return llm.complete_with_tools(
                        system, user, tools, execute_tool,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        max_iterations=max_iterations,
                        on_tool_call=on_tool_call,
                        attachments=attachments,
                        _task_type=tt,
                        tools_provider=tools_provider,
                    )
                return _fb

            out, used_id, used_model, via_failover = self._call_with_failover_chain(
                primary_fn=_primary_call,
                fallback_factory=_make_fallback,
            )
            if via_failover:
                self._track_active_model_call(
                    provider_id=used_id, model=used_model,
                )
            else:
                self._track_active_model_call()
            self._save_state()
            return out

        choice, reason = self._pick(task_type)
        self.state["last_reason"] = reason

        # Pick the LLM for this turn first so we can probe its capabilities.
        target = self.model_a if choice == "a" else self.model_b
        if not _supports_tools(target, tools, tools_provider=tools_provider):
            # The previous code here `return self.call(...)` SILENTLY
            # stripped tools and ran a plain completion when the
            # routed-to model lacked complete_with_tools. That's
            # actively wrong for tool-required tasks: arithmetic
            # demands `calc`, self-analysis demands `read_file`, etc.
            # — losing tools means hallucinating where the model
            # would otherwise have called the right tool.
            #
            # Two safe paths instead:
            #   - If we picked B but A is available AND tool-capable,
            #     escalate to A (the same fallback shape `_pick`
            #     already uses for budget / health failures).
            #   - Otherwise raise LLMError so the caller knows tools
            #     are unavailable. Empty `tools` is impossible here
            #     because _supports_tools only returns False if the
            #     class lacks the override — empty tools would have
            #     been short-circuited earlier in the function.
            if (
                choice == "b"
                and _supports_tools(self.model_a, tools, tools_provider=tools_provider)
                and self._api_available()
            ):
                target = self.model_a
                self.state["last_reason"] = (
                    f"{reason} → B lacks tool support, escalate A"
                )
            else:
                raise LLMError(
                    f"Selected model ({choice.upper()}) does not support "
                    f"tool use, and no tool-capable fallback is available. "
                    f"Configure a tool-capable model A (Anthropic / OpenAI / "
                    f"Codex / Cohere / Bedrock) or remove tools from this task."
                )

        try:
            out = target.complete_with_tools(
                system, user, tools, execute_tool,
                max_tokens=max_tokens,
                temperature=temperature,
                max_iterations=max_iterations,
                on_tool_call=on_tool_call,
                attachments=attachments if choice == "a" else None,
                _task_type=tt,
                tools_provider=tools_provider,
            )
            if choice == "a":
                self.state["api_calls_today"] += 1
                self.state["api_cost_today"] += float(
                    self.cfg_router.get("estimated_cost_per_call_usd", 0.01)
                )
                self.state["total_a_calls"] += 1
            else:
                self.state["model_b_calls_today"] += 1
                self.state["total_b_calls"] += 1
            self._save_state()
            return out
        except LLMError as e:
            if choice == "a":
                self._api_cache = (time.time(), False)
            # Audit follow-up 2026-05-28: terminal-error self-surface.
            self._capture_terminal_llm_error(e, active=active)
            raise


# ---------- singleton ----------
_router: DualModelRouter | None = None


def router() -> DualModelRouter:
    global _router
    if _router is None:
        _router = DualModelRouter()
    return _router


def reset_router() -> None:
    """Reset the singleton (needed for tests and after a config change)."""
    global _router
    _router = None
