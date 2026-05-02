"""Dual-model router: Claude Sonnet (brain) + Qwen 7B (apprentice).

Логика выбора провайдера на каждый вызов:
  1. TaskType ∈ MODEL_A_TASKS → кандидат = A (Claude)
  2. TaskType ∈ MODEL_B_TASKS → кандидат = B (Qwen)
  3. verification + always_use_model_a=true → жёстко A (override)
  4. shift_schedule (auto_shift_after_finetune) → часть A-задач случайно уходит на B
  5. daily_api_budget_usd превышен → fallback на B
  6. Claude API недоступен → fallback на B (если fallback_to_local=true)

State (per-day счётчики и суммарные вызовы) персистится в knowledge/router_state.json.
"""
from __future__ import annotations
import json
import os
import random
import threading
import time
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx

from .config import CONFIG


class LLMError(RuntimeError):
    pass


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
        # Global totals
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0
        self._total_calls = 0

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

    def reset_request(self) -> None:
        """Reset per-request counters (called at start of agent.run())."""
        with self._lock:
            self._request_input = 0
            self._request_output = 0
            self._request_cache_read = 0
            self._request_cache_create = 0
            self._request_cost = 0.0
            self._request_calls = 0

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


# ---------- базовый парсер JSON ----------
def _parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        raise LLMError(f"LLM вернул не-JSON: {raw[:200]}")
    try:
        return json.loads(raw[start : end + 1])
    except json.JSONDecodeError as e:
        raise LLMError(f"Ошибка парсинга JSON: {e}\n{raw[:300]}") from e


# ---------- провайдеры ----------
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
            raise LLMError(f"Не задан {key_env} в окружении/.env")
        self.model = cfg["model"]
        self.default_max = cfg.get("max_tokens", 2000)
        self.default_temp = cfg.get("temperature", 0.3)
        self.url = "https://api.anthropic.com/v1/messages"

    # ----- low-level POST -----
    def _post(self, payload: dict, *, _max_retries: int = 5) -> dict:
        """POST to Anthropic API with automatic retry on transient errors.

        Retries on: 429 (rate limit), 529 (overloaded), 500/502/503 (server),
        and connection/timeout errors. Uses exponential backoff with the
        Retry-After header when provided.
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        RETRYABLE_STATUSES = {429, 500, 502, 503, 529}
        last_error: Exception | None = None

        for attempt in range(_max_retries + 1):
            try:
                r = httpx.post(self.url, json=payload, headers=headers, timeout=120.0)
                r.raise_for_status()
                return r.json()
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                if status in RETRYABLE_STATUSES and attempt < _max_retries:
                    # Use Retry-After header if present, otherwise exponential backoff
                    retry_after = e.response.headers.get("retry-after")
                    if retry_after:
                        try:
                            wait = min(float(retry_after), 60.0)
                        except ValueError:
                            wait = min(2 ** attempt * 2, 60.0)
                    else:
                        wait = min(2 ** attempt * 2, 60.0)
                    import logging
                    logging.getLogger("llm").warning(
                        f"Anthropic API {status}, retry {attempt+1}/{_max_retries} "
                        f"in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    continue
                # Non-retryable status or retries exhausted
                body = (e.response.text or "").strip()
                try:
                    err = e.response.json().get("error", {})
                    detail = f"{err.get('type', '?')}: {err.get('message', body)}"
                except Exception:
                    detail = body or str(e)
                raise LLMError(
                    f"Anthropic API {status} "
                    f"(model={self.model!r}): {detail}"
                ) from e
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < _max_retries:
                    wait = min(2 ** attempt * 2, 60.0)
                    import logging
                    logging.getLogger("llm").warning(
                        f"Anthropic connection error: {e}, retry {attempt+1}/{_max_retries} "
                        f"in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    continue
                raise LLMError(f"Anthropic API error after {_max_retries} retries: {e}") from e
        # Should not reach here, but just in case
        raise LLMError(f"Anthropic API failed after {_max_retries} retries: {last_error}")

    @staticmethod
    def _build_user_content(user: str, attachments: list[str] | None) -> "str | list[dict]":
        """Anthropic content array with images inlined as base64.

        For audio attachments we use the transcript (Anthropic doesn't
        accept raw audio); fallback to placeholder if no transcript yet.
        """
        resolved = _resolve_attachments(attachments)
        if not resolved:
            return user
        import base64 as _b64
        blocks: list[dict] = []
        for meta, data in resolved:
            if meta.kind == "image":
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": meta.mime_type,
                        "data": _b64.b64encode(data).decode("ascii"),
                    },
                })
            elif meta.kind == "audio":
                if meta.transcript:
                    blocks.append({"type": "text", "text": f"[voice transcript]\n{meta.transcript}"})
                else:
                    blocks.append({"type": "text", "text": "[voice attachment, transcript unavailable]"})
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
            raise LLMError(f"Неожиданный формат ответа: {data}") from e

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
    ) -> str:
        """Multi-turn tool-use loop.

        Шаги:
          1. Отправляем messages + tools.
          2. Если LLM вернула stop_reason=tool_use, исполняем все tool_use
             блоки и кладём tool_result обратно в messages.
          3. Повторяем до stop_reason=end_turn (или max_iterations).

        Возвращает финальный текст ассистента (склеенные text-блоки).
        Безопасно деградирует, если tools пустые или модель сразу
        возвращает текст. `attachments` (sha256 list) attach images on
        the first user turn.
        """
        first_content = self._build_user_content(user, attachments)
        messages: list[dict] = [{"role": "user", "content": first_content}]
        final_text = ""
        for _iter in range(max_iterations):
            payload = {
                "model": self.model,
                "max_tokens": max_tokens or self.default_max,
                "temperature": temperature if temperature is not None else self.default_temp,
                "system": system,
                "messages": messages,
            }
            if tools:
                payload["tools"] = tools
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

            # Собираем текст и tool_use блоки
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

            # Кладём ответ ассистента (со всеми блоками) и tool_result в messages
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
                result_block: dict = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_text,
                }
                if is_error:
                    result_block["is_error"] = True
                tool_results.append(result_block)
            messages.append({"role": "user", "content": tool_results})

        # Hit max_iterations with the model still wanting to call tools.
        # Force one final tool-less synthesis call so the user gets a
        # real answer instead of the last preamble.
        synth_payload = {
            "model": self.model,
            "max_tokens": max_tokens or self.default_max,
            "temperature": temperature if temperature is not None else self.default_temp,
            "system": system,
            "messages": messages,
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
        auth = self._get_auth_headers()
        headers = {
            **auth,
            "Content-Type": "application/json",
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
                status = e.response.status_code
                if status in RETRYABLE and attempt < _max_retries:
                    retry_after = e.response.headers.get("retry-after")
                    wait = float(retry_after) if retry_after else min(2 ** attempt * 2, 60.0)
                    wait = min(wait, 60.0)
                    time.sleep(wait)
                    continue
                body = e.response.text[:500]
                raise LLMError(f"OpenAI API {status} ({self.provider_name}/{self.model}): {body}") from e
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_error = e
                if attempt < _max_retries:
                    time.sleep(min(2 ** attempt * 2, 60.0))
                    continue
                raise LLMError(f"OpenAI API error after {_max_retries} retries: {e}") from e
        raise LLMError(f"OpenAI API failed: {last_error}")

    @staticmethod
    def _build_user_content(user: str, attachments: list[str] | None) -> "str | list[dict]":
        """OpenAI chat-completions content array with image_url data URIs.

        Audio attachments fall back to their transcript (text); raw audio
        in chat-completions isn't broadly supported across compat providers.
        """
        resolved = _resolve_attachments(attachments)
        if not resolved:
            return user
        import base64 as _b64
        blocks: list[dict] = []
        for meta, data in resolved:
            if meta.kind == "image":
                b64 = _b64.b64encode(data).decode("ascii")
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{meta.mime_type};base64,{b64}"},
                })
            elif meta.kind == "audio":
                if meta.transcript:
                    blocks.append({"type": "text", "text": f"[voice transcript]\n{meta.transcript}"})
                else:
                    blocks.append({"type": "text", "text": "[voice attachment, transcript unavailable]"})
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
                usage={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                },
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
    ) -> str:
        """OpenAI-style tool-use loop. `attachments` (sha256 list) attach
        images on the first user turn via image_url data URIs."""
        # Convert Anthropic tool format to OpenAI format if needed
        oai_tools = []
        for t in tools:
            if "function" in t:
                oai_tools.append(t)
            else:
                oai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema", {}),
                    },
                })

        first_user = self._build_user_content(user, attachments)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": first_user},
        ]
        final_text = ""
        for _iter in range(max_iterations):
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
                    usage={
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    },
                    duration_ms=duration_ms,
                    prompt_preview=user[:300] if _iter == 0 else f"(tool iteration {_iter})",
                )

            choice = data["choices"][0]
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
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result_text,
                })

        # Forced tool-less synthesis at the cap.
        synth_payload = {
            "model": self.model,
            "max_tokens": max_tokens or self.default_max,
            "temperature": temperature if temperature is not None else self.default_temp,
            "messages": messages,
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
                    usage={
                        "input_tokens": usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                    },
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

    def _build_payload(
        self, system: str, input_items: list[dict], tools: list[dict] | None,
        max_tokens: int | None, temperature: float | None,
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
        """
        resolved = _resolve_attachments(attachments)
        blocks: list[dict] = []
        if resolved:
            import base64 as _b64
            for meta, data in resolved:
                if meta.kind == "image":
                    b64 = _b64.b64encode(data).decode("ascii")
                    blocks.append({
                        "type": "input_image",
                        "image_url": f"data:{meta.mime_type};base64,{b64}",
                    })
                elif meta.kind == "audio":
                    if meta.transcript:
                        blocks.append({"type": "input_text", "text": f"[voice transcript]\n{meta.transcript}"})
                    else:
                        blocks.append({"type": "input_text", "text": "[voice attachment, transcript unavailable]"})
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
        payload = self._build_payload(system, input_items, None, max_tokens, temperature)
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
    ) -> str:
        # Convert tools to Responses API flat format. Accept either:
        #   {"name", "description", "input_schema"}                  (Anthropic-ish)
        #   {"type":"function","function":{"name","description","parameters"}}  (Chat Completions)
        flat_tools: list[dict] = []
        for t in tools or []:
            if t.get("type") == "function" and isinstance(t.get("function"), dict):
                fn = t["function"]
                flat_tools.append({
                    "type": "function",
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                    "strict": False,
                })
            else:
                flat_tools.append({
                    "type": "function",
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema") or t.get("parameters") or {},
                    "strict": False,
                })

        input_items: list[dict] = [
            {
                "type": "message",
                "role": "user",
                "content": self._build_user_content_blocks(user, attachments),
            }
        ]

        final_text = ""
        for _iter in range(max_iterations):
            payload = self._build_payload(
                system, input_items, flat_tools or None, max_tokens, temperature
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
                input_items.append({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": result_text,
                })

        # Forced tool-less synthesis at the cap — call again with no tools.
        synth_payload = self._build_payload(
            system, input_items, None, max_tokens, temperature,
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
    ) -> str:
        _ = attachments  # vision not supported on this backend
        # Convert tools to Anthropic shape: {name, description, input_schema}
        anthropic_tools = []
        for t in tools or []:
            if "input_schema" in t:
                anthropic_tools.append(t)
            elif t.get("type") == "function" and isinstance(t.get("function"), dict):
                fn = t["function"]
                anthropic_tools.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {}),
                })

        messages: list[dict] = [{"role": "user", "content": user}]
        final_text = ""
        for _iter in range(max_iterations):
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
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_text,
                    **({"is_error": True} if is_error else {}),
                })
            messages.append({"role": "user", "content": tool_results})

        # Forced tool-less synthesis at the cap.
        synth_payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or self.default_max,
            "temperature": temperature if temperature is not None else self.default_temp,
            "system": system,
            "messages": messages,
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
    ) -> str:
        _ = attachments  # vision not supported on this backend
        # Cohere v2 accepts OpenAI-style {type:"function", function:{...}} tools.
        coh_tools: list[dict] = []
        for t in tools or []:
            if t.get("type") == "function" and isinstance(t.get("function"), dict):
                coh_tools.append(t)
            else:
                coh_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.get("name", ""),
                        "description": t.get("description", ""),
                        "parameters": t.get("input_schema") or t.get("parameters") or {},
                    },
                })

        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        final_text = ""
        for _iter in range(max_iterations):
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
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": [{"type": "text", "text": result_text}],
                })
        # Forced tool-less synthesis at the cap.
        synth_payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.default_max,
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
                            if c.get("type") == "output_text" and c.get("text"):
                                text_chunks.append(c["text"])
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


# ---------- DualModelRouter ----------
class DualModelRouter:
    def __init__(self, state_path: Path | None = None):
        self.cfg_a = CONFIG.model_a
        self.cfg_b = CONFIG.model_b      # может быть None (claude_only)
        self.cfg_router = CONFIG.router
        self.cfg_verification = CONFIG.verification
        self.mode = CONFIG.mode

        self._model_a: AnthropicLLM | None = None
        self._model_b: OllamaLLM | None = None
        self._active_llm: BaseLLM | None = None
        self._active_cfg_hash: str = ""  # to detect changes
        self._api_cache: tuple[float, bool] = (0.0, True)
        # Если model_b выключен (claude_only) — сразу помечаем как недоступный
        self._ollama_cache: tuple[float, bool] = (9e18, self.cfg_b is not None)

        if state_path is None:
            from .knowledge_manager import KM
            state_path = KM.base / "router_state.json"
        self.state_path = state_path
        self.state = self._load_state()

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
                f"Model B выключен в режиме mode: {self.mode}. "
                "Переключи mode в config.yaml на local_full / cloud_finetune / local_cpu."
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
        for k, v in default.items():
            raw.setdefault(k, v)
        return raw

    def _save_state(self) -> None:
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(self.state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass

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

    # ---- вспомогательное ----
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
        """Проверка, что локальная Ollama запущена и отвечает."""
        # В режиме claude_only локальная модель выключена
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
        """Возвращает % A-задач, которые должны уйти на B (исходя из текущей версии Qwen)."""
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
        """Возвращает (provider, reason)."""
        # 1) hard override: verification → A
        if (
            task_type == TaskType.VERIFICATION
            and self.cfg_verification.get("always_use_model_a", True)
        ):
            choice, reason = "a", "verification: forced A"
        else:
            # 2) база по task set
            if task_type in MODEL_A_TASKS:
                choice, reason = "a", f"{task_type.value}: default A"
            elif task_type in MODEL_B_TASKS:
                choice, reason = "b", f"{task_type.value}: default B"
            else:
                choice, reason = "a", f"{task_type.value}: unknown → A"

            # 3) shift_schedule — часть A-задач уходит на B (только если B доступен)
            if choice == "a":
                pct_b = self._current_shift_pct_b()
                if (
                    pct_b > 0
                    and self._ollama_available()
                    and random.random() * 100 < pct_b
                ):
                    choice = "b"
                    reason = f"shift_schedule: {pct_b:.0f}% → B"

            # 4) budget — fallback только если Ollama поднят
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
                reason = "Claude API недоступен → fallback B"
            else:
                raise LLMError(
                    "Claude API недоступен, Qwen/Ollama также недоступен "
                    "(или fallback_to_local=false)"
                )

        # 6) если выбран B, но Ollama down — эскалация на A (если A доступен)
        if choice == "b" and not self._ollama_available():
            if self._api_available():
                choice = "a"
                reason = f"{reason} → Ollama down, escalate A"
            else:
                raise LLMError("Обе модели недоступны (Claude API и Ollama)")

        return choice, reason

    # ---- публичные вызовы ----
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
        # Check if user selected a specific model
        active = self._get_active_llm()
        if active is not None:
            tt = task_type.value
            self.state["last_reason"] = f"active model: {self._active_cfg_hash}"
            out = active.complete(
                system, user, max_tokens=max_tokens, temperature=temperature,
                attachments=attachments, _task_type=tt,
            )
            self.state["api_calls_today"] += 1
            self.state["total_a_calls"] += 1
            self._save_state()
            return out

        choice, reason = self._pick(task_type)
        self.state["last_reason"] = reason
        tt = task_type.value
        try:
            if choice == "a":
                out = self.model_a.complete(
                    system, user, max_tokens=max_tokens, temperature=temperature,
                    attachments=attachments, _task_type=tt,
                )
                self.state["api_calls_today"] += 1
                self.state["api_cost_today"] += float(
                    self.cfg_router.get("estimated_cost_per_call_usd", 0.01)
                )
                self.state["total_a_calls"] += 1
            else:
                # Local Qwen (Ollama base) doesn't accept image attachments;
                # they'll be silently dropped by the LLM. That's correct
                # behaviour for non-vision models.
                out = self.model_b.complete(
                    system, user, max_tokens=max_tokens, temperature=temperature,
                    _task_type=tt,
                )
                self.state["model_b_calls_today"] += 1
                self.state["total_b_calls"] += 1
            self._save_state()
            return out
        except LLMError:
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
            self._api_cache = (time.time(), False)
            raise

    def call_json(self, task_type: TaskType, system: str, user: str, **kw) -> dict:
        raw = self.call(
            task_type,
            system + "\n\nОтвечай ТОЛЬКО валидным JSON, без markdown-обёрток.",
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
    ) -> str:
        """Tool-use loop via selected model.

        If an active model is set and supports tools, use it. Otherwise
        falls back to A/B routing — and **B keeps its tool-use loop**
        when its LLM class implements `complete_with_tools` (fix #4 from
        the self-review). Only models that lack tool support fall through
        to plain `call()`.
        """
        tt = task_type.value

        # Check active model first
        active = self._get_active_llm()
        if active is not None and tools:
            self.state["last_reason"] = f"active model: {self._active_cfg_hash}"
            try:
                out = active.complete_with_tools(
                    system, user, tools, execute_tool,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    max_iterations=max_iterations,
                    on_tool_call=on_tool_call,
                    attachments=attachments,
                    _task_type=tt,
                )
                self.state["api_calls_today"] += 1
                self.state["total_a_calls"] += 1
                self._save_state()
                return out
            except LLMError:
                raise

        choice, reason = self._pick(task_type)
        self.state["last_reason"] = reason

        # Pick the LLM for this turn first so we can probe its capabilities.
        target = self.model_a if choice == "a" else self.model_b
        supports_tools = (
            tools
            and hasattr(target, "complete_with_tools")
            and target.complete_with_tools.__qualname__.split(".")[0] != "BaseLLM"
        )

        if not supports_tools:
            return self.call(
                task_type, system, user,
                max_tokens=max_tokens, temperature=temperature,
                attachments=attachments,
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
        except LLMError:
            if choice == "a":
                self._api_cache = (time.time(), False)
            raise


# ---------- singleton ----------
_router: DualModelRouter | None = None


def router() -> DualModelRouter:
    global _router
    if _router is None:
        _router = DualModelRouter()
    return _router


def reset_router() -> None:
    """Сбросить singleton (нужно для тестов и после смены конфига)."""
    global _router
    _router = None
