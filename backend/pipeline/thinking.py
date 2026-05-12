"""Thinking module — second stage for task-intent turns.

Runs once per task before the solver. Picks `question_type`
(factual / calculation / file_operation / self_analysis / …),
decides which tools to call and why, lists knowledge topics to
load, rates own confidence 0-100, optionally decomposes into
subtasks. The returned `ThinkingResult` drives:
  - knowledge loading in `Agent.run`
  - tool selection in `_solve`
  - the pipeline-mode dispatch in `_pick_pipeline_mode`
    (confidence < 60 OR subtasks → deep_agent)

Identity preamble is mixed in via `_with_identity` so the thinker
sees the user's pinned language preference + name + role — without
that it can't tell whether "Hrant?" is a name reference vs an
unknown word.
"""
from __future__ import annotations

import time as _t

from ..llm import TaskType
from ..models import ThinkingResult
from ..prompts import THINKING_SYSTEM


class ThinkingMixin:
    """Provides `_think(task, core)` to `Agent`.

    Reads `self._shared_context`, `self._attachment_marker`,
    `self._arithmetic_marker`, `self._record_llm_call`, and
    `self.progress` from the host Agent. Late-imports
    `_is_self_question`, `_looks_like_self_analysis_request`,
    `_capabilities_block`, `_with_identity`, `router`, `TOKENS`
    via `backend.agent` so test patches on those names still
    intercept the calls here.
    """

    def _think(self, task: str, core: str) -> "ThinkingResult":
        """Universal thinking protocol: reason about ANY request
        before acting. The result drives knowledge loading, tool
        selection, solver prompting, and pipeline-mode dispatch.
        """
        from ..agent import (
            _capabilities_block,
            _is_self_question,
            _looks_like_self_analysis_request,
            _with_identity,
            router,
            TOKENS,
        )

        self.progress("think", "thinking...")
        # Self-analysis variants need the FULL source-map view +
        # wider convo/memory windows + skip-stale-memory mode.
        # Everything else uses the compact capabilities block (saves
        # ~3 KB per turn).
        is_selfish = (
            _is_self_question(task)
            or _looks_like_self_analysis_request(task)
        )
        caps = _capabilities_block(compact=not is_selfish)
        ctx = self._shared_context(
            task,
            core,
            n_conv=6 if is_selfish else 3,
            n_memory=10 if is_selfish else 4,
            for_self_analysis=is_selfish,
        )
        marker = self._attachment_marker() + self._arithmetic_marker(task)
        user = (
            f"{ctx}\n\n"
            f"# MY CAPABILITIES\n{caps}\n\n"
            f"# USER REQUEST\n{marker}{task}"
        )

        _t0 = _t.monotonic()
        think_system = _with_identity(THINKING_SYSTEM)
        usage_before = TOKENS.request_usage()
        data = router().call_json(
            TaskType.TASK_ANALYSIS,
            think_system,
            user,
            max_tokens=1000,
            temperature=0.2,
        )
        self._record_llm_call(
            label="_think",
            task_type=TaskType.TASK_ANALYSIS,
            system=think_system,
            user=user,
            response=str(data),
            duration_ms=int((_t.monotonic() - _t0) * 1000),
            usage_before=usage_before,
        )
        result = ThinkingResult(
            question_type=str(data.get("question_type", "factual")).strip(),
            core_question=str(data.get("core_question", task)).strip(),
            already_know=list(data.get("already_know") or []),
            knowledge_gaps=list(data.get("knowledge_gaps") or []),
            approach=str(data.get("approach", "")).strip(),
            tools_needed=list(data.get("tools_needed") or []),
            tools_reasoning=str(data.get("tools_reasoning", "")).strip(),
            required_topics=list(data.get("required_topics") or []),
            plan=list(data.get("plan") or []),
            confidence=int(data.get("confidence", 50)),
            reasoning=str(data.get("reasoning", "")).strip(),
            subtasks=list(data.get("subtasks") or []),
        )
        if result.subtasks:
            self.progress(
                "decompose",
                f"complex task → {len(result.subtasks)} subtasks",
            )
        return result
