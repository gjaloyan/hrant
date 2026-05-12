"""Self-critic stage — verification + critique formatting.

`_verify` invokes `backend.verifier.verify` to grade the answer
against notes + tool outputs + structured claims (P0 Phase C). It's
only called from the deep_agent pipeline tier; task_mode skips it
and uses a placeholder VR built from thinking.confidence.

`_build_critique` formats verifier output (unverified claims +
contradictions + previous answer) into a CRITIQUE block the solver
sees on retry — that's how the self-critic loop conveys "what was
wrong" to the next attempt.

Both methods stay close to `Agent.run` semantically but live here
to keep agent.py focused on the orchestrator.
"""
from __future__ import annotations

from ..llm import TaskType
from ..models import Note, VerificationResult


class SelfCriticMixin:
    """Provides `_build_critique(vr, prev_answer)` and
    `_verify(task, answer, notes, tool_context="")` to `Agent`.

    `_verify` reads `self._trace`, `self._last_solver_claims`,
    `self._record_llm_call`, `self._notes_block`, `self.progress`
    from the host Agent. Late-imports `verify`, `router`, `TOKENS`,
    `CONFIG` via `backend.agent` to keep test patches working.
    """

    @staticmethod
    def _build_critique(vr: VerificationResult, prev_answer: str) -> str:
        """Build a CRITIQUE block from verifier feedback for the
        retry solver. Static — doesn't need self."""
        parts = [
            "# CRITIQUE OF YOUR PREVIOUS ANSWER",
            f"Your previous answer scored {vr.confidence}% confidence.",
            "The verifier found the following problems:\n",
        ]
        if vr.unverified_claims:
            parts.append("## Unverified claims (no evidence found):")
            for c in vr.unverified_claims:
                parts.append(f"- {c}")
        if vr.contradictions:
            parts.append("\n## Contradictions (conflicts with sources):")
            for c in vr.contradictions:
                parts.append(f"- {c}")
        parts.append(
            f"\n## Your previous answer (to revise):\n{prev_answer[:2000]}"
        )
        parts.append(
            "\nFix these issues. Use tools to find evidence. "
            "Remove claims you cannot support."
        )
        return "\n".join(parts)

    def _verify(
        self,
        task: str,
        answer: str,
        notes: "list[Note]",
        tool_context: str = "",
    ) -> VerificationResult:
        """Hand the answer + tool evidence + structured claims to
        the verifier. Returns a VerificationResult with confidence,
        verified/unverified/contradicted claim lists, and the notes
        that were checked. When the verifier subsystem is disabled
        in CONFIG, returns 100% confidence without doing any LLM
        work (a useful escape hatch for offline / debug runs).
        """
        from ..agent import CONFIG, TOKENS
        from ..verifier import verify

        if not CONFIG.verification["enabled"]:
            return VerificationResult(
                confidence=100,
                notes_used=[n.frontmatter.topic for n in notes],
            )
        self.progress("verify", "verifying answer...")

        usage_before = TOKENS.request_usage()

        def _capture(system, user, response, duration_ms):
            self._record_llm_call(
                label="_verify",
                task_type=TaskType.VERIFICATION,
                system=system,
                user=user,
                response=response,
                duration_ms=duration_ms,
                usage_before=usage_before,
            )

        # P0 Phase C: hand the verifier the solver's structured
        # claims + tool-call order so it can rule per-claim against
        # the exact evidence the solver cited. Both args are
        # best-effort — missing solver tail or empty trace falls
        # back to Phase A (legacy regex-based extraction).
        solver_claims = getattr(self, "_last_solver_claims", None)
        tool_call_order: list[dict] = []
        for step in self._trace:
            tc = step.tool_call
            if tc is None:
                continue
            tool_call_order.append(
                {
                    "name": tc.name,
                    "args": tc.args or {},
                    "result": tc.result or "",
                    "is_error": bool(tc.is_error),
                }
            )

        return verify(
            question=task,
            answer=answer,
            notes_text=self._notes_block(notes),
            used_topics=[n.frontmatter.topic for n in notes],
            tool_context=tool_context,
            on_llm_call=_capture,
            solver_claims=solver_claims,
            tool_call_order=tool_call_order,
        )
