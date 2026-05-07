"""Claim/Evidence layer — Phase A.

Assembles `Claim` + `EvidenceItem` objects from data the system
ALREADY produces (the verifier's claim buckets + the thinking trace's
tool calls). Phase B will have the solver emit claims and evidence
directly with proper end-to-end binding; this module is the seam that
lets us populate the new fields on `AgentAnswer` today, without
changing the solver's output contract.

The shapes match `backend.models.Claim` / `backend.models.EvidenceItem`.
Anything in here is pure data transformation — no LLM calls, no tool
calls. That makes it cheap to run on every turn and easy to test.
"""
from __future__ import annotations

import re
from typing import Iterable

from .models import (
    Claim,
    EvidenceItem,
    ThinkingStep,
    VerificationResult,
)


# Per-source-type cap on the inline `quote` field. Tool results in the
# trace are already truncated upstream (Round 8: 4k preview); we
# re-clip here so a future tweak that increases the trace cap doesn't
# silently inflate every AgentAnswer payload too. 800 is enough for
# the model to anchor a claim, short enough to stay out of the
# response-bandwidth conversation.
_QUOTE_CAP = 800

# Default risk per status bucket. Verifier already classifies an
# unverified claim as "uncertain" and a contradiction as "wrong"; risk
# is just the consumer-facing label.
_STATUS_RISK = {
    "verified": "low",
    "unverified": "medium",
    "contradicted": "high",
}


def build_claims_and_evidence(
    verification: VerificationResult,
    thinking_trace: list[ThinkingStep],
    *,
    user_message: str = "",
) -> tuple[list[Claim], list[EvidenceItem]]:
    """Build a Claim list + Evidence list from existing turn data.

    `verification` carries the buckets the verifier already produces
    (`verified_claims`, `unverified_claims`, `contradictions`).
    `thinking_trace` carries the tool calls the agent ran to get
    grounding. `user_message` (optional) is included as one
    "user" evidence item so a claim that quotes the user has a real
    source_ref to point at.

    Phase A does NOT attempt to bind specific evidence to specific
    claims — `Claim.evidence_ids` stays empty. The structures exist
    so the WebUI / Telegram / eventual claim-by-claim verifier have a
    stable shape to consume; Phase B fills the binding in.
    """
    evidence = list(_evidence_from_trace(thinking_trace))
    if user_message and user_message.strip():
        evidence.append(EvidenceItem(
            id=f"ev_user_{len(evidence):03d}",
            source_type="user",
            source_ref="user_turn",
            quote=user_message.strip()[:_QUOTE_CAP],
            confidence=1.0,
        ))
    claims = list(_claims_from_verification(verification))
    return claims, evidence


def _evidence_from_trace(trace: list[ThinkingStep]) -> Iterable[EvidenceItem]:
    """One EvidenceItem per tool call in the trace. Errors get
    `confidence=0.0` so consumers can tell at a glance that the
    grounding came from a failed call."""
    n = 0
    for step in trace or []:
        tc = step.tool_call
        if tc is None:
            continue
        n += 1
        ref = _format_tool_ref(tc.name, tc.args)
        quote = (tc.result or "").strip()
        if len(quote) > _QUOTE_CAP:
            quote = quote[:_QUOTE_CAP] + "…"
        yield EvidenceItem(
            id=f"ev_tool_{n:03d}",
            source_type="tool",
            source_ref=ref,
            quote=quote,
            confidence=0.0 if tc.is_error else 1.0,
        )


def _format_tool_ref(name: str, args: dict) -> str:
    """Compact textual ref so a consumer can ground a claim:
       `read_file:backend/llm.py:1026-1180`
       `locate_symbol:Agent.run@backend/agent.py`
       `calc:2+2`
    Keeps the most identifying arg in front; falls back to bare name.
    """
    args = args or {}
    if name in {"read_file", "view_file"}:
        path = str(args.get("path", "")).strip()
        s = args.get("start_line")
        e = args.get("end_line")
        if path and s and e:
            return f"{name}:{path}:{s}-{e}"
        if path:
            return f"{name}:{path}"
    if name == "locate_symbol":
        sym = str(args.get("name", "")).strip()
        path = str(args.get("path", "")).strip()
        if sym and path:
            return f"{name}:{sym}@{path}"
    if name == "calc":
        expr = str(args.get("expression") or args.get("expr") or "").strip()[:60]
        if expr:
            return f"{name}:{expr}"
    if name in {"web_search", "fetch_url"}:
        target = str(args.get("query") or args.get("url") or "").strip()[:80]
        if target:
            return f"{name}:{target}"
    return name


def _claims_from_verification(v: VerificationResult) -> Iterable[Claim]:
    """Map the verifier's three buckets into Claim objects with
    a stable id and risk label.

    Lists are zipped with their indices to make ids deterministic
    across re-renders — the same answer should produce the same
    claim ids so a UI can highlight diffs across re-runs.
    """
    n = 0
    for status, items in (
        ("verified", v.verified_claims),
        ("unverified", v.unverified_claims),
        ("contradicted", v.contradictions),
    ):
        for text in items or []:
            text = (text or "").strip()
            if not text:
                continue
            n += 1
            yield Claim(
                id=f"c_{n:03d}",
                text=text[:_QUOTE_CAP],
                status=status,
                evidence_ids=[],  # Phase B will bind these
                risk=_STATUS_RISK[status],
            )
