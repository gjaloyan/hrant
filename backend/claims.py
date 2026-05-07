"""Claim/Evidence layer.

Phase A (`build_claims_and_evidence`): assemble Claim + EvidenceItem
from data the system already produces (verifier buckets + thinking
trace tool calls + user message). No solver cooperation required.

Phase B (`SOLVER_CLAIMS_DIRECTIVE`, `extract_solver_claims_block`):
the solver appends a `---CLAIMS---` JSON tail to its answer, naming
which tool calls support each claim it makes. We parse the tail,
strip it from the visible answer, and feed the result to the
builder so each Claim gets `evidence_ids` bound to specific tool
EvidenceItems. If the LLM ignores the directive (older model,
mid-stream truncation, JSON syntax error), we silently fall back
to Phase A — no broken answers ever reach the user.

Anything in here is pure data transformation — no LLM calls, no
tool calls. That makes it cheap to run on every turn and easy to
test.
"""
from __future__ import annotations

import json
import re
from typing import Iterable, Optional

from .models import (
    Claim,
    EvidenceItem,
    ThinkingStep,
    VerificationResult,
)


# The block solver appends to its answer. Strict marker so we don't
# false-positive on prose that happens to contain JSON. The model is
# allowed to omit the block entirely — Phase A path takes over.
SOLVER_CLAIMS_MARKER = "---CLAIMS---"

# What we tell the solver to do. Designed to be cheap to ignore (one
# extra paragraph of instructions; if the model produces no tail, we
# fall back to Phase A behaviour) and cheap to parse when it does.
SOLVER_CLAIMS_DIRECTIVE = (
    f"\n\n## CLAIMS BLOCK (mandatory)\n"
    f"After your prose answer, on its own line, output exactly:\n\n"
    f"    {SOLVER_CLAIMS_MARKER}\n\n"
    f"followed by a single compact JSON object listing the key factual "
    f"claims your answer makes and which tool calls (if any) support each:\n\n"
    f'    {{"claims":[\n'
    f'      {{"text":"<short claim, ≤200 chars>","evidence":["tool_1","tool_3"]}},\n'
    f'      {{"text":"...","evidence":[]}}\n'
    f"    ]}}\n\n"
    f"Rules:\n"
    f"- Reference tool calls by their 1-based index in the order you "
    f"called them this turn (`tool_1`, `tool_2`, …). The first tool "
    f"you call is `tool_1`.\n"
    f"- A claim with no tool grounding gets `\"evidence\":[]` — that "
    f"signals to the verifier it's an inference, not a quoted fact.\n"
    f"- Keep claims atomic: one assertion per entry, ≤200 chars.\n"
    f"- 3–8 claims is typical; up to 15 if the answer is long.\n"
    f"- This block is consumed programmatically — no prose around the "
    f"JSON, no markdown fences, no trailing commas. The block goes AT "
    f"THE END of your answer, after everything else.\n"
    f"- If you can't produce valid JSON, omit the block — the verifier "
    f"will fall back to extracting claims from your prose."
)


# Regex anchored on a line of its own so a marker accidentally inside
# a code block doesn't fire. The JSON body is captured greedily up to
# end-of-string — we trust the marker boundary since the directive
# tells the model to put nothing after the JSON.
_CLAIMS_TAIL_RE = re.compile(
    rf"\n*{re.escape(SOLVER_CLAIMS_MARKER)}\s*\n+(\{{.*?\}})\s*$",
    re.DOTALL,
)


def extract_solver_claims_block(
    answer: str,
) -> tuple[str, Optional[list[dict]]]:
    """Pull the trailing claims JSON out of a solver answer.

    Returns `(stripped_answer, parsed_claims_or_None)`. The stripped
    answer is what the user sees — never includes the marker or JSON.
    Parsed claims is a list of `{"text": ..., "evidence": [...]}` or
    None when the block is missing/malformed. Either way the answer
    text comes back cleanly stripped so the user never sees the raw
    JSON tail.
    """
    if not answer or SOLVER_CLAIMS_MARKER not in answer:
        return answer or "", None

    m = _CLAIMS_TAIL_RE.search(answer)
    if m is None:
        # Marker present but no parseable JSON tail — strip everything
        # from the marker onward so the user doesn't see a half-block.
        idx = answer.rfind("\n" + SOLVER_CLAIMS_MARKER)
        if idx == -1:
            idx = answer.rfind(SOLVER_CLAIMS_MARKER)
        cleaned = answer[:idx].rstrip() if idx >= 0 else answer
        return cleaned, None

    json_blob = m.group(1)
    cleaned = answer[: m.start()].rstrip()
    try:
        parsed = json.loads(json_blob)
    except json.JSONDecodeError:
        return cleaned, None

    raw = parsed.get("claims") if isinstance(parsed, dict) else None
    if not isinstance(raw, list):
        return cleaned, None

    out: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        ev_raw = item.get("evidence") or []
        ev = [str(e).strip() for e in ev_raw if isinstance(e, (str, int))]
        out.append({"text": text, "evidence": ev})
    return cleaned, out


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
    solver_claims: Optional[list[dict]] = None,
) -> tuple[list[Claim], list[EvidenceItem]]:
    """Build a Claim list + Evidence list from a turn's data.

    `verification` carries the buckets the verifier produces
    (`verified_claims`, `unverified_claims`, `contradictions`).
    `thinking_trace` carries tool calls. `user_message` is added as
    one "user" evidence item.

    `solver_claims` (Phase B): when the solver emits a structured
    `---CLAIMS---` tail, pass the parsed list here. We then build
    Claim objects from solver text (preserving the claim wording
    the LLM committed to) and bind each claim's `evidence_ids` to
    the matching tool EvidenceItems via the `tool_N` references.
    Status comes from the verifier — solver doesn't get to mark its
    own claim "verified". When solver_claims is None or empty, falls
    back to Phase A behaviour: claims built from the verifier
    buckets, no evidence binding.
    """
    evidence = list(_evidence_from_trace(thinking_trace))
    # Build a tool_N → evidence_id map up-front. Indices match the
    # order the model called tools, which is also the order they
    # appear in the trace, which is also the order
    # `_evidence_from_trace` walks. So `tool_1` → `evidence[0].id`.
    tool_evs = [e for e in evidence if e.source_type == "tool"]
    tool_ref_to_ev_id = {f"tool_{i + 1}": tool_evs[i].id for i in range(len(tool_evs))}

    if user_message and user_message.strip():
        evidence.append(EvidenceItem(
            id=f"ev_user_{len(evidence):03d}",
            source_type="user",
            source_ref="user_turn",
            quote=user_message.strip()[:_QUOTE_CAP],
            confidence=1.0,
        ))

    if solver_claims:
        claims = list(_claims_from_solver(solver_claims, verification, tool_ref_to_ev_id))
    else:
        claims = list(_claims_from_verification(verification))
    return claims, evidence


def _claims_from_solver(
    solver_claims: list[dict],
    verification: VerificationResult,
    tool_ref_to_ev_id: dict[str, str],
) -> Iterable[Claim]:
    """Build Claim objects from solver-emitted entries.

    Status is decided by the verifier, NOT the solver — we look up
    each solver claim in the verifier's three buckets by normalised
    text match, falling back to "unverified" when no match. This
    preserves the verifier's authority over what's actually grounded
    while keeping the solver's claim wording (which usually matches
    the answer prose exactly).
    """
    bucket_index = _build_verification_index(verification)
    n = 0
    for item in solver_claims:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        n += 1
        ev_refs = item.get("evidence") or []
        evidence_ids = [
            tool_ref_to_ev_id[ref]
            for ref in ev_refs
            if ref in tool_ref_to_ev_id
        ]
        status = _lookup_status(text, bucket_index)
        risk = _STATUS_RISK[status]
        # Evidence-aware risk bump: a solver claim with no evidence and
        # an unverified status is more suspicious than one with at least
        # one tool-grounded evidence id, even if both are "unverified"
        # textually. Verifier will refine this in Phase C.
        if status == "unverified" and not evidence_ids:
            risk = "high"
        yield Claim(
            id=f"c_{n:03d}",
            text=text[:_QUOTE_CAP],
            status=status,
            evidence_ids=evidence_ids,
            risk=risk,
        )


def _normalise_for_match(s: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace — used to
    match a solver's claim string against a verifier bucket entry
    even when the wordings drift slightly."""
    s = s.lower()
    s = re.sub(r"[^\w\s]+", " ", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _build_verification_index(v: VerificationResult) -> dict[str, str]:
    """Map normalised text → status for fast lookup. Last writer wins
    in the (unlikely) case the same string appears in multiple
    buckets — the precedence order matches verifier severity:
    contradicted > unverified > verified."""
    idx: dict[str, str] = {}
    for status, items in (
        ("verified", v.verified_claims),
        ("unverified", v.unverified_claims),
        ("contradicted", v.contradictions),
    ):
        for item in items or []:
            n = _normalise_for_match(item)
            if n:
                idx[n] = status
    return idx


def _lookup_status(claim_text: str, bucket_index: dict[str, str]) -> str:
    """Best-effort: exact normalised match first, then a substring
    pass either direction. Falls back to 'unverified' so an
    unmatched solver claim is treated cautiously."""
    n = _normalise_for_match(claim_text)
    if not n:
        return "unverified"
    if n in bucket_index:
        return bucket_index[n]
    for k, status in bucket_index.items():
        if k in n or n in k:
            return status
    return "unverified"


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
