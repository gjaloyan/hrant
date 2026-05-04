"""Self-verification of agent answers against loaded notes and tool outputs."""
from __future__ import annotations
import json
import re

from .llm import TaskType, router
from .models import VerificationResult


# Pulls names of classes / functions / module-level vars / dotted attrs
# out of a Python-ish tool output. We're not parsing the AST — that
# would be overkill for fuzzy verification — just enough patterns that
# the verifier LLM has an explicit "things that already exist" list.
_IDENT_PATTERNS = (
    re.compile(r"\bclass\s+([A-Za-z_]\w*)"),
    re.compile(r"\bdef\s+([A-Za-z_]\w*)"),
    # MODULE_CONST = ... — module-level uppercase constants. Allows
    # leading whitespace because file dumps from read_file may include
    # indentation or `cat -n` line-number prefixes.
    re.compile(r"^\s*([A-Z][A-Z0-9_]+)\s*=", re.MULTILINE),
    re.compile(r"\bself\.([A-Za-z_]\w*)\s*="),
)


def _extract_code_identifiers(tool_context: str, *, max_idents: int = 200) -> list[str]:
    """Pull class / function / attr names from tool output.

    Returned sorted, deduplicated, capped at `max_idents` so the
    verifier prompt doesn't balloon when a 12k-char file dump goes in.
    Empty string in → [] out, never raises.
    """
    if not tool_context:
        return []
    found: set[str] = set()
    for pattern in _IDENT_PATTERNS:
        for m in pattern.finditer(tool_context):
            for grp in m.groups():
                if grp and len(grp) >= 2:
                    found.add(grp)
                    if len(found) >= max_idents:
                        break
            if len(found) >= max_idents:
                break
        if len(found) >= max_idents:
            break
    return sorted(found)

VERIFIER_SYSTEM = """You are a strict fact-checker for an AI assistant's
answers. Your job is to surface hallucinations the assistant may have
introduced — including false claims about what the source code, notes,
or tool outputs DO or DO NOT contain.

You are given: (1) a user question, (2) an assistant's answer, (3) source
notes, and optionally (4) tool outputs (file contents the assistant read,
web search results, etc.).

Classify EVERY substantive claim in the answer into one of three buckets:

  verified       — claim is directly supported by a note OR a tool output.
                   For positive claims, the supporting text must actually
                   contain or imply the claim — not just share keywords.
  unverified     — claim is not confirmed and not contradicted by any
                   source you have. Default for absence-of-evidence cases.
  contradiction  — claim contradicts a note or a tool output. Use this
                   aggressively. A claim that something is "missing",
                   "absent", "not handled", or "needs to be added" is a
                   CONTRADICTION whenever the tool output shows that
                   thing IS already there.

IMPORTANT — negative existence claims ("X is missing", "code doesn't
handle Y", "no validation for Z", or proposed "fixes" that add what the
code allegedly lacks):

  Step 1. Identify what the assistant says is absent.
  Step 2. Search the tool output for the exact identifier, the related
          function name, the matching pattern. Don't rely on keyword
          overlap alone — read the lines.
  Step 3. If you find evidence the thing IS already present in the file,
          mark this claim as a CONTRADICTION. This is the most common
          source of code-review hallucinations: the assistant proposes
          adding code that's already there because it forgot a previous
          fix or didn't read carefully enough.
  Step 4. If you can't find it but the tool output covers the relevant
          file/section, mark UNVERIFIED — absence-of-evidence is not
          evidence-of-absence, and you should not promote a "missing"
          claim to verified just because you also didn't see it.
  Step 5. If the tool output didn't cover the relevant area at all,
          mark UNVERIFIED — the assistant may be right or wrong; you
          have no basis to confirm either way.

For "fix" suggestions specifically: when the assistant proposes a code
change "to add X", check whether X already exists in the tool output.
If yes → that's a contradiction with the implicit claim "X is missing".

IMPORTANT: Tool outputs (file contents, search results) are PRIMARY
evidence. Notes are SECONDARY. When they conflict, tool output wins.

Return strictly JSON with the three claim lists and which topics you used.
DO NOT return a confidence number — the caller computes it deterministically
from verified / unverified / contradiction counts.

{
  "verified_claims":   ["..."],
  "unverified_claims": ["..."],
  "contradictions":    ["..."],
  "notes_used":        ["topic1", "topic2"]
}"""


def _compute_confidence(verified: int, unverified: int, contradictions: int) -> int:
    """Deterministic confidence from claim counts.

    Pulled out of the LLM prompt (where it lived as a formula the model
    was supposed to evaluate but routinely got wrong) into Python so the
    same claim split always yields the same score. Formula matches what
    used to be in the prompt: contradictions are weighted 2× because
    they're worse than no evidence — they're evidence against.

        confidence = 100 * verified / (verified + unverified + 2*contradictions)

    Edge: zero claims of any kind → 0 (the verifier saw nothing).
    """
    denom = verified + unverified + 2 * contradictions
    if denom <= 0:
        return 0
    return max(0, min(100, round(100.0 * verified / denom)))


def verify(
    question: str,
    answer: str,
    notes_text: str,
    used_topics: list[str],
    tool_context: str = "",
) -> VerificationResult:
    if not notes_text.strip() and not tool_context.strip():
        return VerificationResult(
            confidence=0,
            unverified_claims=["no loaded notes or tool outputs — all claims unverified"],
            notes_used=[],
        )

    tool_section = ""
    if tool_context.strip():
        # Pre-extract identifiers so the LLM has an explicit "already
        # in code" keyword set. Catches the common review hallucination
        # "fix: add reject category" when reject is right there at line
        # 248 — without this list the model has to spot the line in a
        # 12k-char dump, which it routinely fails to do.
        idents = _extract_code_identifiers(tool_context)
        idents_section = ""
        if idents:
            idents_section = (
                "\n\nEXTRACTED IDENTIFIERS — ALREADY PRESENT IN THE CODE\n"
                "(if the assistant proposes adding any of these as a 'fix', "
                "that is a CONTRADICTION):\n"
                + ", ".join(idents)
            )
        tool_section = (
            f"\n\nTOOL OUTPUTS (file contents, search results — primary evidence):\n"
            f"{tool_context}{idents_section}"
        )

    user = f"""QUESTION:
{question}

ASSISTANT'S ANSWER:
{answer}

SOURCE NOTES:
{notes_text}{tool_section}

Available topics: {', '.join(used_topics)}"""
    try:
        data = router().call_json(
            TaskType.VERIFICATION,
            VERIFIER_SYSTEM, user, max_tokens=1500, temperature=0.0,
        )
    except Exception as e:
        return VerificationResult(
            confidence=50,
            unverified_claims=[f"verifier error: {e}"],
            notes_used=used_topics,
        )
    verified = list(data.get("verified_claims", []))
    unverified = list(data.get("unverified_claims", []))
    contradictions = list(data.get("contradictions", []))
    return VerificationResult(
        confidence=_compute_confidence(
            len(verified), len(unverified), len(contradictions)
        ),
        verified_claims=verified,
        unverified_claims=unverified,
        contradictions=contradictions,
        notes_used=list(data.get("notes_used", used_topics)),
    )
