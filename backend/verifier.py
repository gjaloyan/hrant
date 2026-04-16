"""Self-verification of agent answers against loaded notes and tool outputs."""
from __future__ import annotations
import json

from .llm import TaskType, router
from .models import VerificationResult

VERIFIER_SYSTEM = """You are a strict fact-checker.
You are given: (1) a user question, (2) an assistant's answer, (3) source notes,
and optionally (4) tool outputs (e.g. file contents the assistant read).

Your task: check EVERY claim in the answer against the available evidence.

Rules:
- verified — claim is directly supported by a note OR by a tool output;
- unverified — claim is not confirmed and not contradicted by any source;
- contradiction — claim contradicts a note or tool output.

IMPORTANT: Tool outputs (file contents, search results) are PRIMARY evidence.
If the assistant read a file via read_file and makes claims about its contents,
those claims CAN be verified against the tool output.

Return strictly JSON:
{
  "confidence": 0-100,
  "verified_claims": ["..."],
  "unverified_claims": ["..."],
  "contradictions": ["..."],
  "notes_used": ["topic1", "topic2"]
}
confidence = 100 * verified / (verified + unverified + 2*contradictions)."""


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
        tool_section = f"\n\nTOOL OUTPUTS (file contents, search results — primary evidence):\n{tool_context}"

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
    return VerificationResult(
        confidence=int(data.get("confidence", 0)),
        verified_claims=list(data.get("verified_claims", [])),
        unverified_claims=list(data.get("unverified_claims", [])),
        contradictions=list(data.get("contradictions", [])),
        notes_used=list(data.get("notes_used", used_topics)),
    )
