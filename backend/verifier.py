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


# Patterns the agent uses when proposing to "add" something that's
# allegedly missing. Each pattern has the candidate identifier in
# group 1. We anchor on lowercase verbs / negation phrasings in EN +
# RU; the trailing identifier is matched permissively (Python-shape).
_FALSE_ABSENCE_PATTERNS = (
    # add/create X (verb-first, identifier follows)
    re.compile(r"\b(?:add|added|adding|create|creating)\s+(?:a\s+|an\s+|the\s+)?[`']?([A-Za-z_]\w{2,})", re.IGNORECASE),
    # missing/not-implemented X (verb-first, identifier follows)
    re.compile(r"\b(?:missing|absent|not\s+implemented|not\s+called|not\s+invoked|not\s+handled|not\s+used)\s*[:`']?\s*([A-Za-z_]\w{2,})", re.IGNORECASE),
    # X is missing / X doesn't exist (identifier-first)
    re.compile(r"[`']?(?:[\w.]*\.)?([A-Za-z_]\w{2,})[`']?\s+(?:is\s+missing|doesn['’]?t\s+exist|isn['’]?t\s+(?:there|defined|implemented|called)|not\s+called|not\s+invoked|is\s+not\s+called)", re.IGNORECASE),
    re.compile(r"\b(?:there['’]?s\s+no|no\s+such)\s+[`']?([A-Za-z_]\w{2,})", re.IGNORECASE),
    re.compile(r"\b(?:fix\s*:?\s*add|recommend(?:ation)?\s*:?\s*add|TODO\s*:?\s*add)\s+[`']?([A-Za-z_]\w{2,})", re.IGNORECASE),
    # Russian: добавь/создай X (verb-first)
    re.compile(r"\b(?:добав(?:ь|ить|им|лен)|создать?|реализовать)\s+[`']?([A-Za-z_]\w{2,})", re.IGNORECASE),
    # Russian: identifier-first — `X не вызывается`, `Foo.bar не реализован`,
    # `mod.func не существует`. Allows a dotted prefix; group 1 captures
    # the trailing identifier (the part that's in EXTRACTED IDENTIFIERS).
    re.compile(r"[`']?(?:[\w.]*\.)?([A-Za-z_]\w{2,})[`']?\s+не\s+(?:вызыва(?:ется|ют)|реализован|обрабатыва(?:ется|ют)|существует|использует(?:ся|ют))", re.IGNORECASE),
)


def detect_false_absence_contradictions(
    answer: str,
    identifiers: list[str],
) -> list[str]:
    """Deterministic catch for the 'agent suggests adding X when X is
    already in the code' hallucination. Scans the answer for common
    "add/missing X" phrasings, checks each candidate against the set
    of identifiers extracted from tool output, and returns one
    contradiction string per match.

    LLM-side verification (the prompt rules in VERIFIER_SYSTEM) is
    still the primary path, but Sonnet has been observed missing these
    even with explicit guidance. This is the belt to that suspenders.
    Empty list when there's no overlap, so quiet on healthy answers.
    """
    if not answer or not identifiers:
        return []

    def _norm(s: str) -> str:
        """Canonical form for cross-matching identifiers in answer text
        against identifiers extracted from source. Both sides may use
        different conventions for the SAME concept:
          - `FILE_CACHE` (module const) vs `_file_cache` (private dict)
          - `TokenTracker` (class) vs `_token_tracker` (instance var)
          - `web_search` (snake) vs `WebSearch` (Pascal) vs `_webSearch`
        Lowercase + strip ALL underscores collapses snake_case,
        SCREAMING_CASE, _private, and CamelCase to one bucket, so
        "add `_token_tracker`" matches `TokenTracker` in extracted.
        """
        return s.replace("_", "").lower()

    def _looks_like_code_identifier(s: str) -> bool:
        """A candidate from the answer must look code-shaped before
        we treat it as an identifier reference. Otherwise plain
        English words ("tool", "view", "logic") collide with class
        names like `Tool`, `View`, `Logic` after case-normalization
        and produce a false-positive contradiction. A real identifier
        usually has at least one of:
          - an underscore (snake / SCREAMING_CASE / _private)
          - an internal uppercase (PascalCase / camelCase, NOT just
            a capitalized first word in prose like "Tool")
          - a digit
        Plus a length floor — a 3-letter lowercase word is too
        likely to be prose."""
        if not s or len(s) < 4:
            return False
        if "_" in s:
            return True
        if any(ch.isdigit() for ch in s):
            return True
        # Internal uppercase: a capital letter at any position other
        # than the first. `Tool` (just sentence-cased) -> False.
        # `TokenTracker`, `webSearch` -> True.
        for i, ch in enumerate(s):
            if i > 0 and ch.isupper():
                return True
        return False

    norm_to_original: dict[str, str] = {}
    for ident in identifiers:
        if ident and _looks_like_code_identifier(ident):
            norm_to_original.setdefault(_norm(ident), ident)
    seen: set[str] = set()
    out: list[str] = []
    for pattern in _FALSE_ABSENCE_PATTERNS:
        for m in pattern.finditer(answer):
            candidate = m.group(1)
            if not _looks_like_code_identifier(candidate):
                continue
            key = _norm(candidate)
            if key in norm_to_original and key not in seen:
                seen.add(key)
                source_form = norm_to_original[key]
                # Snippet of surrounding context for the verifier UI.
                start = max(0, m.start() - 40)
                end = min(len(answer), m.end() + 40)
                snippet = answer[start:end].replace("\n", " ").strip()
                # Show both names if they differ — clarifies that the
                # match was case/underscore-normalized.
                ref = (
                    f"'{candidate}' (matches '{source_form}')"
                    if candidate != source_form
                    else f"'{candidate}'"
                )
                out.append(
                    f"Answer claims {ref} is missing or proposes adding it, "
                    f"but it is already present in the code per tool output. "
                    f"Context: …{snippet}…"
                )
    return out


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


# Verifier prompt compression
# ---------------------------
# Round 5 made the verifier see all retry + subtask evidence — useful
# for catching cross-attempt contradictions, but the prompt grew to
# 60-80 KB. Most of that body is code the verifier never quotes back;
# only snippets around identifiers the assistant actually MENTIONED
# in its answer matter for fact-checking.
#
# Strategy:
#   1. Pull mentioned identifiers from the answer itself (same regex
#      as `_extract_code_identifiers` but applied to the answer side
#      so we know what the verifier needs to fact-check).
#   2. For each mention, keep ±5 lines of context from the tool
#      output where that identifier appears.
#   3. If the compressed body is still bigger than `keep_full_under`,
#      stop — return the original body untruncated. Compression isn't
#      worth it for short outputs.
#   4. If nothing in the answer maps to the tool output (rare),
#      keep the original — verifier needs SOMETHING to compare.

_ANSWER_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{3,})\b")
_CONTEXT_LINES_AROUND = 5
_FULL_KEEP_UNDER = 8000


def _compress_tool_context(
    answer: str,
    tool_context: str,
    *,
    max_chars: int = 12000,
) -> str:
    """Replace raw tool_context with snippets around identifiers the
    assistant's answer actually mentions. Falls back to the original
    string when:
      - tool_context is short (compression overhead not worth it),
      - no answer-side identifiers overlap with tool_context,
      - the compressed result somehow ends up larger than the source
        (pathological inputs).
    """
    if not tool_context or len(tool_context) < _FULL_KEEP_UNDER:
        return tool_context
    answer_idents = {
        m.group(1) for m in _ANSWER_IDENT_RE.finditer(answer or "")
        if len(m.group(1)) >= 4
    }
    if not answer_idents:
        return tool_context[:max_chars]

    lines = tool_context.splitlines()
    keep_idx: set[int] = set()
    found_any = False
    for i, line in enumerate(lines):
        for ident in answer_idents:
            if ident in line:
                found_any = True
                lo = max(0, i - _CONTEXT_LINES_AROUND)
                hi = min(len(lines), i + _CONTEXT_LINES_AROUND + 1)
                keep_idx.update(range(lo, hi))
                break

    if not found_any:
        # The answer cites nothing the tool output covers — verifier
        # still needs evidence to check the rest, so keep a head slice.
        return tool_context[:max_chars]

    out_parts: list[str] = []
    prev_i = -2
    for i in sorted(keep_idx):
        if i != prev_i + 1:
            out_parts.append("…")
        out_parts.append(lines[i])
        prev_i = i
    compressed = "\n".join(out_parts)
    # Don't compress UP — if our snippets ended up bigger than the
    # source (only possible on weird short inputs), use the source.
    if len(compressed) >= len(tool_context):
        return tool_context[:max_chars]
    return compressed[:max_chars]


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
    on_llm_call=None,
) -> VerificationResult:
    """Verify the answer against notes + tool output.

    `on_llm_call(system, user, response, duration_ms)` is an optional
    callback fired after the verifier's LLM call returns. Used by the
    agent's dev-mode capture to record the verifier's actual prompts
    without re-constructing them. Best-effort — exceptions swallowed.
    """
    if not notes_text.strip() and not tool_context.strip():
        return VerificationResult(
            confidence=0,
            unverified_claims=["no loaded notes or tool outputs — all claims unverified"],
            notes_used=[],
        )

    tool_section = ""
    extracted_idents: list[str] = []
    if tool_context.strip():
        # Pre-extract identifiers so the LLM has an explicit "already
        # in code" keyword set. Catches the common review hallucination
        # "fix: add reject category" when reject is right there at line
        # 248 — without this list the model has to spot the line in a
        # 12k-char dump, which it routinely fails to do.
        extracted_idents = _extract_code_identifiers(tool_context)
        idents_section = ""
        if extracted_idents:
            idents_section = (
                "\n\nEXTRACTED IDENTIFIERS — ALREADY PRESENT IN THE CODE\n"
                "(if the assistant proposes adding any of these as a 'fix', "
                "that is a CONTRADICTION):\n"
                + ", ".join(extracted_idents)
            )
        # Compress the raw tool_context: a 60-80k accumulated buffer
        # (post Round 5 retry/subtask accumulation) is mostly source
        # code the verifier never quotes back. Keep snippets around
        # the identifiers the answer ACTUALLY mentions; everything
        # else collapses to a one-line summary. The full body is only
        # kept when nothing in the answer maps to the tool output at
        # all (the rare "answer cites nothing concrete" case).
        compressed = _compress_tool_context(answer, tool_context)
        tool_section = (
            f"\n\nTOOL OUTPUTS (file contents, search results — primary evidence):\n"
            f"{compressed}{idents_section}"
        )

    user = f"""QUESTION:
{question}

ASSISTANT'S ANSWER:
{answer}

SOURCE NOTES:
{notes_text}{tool_section}

Available topics: {', '.join(used_topics)}"""
    try:
        import time as _t
        _t0 = _t.monotonic()
        data = router().call_json(
            TaskType.VERIFICATION,
            VERIFIER_SYSTEM, user, max_tokens=1500, temperature=0.0,
        )
        if on_llm_call is not None:
            try:
                on_llm_call(
                    VERIFIER_SYSTEM,
                    user,
                    str(data),
                    int((_t.monotonic() - _t0) * 1000),
                )
            except Exception:
                pass
    except Exception as e:
        return VerificationResult(
            confidence=50,
            unverified_claims=[f"verifier error: {e}"],
            notes_used=used_topics,
        )
    verified = list(data.get("verified_claims", []))
    unverified = list(data.get("unverified_claims", []))
    contradictions = list(data.get("contradictions", []))

    # Belt-and-suspenders: deterministic detector for the
    # "agent claims X is missing, but X is in the code" pattern.
    # The LLM verifier was observed missing these even with the
    # negative-existence rule + EXTRACTED IDENTIFIERS list spelled
    # out for it (Sonnet, real session). Anything caught here is
    # promoted to a contradiction; if the LLM also caught it, we
    # dedup so the same issue doesn't double-count toward confidence.
    auto_contradictions = detect_false_absence_contradictions(answer, extracted_idents)
    if auto_contradictions:
        existing_lower = {c.lower() for c in contradictions}
        for c in auto_contradictions:
            # Cheap dedup: if the auto detector and the LLM both flagged
            # the same identifier, the LLM's natural-language version
            # already covers it. Match on the quoted identifier word.
            ident_token = c.split("'")[1] if "'" in c else ""
            if ident_token and any(
                ident_token.lower() in c_existing.lower() for c_existing in contradictions
            ):
                continue
            if c.lower() not in existing_lower:
                contradictions.append(c)
                existing_lower.add(c.lower())
    return VerificationResult(
        confidence=_compute_confidence(
            len(verified), len(unverified), len(contradictions)
        ),
        verified_claims=verified,
        unverified_claims=unverified,
        contradictions=contradictions,
        notes_used=list(data.get("notes_used", used_topics)),
    )
