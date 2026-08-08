"""Self-verification of agent answers against loaded notes and tool outputs."""
from __future__ import annotations
import re

from .llm import TaskType, router
from .models import VerificationResult


# Per-claim section caps. Tool result quotes are already truncated
# upstream (Round 8: 4k preview), but we re-clip here to keep the
# verifier prompt bounded even when the LLM cites multiple long
# tool results. 1500 per evidence quote is enough to ground a
# specific claim and short enough that 8-claim turns stay under 12k
# total in the structured section.
_VERIFIER_QUOTE_CAP = 1500
_VERIFIER_MAX_CLAIMS_RENDERED = 15


def _format_structured_claims_section(
    solver_claims: list[dict],
    tool_call_order: list[dict],
) -> str:
    """Render the SOLVER CLAIMS block for the verifier prompt.

    `solver_claims` is the parsed list from the solver's
    `---CLAIMS---` tail: `[{"text": str, "evidence": ["tool_1", ...]}, ...]`.
    `tool_call_order` is `[{"name": str, "args": dict, "result": str}, ...]`
    in the order the tools were called this turn — `tool_1` maps to
    index 0, `tool_2` to index 1, and so on.

    Output is a markdown-ish block the verifier reads alongside the
    answer. For each claim we name the cited tools and quote what
    they returned, so the LLM rules per-claim against the evidence
    the solver itself committed to.
    """
    if not solver_claims:
        return ""
    lines: list[str] = [
        "",
        "",
        "STRUCTURED CLAIMS (the assistant grouped its answer into atomic claims, each tagged with the tool calls it relied on — rule each one independently against the cited evidence):",
    ]
    for idx, item in enumerate(solver_claims[:_VERIFIER_MAX_CLAIMS_RENDERED], 1):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        ev_refs = item.get("evidence") or []
        lines.append(f"\nCLAIM {idx}: {text}")
        if not ev_refs:
            lines.append("  Evidence: (none — solver marks this as inference, not a quoted fact)")
            continue
        for ref in ev_refs:
            tool = _resolve_tool_ref(ref, tool_call_order)
            if tool is None:
                lines.append(f"  Evidence {ref}: (UNRESOLVED — solver cited this ref but no such tool call exists this turn)")
                continue
            label = _format_tool_label(tool)
            quote = (tool.get("result") or "").strip()
            if len(quote) > _VERIFIER_QUOTE_CAP:
                quote = quote[:_VERIFIER_QUOTE_CAP] + "…"
            err_marker = " [TOOL ERROR]" if tool.get("is_error") else ""
            lines.append(f"  Evidence {ref} = {label}{err_marker}:")
            lines.append(f"    {quote}")
    if len(solver_claims) > _VERIFIER_MAX_CLAIMS_RENDERED:
        skipped = len(solver_claims) - _VERIFIER_MAX_CLAIMS_RENDERED
        lines.append(f"\n[+{skipped} more claims omitted — render cap is {_VERIFIER_MAX_CLAIMS_RENDERED}]")

    lines.append(
        "\nWHEN A STRUCTURED CLAIMS SECTION IS PRESENT, your verified_claims / "
        "unverified_claims / contradictions arrays MUST quote the CLAIM TEXT "
        "exactly (so the agent can map your ruling back). Decide each:\n"
        "- VERIFIED: the cited evidence supports the claim AND the claim is "
        "consistent with notes/tool outputs.\n"
        "- UNVERIFIED: the claim has no evidence (solver marked it inference) "
        "OR the cited evidence is insufficient / off-topic.\n"
        "- CONTRADICTION: cited evidence directly contradicts the claim, OR "
        "the claim asserts the absence of a feature whose name is in "
        "EXTRACTED IDENTIFIERS, OR cited evidence shows tool error.\n"
    )
    return "\n".join(lines)


def _resolve_tool_ref(ref: str, tool_call_order: list[dict]) -> dict | None:
    """`tool_3` → tool_call_order[2]. Returns None for malformed refs
    or out-of-range indices."""
    if not ref or not ref.startswith("tool_"):
        return None
    try:
        idx = int(ref.split("_", 1)[1]) - 1
    except (ValueError, IndexError):
        return None
    if idx < 0 or idx >= len(tool_call_order):
        return None
    return tool_call_order[idx]


def _format_tool_label(tool: dict) -> str:
    """Compact label like `read_file(backend/llm.py:1026-1180)` so the
    verifier's prompt shows what was called, not just the index."""
    name = tool.get("name", "?")
    args = tool.get("args") or {}
    if name in {"read_file", "view_file"}:
        path = str(args.get("path", ""))
        s = args.get("start_line")
        e = args.get("end_line")
        if path and s and e:
            return f"{name}({path}:{s}-{e})"
        if path:
            return f"{name}({path})"
    if name == "locate_symbol":
        sym = str(args.get("name", ""))
        path = str(args.get("path", ""))
        if sym and path:
            return f"{name}({sym}@{path})"
    if name == "calc":
        expr = str(args.get("expression") or args.get("expr") or "")[:60]
        return f"{name}({expr})"
    if name in {"web_search", "fetch_url"}:
        target = str(args.get("query") or args.get("url") or "")[:80]
        return f"{name}({target})"
    return name


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
# Audit 2026-06-10 (I10): tightened caps. Was 8000 / 12000 — at those
# values a fact-check on a heavy code-read turn could put 12 KB into the
# verifier's classifier prompt, which is dominated by signal in the FIRST
# 6 KB (the relevant snippets around mentioned identifiers). Lower
# bounds: compress starting at 5 KB, cap output at 8 KB. Saves ~4 KB /
# verifier call without measurably degrading claim-grounding.
_FULL_KEEP_UNDER = 5000


def _compress_tool_context(
    answer: str,
    tool_context: str,
    *,
    max_chars: int = 8000,
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

Classify EVERY substantive claim in the answer into one of FOUR buckets:

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
  projection     — an EXPLICITLY-HEDGED forecast or conditional outcome:
                   a statement about the FUTURE or a scenario, marked as
                   uncertain ("may", "likely", "could", "~30% chance",
                   "if X then Y", "bullish/base/bearish scenario"). These
                   are legitimate analysis, NOT hallucinations — an analyst
                   SHOULD make hedged projections. Put a claim here ONLY
                   when it is (a) about the future/a conditional, AND
                   (b) clearly hedged, AND (c) NOT presented as certain.
                   A future claim stated as a CERTAINTY ("PEPE WILL hit
                   $1") is NOT a projection — it is unverified or a
                   contradiction. A wrong PRESENT fact is never a
                   projection. Do not let projections excuse fabricated
                   present-tense facts (prices, levels, filings) — those
                   still go to verified / unverified / contradiction.

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
  "projections":       ["..."],
  "notes_used":        ["topic1", "topic2"]
}"""


def _compute_confidence(
    verified: int,
    unverified: int,
    contradictions: int,
    projections: int = 0,
) -> int:
    """Deterministic confidence from claim counts.

    Pulled out of the LLM prompt (where it lived as a formula the model
    was supposed to evaluate but routinely got wrong) into Python so the
    same claim split always yields the same score. Contradictions are
    weighted 2× because they're worse than no evidence.

        confidence = 100 * verified / (verified + unverified + 2*contradictions)

    Forecast calibration (Q2 level 2, 2026-06-15): `projections` —
    explicitly-hedged, premise-grounded forecasts — are EXCLUDED from
    the math. They're legitimate analysis, not hallucinations, so they
    must not drag confidence the way unverified FACTUAL claims do. The
    data that forced this: a detailed PEPE year-analysis scored 75 with
    95 'unverified' claims that were almost all hedged scenarios — and
    a RICHER analysis (after the catalyst-weighting fix) pushed that to
    95 unverified, i.e. better work looked worse. Confidence here now
    measures factual grounding + hallucination-freedom; the forecast's
    inherent uncertainty is communicated by the answer's own hedging
    and the projection cap the caller applies, not by tanking the
    score. A confident fabrication about the future still lands in
    unverified/contradiction (full weight), so this can't be gamed.

    Edge: zero verified/unverified/contradiction → 0 (nothing to grade),
    even if there are projections (a pure-guess answer with no grounded
    facts shouldn't ride high on hedged speculation alone).
    """
    denom = verified + unverified + 2 * contradictions
    if denom <= 0:
        return 0
    return max(0, min(100, round(100.0 * verified / denom)))


# Cap applied when the answer mentions specific named entities that
# can't be located in any source (notes or tool output). Set high
# enough that legitimate but-novel answers (training-data recall)
# don't get punished too hard; set low enough that a Kvadrigalt-style
# fabrication can't ship at 95% confidence. 50 sits at the chat
# critic_threshold so the agent triggers a retry / surfaces a
# low-confidence banner instead of presenting the answer as solid.
UNGROUNDED_CONFIDENCE_CAP = 50

# Forecast cap (Q2 level 2, 2026-06-15): ceiling for projection-
# dominated answers (a year-ahead forecast is never near-certain),
# set above the critic-retry threshold so good analysis still ships.
_PROJECTION_CONFIDENCE_CAP = 85

# Floor applied when the agent read substantial source material AND
# the LLM verifier produced verified_claims with zero contradictions.
# Production audit hit clean self-analysis answers at 67% confidence
# — the verifier-LLM is over-cautious on long technical answers
# because it flags small phrasing claims as 'unverified' against
# the source. Floor at 75 so demonstrably-grounded answers don't
# trip the critic-retry threshold.
SOURCE_GROUNDED_TOOL_CTX_MIN = 20_000
SOURCE_GROUNDED_CONFIDENCE_FLOOR = 75


def _proper_nouns(text: str) -> list[str]:
    """Capitalised tokens that look like proper nouns / domain terms.

    Heuristic: a CapitalizedToken of 4+ characters, not at the very
    start of a sentence (where a leading capital is meaningless). We
    use this for grounding checks — entities that appear in the answer
    must trace back to something in notes / tool output, otherwise
    the answer reads as fabrication."""
    import re as _re
    out: list[str] = []
    seen: set[str] = set()
    # Markdown bold (**Name**) is the agent's preferred emphasis; pull
    # those first so we don't miss them after stripping punctuation.
    # Allow multi-word entities like `**Drift Test**` — store the
    # whole phrase AND also the individual capitalised words so a
    # haystack that only has one of them still grounds.
    for m in _re.finditer(
        r"\*\*([A-Z][A-Za-z][A-Za-z0-9\- ]{1,40})\*\*", text,
    ):
        phrase = m.group(1).strip()
        if phrase.lower() not in seen:
            seen.add(phrase.lower())
            out.append(phrase)
        # Also break the phrase on whitespace and add each capitalised
        # word individually — improves grounding for "Drift Test" when
        # only "drift" appears in notes.
        for part in phrase.split():
            if (
                part and part[0].isupper() and len(part) > 2
                and part.lower() not in seen
            ):
                seen.add(part.lower())
                out.append(part)
    # Then bare CapitalizedToken occurrences after some prior char
    # (avoids capturing the first word of a sentence which is just
    # English grammar, not a proper noun).
    for m in _re.finditer(r"(?:[\.\?\!\—\-,;:]\s|\(|\s)([A-Z][A-Za-z][A-Za-z0-9\-]{3,})", text):
        tok = m.group(1)
        if tok.lower() not in seen:
            seen.add(tok.lower())
            out.append(tok)
    # Filter common English starts/connectors that slip past the
    # boundary check — they're capitalized but not proper nouns.
    _STOP = {
        "The", "Then", "This", "These", "Those", "There", "That", "They",
        "And", "But", "For", "From", "However", "Note", "When", "While",
        "What", "Where", "Which", "Why", "Who", "With", "Without", "Step",
        "First", "Second", "Third", "Each", "Both", "Either", "Some",
        "Many", "Most", "All", "Any", "More", "Less", "After", "Before",
        "Because", "Since", "Until", "Just", "Only", "Even", "Also",
    }
    return [t for t in out if t not in _STOP]


def _is_ungrounded(answer: str, notes_text: str, tool_context: str) -> bool:
    """True when the answer's proper-noun-like terms aren't backed by
    any source. Used as a defence against the verifier-LLM accepting
    fabricated content as 'verified'.

    Threshold: an answer is considered grounded when at least HALF
    of its proper nouns appear (case-insensitive substring) in
    notes or tool output. The initial cut at 1/3 was too lenient —
    a follow-up audit caught an answer about an invented
    'Plasmodyne protocol' shipping at 95% confidence because one of
    three proper nouns ('Kvadrigalt') matched a leftover note from
    an earlier turn while the other two ('Plasmodyne',
    'Theodorinka2') were pure fabrication. 50% catches that case.

    Excludes: answers with no proper nouns at all (pure prose; no
    grounding to check) and empty-source answers (separately
    handled by the existing notes/tool empty short-circuit)."""
    nouns = _proper_nouns(answer)
    if not nouns:
        return False
    haystack = (notes_text + "\n" + tool_context).lower()
    if not haystack.strip():
        return True
    grounded = sum(1 for n in nouns if n.lower() in haystack)
    # Half-rounded-up: 1/2, 1/3 (no — too lenient), 2/4, 2/5, 3/6...
    needed = max(1, (len(nouns) + 1) // 2)
    return grounded < needed


def verify(
    question: str,
    answer: str,
    notes_text: str,
    used_topics: list[str],
    tool_context: str = "",
    on_llm_call=None,
    *,
    solver_claims: list[dict] | None = None,
    tool_call_order: list[dict] | None = None,
) -> VerificationResult:
    """Verify the answer against notes + tool output.

    `on_llm_call(system, user, response, duration_ms)` is an optional
    callback fired after the verifier's LLM call returns. Used by the
    agent's dev-mode capture to record the verifier's actual prompts
    without re-constructing them. Best-effort — exceptions swallowed.

    P0 Phase C: when `solver_claims` is provided (the structured
    `[{"text", "evidence": ["tool_N"]}]` list parsed from the
    solver's `---CLAIMS---` tail) AND `tool_call_order` lists the
    tools called this turn (in order, with name+args+result), the
    verifier prompt gains a STRUCTURED CLAIMS section binding each
    claim to its cited evidence. The LLM rules per-claim instead of
    extracting claims from prose. Output schema unchanged — the
    rulings still land in verified_claims / unverified_claims /
    contradictions, just with sharper attribution.
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

    structured_section = ""
    if solver_claims:
        structured_section = _format_structured_claims_section(
            solver_claims, tool_call_order or [],
        )

    user = f"""QUESTION:
{question}

ASSISTANT'S ANSWER:
{answer}{structured_section}

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
    projections = list(data.get("projections", []))

    # False-absence detection ("agent proposes adding X that already
    # exists") is handled by the VERIFIER_SYSTEM prompt itself (the
    # negative-existence rule, Steps 1-5) given the EXTRACTED
    # IDENTIFIERS list in the prompt — no deterministic keyword backup.
    confidence = _compute_confidence(
        len(verified), len(unverified), len(contradictions),
        len(projections),
    )
    # Forecast cap (Q2 level 2): when an answer is projection-DOMINATED
    # (more hedged forecasts than verified facts), it is an inherently-
    # uncertain analysis no matter how clean its facts. Cap at 85 so a
    # year-ahead forecast never reads as near-certain, while still
    # scoring well above the critic-retry threshold. A fact-dominated
    # answer with a few projections is unaffected.
    if projections and len(projections) > len(verified) and confidence > _PROJECTION_CONFIDENCE_CAP:
        confidence = _PROJECTION_CONFIDENCE_CAP
    # Grounding guard: cap confidence when the answer's named entities
    # don't trace back to ANY source. Production audit caught the
    # verifier-LLM reporting 95% confidence on a fully fabricated
    # 4-phase protocol ('Kvadrigalt' / 'Theodorinka' — both invented
    # by the user as a test). The LLM had irrelevant notes loaded
    # ('Scary Movie', 'static web page') so the empty-source short-
    # circuit at line 467 didn't fire — it was the deterministic
    # text-grounding check that was missing.
    #
    # Skip the check when tool_context is substantial. Re-audit
    # follow-up: a CORRECT self-analysis answer ("written in Python,
    # routes LLM in backend/llm.py") got 50%-capped because the
    # proper-noun extractor missed some grounded terms while
    # picking up an explicitly-disconfirmed name (LLMRouter — agent
    # explicitly said it does NOT exist). The cap is for the case
    # where notes+tools were sparse or irrelevant. When the agent
    # read substantial source material, the grounding signal is
    # already strong — defer to the LLM verifier's verdict.
    if (
        confidence > UNGROUNDED_CONFIDENCE_CAP
        and len(tool_context) < SOURCE_GROUNDED_TOOL_CTX_MIN
        and _is_ungrounded(answer, notes_text, tool_context)
    ):
        confidence = UNGROUNDED_CONFIDENCE_CAP
        unverified.append(
            "answer mentions specific named entities that cannot be "
            "located in any loaded note or tool output — likely "
            "fabrication or training-data recall without grounding"
        )

    # Strong-grounding floor: when the agent read substantial source
    # material (~20KB+ of tool output) AND the verifier found at
    # least one verified claim with zero contradictions, the answer
    # has solid evidence behind it. Production audit hit a clean
    # "what programming language are you written in?" turn at 67%
    # confidence — formula-correct (verifier flagged a few minor
    # claims as unverified) but misleading next to read_file output
    # that quoted the actual `from __future__ import annotations`
    # line. Floor at SOURCE_GROUNDED_CONFIDENCE_FLOOR so the
    # critic-retry threshold doesn't trip on demonstrably-grounded
    # answers.
    #
    # 2026-08-08: the floor was a pure conjunction with no ratio term, so ONE
    # verified claim among forty unverified ones was lifted to 75 as long as
    # the turn had read 20 KB — ordinary for any coding turn. Measured on a
    # stubbed verifier: {verified: 1, unverified: 40} scored 2 by formula and
    # was REPORTED as 75, and one byte less tool output reported it as 2. A
    # 73-point swing on an unrelated input.
    #
    # That is not cosmetic. answer_critic only fires below 60, so the critique
    # pass that exists to fix unsupported claims was skipped precisely on the
    # turns with forty of them, and the daily low-confidence flag never
    # tripped. The floor is meant to rescue an over-cautious 67, not to
    # manufacture a 75 — so it now lifts by a bounded amount and only when the
    # claim mix is actually favourable.
    if (
        len(tool_context) >= SOURCE_GROUNDED_TOOL_CTX_MIN
        and len(verified) > 0
        and len(verified) >= len(unverified)
        and len(contradictions) == 0
        and confidence < SOURCE_GROUNDED_CONFIDENCE_FLOOR
    ):
        confidence = min(SOURCE_GROUNDED_CONFIDENCE_FLOOR, confidence + 25)

    return VerificationResult(
        confidence=confidence,
        verified_claims=verified,
        unverified_claims=unverified,
        contradictions=contradictions,
        projections=projections,
        notes_used=list(data.get("notes_used", used_topics)),
    )
