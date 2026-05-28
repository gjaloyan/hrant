"""The LLM pipeline that turns an ActivityBundle into a Digest.

Phase 16A scope — six steps:
  1. Render the bundle as a compact transcript the LLM can read
  2. Narrative: one LLM call producing a 3-5 sentence narrative
  3. Extract facts: JSON output with confidence + topics + category
  4. Dedup new facts against existing `memory_facts.jsonl`
  5. Profile updates: appends to `user.md` (global) and per-speaker
     profiles for any active Telegram speakers
  6. Open threads: one LLM call surfacing unresolved questions

NOT in Phase 16A (will land in 16B/16C):
  - Pruning candidates → `_trash/`
  - Knowledge-graph edge proposals
  - Cross-link inference between new facts and existing topics

Cost cap: per the user's spec, unlimited. Token usage is still
tracked + reported in the digest so it shows in the WebUI.

Failure semantics: any LLM step can fail (provider down even
after failover, JSON parse error). The pipeline catches those,
records the error in the digest, and writes a `failed` status —
it does NOT raise. Scheduler updates state with last_run_status
so the user sees what went wrong in the WebUI banner.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date as _date
from pathlib import Path
from typing import Optional

from .. import paths
from . import config, digest as _digest_mod, gather

log = logging.getLogger(__name__)


# ─── System prompts ──────────────────────────────────────────────────


NARRATIVE_SYSTEM = """You are Hrant's daily memory consolidation module.

You receive a transcript of the last ~24h of conversations and turns
across all of the user's channels (WebUI + Telegram, etc.).

Write a TIGHT, factual 3-5 sentence narrative of what happened.
Examples of good lines:
  - "Worked on a multi-provider failover system for the LLM router."
  - "Asked about CLI ergonomics; renamed `hrant service` to `hrant gateway`."
  - "Discussed daily memory consolidation design; settled on Phase 16C scope."

Rules:
- Russian or English, match the user's language.
- Skip greetings, small talk, agent reasoning, tool failures.
- Refer to the user as "the user" (or the equivalent in the response language), not "you".
- Do NOT include facts the agent learned about itself."""


FACTS_SYSTEM = """You extract durable, long-term facts from a day of conversations.

Return strictly JSON:
{
  "new_facts": [
    {
      "text": "User uses Tailscale for cross-device service discovery (host 100.124.210.21).",
      "related_topics": ["network", "preferences", "tailscale"],
      "confidence": 0.95,
      "category": "preference"
    }
  ]
}

Categories: role | location | preference | project | technical | event | relationship | rule | general.

Rules:
- Skip greetings, small talk, agent reasoning. Skip transient state
  (what the user asked today is NOT a fact; what they ESTABLISHED
  about themselves or the world IS).
- Russian or English to match the source.
- Confidence ≥0.8 for things worth promoting to long-term memory.
- Max 12 facts total.
- Do NOT extract facts about the agent's own identity or values.
- Each fact must be a single declarative sentence."""


OPEN_THREADS_SYSTEM = """You identify UNRESOLVED conversational threads from a day's transcript.

Return strictly JSON:
{ "open_threads": ["short phrase describing one open thread", ...] }

Examples:
  "User asked about CHANGELOG migration tool — never implemented"
  "Telegram bot still missing /retry command"
  "Discussed knowledge graph but no design doc yet"

Rules:
- Max 5 threads. Each is one sentence.
- Skip threads that were RESOLVED in the same session.
- Skip the agent's own internal todos.
- Russian or English to match the source."""


PROFILE_UPDATE_SYSTEM = """You write a short profile-update entry for one user, based on a day of conversations.

You'll get the user's existing profile + today's transcript filtered
to just that user's turns.

Return strictly JSON:
{
  "should_update": true/false,
  "entry": "Short paragraph of NEW info learned about this user today. Empty if should_update is false."
}

Rules:
- ONLY include facts/preferences/projects NOT already in the existing profile.
- 1-3 sentences max. Concrete, not vague.
- Russian or English to match the source.
- should_update=false if nothing genuinely new was learned (don't pad)."""


# ─── Helpers ─────────────────────────────────────────────────────────


_TRANSCRIPT_USER_CAP = 400        # per-message truncation
_TRANSCRIPT_AGENT_CAP = 400
_TRANSCRIPT_TOTAL_CAP = 24_000    # whole-transcript ceiling


def _render_transcript(bundle: gather.ActivityBundle) -> str:
    """Compact one-line-per-turn rendering. Job order preserved
    (oldest first → most-recent last) so the LLM sees the day in
    chronological order."""
    chronological = sorted(bundle.jobs, key=lambda j: j.created_at)
    lines: list[str] = []
    for j in chronological:
        # Channel + speaker tag so the LLM can distinguish "this is
        # Telegram user A" from "this is WebUI session".
        tag = f"[{j.channel}/{j.speaker_id}]"
        prompt = (j.prompt or "").replace("\n", " ").strip()
        if prompt:
            lines.append(f"{tag} USER: {prompt[:_TRANSCRIPT_USER_CAP]}")
        answer = (j.response or "").replace("\n", " ").strip()
        if answer:
            lines.append(f"{tag} AGENT: {answer[:_TRANSCRIPT_AGENT_CAP]}")
        if j.status == "failed" and j.error:
            lines.append(f"{tag} (turn failed: {j.error[:200]})")
    text = "\n".join(lines)
    if len(text) > _TRANSCRIPT_TOTAL_CAP:
        # Keep the most-recent activity — older turns get truncated.
        text = "[…earlier turns truncated…]\n" + text[-_TRANSCRIPT_TOTAL_CAP:]
    return text


_FACT_CACHE: dict[str, tuple[float, int, set[str]]] = {}
# `path → (mtime, size, summaries_set)` so repeated reads of the
# same file inside a single process don't re-tokenise it from disk.
# Audit #16: previously every consolidation re-read + re-scanned
# the whole file. Cache key is mtime+size so concurrent appends
# (autonomic lever runs while a consolidation tick is in flight)
# invalidate cleanly. RLock not needed at single-process scale
# but if the cache grows multiple-key, gate with one.


def _existing_fact_summaries(limit: int = 5000) -> set[str]:
    """Recent memory_facts.jsonl summaries lower-cased, for dedup.

    Audit #8 fix: bumped from 500 → 5000. At 5 promoted facts/day,
    500 lines covered ~100 days of history; at the personal-agent
    scale (5–10 years of daily consolidation) the cap is now well
    out of reach. Reads ~1MB max on the rare full-cap day — still
    cheap. For larger deployments we'd need a real index.

    Audit #16: cached by (mtime, size) so two consolidations in the
    same process don't re-parse the file. The cache key invalidates
    on any disk change so concurrent appends (autonomic lever
    running while a consolidation tick is in flight) refresh
    correctly."""
    p = paths.knowledge_dir() / "memory_facts.jsonl"
    if not p.exists():
        return set()
    # Cache lookup: same (mtime, size) → reuse previous result.
    try:
        stat = p.stat()
        cache_key = (stat.st_mtime, stat.st_size)
    except OSError:
        cache_key = None
    if cache_key is not None:
        cached = _FACT_CACHE.get(str(p))
        if cached is not None and (cached[0], cached[1]) == cache_key:
            return set(cached[2])  # copy so callers can mutate freely
    out: set[str] = set()
    try:
        lines = p.read_text(encoding="utf-8").splitlines()[-limit:]
    except OSError:
        return out
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        s = obj.get("summary") or obj.get("text")
        if s:
            out.add(str(s).strip().lower())
    # Populate cache (size-bounded by clearing when too many keys).
    if cache_key is not None:
        if len(_FACT_CACHE) > 8:
            _FACT_CACHE.clear()
        _FACT_CACHE[str(p)] = (cache_key[0], cache_key[1], set(out))
    return out


# Audit #15: rotate memory_facts.jsonl when it grows past this
# many lines. At 5 facts/day, 50k lines = ~27 years of history —
# big enough to never hit accidentally, small enough that the
# dedup full-file scan stays cheap. Once exceeded, the file is
# renamed to memory_facts.jsonl.<YYYY-MM-DD>.archive and a fresh
# one starts. Archives are preserved on disk — no data ever
# deleted, just chunked.
_MEMORY_FACTS_ROTATE_LINES = int(
    __import__("os").environ.get("HRANT_MEMORY_FACTS_ROTATE", 50_000)
)


def _maybe_rotate_memory_facts(p) -> None:
    """If `p` exceeds the rotation cap, rename it to
    `memory_facts.jsonl.<YYYY-MM-DD>.archive` so the active file
    stays bounded. Best-effort: rotation failure logs and lets the
    file keep growing rather than blocking the consolidation."""
    try:
        if not p.exists():
            return
        # Cheap upper-bound check: file size in bytes / ~200 bytes
        # per row >= rotate threshold? If yes, do the precise line
        # count. Avoids reading the whole file every consolidation
        # just to confirm it's not full.
        if p.stat().st_size < _MEMORY_FACTS_ROTATE_LINES * 100:
            return
        with p.open("r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        if line_count < _MEMORY_FACTS_ROTATE_LINES:
            return
        from datetime import datetime as _dt
        stamp = _dt.now().strftime("%Y-%m-%d")
        archive = p.with_suffix(f".jsonl.{stamp}.archive")
        # If today's archive already exists (rare — would require
        # writing >50k facts in a day), suffix with hh-mm-ss to
        # avoid clobbering it.
        if archive.exists():
            archive = p.with_suffix(
                f".jsonl.{_dt.now().strftime('%Y-%m-%d_%H-%M-%S')}.archive",
            )
        p.rename(archive)
        log.info(
            "memory_facts rotated: %d lines moved to %s",
            line_count, archive.name,
        )
    except Exception as e:
        log.warning("memory_facts rotation skipped (%s)", e)


def _append_memory_fact(text: str, category: str, confidence: float,
                       topics: list[str], date_str: str) -> None:
    """Append one row to `memory_facts.jsonl`, shape compatible
    with the existing autonomic lever. Rotates the file when it
    exceeds _MEMORY_FACTS_ROTATE_LINES so the dedup scan stays
    bounded."""
    p = paths.knowledge_dir() / "memory_facts.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    _maybe_rotate_memory_facts(p)
    entry = {
        "summary": text,
        "text": text,                    # alias for forward-compat
        "triples": [],                   # graph stub for Phase 16C
        "tags": list(topics),
        "category": category,
        "confidence": float(confidence),
        "ts": time.time(),
        "source_turn": f"consolidation:{date_str}",
    }
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _speaker_transcript(bundle: gather.ActivityBundle, speaker_id: str) -> str:
    """Subset of the transcript for one speaker — used by the
    per-speaker profile-update step so we don't accidentally write
    user A's facts into user B's profile."""
    matching = [j for j in bundle.jobs if j.speaker_id == speaker_id]
    if not matching:
        return ""
    sub_bundle = gather.ActivityBundle(
        window_start_ts=bundle.window_start_ts,
        window_end_ts=bundle.window_end_ts,
        jobs=matching,
    )
    return _render_transcript(sub_bundle)


def _profile_path_for(speaker_id: str) -> Path:
    """Telegram speakers get isolated per-chat profile files;
    every other channel (WebUI, voice, future channels) writes
    to the global `user.md`. Matches the user's stated rule:
    "globally — telegram has only its own user.md per chat".

    Audit #3 fix: pre-fix only `webui:default` mapped to the
    global file, which meant any future non-default WebUI speaker
    or voice speaker would silently end up with its own per-
    speaker profile — diverging from the global file and
    surprising the user."""
    if speaker_id.startswith("telegram:"):
        return config.profile_path_for_speaker(speaker_id)
    return config.user_md_path()


def _append_profile(path: Path, entry: str, date_str: str) -> int:
    """Append a dated section to the profile file. Returns
    pre-write file size for the digest's audit trail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pre_size = path.stat().st_size if path.exists() else 0
    block = f"\n## {date_str} — daily update\n\n{entry.strip()}\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(block)
    return pre_size


# ─── The pipeline ────────────────────────────────────────────────────


def _call_router_json(system: str, user: str, *, max_tokens: int) -> dict:
    """Wrap router().call_json with our task type. Failover chain
    automatically applies if the user enabled it."""
    from ..llm import router, TaskType  # local import — avoid boot order issues
    return router().call_json(
        TaskType.COMPLEX_SOLVING,
        system,
        user,
        max_tokens=max_tokens,
        temperature=0.2,
    )


def _call_router_text(system: str, user: str, *, max_tokens: int) -> str:
    from ..llm import router, TaskType
    return router().call(
        TaskType.COMPLEX_SOLVING,
        system,
        user,
        max_tokens=max_tokens,
        temperature=0.3,
    )


def run(
    *,
    bundle: gather.ActivityBundle,
    digest_date: Optional[str] = None,
    dry_run: bool = False,
) -> _digest_mod.Digest:
    """Execute the pipeline and return the populated Digest.

    `dry_run=True` skips ALL side-effects: no memory_facts append,
    no profile writes. The returned Digest still contains everything
    the LLM produced, so the caller can preview before applying.

    The function never raises — pipeline failures land in the
    digest's `error` field with `status="failed"`. Returning a
    failed Digest is more useful to the caller than an exception:
    they can still persist what completed (often the narrative
    works even when extract-facts fails).
    """
    date_str = digest_date or _digest_mod.today_str(bundle.window_end_ts)
    d = _digest_mod.Digest(
        date=date_str,
        started_at=time.time(),
        window_start_ts=bundle.window_start_ts,
        window_end_ts=bundle.window_end_ts,
        speakers_active=bundle.speakers,
        channels_active=bundle.channels,
        turns_analyzed=bundle.turn_count,
        jobs_failed=bundle.failed_count,
        jobs_interrupted=bundle.interrupted_count,
        status="in_progress",
    )

    # Token tracking: snapshot the request counters before any
    # consolidation LLM call so we can compute a delta at the end.
    # Pre-fix the digest claimed `tokens_used: 0` even when the
    # narrative was a multi-paragraph LLM output (caught in the
    # production audit). The consolidation pipeline runs outside any
    # request context, so request_usage() may already hold cross-call
    # state — subtract the snapshot to get just our delta.
    from ..llm import TOKENS as _TOKENS
    _tokens_before = _TOKENS.request_usage()

    # Zero-activity short-circuit: still write a digest so the UI
    # shows "consolidation ran, nothing happened". Cheap, no LLM
    # calls — saves cost on long quiet weekends.
    if bundle.turn_count == 0:
        d.narrative = "No activity in the consolidation window."
        d.status = "success"
        d.skip_reason = "no_activity"
        d.tokens_used = 0
        d.estimated_cost_usd = 0.0
        d.completed_at = time.time()
        return d

    transcript = _render_transcript(bundle)

    # Step 2: narrative
    try:
        d.narrative = _call_router_text(
            NARRATIVE_SYSTEM,
            f"TRANSCRIPT (last 24h):\n\n{transcript}",
            max_tokens=400,
        ).strip()
    except Exception as e:
        log.warning("consolidation.narrative failed: %s", e)
        d.status = "failed"
        d.error = f"narrative step: {e}"
        d.completed_at = time.time()
        return d  # without a narrative the rest is meaningless

    # Step 3: extract facts
    try:
        facts_resp = _call_router_json(
            FACTS_SYSTEM,
            f"TRANSCRIPT:\n{transcript}",
            max_tokens=2000,
        )
        raw_facts = facts_resp.get("new_facts") or []
    except Exception as e:
        log.warning("consolidation.facts failed: %s", e)
        raw_facts = []
        d.error = (d.error or "") + f" facts step: {e};"

    # Step 4: dedup + promote (+ insert into knowledge graph)
    existing = _existing_fact_summaries()
    # Audit #5: collect graph upserts and save ONCE at the end of
    # the loop instead of save()-ing after every promoted fact.
    # Pre-fix wrote the whole graph file ~10× per consolidation;
    # one save reduces I/O by an order of magnitude.
    any_graph_added = False
    for raw in raw_facts:
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        conf = float(raw.get("confidence") or 0.0)
        category = str(raw.get("category") or "general")
        topics = [str(t) for t in (raw.get("related_topics") or []) if t]
        fact = _digest_mod.DigestFact(
            text=text, related_topics=topics, confidence=conf, category=category,
        )
        from .config import PROMOTE_CONFIDENCE_THRESHOLD
        if conf < PROMOTE_CONFIDENCE_THRESHOLD:
            fact.reason_if_skipped = "low_confidence"
        elif text.lower() in existing:
            fact.reason_if_skipped = "duplicate"
        else:
            if not dry_run:
                _append_memory_fact(text, category, conf, topics, date_str)
                # Phase 16C: mirror into the knowledge graph in
                # memory; we save once after the loop.
                try:
                    from ..graph import builder as _graph_builder
                    _graph_builder.add_fact(
                        text=text,
                        related_topics=topics,
                        category=category,
                        confidence=conf,
                        source=f"consolidation:{date_str}",
                    )
                    any_graph_added = True
                    # Audit #6: tag the entry with its edge kind so
                    # the WebUI can render is_about vs relates_to
                    # entries differently. Pre-fix the step-4 rows
                    # had no `kind` field while the step-5.5 rows
                    # did — the consumer had to sniff the shape.
                    d.links_added.append({
                        "kind": "is_about",
                        "fact": text,
                        "topics": topics,
                    })
                except Exception as e:
                    log.warning("graph.add_fact failed for %r: %s", text[:60], e)
            fact.promoted = True
            existing.add(text.lower())
        d.new_facts.append(fact)
    # Single flush to disk for the whole batch of graph upserts.
    if any_graph_added and not dry_run:
        try:
            from ..graph.store import GRAPH as _G
            _G.save()
        except Exception as e:
            log.warning("graph.save after consolidation batch failed: %s", e)

    # Step 5: per-speaker profile updates
    for speaker in bundle.speakers:
        sub_transcript = _speaker_transcript(bundle, speaker)
        if not sub_transcript:
            continue
        target_path = _profile_path_for(speaker)
        try:
            current_profile = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
        except OSError:
            current_profile = ""
        try:
            resp = _call_router_json(
                PROFILE_UPDATE_SYSTEM,
                (
                    f"EXISTING PROFILE:\n{current_profile[:3000]}\n\n"
                    f"TRANSCRIPT FOR {speaker}:\n{sub_transcript}"
                ),
                max_tokens=600,
            )
        except Exception as e:
            log.warning("consolidation.profile_update(%s) failed: %s", speaker, e)
            continue
        if not bool(resp.get("should_update")):
            continue
        entry = str(resp.get("entry") or "").strip()
        if not entry:
            continue
        pre_size = 0
        if not dry_run:
            pre_size = _append_profile(target_path, entry, date_str)
        d.profile_updates.append(_digest_mod.ProfileUpdate(
            speaker_id=speaker,
            profile_path=str(target_path),
            appended_text=entry,
            pre_size_bytes=pre_size,
        ))

    # Step 5.5: LLM-proposed `relates_to` edges between today's
    # facts and the graph's hubs (Phase 16C.1). Best-effort: if the
    # proposer raises, log and continue — the graph keeps whatever
    # edges it has from `add_fact` above. Skipped when dry_run is on
    # to avoid burning tokens on a preview that doesn't persist.
    if not dry_run:
        try:
            from ..graph import proposer as _graph_proposer
            promoted_texts = [f.text for f in d.new_facts if f.promoted]
            proposed = _graph_proposer.propose_links(
                new_fact_texts=promoted_texts,
                digest_date=date_str,
            )
            # Each entry is `{from, to, reason, kind}` — extends
            # the `links_added` list already populated by the
            # is_about edges from step 4.
            for link in proposed:
                d.links_added.append(link)
        except Exception as e:
            log.warning("consolidation.propose_links failed: %s", e)

    # Step 6: open threads
    try:
        threads_resp = _call_router_json(
            OPEN_THREADS_SYSTEM,
            f"TRANSCRIPT:\n{transcript}",
            max_tokens=500,
        )
        d.open_threads = [
            str(t) for t in (threads_resp.get("open_threads") or []) if t
        ][:5]
    except Exception as e:
        log.warning("consolidation.threads failed: %s", e)

    # Token / cost delta — see snapshot taken at the top of run().
    _tokens_after = _TOKENS.request_usage()
    d.tokens_used = max(
        0, int(_tokens_after.get("total_tokens", 0))
            - int(_tokens_before.get("total_tokens", 0)),
    )
    d.estimated_cost_usd = max(
        0.0, float(_tokens_after.get("cost_usd", 0.0))
            - float(_tokens_before.get("cost_usd", 0.0)),
    )

    d.status = "success" if d.error is None else "partial"
    d.completed_at = time.time()
    return d
