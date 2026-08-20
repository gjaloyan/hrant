"""Layer 0 reflex engine — rule-based pure-Python decisions per tick."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from .types import StateSnapshot, TickDecision, TickDecisionSource

log = logging.getLogger(__name__)


@dataclass
class LayerZeroRule:
    name: str
    predicate: Callable[[StateSnapshot], bool]
    lever: str
    params: dict = field(default_factory=dict)
    cooldown_seconds: float = 30.0


class Layer0Engine:
    def __init__(self, rules: list[LayerZeroRule]) -> None:
        self._rules = list(rules)
        self._last_fired: dict[str, float] = {}

    def evaluate(self, state: StateSnapshot) -> TickDecision:
        now = time.monotonic()
        last_cooldown_hit: TickDecision | None = None
        for rule in self._rules:
            try:
                matched = bool(rule.predicate(state))
            except Exception as exc:
                log.warning("Layer0 rule %r predicate raised: %s", rule.name, exc)
                continue
            if not matched:
                continue
            last = self._last_fired.get(rule.name)
            if last is not None and (now - last) < rule.cooldown_seconds:
                if last_cooldown_hit is None:
                    last_cooldown_hit = TickDecision(
                        source=TickDecisionSource.L0_REFLEX,
                        lever=None,
                        params={},
                        reason=f"cooldown:{rule.name}",
                        rule_name=rule.name,
                    )
                continue
            self._last_fired[rule.name] = now
            return TickDecision(
                source=TickDecisionSource.L0_REFLEX,
                lever=rule.lever,
                params=dict(rule.params),
                reason=f"rule_matched:{rule.name}",
                rule_name=rule.name,
            )
        if last_cooldown_hit is not None:
            return last_cooldown_hit
        return TickDecision(
            source=TickDecisionSource.L0_REFLEX,
            lever=None,
            params={},
            reason="idle_no_rules_matched",
        )


def _has_watched_channels() -> bool:
    """True when this deployment follows at least one channel.

    Read live rather than captured at import: the owner can start
    following a channel without a restart, and a predicate frozen at boot
    would ignore them until one.
    """
    try:
        from ..channel_watch import watched
        return bool(watched())
    except Exception:
        return False


def default_rules() -> list[LayerZeroRule]:
    return [
        LayerZeroRule(
            name="disk_low",
            predicate=lambda s: s.disk_free_gb < 2.0,
            lever="FIRE_SERVER_HEALTH",
            params={"reason": "disk_low"},
            cooldown_seconds=300.0,
        ),
        LayerZeroRule(
            name="memory_low",
            predicate=lambda s: s.memory_free_gb < 0.5,
            lever="FIRE_SERVER_HEALTH",
            params={"reason": "memory_low"},
            cooldown_seconds=300.0,
        ),
        LayerZeroRule(
            name="cpu_high",
            predicate=lambda s: s.cpu_load_1m > 4.0,
            lever="FIRE_SERVER_HEALTH",
            params={"reason": "cpu_high"},
            cooldown_seconds=300.0,
        ),
        LayerZeroRule(
            # 2026-08-09/10. FIRE_SERVICE_REPAIR was registered with no rule
            # and could never fire. It is wired here, in the reactive block,
            # only after five defects were fixed — as it stood it would have
            # restarted HEALTHY services on a timer and logged repairs that
            # never happened.
            #
            # `failed` is the narrow, correct trigger: systemd retries
            # crash-loops itself far faster than any tick (Restart=always,
            # RestartUSec 5-10s; prod has units at 174k restarts) and only
            # gives up once StartLimitBurst is exhausted. That give-up state
            # is the one thing nothing else on the box recovers from.
            #
            # Measured on prod: zero failed units in either manager, so this
            # predicate is FALSE today. That is the property that lets it sit
            # this high — a reactive rule that is true every tick would
            # starve the 20 working levers below it, which is exactly the
            # 2026-06-12 starvation bug documented further down.
            name="service_failed",
            predicate=lambda s: bool(getattr(s, "failed_services", None)),
            lever="FIRE_SERVICE_REPAIR",
            params={},
            cooldown_seconds=600.0,
        ),
        LayerZeroRule(
            name="errors_present",
            predicate=lambda s: len(s.recent_errors) > 0,
            lever="FIRE_ERROR_TRIAGE",
            params={},
            cooldown_seconds=120.0,
        ),
        LayerZeroRule(
            # Starvation fix (2026-06-12): this rule lived LAST in
            # the list, and the engine is first-match-wins. After a
            # service restart every periodic rule's cooldown is
            # clear, so they all queue ahead by list order and a DUE
            # REMINDER waited 5+ minutes for a tick slot (caught
            # live: integrity + goal_propose + capability_scan each
            # took a tick while a due message sat pending). Now the
            # FIRST of the always-true periodic rules: reactive rules
            # (disk/memory/cpu/errors — the four above) still preempt
            # genuinely urgent conditions, but among housekeeping,
            # user-facing delivery wins every time.
            name="scheduled_messages_tick",
            predicate=lambda s: True,
            lever="FIRE_SCHEDULED_MESSAGES",
            params={},
            cooldown_seconds=60.0,
        ),
        LayerZeroRule(
            # 2026-08-09 audit. FIRE_GRAPH_REBUILD has been registered since
            # May with NO rule naming it — zero fires, ever. Wired as a
            # RECOVERY REFLEX, not a periodic tick, for two measured reasons.
            #
            # Its own drift gate is false on prod today (2801 facts vs 6877
            # graph nodes), so a periodic rule would log "no_drift" forever
            # while consuming a tick slot. And builder.rebuild() is
            # destructive: it clear()s the graph and re-derives only from
            # memory_facts/skills/goals, so it would silently delete the
            # LLM-proposed `relates_to` edges that graph/proposer.py writes
            # into the same file. The 12-nodes-vs-1453-facts numbers in the
            # lever's docstring are stale; that drift was repaired long ago.
            #
            # What nothing else covers is catastrophic loss — the only other
            # rebuild caller in the tree is the manual POST
            # /api/kgraph/rebuild. 500 is ~7% of the live 6877 and
            # unreachable by normal churn (nodes only grow via upsert): it is
            # reached by a wipe, a truncation, or a legacy-schema regression.
            # By the time it fires there is nothing curated left to destroy,
            # which is what makes the destructive rebuild acceptable here.
            #
            # Conditional and false in steady state, so it cannot starve the
            # always-true block below it despite sitting this high. Below
            # scheduled_messages_tick per the 2026-06-12 starvation fix:
            # user-facing delivery preempts repair.
            # `kb_notes_count > 0` matters: on a FRESH install both counts
            # are 0, and "no graph yet" is not "the graph was wiped" — there
            # is nothing to recover and the rule would burn an hourly tick
            # forever on a new box. The real recovery signal is knowledge
            # present WITHOUT a graph (prod: 15 notes, 6877 nodes today; after
            # a wipe: 15 notes, 0 nodes). Caught by an integration test that
            # builds a real snapshot over an empty knowledge root.
            name="graph_collapsed",
            predicate=lambda s: s.kb_graph_nodes < 500 and s.kb_notes_count > 0,
            lever="FIRE_GRAPH_REBUILD",
            params={},
            cooldown_seconds=3600.0,
        ),
        LayerZeroRule(
            name="integrity_tick",
            predicate=lambda s: True,
            lever="FIRE_INTEGRITY_HEARTBEAT",
            params={},
            cooldown_seconds=300.0,
        ),
        LayerZeroRule(
            name="goal_propose_tick",
            predicate=lambda s: True,
            lever="FIRE_GOAL_PROPOSE",
            params={},
            cooldown_seconds=3600.0,
        ),
        # `consolidation_tick` was retired 2026-05-27 (audit T3.2).
        # The dedicated consolidation scheduler in
        # `backend/consolidation/scheduler.py` already runs daily +
        # idle-aware. The layer-0 rule fired ~86 times/day and was
        # ~99% no-ops ("no_unconsolidated_sessions"), polluting the
        # lever_log without doing additional work. The lever class
        # stays registered so it can still be invoked manually via
        # the autonomic API.
        LayerZeroRule(
            name="capability_scan_tick",
            predicate=lambda s: True,
            lever="FIRE_CAPABILITY_SCAN",
            params={},
            cooldown_seconds=21600.0,
        ),
        LayerZeroRule(
            name="self_study_tick",
            predicate=lambda s: True,
            lever="FIRE_SELF_STUDY",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="graph_maintenance_tick",
            predicate=lambda s: True,
            lever="FIRE_GRAPH_MAINTENANCE",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="proactive_learn_tick",
            predicate=lambda s: True,
            lever="FIRE_PROACTIVE_LEARN",
            params={},
            cooldown_seconds=3600.0,
        ),
        LayerZeroRule(
            # 2026-07-06 audit: the lever was registered but had NO rule —
            # never fired once, and 385 pending proposals piled up since May.
            name="stale_proposals_tick",
            predicate=lambda s: True,
            lever="FIRE_STALE_PROPOSALS",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            # Drive user/learning goals to completion, one subtask per fire
            # (2026-06-25 audit: those goal types were created but never
            # executed by anything). Long cooldown bounds the spend.
            name="goal_drive_tick",
            predicate=lambda s: True,
            lever="FIRE_GOAL_DRIVE",
            params={},
            cooldown_seconds=7200.0,
        ),
        LayerZeroRule(
            name="note_curation_tick",
            predicate=lambda s: True,
            lever="FIRE_NOTE_CURATION",
            params={},
            cooldown_seconds=604800.0,
        ),
        LayerZeroRule(
            name="model_eval_tick",
            predicate=lambda s: True,
            lever="FIRE_MODEL_EVAL",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="session_archive_tick",
            predicate=lambda s: True,
            lever="FIRE_SESSION_ARCHIVE",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="cost_audit_tick",
            predicate=lambda s: True,
            lever="FIRE_COST_AUDIT",
            params={},
            cooldown_seconds=3600.0,
        ),
        LayerZeroRule(
            name="self_reflection_tick",
            predicate=lambda s: True,
            lever="FIRE_SELF_REFLECTION",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            # Weekly, not nightly. Character is not a daily deliverable, and
            # a lever that mails its person a new personality every morning
            # gets its proposals dismissed unread. It also self-suppresses
            # while a revision is pending.
            name="character_reflection_tick",
            predicate=lambda s: True,
            lever="FIRE_CHARACTER_REFLECTION",
            params={},
            cooldown_seconds=604800.0,
        ),
        LayerZeroRule(
            name="finetune_qc_tick",
            predicate=lambda s: True,
            lever="FIRE_FINETUNE_QC",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="gap_detection_tick",
            predicate=lambda s: True,
            lever="FIRE_GAP_DETECTION",
            params={},
            cooldown_seconds=86400.0,
        ),
        LayerZeroRule(
            name="log_rotation_tick",
            predicate=lambda s: True,
            lever="FIRE_LOG_ROTATION",
            params={},
            cooldown_seconds=86400.0,  # daily
        ),
        LayerZeroRule(
            name="goal_executor_tick",
            predicate=lambda s: True,
            lever="FIRE_GOAL_EXECUTOR",
            params={},
            cooldown_seconds=86400.0,  # daily
        ),
        LayerZeroRule(
            name="fact_embedding_backfill_tick",
            predicate=lambda s: True,
            lever="FIRE_FACT_EMBEDDING_BACKFILL",
            params={},
            cooldown_seconds=86400.0,  # daily
        ),
        LayerZeroRule(
            # 2026-08-10. FIRE_MEMORY_CONSOLIDATION was registered with no
            # rule and never ran. Wired only after the two things that made
            # it dangerous were fixed.
            #
            # It shares an append-only fact store with
            # backend/consolidation/pipeline.py, which runs daily and works.
            # Its dedup horizon was 200 lines against the pipeline's 5000, so
            # it would have re-added every fact the pipeline wrote more than
            # 200 lines ago — two writers, the shorter horizon silently
            # duplicating the longer one's work. Both now read 5000, and both
            # stamp a `writer` field: the store had NO writer attribution at
            # all until today, which is why synthetic rows sat in it
            # unnoticed for months.
            #
            # The trigger is real work rather than a timer: sessions that
            # ended without being consolidated. FALSE on an idle box, so this
            # LLM-costing lever cannot fire on a tick with nothing to do.
            name="unconsolidated_sessions_tick",
            predicate=lambda s: getattr(s, "unconsolidated_sessions", 0) > 0,
            lever="FIRE_MEMORY_CONSOLIDATION",
            params={"max_sessions": 3},
            cooldown_seconds=21600.0,  # 6h
        ),
        LayerZeroRule(
            # 2026-08-09 audit. FACT embeddings got the daily rule above;
            # NOTE embeddings never got one, so FIRE_EMBEDDING_BACKFILL sat
            # registered with zero fires since May. The embedder lives on a
            # LAN box rather than localhost, so a transient failure during
            # note creation leaves that note permanently unsearchable — and
            # both knowledge_manager._embed_note and the startup probe defer
            # the repair to THIS lever, which could never run.
            #
            # Last in the list on purpose: pure housekeeping with no latency
            # requirement, so being last costs it nothing, while any higher
            # position would push one more always-true rule ahead of
            # everything below it after a restart (the 2026-06-12 starvation
            # fix). Skips cheaply without touching the network when coverage
            # is already complete — prod is 15/15 today.
            name="note_embedding_backfill_tick",
            predicate=lambda s: True,
            lever="FIRE_EMBEDDING_BACKFILL",
            params={},
            cooldown_seconds=86400.0,  # daily
        ),
        LayerZeroRule(
            # Collect followed channels often enough that a busy one cannot
            # overflow its page between polls: a public channel page holds
            # only ~16 posts, and at ten minutes even a lively channel stays
            # inside that window.
            #
            # The predicate — not just the lever's preconditions — asks
            # whether anything is followed. A false precondition still
            # consumes a tick slot and files a SKIPPED report, which is the
            # trap FIRE_GRAPH_REBUILD documents further up: a rule that can
            # never do anything should not be selected in the first place.
            #
            # Last in the periodic block, not near the head. Collecting posts
            # for a digest hours away is housekeeping, and the 2026-06-12
            # starvation fix reserved the front for user-facing delivery.
            name="channel_watch_tick",
            predicate=lambda s: _has_watched_channels(),
            lever="POLL_WATCHED_CHANNELS",
            params={},
            cooldown_seconds=600.0,
        ),
        # scheduled_messages_tick moved to the head of the list
        # (right after the safety triad) 2026-06-12 — see the
        # starvation-fix comment there.
    ]
