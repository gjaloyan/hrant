"""`hrant consolidate` subcommand group (Phase 16A).

Extracted from cli.py per audit #21. Handles daily memory
consolidation: status / run [--dry-run] / list / show <date>.
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as _dt


def _print_err(msg: str) -> None:
    from .cli import _print_err as f
    f(msg)


def cmd_consolidate_status(args: argparse.Namespace) -> int:
    """`hrant consolidate status` — when does it fire, what was the
    last run, why is it (not) firing right now."""
    from .consolidation import scheduler as _sched
    from .cli_colors import c
    s = _sched.status()
    st = s["state"]
    print()
    print(c.heading("  Daily memory consolidation"))
    print()
    if st["last_run_at"]:
        ts = _dt.datetime.fromtimestamp(st["last_run_at"]).strftime("%Y-%m-%d %H:%M:%S")
        status_s = st["last_run_status"]
        status_c = {
            "success": c.success(status_s),
            "partial": c.warn(status_s),
            "failed": c.error(status_s),
            "skipped": c.muted(status_s),
        }.get(status_s, status_s)
        print(f"  {c.muted('last run:'):<24} {ts}  ({status_c})")
        if st["last_run_digest"]:
            print(f"  {c.muted('digest:'):<24} {st['last_run_digest']}")
        if st["last_run_jobs_analyzed"]:
            print(f"  {c.muted('turns analyzed:'):<24} {st['last_run_jobs_analyzed']}")
        if st["last_run_facts_added"]:
            print(f"  {c.muted('facts added:'):<24} {st['last_run_facts_added']}")
        if st["last_run_error"]:
            print(f"  {c.error('error:'):<24} {st['last_run_error']}")
    else:
        print(f"  {c.muted('last run:'):<24} {c.muted('never')}")
    print()
    if s["would_fire_now"]:
        print(f"  {c.muted('status:'):<24} {c.accent_bright('READY')} — would fire on next tick")
    else:
        print(f"  {c.muted('status:'):<24} {c.muted('waiting:')} {s['gate_reason']}")
    cd = int(s["cooldown_remaining_seconds"])
    if cd > 0:
        h, rem = divmod(cd, 3600)
        m, _sec = divmod(rem, 60)
        print(f"  {c.muted('cooldown:'):<24} {h}h {m}m remaining")
    idle = s["idle_for_seconds"]
    if idle is not None:
        print(f"  {c.muted('idle for:'):<24} {int(idle)}s "
              f"(threshold: {int(s['config']['idle_threshold_seconds'])}s)")
    print()
    return 0


def cmd_consolidate_run(args: argparse.Namespace) -> int:
    """`hrant consolidate run` — fire a consolidation right now,
    bypassing the idle/24h gates. `--dry-run` shows what would be
    added without writing to memory_facts / profiles."""
    from .consolidation import scheduler as _sched
    from .cli_colors import c
    print(f"  {c.muted('running consolidation...')}")
    d = asyncio.run(_sched.fire_now(dry_run=bool(args.dry_run)))
    status_label = {
        "success": c.success("✓ success"),
        "partial": c.warn("⚠ partial — some steps failed"),
        "failed": c.error("✗ failed"),
        "skipped": c.muted(f"skipped: {d.skip_reason or '?'}"),
    }.get(d.status, d.status)
    print(f"  {status_label}")
    print(f"  {c.muted('turns analyzed:')} {d.turns_analyzed}")
    print(f"  {c.muted('new facts:')} "
          f"{sum(1 for f in d.new_facts if f.promoted)} promoted, "
          f"{sum(1 for f in d.new_facts if not f.promoted)} skipped")
    print(f"  {c.muted('profile updates:')} {len(d.profile_updates)}")
    print(f"  {c.muted('open threads:')} {len(d.open_threads)}")
    if d.narrative:
        print()
        print(c.muted("  narrative:"))
        for line in d.narrative.split("\n"):
            print(f"    {line}")
    if d.error:
        print()
        _print_err(d.error)
    return 0 if d.status in ("success", "partial") else 1


def cmd_consolidate_list(args: argparse.Namespace) -> int:
    from .consolidation import digest as _digest_mod
    from .cli_colors import c
    rows = _digest_mod.list_all()[:args.limit]
    if not rows:
        print(c.muted("  no digests yet — run `hrant consolidate run`"))
        return 0
    print()
    print(c.heading("  Memory digests (newest first)"))
    print()
    for r in rows:
        status_c = {
            "success": c.success(r["status"]),
            "partial": c.warn(r["status"]),
            "failed": c.error(r["status"]),
        }.get(r["status"], c.muted(r["status"]))
        preview = r["narrative_preview"][:80]
        print(
            f"  {c.muted(r['date'])}  {status_c:<22}  "
            f"{r['turns_analyzed']:3d} turns, "
            f"{r['new_facts_count']:2d} facts  "
            f"{c.muted(preview)}"
        )
    print()
    return 0


def cmd_consolidate_show(args: argparse.Namespace) -> int:
    from .consolidation import digest as _digest_mod
    from .cli_colors import c
    d = _digest_mod.read(args.date)
    if d is None:
        _print_err(f"no digest for {args.date}")
        return 1
    print()
    print(c.heading(f"  Digest for {d.date}"))
    print(f"  {c.muted('status:')} {d.status}")
    print(f"  {c.muted('turns analyzed:')} {d.turns_analyzed}")
    print(f"  {c.muted('speakers:')} {', '.join(d.speakers_active)}")
    print()
    print(c.muted("  narrative:"))
    for line in (d.narrative or "(empty)").split("\n"):
        print(f"    {line}")
    if d.new_facts:
        print()
        print(c.muted(f"  facts ({len(d.new_facts)}):"))
        for f in d.new_facts:
            marker = c.success("✓") if f.promoted else c.muted("·")
            note = f" {c.muted(f'({f.reason_if_skipped})')}" if f.reason_if_skipped else ""
            print(f"    {marker} [{c.muted(f.category)}] {f.text}{note}")
    if d.open_threads:
        print()
        print(c.muted(f"  open threads ({len(d.open_threads)}):"))
        for t in d.open_threads:
            print(f"    · {t}")
    if d.profile_updates:
        print()
        print(c.muted(f"  profile updates ({len(d.profile_updates)}):"))
        for up in d.profile_updates:
            print(f"    {up.speaker_id} → {up.profile_path}")
    if d.error:
        print()
        _print_err(d.error)
    print()
    return 0
