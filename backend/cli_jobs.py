"""`hrant jobs` subcommand group.

Extracted from `backend/cli.py` as part of the audit-#21 cleanup
(the monolithic `cli.py` had grown to 2417 lines). The argparse
plumbing still lives in `cli.py`; this module exports the
handler functions + the row formatter the handlers share.
"""
from __future__ import annotations

import argparse
import datetime as _dt


def _print_ok(msg: str) -> None:
    from .cli import _print_ok as f
    f(msg)


def _print_warn(msg: str) -> None:
    from .cli import _print_warn as f
    f(msg)


def _print_err(msg: str) -> None:
    from .cli import _print_err as f
    f(msg)


def _format_job_row(job, *, width_id: int = 14, width_status: int = 12) -> str:
    """One-line summary of a job for the `hrant jobs list` table.

    Audit #22: ANSI escape codes don't count as visible chars but
    f-string `:<N` padding counts them anyway. Use `pad_visible`
    so columns line up whether colors are on or off."""
    from .cli_colors import c, pad_visible
    age = _dt.datetime.fromtimestamp(job.created_at).strftime("%m-%d %H:%M")
    status_colored = {
        "queued":      c.muted(job.status),
        "running":     c.accent_bright(job.status),
        "completed":   c.success(job.status),
        "failed":      c.error(job.status),
        "interrupted": c.warn(job.status),
        "cancelled":   c.muted(job.status),
    }.get(job.status, job.status)
    prompt = (job.prompt or "").replace("\n", " ").strip()[:60]
    return (
        f"  {pad_visible(c.muted(job.id), width_id)}  "
        f"{pad_visible(status_colored, width_status)}  "
        f"{c.muted(age)}  "
        f"{pad_visible(c.muted(job.channel), 10)}  "
        f"{prompt}"
    )


def cmd_jobs_list(args: argparse.Namespace) -> int:
    """`hrant jobs list` — table of recent jobs."""
    from . import jobs as _jobs
    from .cli_colors import c
    rows = _jobs.JOBS.list(
        status=args.status,
        channel=args.channel,
        limit=args.limit,
        offset=0,
    )
    if not rows:
        print(c.muted("  no jobs match"))
        return 0
    print()
    print(c.heading("  Jobs (newest first)"))
    print()
    header = (
        f"  {'id':<14}  {'status':<12}  {'created':<14}  "
        f"{'channel':<10}  prompt"
    )
    print(c.muted(header))
    print(c.muted(f"  {'-'*100}"))
    for j in rows:
        print(_format_job_row(j))
    print()
    return 0


def cmd_jobs_show(args: argparse.Namespace) -> int:
    """`hrant jobs show <id>` — full record of one job."""
    from . import jobs as _jobs
    from .cli_colors import c
    job = _jobs.JOBS.get(args.job_id)
    if job is None:
        _print_err(f"no job with id '{args.job_id}'")
        return 1

    def _ts(t):
        return _dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S") if t else "—"
    print()
    print(c.heading(f"  Job {job.id}"))
    print(f"  {c.muted('status:'):<14} {job.status}")
    print(f"  {c.muted('channel:'):<14} {job.channel}")
    print(f"  {c.muted('speaker:'):<14} {job.speaker_id}")
    print(f"  {c.muted('created:'):<14} {_ts(job.created_at)}")
    print(f"  {c.muted('started:'):<14} {_ts(job.started_at)}")
    print(f"  {c.muted('completed:'):<14} {_ts(job.completed_at)}")
    if job.retry_count:
        print(f"  {c.muted('retried:'):<14} {job.retry_count}x")
    if job.interrupted_count:
        print(f"  {c.muted('interrupted:'):<14} {c.warn(str(job.interrupted_count) + 'x')}")
    print()
    print(c.muted("  prompt:"))
    print("    " + (job.prompt or "").replace("\n", "\n    "))
    if job.response:
        print()
        print(c.muted("  response:"))
        print("    " + job.response.replace("\n", "\n    "))
    if job.error:
        print()
        print(c.error("  error:"))
        print("    " + job.error.replace("\n", "\n    "))
    if job.tool_calls:
        print()
        print(c.muted(f"  tool_calls ({len(job.tool_calls)}):"))
        for tc in job.tool_calls:
            ok = c.success("ok") if tc.get("ok") else c.error("fail")
            print(f"    {ok}  {tc.get('name','?')}  {c.muted(tc.get('args_summary','')[:80])}")
    if job.attempts:
        print()
        print(c.muted(f"  attempts ({len(job.attempts)}):"))
        for a in job.attempts:
            ok = c.success("ok") if a.get("ok") else c.error("fail")
            print(f"    {ok}  {a.get('provider_id','?')} / {a.get('model','?')}  {c.muted(a.get('error','') or '')}")
    print()
    return 0


def cmd_jobs_retry(args: argparse.Namespace) -> int:
    """`hrant jobs retry <id>` — clone as a new queued job. Does NOT
    run it; the user replays the prompt through their channel of
    choice. Prints the new id."""
    from . import jobs as _jobs
    from .cli_colors import c
    new = _jobs.JOBS.retry(args.job_id)
    if new is None:
        _print_err(f"no job with id '{args.job_id}'")
        return 1
    _print_ok(f"new job: {new.id}")
    print(c.muted(f"  the prompt is queued; re-send via WebUI or your channel "
                  f"to run it"))
    return 0


def cmd_jobs_cancel(args: argparse.Namespace) -> int:
    from . import jobs as _jobs
    job = _jobs.JOBS.get(args.job_id)
    if job is None:
        _print_err(f"no job with id '{args.job_id}'")
        return 1
    if job.status in _jobs.TERMINAL_STATUSES:
        _print_warn(f"job already terminal ({job.status}); nothing to cancel")
        return 0
    _jobs.JOBS.mark_cancelled(args.job_id)
    _print_ok(f"cancelled {args.job_id}")
    return 0


def cmd_jobs_delete(args: argparse.Namespace) -> int:
    from . import jobs as _jobs
    if not _jobs.JOBS.delete(args.job_id):
        _print_err(f"no job with id '{args.job_id}'")
        return 1
    _print_ok(f"deleted {args.job_id}")
    return 0
