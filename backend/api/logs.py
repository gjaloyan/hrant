"""Logs tab — REST snapshot + sources + download.

The owner-gated endpoints surface the live `LogBus` so the WebUI can
render a unified feed of Python logging / tool calls / job state /
supervisor events / agent progress. The SSE stream endpoint lives
in this same module (added in Task 6).

Spec: docs/superpowers/specs/2026-05-21-logs-tab-design.md
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import Response

from ..log_bus import VALID_LEVELS, VALID_SOURCES, BUS
from ._auth import require_owner_for_writes


log = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/logs")
def get_logs(
    level: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    search: str = Query(default=""),
    limit: int = Query(default=500, ge=1, le=20_000),
    before_ts: float = Query(default=0.0, ge=0.0),
):
    """Snapshot of the ring buffer with optional filters. Returns
    newest-last so the UI can append in chronological order."""
    require_owner_for_writes(action="viewing agent logs")
    levels = [level] if level else None
    sources = [source] if source else None
    events = BUS.tail(
        level=levels, source=sources, search=search,
        limit=limit, before_ts=before_ts,
    )
    return {"events": events, "count": len(events)}


@router.get("/api/logs/sources")
def get_log_sources():
    """Static enums for the UI dropdowns."""
    require_owner_for_writes(action="viewing log source list")
    return {
        "levels": list(VALID_LEVELS),
        "sources": list(VALID_SOURCES),
    }


@router.get("/api/logs/download")
def download_logs(format: str = Query(default="jsonl", pattern="^(jsonl|txt)$")):
    """Dump the current ring buffer. `jsonl` = one JSON object per
    line (machine-readable); `txt` = human-readable lines."""
    require_owner_for_writes(action="downloading agent logs")
    events = BUS.tail(limit=0)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if format == "jsonl":
        body = "\n".join(
            json.dumps(e, ensure_ascii=False) for e in events
        ) + "\n"
        return Response(
            content=body,
            media_type="application/x-ndjson",
            headers={
                "content-disposition": f'attachment; filename="agent-logs-{stamp}.jsonl"',
            },
        )
    lines = []
    for e in events:
        iso = datetime.fromtimestamp(float(e.get("ts") or 0)).isoformat(
            timespec="milliseconds",
        )
        level = (e.get("level") or "info").upper()
        src = e.get("source") or ""
        logger_name = e.get("logger") or ""
        msg = e.get("message") or ""
        lines.append(f"{iso} {level:<8} {src:<10} {logger_name}  {msg}")
    body = "\n".join(lines) + "\n"
    return Response(
        content=body,
        media_type="text/plain; charset=utf-8",
        headers={
            "content-disposition": f'attachment; filename="agent-logs-{stamp}.txt"',
        },
    )
