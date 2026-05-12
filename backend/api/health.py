"""Aggregated /api/health endpoint.

Different from `/api/status` (which dumps router stats and topic counts
for the WebUI) — this one is the *operational* readiness probe a
process monitor / Docker healthcheck / `hrant status` consumes.

Each component reports one of four states:

    ok               — fully operational
    degraded         — usable but with a caveat (e.g. WAV fallback)
    down             — configured but unreachable / broken
    not_configured   — feature deliberately off (no URL set, etc.)

The aggregate `status` field collapses the per-component states using
a worst-of rollup: any `down` makes the whole thing `down`; any
`degraded` makes it `degraded`; only when every configured component
is `ok` does the aggregate read `ok`. `not_configured` components do
not drag the aggregate down — a deployment without Piper isn't
unhealthy, just text-only.

The endpoint is intentionally cheap (single-digit milliseconds per
component). External probes (Whisper/Piper/Ollama) get a short
timeout so a hung upstream can't stall the health check itself.
"""
from __future__ import annotations

import logging
import platform
import time
from typing import Any

from fastapi import APIRouter

log = logging.getLogger(__name__)

router = APIRouter()


# Wall-clock for uptime. Reset when the module is imported (i.e.
# server boot). Good enough — uvicorn reload counts as a fresh boot
# anyway.
_BOOT_TIME = time.time()


def _ok(detail: str = "") -> dict:
    return {"status": "ok", "detail": detail}


def _degraded(detail: str) -> dict:
    return {"status": "degraded", "detail": detail}


def _down(detail: str) -> dict:
    return {"status": "down", "detail": detail}


def _not_configured(detail: str = "") -> dict:
    return {"status": "not_configured", "detail": detail}


def _check_agent_core() -> dict:
    """Can we import the agent module? Cheapest possible smoke test."""
    try:
        from .. import agent as _agent  # noqa: F401
        return _ok()
    except Exception as e:
        return _down(f"import failed: {e}")


def _check_model() -> dict:
    """Active model + provider routing. `down` means the agent can't
    talk to any LLM — the most load-bearing failure mode."""
    try:
        from ..config import CONFIG
        from ..providers import ACTIVE_MODEL, get_providers
        active = ACTIVE_MODEL.get()
        if active:
            mid = f"{active.get('provider_id')}/{active.get('model')}"
        else:
            mid = CONFIG.model_a.get("model") or "(unconfigured)"
        provs = [p for p in get_providers() if p.get("enabled", True)]
        if not provs:
            return _down("no enabled providers")
        return _ok(f"active={mid}, providers={len(provs)}")
    except Exception as e:
        return _down(f"model check failed: {e}")


def _check_stt() -> dict:
    """Whisper STT — `not_configured` when no URL set, `down` when the
    URL is set but health probe fails."""
    try:
        from ..transcriber import load_config as _load_t
        cfg = _load_t() or {}
        url = (cfg.get("local_whisper") or {}).get("url") or ""
        if not url:
            return _not_configured("no Whisper URL configured")
        from ..discovery import KNOWN_SERVICES, probe_service
        spec = KNOWN_SERVICES["whisper"]
        # The URL is a full base URL; pull host/port out so we can
        # reuse probe_service which expects host+port. Falling back
        # to the raw URL when parsing fails keeps the check forgiving.
        from urllib.parse import urlparse
        u = urlparse(url)
        host = u.hostname or url
        port = u.port or spec.default_port
        r = probe_service(spec, host=host, port=port, timeout=1.5)
        if r.get("ok"):
            return _ok(f"url={url}")
        return _down(f"probe failed: {r.get('reason')}")
    except Exception as e:
        return _down(f"stt check failed: {e}")


def _check_tts() -> dict:
    """Piper TTS. Same shape as Whisper."""
    try:
        from ..tts import load_config as _load_p
        cfg = _load_p() or {}
        url = (cfg.get("local_piper") or {}).get("url") or ""
        if not url:
            return _not_configured("no Piper URL configured")
        from ..discovery import KNOWN_SERVICES, probe_service
        spec = KNOWN_SERVICES["piper"]
        from urllib.parse import urlparse
        u = urlparse(url)
        host = u.hostname or url
        port = u.port or spec.default_port
        r = probe_service(spec, host=host, port=port, timeout=1.5)
        if r.get("ok"):
            return _ok(f"url={url}")
        return _down(f"probe failed: {r.get('reason')}")
    except Exception as e:
        return _down(f"tts check failed: {e}")


def _check_ffmpeg() -> dict:
    """ffmpeg is only used to make Telegram voice replies into native
    voice bubbles. Missing → degraded (WAV fallback works) not down."""
    try:
        from ..tts import _ffmpeg_available
        return _ok("present") if _ffmpeg_available() else _degraded(
            "missing — Telegram voice replies fall back to WAV"
        )
    except Exception as e:
        return _down(f"ffmpeg probe failed: {e}")


def _check_telegram() -> dict:
    """Aggregate state of every telegram-typed channel. `ok` when at
    least one bot is configured AND running; `degraded` when one is
    configured but stopped; `not_configured` when there are none."""
    try:
        from ..channels import CHANNELS, get_channels
        tg = [c for c in get_channels() if c.get("type") == "telegram"]
        if not tg:
            return _not_configured("no telegram channels configured")
        statuses = CHANNELS.status_all() or {}
        running = [c for c in tg if statuses.get(c["id"]) == "running"]
        if running:
            return _ok(
                f"{len(running)}/{len(tg)} running"
            )
        return _degraded(
            f"0/{len(tg)} running — start with `hrant chat` or via WebUI"
        )
    except Exception as e:
        return _down(f"telegram check failed: {e}")


def _check_workspace() -> dict:
    """Can we write to workspace/? `down` would mean the agent can't
    take any uploads or write notes."""
    try:
        from ..workspace import INBOX, NOTES, OUTBOX, TURNS, get_workspace
        ws = get_workspace()
        if not ws.root.exists():
            return _down(f"workspace root missing: {ws.root}")
        # Tiny write-probe — most likely failure on shared boxes is a
        # permission flip after a deploy.
        probe = ws.root / ".health_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except Exception as e:
            return _down(f"workspace not writable: {e}")
        sub_counts = {}
        for sub in (INBOX, OUTBOX, NOTES, TURNS):
            d = ws.root / sub
            if d.exists():
                sub_counts[sub] = sum(
                    1 for p in d.iterdir()
                    if p.is_file() and not p.name.endswith(".meta.json")
                )
        return _ok(f"counts={sub_counts}")
    except Exception as e:
        return _down(f"workspace check failed: {e}")


# Worst-of rollup ordering. Indices later → "worse" in the aggregate.
# not_configured does NOT drag the aggregate down — see module docstring.
_RANK = {"ok": 0, "degraded": 1, "down": 2}


def _aggregate(components: dict[str, dict]) -> str:
    worst = "ok"
    for c in components.values():
        s = c.get("status", "ok")
        if s == "not_configured":
            continue
        if _RANK.get(s, 0) > _RANK[worst]:
            worst = s
    return worst


@router.get("/api/health")
def health() -> dict[str, Any]:
    """Operational health check. Suitable for `docker healthcheck`,
    a systemd `ExecStartPost`, or a Tailscale Funnel sidekick.

    Response shape:
        {
          "status": "ok|degraded|down",
          "version": "0.1.0",
          "uptime_seconds": 1234,
          "components": {
            "agent_core": {"status": "ok", "detail": ""},
            "model":      {"status": "ok", "detail": "active=…"},
            "stt":        {"status": "not_configured", "detail": "…"},
            "tts":        {"status": "ok", "detail": "url=…"},
            "ffmpeg":     {"status": "degraded", "detail": "missing"},
            "telegram":   {"status": "ok", "detail": "1/1 running"},
            "workspace":  {"status": "ok", "detail": "counts={…}"}
          }
        }
    """
    from ..cli import VERSION
    components = {
        "agent_core": _check_agent_core(),
        "model": _check_model(),
        "stt": _check_stt(),
        "tts": _check_tts(),
        "ffmpeg": _check_ffmpeg(),
        "telegram": _check_telegram(),
        "workspace": _check_workspace(),
    }
    return {
        "status": _aggregate(components),
        "version": VERSION,
        "uptime_seconds": int(time.time() - _BOOT_TIME),
        "platform": f"{platform.system()} {platform.release()}",
        "components": components,
    }
