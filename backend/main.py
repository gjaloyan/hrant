"""FastAPI entry point.

Endpoint groups live in `backend/api/` — each module exports an
`APIRouter` mounted below. Autonomic endpoints live alongside the
autonomic engine in `backend/autonomic/api.py` and follow the same
pattern.
"""
from __future__ import annotations
import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Bridge stdlib logging into the LogBus so the WebUI Logs tab sees
# every `log.info(...)` / `log.warning(...)` call as a unified event.
# Idempotent — if main is reloaded under pytest, don't double-attach.
from .log_bus import LogBusHandler as _LogBusHandler  # noqa: E402
_root_logger = logging.getLogger()
if not any(isinstance(h, _LogBusHandler) for h in _root_logger.handlers):
    _root_logger.addHandler(_LogBusHandler())

# Audit P0 #2 fix: suppress INFO-level logs from HTTP-client libraries.
# Default INFO from httpx/httpcore/telegram leaks the full request URL,
# which for the Telegram polling loop embeds the bot token in plaintext:
#   POST https://api.telegram.org/bot<TOKEN>/getUpdates "HTTP/1.1 200 OK"
# Anyone with access to the systemd journal then has the bot token. Set
# these loggers to WARNING so 4xx/5xx errors still surface but per-call
# URL logs don't. If detailed httpx tracing is ever needed, override via
# LOG_LEVEL_HTTPX env var (e.g. LOG_LEVEL_HTTPX=DEBUG for one-off debug).
import os as _os_for_log_levels
_HTTPX_LEVEL = _os_for_log_levels.environ.get("LOG_LEVEL_HTTPX", "WARNING").upper()
for _noisy in ("httpx", "httpcore", "telegram", "telegram.ext.Updater"):
    logging.getLogger(_noisy).setLevel(_HTTPX_LEVEL)


# Snapshot of root + per-module log levels taken once at boot,
# BEFORE any pipeline profile is applied. Used by
# `_apply_logging_overrides` to restore modules to their boot state
# when a profile switch drops them from its `modules` overlay —
# without this, switching from `development` (root=DEBUG +
# backend.unified_agent=DEBUG) to `default` (no logging overlay)
# would leave both loggers stuck at DEBUG forever, growing disk
# usage in logs/ unbounded. Audit Important #10 (2026-05-23).
_LOG_LEVEL_BOOT_SNAPSHOT: dict[str, int] = {}

# Modules touched by the previous overlay — we need to remember the
# set so the next apply knows which modules to RESET vs which to
# leave alone (don't reset modules the user / other code raised
# manually after boot).
_LOG_LEVEL_OVERLAY_MODULES: set[str] = set()


def _snapshot_log_levels() -> None:
    """Capture the boot-time root + per-module levels. Called once."""
    _LOG_LEVEL_BOOT_SNAPSHOT["__root__"] = logging.getLogger().level
    # Snapshot only loggers that exist — pre-created ones for noisy
    # libs (httpx, telegram, etc.) and root itself. Unknown modules
    # default to NOTSET (which inherits root) and we'll capture them
    # lazily the first time an overlay touches them.
    for name in logging.root.manager.loggerDict:  # type: ignore[attr-defined]
        logger = logging.getLogger(name)
        _LOG_LEVEL_BOOT_SNAPSHOT[name] = logger.level


def _apply_logging_overrides(overrides: dict | None) -> None:
    """Apply per-module log levels from the active pipeline profile.

    On every call, restore any module that was raised by a PREVIOUS
    overlay but is not mentioned by this one — otherwise switching
    profile X (root=DEBUG, unified_agent=DEBUG) → profile Y (no
    overlay) would leave both loggers stuck at DEBUG.
    """
    global _LOG_LEVEL_OVERLAY_MODULES

    # Restore root + previously-touched modules to boot snapshot
    # before applying the new overlay. Modules not in the snapshot
    # (created after boot) reset to NOTSET so they inherit root.
    root_default = _LOG_LEVEL_BOOT_SNAPSHOT.get("__root__", logging.INFO)
    logging.getLogger().setLevel(root_default)
    for module in _LOG_LEVEL_OVERLAY_MODULES:
        prev = _LOG_LEVEL_BOOT_SNAPSHOT.get(module, logging.NOTSET)
        logging.getLogger(module).setLevel(prev)
    _LOG_LEVEL_OVERLAY_MODULES = set()

    if not overrides:
        return
    root_level = overrides.get("root")
    if root_level:
        logging.getLogger().setLevel(root_level)
    for module, level in (overrides.get("modules") or {}).items():
        if level:
            # Snapshot the module's CURRENT level the first time we
            # touch it, so we know what to restore later — but only
            # if it wasn't already in the boot snapshot.
            if module not in _LOG_LEVEL_BOOT_SNAPSHOT:
                _LOG_LEVEL_BOOT_SNAPSHOT[module] = logging.getLogger(module).level
            logging.getLogger(module).setLevel(level)
            _LOG_LEVEL_OVERLAY_MODULES.add(module)


# Boot apply: snapshot log levels, seed the five starter profiles
# on first run, then apply the active profile's logging_overrides.
# Wrapped in try/except so a transient store failure doesn't block
# FastAPI startup.
_snapshot_log_levels()
try:
    from .pipeline_profile import (
        active_overrides as _po_active_overrides,
        seed_starter_profiles as _po_seed,
    )
    _po_seed()
    _apply_logging_overrides(_po_active_overrides().get("logging_overrides"))
except Exception as _e:
    logging.getLogger(__name__).warning(
        "pipeline profile boot apply failed: %s", _e,
    )

from .config import CONFIG
from .channels import CHANNELS, get_channels
from .runtime_config import apply_overrides_from_file
from .autonomic.startup import (
    build_scheduler,
    start_autonomic_scheduler,
    stop_autonomic_scheduler,
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    # --- startup ---
    # Apply user's runtime overrides (router budget, verification knobs, …)
    # BEFORE anything else looks at CONFIG. Otherwise the autonomic scheduler
    # / channels caching CONFIG.router etc. would see pre-override defaults.
    try:
        applied = apply_overrides_from_file()
        if applied:
            log.info("runtime overrides applied: %s", list(applied.keys()))
    except Exception as e:
        log.warning("could not apply runtime overrides: %s", e)

    # Audit #18: run job recovery in the BACKGROUND so port-bind
    # isn't blocked by it. The previous version walked every file
    # in jobs/ synchronously here — at 100 jobs ~50ms (fine), at
    # 30k jobs ~15s (port stays closed the whole time). The new
    # task runs after the FastAPI server is accepting requests.
    # In-flight requests can't race with recovery because the
    # recovery only flips state for jobs marked `running` from a
    # PREVIOUS process — those can't be currently running anymore.
    async def _recover_jobs_background():
        recovered: list[str] = []
        try:
            from . import jobs as _jobs
            recovered = await asyncio.to_thread(_jobs.JOBS.recover_interrupted)
            if recovered:
                log.info(
                    "Job recovery: %d turn(s) marked interrupted",
                    len(recovered),
                )
        except Exception as e:
            log.warning("Job recovery error: %s", e)
        return recovered

    _recovery_task = asyncio.create_task(_recover_jobs_background())
    application.state.consolidation_recovery_task = _recovery_task

    # Phase B+C migration: one-shot copy of any legacy
    # channels.json::allowed_users entries into roles.json::trusted.
    # Idempotent — internal marker prevents re-runs. MUST run before
    # channels start so the bots see the unified state on their
    # first message.
    try:
        from . import access as _access
        mig = _access.migrate_legacy_allowed_users()
        if mig.get("migrated"):
            log.info("access migration: promoted %d telegram user(s) to trusted", mig["migrated"])
    except Exception as e:
        log.warning("access migration error: %s", e)

    # T6: any background job left in 'running' state must have died
    # when the service stopped (subprocess child of the previous pid).
    # Flip those to 'interrupted' so the registry doesn't lie.
    try:
        from .tools import background_jobs as _bg
        n_interrupted = _bg.STORE.mark_interrupted_on_startup()
        if n_interrupted:
            log.info(
                "background-jobs: marked %d running-on-shutdown job(s) "
                "as interrupted", n_interrupted,
            )
    except Exception as e:
        log.warning("background-jobs interrupted-sweep error: %s", e)

    # Audit T6 Phase 2: start the background-job watchdog. One
    # daemon thread that ticks every 60s and checks running jobs
    # for vanished PIDs, progress milestones, and stale
    # (no log growth) state. Idempotent — safe across reloads.
    try:
        from . import bg_job_watchdog as _bgw
        _bgw.start_watchdog()
    except Exception as e:
        log.warning("bg-job watchdog start failed: %s", e)

    # Audit follow-up: tighten file modes on sensitive configs that may
    # exist from before write_secret_json() landed. Best-effort; failures
    # never block startup. POSIX-only — Windows ignores os.chmod for mode
    # bits and relies on user-profile ACLs instead.
    try:
        from . import paths as _paths
        kd = _paths.knowledge_dir()
        for _fname in (
            "providers.json",
            "telegram_chat_ids.json",
            "pairing.json",
            "tts_config.json",
            "transcriber_config.json",
            "active_model.json",
            "access_log.json",
            "oauth_tokens.json",
        ):
            _paths.secure_existing_file(kd / _fname)
    except Exception as e:
        log.warning("sensitive-file chmod sweep skipped: %s", e)

    log.info("Server starting — auto-starting channels...")
    try:
        channels_list = get_channels()
        auto = [c for c in channels_list if c.get("enabled") and c.get("auto_start")]
        log.info("Found %d channel(s) with auto_start", len(auto))
        for ch in auto:
            try:
                result = CHANNELS.start_channel(ch["id"])
                log.info("Auto-start channel %s: %s", ch["id"], result)
            except Exception as e:
                log.error("Failed to auto-start channel %s: %s", ch["id"], e)
    except Exception as e:
        log.warning("Channel auto-start error: %s", e)

    # Telegram interrupted-job notification (Phase 15A.1).
    # Waits for the background recovery task to finish, then if any
    # Telegram-channel jobs got marked interrupted, sends the user a
    # heads-up so they don't experience silent message loss across a
    # server restart.
    async def _notify_telegram_after_recovery() -> None:
        try:
            recovered_ids = await _recovery_task
        except Exception as e:
            log.warning("recovery task crashed: %s", e)
            return
        if not recovered_ids:
            return
        try:
            from . import jobs as _jobs
            tg_interrupted = [
                j for j in (_jobs.JOBS.get(jid) for jid in recovered_ids)
                if j is not None
                and j.channel == "telegram"
                and (j.reply_to or {}).get("telegram_chat_id")
            ]
            if not tg_interrupted:
                return
            # Let the Telegram bot's event loop finish wiring up.
            # send_text returns False until then.
            await asyncio.sleep(3.0)
            for j in tg_interrupted:
                try:
                    chat_id = int(j.reply_to["telegram_chat_id"])
                except (TypeError, ValueError):
                    continue
                preview = (j.prompt or "").replace("\n", " ").strip()
                if len(preview) > 200:
                    preview = preview[:200] + "…"
                msg = (
                    "⚠️ I was interrupted earlier when you asked:\n\n"
                    f"«{preview}»\n\n"
                    f"Job id: `{j.id}` — open the WebUI Jobs tab "
                    "to retry, or just resend your message."
                )
                try:
                    CHANNELS.send_to_telegram_chat(chat_id, msg)
                except Exception as e:
                    log.warning(
                        "TG interrupted notify failed for %s: %s", j.id, e,
                    )
            log.info(
                "Telegram: scheduled %d interrupted-job notification(s)",
                len(tg_interrupted),
            )
        except Exception as e:
            log.warning("Telegram interrupted-notify scheduler error: %s", e)

    application.state.consolidation_notify_task = (
        asyncio.create_task(_notify_telegram_after_recovery())
    )

    bundle = build_scheduler()
    application.state.autonomic_bundle = bundle
    application.state.autonomic_scheduler = bundle.scheduler
    application.state.autonomic_gate = bundle.gate
    application.state.autonomic_executor = bundle.executor
    application.state.autonomic_builder = bundle.builder
    application.state.autonomic_registry = bundle.registry
    application.state.autonomic_kill_switch = bundle.kill_switch
    application.state.autonomic_lever_log = bundle.lever_log_path
    application.state.autonomic_tick_log = bundle.tick_log_path
    await start_autonomic_scheduler(bundle)

    # Phase 16A: daily memory consolidation scheduler. Adaptive —
    # fires when idle for 15min AND >=24h since last run.
    try:
        from .consolidation import scheduler as _cons_sched
        await _cons_sched.start_scheduler(application)
        log.info("Consolidation scheduler started")
    except Exception as e:
        log.warning("Consolidation scheduler failed to start: %s", e)

    # Embedder startup probe + coverage report. Silent backend
    # failures used to hide here — the audit 2026-05-27 found 13/16
    # notes embedded because llama-cpp was unreachable at note-create
    # time and nothing logged. Probe + WARN early so the operator
    # can see it; the FIRE_EMBEDDING_BACKFILL lever fixes the gap
    # on next tick.
    try:
        from .embedder import EMBEDDER
        from .embedding_backfill import missing_count as _emb_missing
        EMBEDDER.reset()
        status = EMBEDDER.status()
        backend = status.get("backend")
        if backend in (None, "disabled"):
            log.warning(
                "Embedder unavailable (backend=%r, last_error=%r). "
                "Semantic search disabled; FIRE_EMBEDDING_BACKFILL will "
                "retry. Configure via /api/memory/embeddings/config.",
                backend, status.get("last_error"),
            )
        else:
            try:
                coverage = _emb_missing()
            except Exception as cov_exc:
                coverage = {"missing": "?", "embedded": "?", "error": str(cov_exc)}
            log.info(
                "Embedder ready: backend=%s model=%s dim=%s "
                "coverage=%s/%s embedded (missing=%s).",
                backend, status.get("model"), status.get("dim"),
                coverage.get("embedded"), coverage.get("total_notes"),
                coverage.get("missing"),
            )
            missing = coverage.get("missing")
            if isinstance(missing, int) and missing > 0:
                log.warning(
                    "Embedder has %d notes without vectors. The "
                    "FIRE_EMBEDDING_BACKFILL lever will catch them "
                    "on its next tick.", missing,
                )
    except Exception as e:
        log.warning("Embedder startup probe failed: %s", e)

    yield
    # --- shutdown ---
    log.info("Server shutting down — stopping channels...")
    CHANNELS.stop_all()
    await stop_autonomic_scheduler(application.state.autonomic_bundle)
    try:
        from .consolidation import scheduler as _cons_sched
        await _cons_sched.stop_scheduler(application)
    except Exception as e:
        log.warning("Consolidation scheduler shutdown error: %s", e)


app = FastAPI(title="Self-Learning Agent", lifespan=lifespan)


# Audit P0 #1 fix: HTTP speaker-context middleware.
#
# The `require_owner_for_writes` gate trusts a None ContextVar
# (CLI / tests / autonomic). Before this middleware, an HTTP request
# would hit a write endpoint BEFORE `agent.run()` had a chance to set
# the ContextVar, so sp was None → gate passed → unauthenticated
# request slipped through. The gate is only safe when bind is
# 127.0.0.1; flipping to 0.0.0.0 (e.g. via `hrant gateway start --gateway`)
# would open every mutating endpoint to the LAN.
#
# Fix: this middleware sets the ContextVar for EVERY HTTP request
# BEFORE the route handler runs:
#   - local requests (127.0.0.1, ::1, ::ffff:127.0.0.1) → trusted
#     as the WebUI owner. The server is bound to 127.0.0.1 by default;
#     anything reaching us via localhost is the local owner using the
#     WebUI.
#   - remote requests (anything else) → "http:remote-anonymous", which
#     is NOT in `roles.json` owners, so `require_owner_for_writes` will
#     fail-closed with 403.
#
# Defence in depth: even if a future code path bypasses this middleware,
# the auth gate inside the route handler still fails-closed because the
# default speaker is "remote-anonymous" not None.

_LOCAL_REQUEST_HOSTS = frozenset({
    "127.0.0.1", "::1", "::ffff:127.0.0.1", "localhost",
    # FastAPI TestClient identifies as "testclient" — trusted by
    # construction (it only runs in pytest sessions).
    "testclient",
})


@app.middleware("http")
async def _speaker_context_middleware(request, call_next):
    from .roles import set_current_speaker, reset_current_speaker
    client = getattr(request, "client", None)
    host = (getattr(client, "host", None) or "").strip()
    if host in _LOCAL_REQUEST_HOSTS:
        # Local WebUI / curl from the box itself — trust as owner.
        speaker_id = "webui:default"
    else:
        # Remote IP reaching us — only possible when bind is 0.0.0.0
        # AND the LAN has access. Default to anonymous; require_owner
        # will refuse.
        speaker_id = f"http:remote-anonymous-{host or 'unknown'}"
    token = set_current_speaker(speaker_id)
    try:
        return await call_next(request)
    finally:
        reset_current_speaker(token)


# Audit P0 #1 fix: narrow CORS. `*` plus the auth-bypass was the
# combo that made the unauthenticated-mutation risk real. We bind to
# 127.0.0.1 by default and serve the WebUI from the same origin, so
# the browser doesn't need cross-origin permissions for the API. If
# someone deploys behind a reverse proxy on a different origin, they
# can extend this list via HRANT_CORS_ORIGINS (comma-separated).
import os as _os_for_cors
_CORS_DEFAULTS = [
    "http://127.0.0.1:3333",
    "http://localhost:3333",
    "http://127.0.0.1:5173",  # vite dev server
    "http://localhost:5173",
]
_cors_env = (_os_for_cors.environ.get("HRANT_CORS_ORIGINS") or "").strip()
_cors_allowed = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env else _CORS_DEFAULTS
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_allowed,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- mount routers ----------
from .api import (  # noqa: E402
    chat,
    knowledge,
    projects,
    finetune,
    status as status_api,
    identity,
    intel,
    goals,
    sessions,
    providers as providers_api,
    channels as channels_api,
    attachments as attachments_api,
    health as health_api,
    voice as voice_api,
    engine as engine_api,
    roles as roles_api,
    skills as skills_api,
    jobs as jobs_api,
    failover as failover_api,
    consolidation as consolidation_api,
    graph as graph_api,
    subagents as subagents_api,
    background_jobs as background_jobs_api,
    reasoning_routing as reasoning_routing_api,
    logs as logs_api,
    pipeline_profiles as pipeline_profiles_api,
)
from .autonomic.api import router as autonomic_router  # noqa: E402

for mod in (
    chat, knowledge, projects, finetune, status_api, identity,
    intel, goals, sessions, providers_api, channels_api, attachments_api,
    health_api, voice_api, engine_api, roles_api, skills_api, jobs_api,
    failover_api, consolidation_api, graph_api, subagents_api,
    background_jobs_api, reasoning_routing_api,
    logs_api, pipeline_profiles_api,
):
    app.include_router(mod.router)
app.include_router(autonomic_router)


# ---------- frontend static files ----------
_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"

if _FRONTEND_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="static")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = _FRONTEND_DIST / full_path
        if full_path and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(_FRONTEND_DIST / "index.html")


def serve() -> None:
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=CONFIG.server["host"],
        port=CONFIG.server["port"],
        reload=False,
    )


if __name__ == "__main__":
    serve()
