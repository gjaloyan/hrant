"""FastAPI entry point.

Endpoint groups live in `backend/api/` — each module exports an
`APIRouter` mounted below. Autonomic endpoints live alongside the
autonomic engine in `backend/autonomic/api.py` and follow the same
pattern.
"""
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

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

    import asyncio as _asyncio_for_recovery
    _recovery_task = _asyncio_for_recovery.create_task(_recover_jobs_background())
    application.state.consolidation_recovery_task = _recovery_task

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
            await _asyncio_for_recovery.sleep(3.0)
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
        _asyncio_for_recovery.create_task(_notify_telegram_after_recovery())
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
)
from .autonomic.api import router as autonomic_router  # noqa: E402

for mod in (
    chat, knowledge, projects, finetune, status_api, identity,
    intel, goals, sessions, providers_api, channels_api, attachments_api,
    health_api, voice_api, engine_api, roles_api, skills_api, jobs_api,
    failover_api, consolidation_api, graph_api, subagents_api,
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
