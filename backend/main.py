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
from .autonomic.startup import (
    build_scheduler,
    start_autonomic_scheduler,
    stop_autonomic_scheduler,
)


@asynccontextmanager
async def lifespan(application: FastAPI):
    # --- startup ---
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

    yield
    # --- shutdown ---
    log.info("Server shutting down — stopping channels...")
    CHANNELS.stop_all()
    await stop_autonomic_scheduler(application.state.autonomic_bundle)


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
)
from .autonomic.api import router as autonomic_router  # noqa: E402

for mod in (
    chat, knowledge, projects, finetune, status_api, identity,
    intel, goals, sessions, providers_api, channels_api, attachments_api,
    health_api, voice_api,
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
