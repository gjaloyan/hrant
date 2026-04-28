"""Channels (Telegram, etc.) — CRUD + start/stop/test."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..channels import CHANNELS, delete_channel, get_channel, get_channels, save_channel

router = APIRouter()


@router.get("/api/channels")
def list_channels():
    channels = get_channels()
    runtime = CHANNELS.status_all()
    for ch in channels:
        ch["runtime_status"] = runtime.get(ch["id"], "stopped")
    return {"channels": channels}


@router.get("/api/channels/{channel_id}")
def get_channel_api(channel_id: str):
    ch = get_channel(channel_id)
    if not ch:
        raise HTTPException(404, "channel not found")
    ch["runtime_status"] = CHANNELS.channel_status(channel_id)
    return ch


class ChannelCreateRequest(BaseModel):
    id: str
    name: str
    type: str  # "telegram"
    enabled: bool = False
    auto_start: bool = False
    config: dict = {}


@router.post("/api/channels")
def create_channel(body: ChannelCreateRequest):
    return save_channel(body.model_dump())


class ChannelUpdateRequest(BaseModel):
    name: str | None = None
    enabled: bool | None = None
    auto_start: bool | None = None
    config: dict | None = None


@router.put("/api/channels/{channel_id}")
def update_channel(channel_id: str, body: ChannelUpdateRequest):
    ch = get_channel(channel_id)
    if not ch:
        raise HTTPException(404, "channel not found")
    if body.name is not None:
        ch["name"] = body.name
    if body.enabled is not None:
        ch["enabled"] = body.enabled
    if body.auto_start is not None:
        ch["auto_start"] = body.auto_start
    if body.config is not None:
        ch["config"] = body.config
    save_channel(ch)
    return ch


@router.delete("/api/channels/{channel_id}")
def delete_channel_api(channel_id: str):
    CHANNELS.stop_channel(channel_id)
    if not delete_channel(channel_id):
        raise HTTPException(404, "channel not found")
    return {"ok": True}


@router.post("/api/channels/{channel_id}/start")
def start_channel(channel_id: str):
    result = CHANNELS.start_channel(channel_id)
    if not result["ok"]:
        raise HTTPException(400, result["error"])
    return result


@router.post("/api/channels/{channel_id}/stop")
def stop_channel(channel_id: str):
    return CHANNELS.stop_channel(channel_id)


@router.post("/api/channels/{channel_id}/test")
async def test_channel(channel_id: str):
    """Test channel connection (validate token, etc.)."""
    ch = get_channel(channel_id)
    if not ch:
        raise HTTPException(404, "channel not found")

    if ch["type"] == "telegram":
        token = ch.get("config", {}).get("bot_token", "")
        if not token:
            return {"ok": False, "error": "No bot token configured"}
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
                data = resp.json()
                if data.get("ok"):
                    bot_info = data["result"]
                    return {
                        "ok": True,
                        "bot_name": bot_info.get("first_name", ""),
                        "bot_username": bot_info.get("username", ""),
                    }
                return {"ok": False, "error": data.get("description", "Unknown error")}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return {"ok": False, "error": f"Unknown channel type: {ch['type']}"}
