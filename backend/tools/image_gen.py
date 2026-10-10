"""Image generation and editing through the ChatGPT/Codex subscription.

Ported from Hermes Agent's `plugins/image_gen/openai-codex` (MIT, Nous
Research): the same native Codex `images/generations` and `images/edits`
endpoints the official Codex client uses, with `gpt-image-2`. No API key —
the OAuth login the chat provider already uses pays for it.

Why it exists here (2026-09-28, the owner's last conversation before the
review): three room photos, "design this room, keep the concrete ceiling",
then "can you generate the pictures?". The agent had no way to make an
image. It spent 32 LLM calls looking for ComfyUI, opened chatgpt.com in a
headless browser, read another agent's skill file, and ended by asking the
owner which image service to use.

Edits take the user's own photos as `input_images` (attachment sha256 or a
file path), which is exactly the "redesign this room" case.
"""
from __future__ import annotations

import base64
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

API_MODEL = "gpt-image-2"
DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
SIZES = {"square": "1024x1024", "landscape": "1536x1024", "portrait": "1024x1536"}
QUALITIES = ("low", "medium", "high")
MAX_INPUT_IMAGES = 16
MAX_INPUT_BYTES = 25 * 1024 * 1024
_ACCEPTED_MIME = {"image/png", "image/jpeg", "image/gif", "image/webp"}


def _sniff_mime(raw: bytes) -> Optional[str]:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def _image_bytes(ref: str) -> tuple[bytes, str]:
    """Bytes of one input image: an attachment sha256 or a local path."""
    ref = (ref or "").strip()
    if not ref:
        raise ValueError("blank image reference")
    from ..attachments import ATTACHMENTS
    meta = ATTACHMENTS.get_meta(ref) if len(ref) == 64 else None
    if meta is not None:
        data = ATTACHMENTS.get_bytes(ref)
        if data is None:
            raise ValueError(f"attachment {ref[:12]}… has no stored bytes")
        name = ref[:12]
    else:
        p = Path(ref).expanduser()
        if not p.is_file():
            raise ValueError(f"not an attachment sha256 and not a file: {ref}")
        data = p.read_bytes()
        name = p.name
    if not data:
        raise ValueError(f"image {name} is empty")
    if len(data) > MAX_INPUT_BYTES:
        raise ValueError(f"image {name} is larger than 25 MB")
    mime = _sniff_mime(data)
    if mime not in _ACCEPTED_MIME:
        raise ValueError(f"image {name} is not PNG/JPEG/GIF/WEBP")
    return data, mime


def _input_image_entry(ref: str) -> dict:
    data, mime = _image_bytes(ref)
    return {"image_url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}


def build_request(*, prompt: str, size: str, quality: str,
                  input_images: Optional[list[dict]] = None) -> tuple[str, dict]:
    """(endpoint path, JSON body) — `images/edits` when there are inputs."""
    body: dict[str, Any] = {
        "prompt": prompt, "model": API_MODEL, "n": 1,
        "quality": quality, "size": size, "background": "opaque",
    }
    if input_images:
        body["images"] = input_images
        return "images/edits", body
    return "images/generations", body


def _base_url() -> str:
    """The Codex provider's base URL, so images go where chat goes."""
    try:
        from ..providers import get_providers
        for p in get_providers() or []:
            if p.get("auth_type") == "codex_subscription" and p.get("enabled", True):
                return (p.get("base_url") or DEFAULT_BASE_URL).rstrip("/")
    except Exception:
        pass
    return DEFAULT_BASE_URL


def _headers(token: str, account_id: str) -> dict:
    # OpenAI asks third-party harnesses to identify themselves rather than
    # pass as the official client; Hermes does the same.
    try:
        from ..version import get_version
        version = get_version()
    except Exception:
        version = "0"
    h = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": f"HrantAgent/{version}",
        "originator": "hrant-agent",
        "x-codex-image-turn-id": str(uuid.uuid4()),
    }
    if account_id:
        h["ChatGPT-Account-ID"] = account_id
    return h


def _post(path: str, body: dict) -> dict:
    """POST with the Codex login; one forced token refresh on a 401."""
    import httpx
    from ..providers import CODEX_AUTH

    timeout = httpx.Timeout(300.0, connect=30.0, read=300.0, write=60.0, pool=30.0)
    url = f"{_base_url()}/{path}"
    token, account_id = CODEX_AUTH.get_access_token()
    for attempt in range(2):
        r = httpx.post(url, json=body, headers=_headers(token, account_id), timeout=timeout)
        if r.status_code == 401 and attempt == 0:
            token, account_id = CODEX_AUTH.force_refresh()
            continue
        if r.status_code >= 400:
            text = r.text or ""
            try:
                msg = (r.json().get("error") or {}).get("message") or text
            except Exception:
                msg = text
            raise RuntimeError(f"Codex images API HTTP {r.status_code}: {str(msg)[:400]}")
        data = r.json()
        if not isinstance(data, dict):
            raise RuntimeError("Codex images API returned a non-object body")
        data["imagegen_request_id"] = r.headers.get("x-codex-imagegen-request-id")
        return data
    raise RuntimeError("Codex images API: still 401 after refreshing the login")


def generate_image(prompt: str, aspect: str = "square", quality: str = "medium",
                   input_images: Optional[list] = None) -> dict:
    """Generate (or, with `input_images`, edit) one image. Returns a dict:
    ok, path, sha256, media_line — or ok=False with `error`."""
    prompt = (prompt or "").strip()
    if not prompt:
        return {"ok": False, "error": "prompt is required"}
    aspect = aspect if aspect in SIZES else "square"
    quality = quality if quality in QUALITIES else "medium"
    refs = [r for r in (input_images or []) if isinstance(r, str) and r.strip()]
    if len(refs) > MAX_INPUT_IMAGES:
        return {"ok": False, "error": f"at most {MAX_INPUT_IMAGES} input images"}
    try:
        entries = [_input_image_entry(r) for r in refs]
    except ValueError as e:
        return {"ok": False, "error": f"input image: {e}"}
    path, body = build_request(prompt=prompt, size=SIZES[aspect], quality=quality,
                               input_images=entries or None)
    try:
        payload = _post(path, body)
    except Exception as e:
        log.warning("generate_image failed: %s", e)
        return {"ok": False, "error": str(e)[:500]}
    items = payload.get("data")
    b64 = (items[0].get("b64_json")
           if isinstance(items, list) and items and isinstance(items[0], dict) else None)
    if not isinstance(b64, str) or not b64:
        return {"ok": False, "error": "the images API returned no image data"}
    raw = base64.b64decode(b64)
    try:
        from ..workspace import get_workspace
        out = get_workspace().save_outbox(f"image_{uuid.uuid4().hex[:8]}.png", raw)
    except Exception as e:
        return {"ok": False, "error": f"could not save the image: {e}"}
    sha = ""
    try:
        from ..attachments import ATTACHMENTS
        sha = ATTACHMENTS.save(raw, "image/png", filename=out.name, kind="image").sha256
    except Exception as e:
        log.debug("generate_image: attachment registration failed: %s", e)
    return {
        "ok": True,
        "path": str(out),
        "sha256": sha,
        "media_line": f"MEDIA:{out}",
        "mode": "edit" if entries else "generate",
        "input_image_count": len(entries),
        "size": SIZES[aspect],
        "quality": quality,
        "reported_size": payload.get("size"),
        "imagegen_request_id": payload.get("imagegen_request_id"),
        "note": ("Deliver it by putting the media_line on its own line in your "
                 "answer. To look at the result, call analyze_image with the sha256."),
    }


def generate_image_json(**kw) -> str:
    return json.dumps(generate_image(**kw), ensure_ascii=False)
