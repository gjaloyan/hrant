"""LLM provider catalog + active-model selection + OAuth wizard."""
from __future__ import annotations
import secrets

import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..config import CONFIG
from ..providers import (
    ACTIVE_MODEL,
    AUTH_TYPES,
    CODEX_AUTH,
    COPILOT_AUTH,
    KNOWN_PRICING,
    OAUTH_PRESETS,
    OAUTH_TOKENS,
    PROVIDER_CONNECT_INFO,
    PROVIDER_TYPES,
    _pkce_store,
    delete_provider,
    generate_pkce,
    get_api_key,
    get_available_models,
    get_provider,
    get_providers,
    save_provider,
    test_provider as run_provider_test,
)
from ._auth import require_owner_for_writes

router = APIRouter()


# ---- list / catalog endpoints ----
@router.get("/api/providers")
def list_providers():
    providers = get_providers()
    for p in providers:
        key = p.get("api_key", "")
        if key and len(key) > 8:
            p["api_key_masked"] = "••••" + key[-6:]
        else:
            p["api_key_masked"] = "(env)" if get_api_key(p) else "(not set)"
        p.pop("api_key", None)
        oauth = p.get("oauth") or {}
        if oauth.get("client_secret"):
            oauth["client_secret_masked"] = "••••" + oauth["client_secret"][-4:]
            del oauth["client_secret"]
        if p.get("auth_type") == "oauth":
            p["oauth_status"] = OAUTH_TOKENS.status(p["id"])
    return {"providers": providers, "types": PROVIDER_TYPES}


@router.get("/api/providers/types")
def provider_types():
    return {"types": PROVIDER_TYPES, "pricing": KNOWN_PRICING}


@router.get("/api/providers/connect-info")
def provider_connect_info():
    return {"connect_info": PROVIDER_CONNECT_INFO}


@router.get("/api/providers/auth-types")
def get_auth_types():
    return {"auth_types": AUTH_TYPES, "oauth_presets": OAUTH_PRESETS}


@router.get("/api/providers/codex/status")
def codex_subscription_status():
    return CODEX_AUTH.status()


@router.get("/api/providers/copilot/status")
def copilot_subscription_status():
    return COPILOT_AUTH.status()


@router.get("/api/providers/codex/models")
def codex_subscription_models():
    fallback = (PROVIDER_TYPES.get("openai_codex", {}) or {}).get("models", [])
    return CODEX_AUTH.models(fallback=fallback)


# ---- OAuth callback (must be before {provider_id} route) ----
@router.get("/api/providers/oauth/callback")
async def oauth_callback(code: str = "", state: str = "", error: str = ""):
    if error:
        html = f"""
        <html><body style="background:#1e293b;color:#e2e8f0;font-family:sans-serif;text-align:center;padding-top:100px">
        <h2>OAuth Error</h2><p>{error}</p>
        <script>setTimeout(()=>window.close(),5000)</script>
        </body></html>
        """
        return HTMLResponse(html, status_code=400)

    if not code or not state:
        raise HTTPException(400, "Missing code or state")

    provider_id = state.rsplit("_", 1)[0] if "_" in state else state
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")

    oauth = p.get("oauth") or {}
    redirect_uri = oauth.get(
        "redirect_uri",
        f"http://localhost:{CONFIG.server['port']}/api/providers/oauth/callback",
    )

    pkce_verifier = _pkce_store.pop(state, None)
    result = OAUTH_TOKENS.exchange_code(provider_id, code, redirect_uri, pkce_verifier=pkce_verifier)

    if result.get("ok"):
        html = """
        <html><body style="background:#1e293b;color:#e2e8f0;font-family:sans-serif;text-align:center;padding-top:100px">
        <h2 style="color:#34d399">Connected!</h2>
        <p>OAuth token received. You can close this tab.</p>
        <script>setTimeout(()=>window.close(),2000)</script>
        </body></html>
        """
        return HTMLResponse(html)
    html = f"""
    <html><body style="background:#1e293b;color:#e2e8f0;font-family:sans-serif;text-align:center;padding-top:100px">
    <h2 style="color:#f87171">Token Exchange Failed</h2>
    <p>{result.get('error', 'Unknown error')}</p>
    <p style="font-size:12px;opacity:0.6">Copy this page URL and paste it in the agent settings to retry.</p>
    </body></html>
    """
    return HTMLResponse(html, status_code=400)


# ---- Ollama local-models pre-routes (before /{provider_id}) ----
@router.get("/api/providers/ollama/models")
def ollama_models():
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        r.raise_for_status()
        data = r.json()
        models = [
            {
                "name": m.get("name", ""),
                "size": m.get("size", 0),
                "modified": m.get("modified_at", ""),
                "family": m.get("details", {}).get("family", ""),
                "parameters": m.get("details", {}).get("parameter_size", ""),
                "quantization": m.get("details", {}).get("quantization_level", ""),
            }
            for m in data.get("models", [])
        ]
        return {"ok": True, "models": models}
    except Exception as e:
        return {"ok": False, "models": [], "error": str(e)}


class OllamaPullRequest(BaseModel):
    model: str


@router.post("/api/providers/ollama/pull")
async def ollama_pull(body: OllamaPullRequest):
    require_owner_for_writes(action="pulling an Ollama model")
    try:
        r = httpx.post(
            "http://localhost:11434/api/pull",
            json={"name": body.model, "stream": False},
            timeout=600.0,
        )
        r.raise_for_status()
        return {"ok": True, "message": f"Model '{body.model}' pulled successfully"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.delete("/api/providers/ollama/models/{model_name:path}")
def ollama_delete_model(model_name: str):
    require_owner_for_writes(action="deleting an Ollama model")
    try:
        r = httpx.delete(
            "http://localhost:11434/api/delete",
            json={"name": model_name},
            timeout=30.0,
        )
        r.raise_for_status()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ---- active model selection ----
@router.get("/api/active-model")
def get_active_model():
    return {"active": ACTIVE_MODEL.get(), "models": get_available_models()}


class SetActiveModelRequest(BaseModel):
    provider_id: str
    model: str


@router.put("/api/active-model")
def set_active_model(req: SetActiveModelRequest):
    require_owner_for_writes(action="pinning the active model")
    try:
        result = ACTIVE_MODEL.set(req.provider_id, req.model)
        return {"ok": True, "active": result}
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.delete("/api/active-model")
def clear_active_model():
    require_owner_for_writes(action="clearing the active model")
    ACTIVE_MODEL.clear()
    return {"ok": True}


# ---- generic /{provider_id} CRUD (must come AFTER specific routes above) ----
@router.get("/api/providers/{provider_id}")
def get_provider_api(provider_id: str):
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")
    key = p.get("api_key", "")
    if key and len(key) > 8:
        p["api_key_masked"] = "••••" + key[-6:]
    else:
        p["api_key_masked"] = "(env)" if get_api_key(p) else "(not set)"
    p.pop("api_key", None)
    return p


class ProviderCreateRequest(BaseModel):
    id: str
    name: str
    type: str
    enabled: bool = True
    api_key: str = ""
    api_key_env: str = ""
    base_url: str = ""
    models: list[str] = []
    default_model: str = ""
    max_tokens: int = 2000
    temperature: float = 0.3
    auth_type: str = "api_key"
    oauth: dict | None = None
    aws: dict | None = None  # {access_key_id, secret_access_key, region}


@router.post("/api/providers")
def create_provider(body: ProviderCreateRequest):
    # Owner-only: registering a provider with an API key is the
    # primary cost-amplification vector if the gateway is bound
    # beyond loopback. An attacker who can hit this can either
    # spend the owner's money (registering their own free-tier
    # key) or harvest the owner's prompts (registering an
    # attacker-controlled base_url).
    require_owner_for_writes(action="registering a provider")
    return {"ok": True, "provider": save_provider(body.model_dump())}


class ProviderUpdateRequest(BaseModel):
    name: str | None = None
    enabled: bool | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None
    models: list[str] | None = None
    default_model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None


@router.put("/api/providers/{provider_id}")
def update_provider_api(provider_id: str, body: ProviderUpdateRequest):
    require_owner_for_writes(action="updating a provider")
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")
    p.update(body.model_dump(exclude_none=True))
    save_provider(p)
    return {"ok": True}


@router.delete("/api/providers/{provider_id}")
def delete_provider_api(provider_id: str):
    require_owner_for_writes(action="deleting a provider")
    if not delete_provider(provider_id):
        raise HTTPException(404, "provider not found")
    return {"ok": True}


# ---- test_provider: dispatcher per provider type ----
@router.post("/api/providers/{provider_id}/test")
async def test_provider(provider_id: str):
    # Provider test fires a real LLM call to validate the credential.
    # Owner-only so an attacker can't probe what credentials work.
    require_owner_for_writes(action="testing a provider connection")
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")
    return run_provider_test(p)


# ---- OAuth: auth-config update + status + authorize-url + flow helpers ----
class OAuthConfigUpdate(BaseModel):
    auth_type: str
    oauth: dict = {}


@router.put("/api/providers/{provider_id}/auth")
def update_provider_auth(provider_id: str, body: OAuthConfigUpdate):
    require_owner_for_writes(action="changing provider auth config")
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")
    p["auth_type"] = body.auth_type
    if body.auth_type == "oauth":
        p["oauth"] = body.oauth
    save_provider(p)
    return {"ok": True}


@router.get("/api/providers/{provider_id}/oauth/status")
def oauth_status(provider_id: str):
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")
    return OAUTH_TOKENS.status(provider_id)


@router.post("/api/providers/{provider_id}/oauth/authorize-url")
def oauth_authorize_url(provider_id: str):
    require_owner_for_writes(action="starting OAuth flow")
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")
    oauth = p.get("oauth") or {}
    authorize_url = oauth.get("authorize_url", "")
    client_id = oauth.get("client_id", "")
    scope = oauth.get("scope", "")
    audience = oauth.get("audience", "")
    _pkce_val = oauth.get("pkce", False)
    use_pkce = _pkce_val is True or str(_pkce_val).lower() == "true"
    redirect_uri = oauth.get(
        "redirect_uri",
        f"http://localhost:{CONFIG.server['port']}/api/providers/oauth/callback",
    )

    if not authorize_url:
        raise HTTPException(400, "Missing authorize_url in OAuth config")

    import urllib.parse

    state = f"{provider_id}_{secrets.token_urlsafe(8)}"
    params: dict[str, str] = {
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "state": state,
    }
    if client_id:
        params["client_id"] = client_id
    if scope:
        params["scope"] = scope
    if audience:
        params["audience"] = audience

    extra_params = oauth.get("extra_params", {})
    if isinstance(extra_params, dict):
        params.update(extra_params)

    if use_pkce:
        verifier, challenge = generate_pkce()
        _pkce_store[state] = verifier
        params["code_challenge"] = challenge
        params["code_challenge_method"] = "S256"

    url = f"{authorize_url}?{urllib.parse.urlencode(params)}"

    parsed_redir = urllib.parse.urlparse(redirect_uri)
    redir_port = parsed_redir.port or 80
    if redir_port != CONFIG.server.get("port", 8000):
        _start_oauth_callback_listener(redir_port, parsed_redir.path or "/auth/callback", provider_id)

    return {"url": url, "redirect_uri": redirect_uri, "state": state, "pkce": use_pkce}


def _start_oauth_callback_listener(port: int, path: str, provider_id: str):
    """Start a temporary HTTP server on the given port to catch OAuth callback."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            import urllib.parse as up
            parsed = up.urlparse(self.path)
            if not parsed.path.rstrip("/").endswith(path.rstrip("/")):
                self.send_response(404)
                self.end_headers()
                return

            params = up.parse_qs(parsed.query)
            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]
            error = params.get("error", [None])[0]

            if error:
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f"""<html><body style="background:#1e293b;color:#e2e8f0;font-family:sans-serif;text-align:center;padding-top:100px">
                <h2 style="color:#f87171">OAuth Error</h2><p>{error}</p>
                <script>setTimeout(()=>window.close(),5000)</script></body></html>""".encode())
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return

            if not code or not state:
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"Missing code or state")
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return

            pid = state.rsplit("_", 1)[0] if "_" in state else state
            p = get_provider(pid)
            if not p:
                self.send_response(404)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"Provider not found")
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return

            oauth = p.get("oauth") or {}
            redir = oauth.get("redirect_uri", f"http://localhost:{port}{path}")
            pkce_verifier = _pkce_store.pop(state, None)
            result = OAUTH_TOKENS.exchange_code(pid, code, redir, pkce_verifier=pkce_verifier)

            if result.get("ok"):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(b"""<html><body style="background:#1e293b;color:#e2e8f0;font-family:sans-serif;text-align:center;padding-top:100px">
                <h2 style="color:#34d399">Connected!</h2>
                <p>OAuth token received. You can close this tab.</p>
                <script>setTimeout(()=>window.close(),2000)</script></body></html>""")
            else:
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                err_msg = result.get("error", "Unknown error")
                self.wfile.write(f"""<html><body style="background:#1e293b;color:#e2e8f0;font-family:sans-serif;text-align:center;padding-top:100px">
                <h2 style="color:#f87171">Token Exchange Failed</h2><p>{err_msg}</p>
                <p style="font-size:12px;opacity:0.6">Copy the redirect URL and paste it in settings.</p></body></html>""".encode())

            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, fmt, *args):
            pass

    def run_server():
        try:
            server = HTTPServer(("127.0.0.1", port), CallbackHandler)
            server.timeout = 300
            server.handle_request()
        except OSError:
            pass

    threading.Thread(target=run_server, daemon=True).start()


class ClientCredentialsRequest(BaseModel):
    pass


@router.post("/api/providers/{provider_id}/oauth/client-credentials")
def oauth_client_credentials(provider_id: str):
    require_owner_for_writes(action="acquiring OAuth tokens")
    result = OAUTH_TOKENS.client_credentials_auth(provider_id)
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "Auth failed"))
    return result


@router.post("/api/providers/{provider_id}/oauth/exchange-url")
def oauth_exchange_url(provider_id: str, body: dict):
    require_owner_for_writes(action="exchanging an OAuth code")
    redirect_url = body.get("url", "").strip()
    if not redirect_url:
        raise HTTPException(400, "Missing redirect URL")

    import urllib.parse
    parsed = urllib.parse.urlparse(redirect_url)
    params = urllib.parse.parse_qs(parsed.query)

    code = params.get("code", [None])[0]
    if not code:
        raise HTTPException(400, "No ?code= parameter found in the URL")

    state = params.get("state", [None])[0]
    pkce_verifier = _pkce_store.pop(state, None) if state else None

    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")

    oauth = p.get("oauth") or {}
    redirect_uri = oauth.get(
        "redirect_uri",
        f"http://localhost:{CONFIG.server['port']}/api/providers/oauth/callback",
    )

    result = OAUTH_TOKENS.exchange_code(provider_id, code, redirect_uri, pkce_verifier=pkce_verifier)
    if result.get("ok"):
        return result
    raise HTTPException(400, result.get("error", "Token exchange failed"))


class ManualTokenRequest(BaseModel):
    access_token: str
    refresh_token: str = ""
    expires_in: int = 86400


@router.post("/api/providers/{provider_id}/oauth/manual-token")
def oauth_manual_token(provider_id: str, body: ManualTokenRequest):
    require_owner_for_writes(action="storing a manual OAuth token")
    p = get_provider(provider_id)
    if not p:
        raise HTTPException(404, "provider not found")

    oauth = p.get("oauth") or {}
    OAUTH_TOKENS._store_token(provider_id, {
        "access_token": body.access_token,
        "refresh_token": body.refresh_token,
        "expires_in": body.expires_in,
        "token_type": "Bearer",
    }, oauth)
    return {"ok": True, "message": "Token saved"}


@router.post("/api/providers/{provider_id}/oauth/revoke")
def oauth_revoke(provider_id: str):
    require_owner_for_writes(action="revoking OAuth tokens")
    OAUTH_TOKENS.revoke(provider_id)
    return {"ok": True}
