"""Interactive setup wizard — what `hrant init` runs by default.

Designed for a non-technical user who's never seen the agent
before. The previous `cmd_init` flow asked for raw env-var names
("ANTHROPIC_API_KEY (current: …)"); this one walks the user
through one choice at a time, opens the right URL in a browser,
validates input live, and offers retries on failure.

Layout:

  Welcome banner
  Step 1: pick ONE provider to start with (skip → finish without)
  Step 1 detail: per-provider sub-flow (api_key / codex / copilot /
                 ollama)
  Step 2: pick which model to use as default for the active provider
  Step 3: optional services (voice / embeddings / telegram /
                              tailscale) — each gated behind a
                              single y/n
  Final summary

Each sub-flow can fail and offer a retry; nothing here ever raises
into `cmd_init`. The wizard always returns 0 — the user can
re-enter `hrant init` to fix anything that didn't take. For
non-interactive runs (cron, CI, tests with stubbed stdin) the
caller passes `--skip-wizard` and `cmd_init` falls back to the
old flat-prompt path.

The wizard is OUTPUT-heavy — every step prints a short header,
guidance, and a clear `[default]` hint so the user knows what
pressing Enter will do. We use `sys.stdout` directly (not the
agent's progress streamer) because this runs before the FastAPI
server boots.
"""
from __future__ import annotations

import logging
import os
import sys
import time
import webbrowser
from typing import Optional

log = logging.getLogger(__name__)


# --- Display helpers ---------------------------------------------------


def _bold(text: str) -> str:
    """ANSI bold when stdout is a TTY; plain text otherwise. Keeps
    scripted captures clean."""
    if sys.stdout.isatty():
        return f"\033[1m{text}\033[0m"
    return text


def _dim(text: str) -> str:
    if sys.stdout.isatty():
        return f"\033[2m{text}\033[0m"
    return text


def _hr() -> None:
    print()
    print(_dim("  " + "─" * 60))
    print()


def _step(num: int, total: int, title: str) -> None:
    print()
    print(_bold(f"  ► Step {num} / {total}: {title}"))
    print()


def _ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def _warn(msg: str) -> None:
    print(f"  ⚠ {msg}")


def _err(msg: str) -> None:
    print(f"  ✗ {msg}")


def _ask_str(prompt: str, default: str = "", *, secret: bool = False) -> str:
    """Read a string from the user. Non-TTY → return default."""
    if not sys.stdin.isatty():
        return default
    hint = ""
    if default:
        hint = f" [{default}]"
    try:
        if secret:
            import getpass
            value = getpass.getpass(f"  {prompt}{hint}: ").strip()
        else:
            value = input(f"  {prompt}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return value or default


def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Yes/no with a clear default. Non-TTY → return default."""
    if not sys.stdin.isatty():
        return default
    suffix = " [Y/n]" if default else " [y/N]"
    try:
        ans = input(f"  {prompt}{suffix}: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    if not ans:
        return default
    return ans.startswith("y")


def _ask_choice(prompt: str, options: list[tuple[str, str]], default_idx: int = 0) -> int:
    """Numbered choice menu. Each option is (short_label, description).
    Returns the chosen index. Non-TTY → returns default_idx."""
    print(f"  {prompt}")
    print()
    for i, (label, desc) in enumerate(options, start=1):
        marker = "→" if (i - 1) == default_idx else " "
        line = f"    {marker} {i}) {_bold(label)}"
        if desc:
            line += f"  {_dim(desc)}"
        print(line)
    print()
    if not sys.stdin.isatty():
        return default_idx
    while True:
        try:
            raw = input(f"  Choice [default {default_idx + 1}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default_idx
        if not raw:
            return default_idx
        try:
            idx = int(raw) - 1
        except ValueError:
            print(f"  {_warn_inline('please enter a number 1..%d' % len(options))}")
            continue
        if 0 <= idx < len(options):
            return idx
        print(f"  {_warn_inline('out of range; try again')}")


def _warn_inline(msg: str) -> str:
    return f"⚠ {msg}"


def _maybe_open_url(url: str, label: str = "this URL") -> None:
    """Offer to open the URL in the user's browser. Quiet no-op on
    headless boxes — `webbrowser.open` returns False but doesn't
    raise."""
    if not url:
        return
    print(f"     URL: {_bold(url)}")
    if _ask_yes_no(f"     Open {label} in your browser?", default=True):
        try:
            webbrowser.open(url)
        except Exception:
            pass  # headless / no display — user copies the URL manually


# --- Provider sub-flows -----------------------------------------------


def _ascii_provider_menu() -> list[tuple[str, str, str]]:
    """The 6 first-run provider choices. Each: (key, label, blurb)."""
    return [
        ("anthropic", "Claude API",
         "Best quality, paid — recommended for first-time users"),
        ("openai", "OpenAI API",
         "Also great quality, paid"),
        ("openai_codex", "ChatGPT Plus/Pro (Codex)",
         "Free if you already have a ChatGPT paid subscription"),
        ("github_copilot", "GitHub Copilot",
         "Free if you already have Copilot via GitHub"),
        ("ollama", "Ollama (local)",
         "Fully free, runs on your machine; slower but private"),
        ("skip", "I'll set this up later",
         "Skip — Hrant won't be able to chat until you configure a provider"),
    ]


def _wizard_provider_api_key(provider_type: str) -> Optional[dict]:
    """Generic API-key sub-flow: show signup URL, accept key, test,
    register. Returns the persisted entry or None on skip/failure."""
    from . import init_helpers as _ih
    from . import providers as _p
    info = _p.PROVIDER_CONNECT_INFO.get(provider_type) or {}
    print()
    if info.get("key_instructions"):
        for line in info["key_instructions"].splitlines():
            print(f"     {line}")
    if info.get("key_url"):
        _maybe_open_url(info["key_url"], "the signup page")
    print()

    # Up to 3 retries on bad-key / network failure so a typo doesn't
    # end the wizard.
    for attempt in range(3):
        key = _ask_str(f"Paste your {provider_type} API key", default="", secret=True)
        if not key:
            _warn("no key entered; skipping this provider")
            return None
        # Test before registering — feedback within 5s.
        print("  testing...")
        test_fn = {
            "anthropic": _ih.test_anthropic_key,
            "openai": _ih.test_openai_key,
        }.get(provider_type)
        if test_fn is None:
            # No type-specific tester — accept the key as-is. The
            # WebUI's "Test" button is the user's next checkpoint.
            ok = True
            msg = "no live test for this provider; accepted as-is"
        else:
            ok, msg = test_fn(key)
        if ok:
            _ok(msg)
            break
        _err(msg)
        if attempt < 2 and _ask_yes_no("Try again?", default=True):
            continue
        return None

    # Persist into .env first, then register a provider entry.
    env_key = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "COHERE_API_KEY",
        "google": "GOOGLE_API_KEY",
        "perplexity": "PERPLEXITY_API_KEY",
        "huggingface": "HF_TOKEN",
        "xai": "XAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider_type, f"{provider_type.upper()}_API_KEY")
    os.environ[env_key] = key.strip()

    # Auto-register the default entry. For Anthropic, get_providers
    # injects one automatically from env — so we don't write a
    # duplicate. For OpenAI use auto_register_openai (idempotent).
    # For everything else, fall back to a generic registration.
    if provider_type == "anthropic":
        from .providers import get_providers
        return next(
            (p for p in get_providers() if p.get("type") == "anthropic"),
            None,
        )
    if provider_type == "openai":
        return _ih.auto_register_openai(key.strip())
    # Generic registration for the rest.
    from .providers import _load_providers, _save_providers
    new_id = f"{provider_type}-default"
    providers_list = _load_providers()
    if not any(p.get("id") == new_id for p in providers_list):
        entry = {
            "id": new_id,
            "name": f"{provider_type.title()} (default)",
            "type": provider_type,
            "auth_type": "api_key",
            "enabled": True,
            "is_default": False,
            "api_key_env": env_key,
            "api_key": "",
            "base_url": "",
            "models": [],
            "default_model": "",
            "max_tokens": 2000,
            "temperature": 0.3,
        }
        providers_list.append(entry)
        _save_providers(providers_list)
        return entry
    return next(p for p in providers_list if p.get("id") == new_id)


def _wizard_provider_codex() -> Optional[dict]:
    """ChatGPT subscription via Codex CLI. Reads ~/.codex/auth.json."""
    from . import providers as _p
    print()
    print("     This uses your existing ChatGPT Plus/Pro account via the")
    print("     upstream `codex` CLI's saved login (~/.codex/auth.json).")
    print()
    status = _p.CODEX_AUTH.status()
    if not status.get("logged_in"):
        _warn("Codex auth file not found.")
        print("     To sign in:")
        print(f"       1) Install Codex CLI from {_bold('https://github.com/openai/codex')}")
        print("       2) Run `codex login` and complete the browser sign-in")
        print("       3) Re-run `hrant init` (or `hrant provider login codex`)")
        return None
    _ok(f"signed in as {status.get('email','(unknown)')}, plan: {status.get('plan_type','?')}")
    # Register if missing.
    from .providers import _load_providers, _save_providers
    providers_list = _load_providers()
    providers_list = [p for p in providers_list if p.get("auth_type") != "codex_subscription"]
    entry = {
        "id": "openai-codex",
        "name": "OpenAI Codex (ChatGPT subscription)",
        "type": "openai_codex",
        "auth_type": "codex_subscription",
        "enabled": True,
        "is_default": False,
        "api_key": "",
        "api_key_env": "",
        "base_url": "",
        "models": ["gpt-5", "gpt-5.5", "o1", "o1-mini"],
        "default_model": "gpt-5",
        "max_tokens": 2000,
        "temperature": 0.3,
    }
    providers_list.append(entry)
    _save_providers(providers_list)
    return entry


def _wizard_provider_copilot() -> Optional[dict]:
    """GitHub Copilot subscription via existing client login."""
    from . import providers as _p
    print()
    print("     This uses your existing GitHub Copilot subscription via your")
    print("     VS Code / JetBrains / `gh` client's saved login.")
    print()
    status = _p.COPILOT_AUTH.status()
    if not status.get("logged_in"):
        _warn("GitHub Copilot auth not found.")
        print("     To sign in, use one of:")
        print("       - VS Code: install GitHub Copilot extension + sign in")
        print("       - JetBrains: install Copilot plugin + sign in")
        print("       - CLI: `gh auth login --scopes copilot`")
        return None
    _ok(f"signed in (user: {status.get('user','?')})")
    from .providers import _load_providers, _save_providers
    providers_list = _load_providers()
    providers_list = [p for p in providers_list if p.get("auth_type") != "copilot_subscription"]
    entry = {
        "id": "github-copilot",
        "name": "GitHub Copilot",
        "type": "github_copilot",
        "auth_type": "copilot_subscription",
        "enabled": True,
        "is_default": False,
        "api_key": "",
        "api_key_env": "",
        "base_url": "",
        "models": ["gpt-4o", "claude-3.5-sonnet"],
        "default_model": "gpt-4o",
        "max_tokens": 2000,
        "temperature": 0.3,
    }
    providers_list.append(entry)
    _save_providers(providers_list)
    return entry


def _wizard_provider_ollama() -> Optional[dict]:
    """Probe localhost, list models, register."""
    print()
    print(f"     Ollama is a free local model runner. {_bold('https://ollama.com/download')}")
    default_url = "http://localhost:11434"
    url = _ask_str("Ollama URL", default=default_url) or default_url
    print(f"  probing {url}/api/tags ...")
    try:
        import httpx
        r = httpx.get(f"{url}/api/tags", timeout=5.0)
    except Exception as e:
        _err(f"Ollama not reachable: {e}")
        print(f"     Start it with: {_bold('ollama serve')} (or install it first)")
        return None
    if r.status_code != 200:
        _err(f"Ollama responded with HTTP {r.status_code}")
        return None
    models = [m.get("name", "") for m in (r.json().get("models") or [])]
    if not models:
        _warn("Ollama is running but no models installed.")
        print(f"     Pull one: {_bold('ollama pull qwen2.5:7b-instruct')}")
        return None
    _ok(f"found {len(models)} model(s): {', '.join(models[:5])}")
    from .providers import _load_providers, _save_providers
    new_id = f"ollama-{int(time.time())}"
    entry = {
        "id": new_id,
        "name": "Ollama (local)",
        "type": "ollama",
        "auth_type": "none",
        "enabled": True,
        "is_default": False,
        "api_key": "",
        "api_key_env": "",
        "base_url": url,
        "models": models,
        "default_model": models[0],
        "max_tokens": 2000,
        "temperature": 0.3,
    }
    providers_list = _load_providers()
    providers_list.append(entry)
    _save_providers(providers_list)
    return entry


# --- Model picker ------------------------------------------------------


_KNOWN_MODELS: dict[str, list[tuple[str, str]]] = {
    # (model_id, short_description) — used when the provider's
    # configured `models` list is empty.
    "anthropic": [
        ("claude-sonnet-4-5", "balanced quality + cost (recommended)"),
        ("claude-opus-4-7", "highest quality, more expensive"),
        ("claude-haiku-4-5", "fastest + cheapest"),
    ],
    "openai": [
        ("gpt-4o-mini", "cheapest, fast"),
        ("gpt-4o", "balanced"),
        ("o1-mini", "reasoning-heavy"),
    ],
    "openai_codex": [
        ("gpt-5", "default"),
        ("gpt-5.5", "newer"),
    ],
    "github_copilot": [
        ("gpt-4o", "OpenAI via Copilot"),
        ("claude-3.5-sonnet", "Anthropic via Copilot"),
    ],
}


def _wizard_model_picker(provider: dict) -> Optional[str]:
    """After a provider's connected, pick the default model. Uses
    the provider's `models` list if populated; otherwise falls back
    to a curated `_KNOWN_MODELS` table."""
    pt = provider.get("type") or ""
    models = provider.get("models") or []
    if not models:
        models = [m for m, _ in _KNOWN_MODELS.get(pt, [])]
    if not models:
        # Truly unknown — just ask freeform.
        return _ask_str("Default model name", default=provider.get("default_model") or "")
    options: list[tuple[str, str]] = []
    for m in models[:10]:
        desc = ""
        for known_m, known_desc in _KNOWN_MODELS.get(pt, []):
            if known_m == m:
                desc = known_desc
                break
        options.append((m, desc))
    if len(options) == 1:
        return options[0][0]
    idx = _ask_choice("Pick the default model:", options, default_idx=0)
    return options[idx][0]


# --- Optional services ------------------------------------------------


def _validate_telegram_token(token: str) -> tuple[bool, str]:
    """Call Telegram's getMe API to verify a token before saving.
    Returns (ok, message). 5s timeout; network failures don't
    raise into the wizard."""
    if not token.strip():
        return False, "(no token)"
    try:
        import httpx
        r = httpx.get(
            f"https://api.telegram.org/bot{token.strip()}/getMe",
            timeout=5.0,
        )
    except Exception as e:
        return False, f"network: {e}"
    if r.status_code != 200:
        return False, f"HTTP {r.status_code}"
    try:
        body = r.json()
    except Exception:
        return False, "non-JSON response"
    if not body.get("ok"):
        return False, body.get("description", "Telegram rejected the token")
    info = body.get("result") or {}
    username = info.get("username") or "(no username)"
    name = info.get("first_name") or ""
    return True, f"@{username} ({name})"


def _wizard_telegram() -> bool:
    """Walk the user through creating + registering a Telegram bot.
    Returns True if a bot was registered. Idempotent — if a
    Telegram channel already exists, asks before creating another."""
    print()
    print("     A Telegram bot lets you chat with Hrant from your phone.")
    print("     You'll need a bot token from @BotFather (free).")
    print()

    # Detect existing Telegram channels so the wizard doesn't silently
    # stack duplicates on re-runs.
    from . import channels as _ch
    existing_tg = [c for c in _ch.get_channels() if c.get("type") == "telegram"]
    if existing_tg:
        labels = ", ".join(c["id"] for c in existing_tg)
        print(f"     You already have {len(existing_tg)} Telegram channel(s): {labels}")
        if not _ask_yes_no("Add another Telegram bot?", default=False):
            return False
    else:
        if not _ask_yes_no("Set up a Telegram bot now?", default=False):
            return False

    print()
    print("     1) Open Telegram, search for @BotFather")
    print("     2) Send /newbot, follow the prompts")
    print("     3) Copy the token it gives you (looks like 123456:ABC-DEF...)")
    print()

    # Up to 3 retries — typos in long tokens are common.
    token = ""
    for attempt in range(3):
        token = _ask_str("Paste your bot token", default="", secret=True)
        if not token:
            _warn("no token entered; skipping")
            return False
        print("  validating via Telegram getMe...")
        ok, msg = _validate_telegram_token(token)
        if ok:
            _ok(f"valid bot: {msg}")
            break
        _err(msg)
        if attempt < 2 and _ask_yes_no("Try again?", default=True):
            continue
        return False

    bot_id = f"telegram-{int(time.time())}"
    _ch.save_channel({
        "id": bot_id,
        "type": "telegram",
        "enabled": True,
        "auto_start": True,
        # NB: channels.py reads `config.bot_token` (not `token`) — keep
        # the key in sync with that consumer (see ChannelManager
        # .start_channel which fetches it under that exact name).
        "config": {"bot_token": token, "allowed_users": []},
    })
    _ok(f"registered Telegram bot '{bot_id}' (auto-start enabled)")
    print("     Note: anyone with your bot's username can message it by default.")
    print("     Add allowed_users in WebUI Settings → Channels to restrict access.")
    return True


def _wizard_tailscale() -> bool:
    """Offer Tailscale-based service discovery (Whisper/Piper/Ollama)."""
    print()
    print("     If you run Whisper / Piper / Ollama on a separate machine")
    print("     reachable via Tailscale, Hrant can auto-discover them.")
    print()
    if not _ask_yes_no("Configure Tailscale discovery now?", default=False):
        return False
    host = _ask_str("Tailscale host IP (e.g. 100.64.0.1)", default="")
    if not host:
        _warn("no host entered; skipping")
        return False
    os.environ["TAILSCALE_HOST"] = host
    from . import init_helpers as _ih
    print(f"  discovering services on {host}...")
    report = _ih.discover_and_apply(host)
    if report.get("error"):
        _err(report["error"])
        return False
    found_count = 0
    for name, r in (report.get("found") or {}).items():
        if r.get("ok"):
            _ok(f"{name}: {r.get('url')}")
            found_count += 1
        else:
            _warn(f"{name}: {r.get('reason', 'not found')}")
    applied = [n for n, status in (report.get("applied") or {}).items() if status == "applied"]
    if applied:
        _ok(f"applied URLs: {', '.join(applied)}")
    return found_count > 0


# --- Main wizard entry point -------------------------------------------


def run_wizard(existing_env: dict[str, str]) -> dict:
    """Drive the whole interactive setup. Returns a dict the caller
    folds back into .env writes:

      {
        "env_updates": {KEY: value, ...},   # to merge into existing_env
        "provider_registered": <id or None>,
        "active_model": <model_id or None>,
        "voice_enabled": bool,    # 14B will fill this in
        "telegram_enabled": bool,
        "tailscale_host": str,
      }

    Idempotent: re-running on a configured install lets the user
    pick a different provider, skip flows they've already done, etc.
    """
    print(_bold("  Welcome to Hrant ✨"))
    print()
    print("  Let's get you set up — about a minute.")
    print()
    print("  Press Enter at any prompt to accept the default (shown in brackets).")
    print("  Ctrl-C at any time aborts; re-run `hrant init` to continue.")

    result: dict = {
        "env_updates": {},
        "provider_registered": None,
        "active_model": None,
        "voice_enabled": False,
        "telegram_enabled": False,
        "tailscale_host": "",
    }

    # --- Step 1: provider ---
    _hr()
    _step(1, 3, "How should Hrant think?")
    choices = _ascii_provider_menu()
    idx = _ask_choice(
        "Pick a provider (you can add more later):",
        [(label, blurb) for _key, label, blurb in choices],
        default_idx=0,
    )
    chosen_key = choices[idx][0]

    provider: Optional[dict] = None
    if chosen_key == "skip":
        _warn("skipping provider setup — agent won't chat until you add one")
    elif chosen_key == "openai_codex":
        provider = _wizard_provider_codex()
    elif chosen_key == "github_copilot":
        provider = _wizard_provider_copilot()
    elif chosen_key == "ollama":
        provider = _wizard_provider_ollama()
    else:
        provider = _wizard_provider_api_key(chosen_key)

    # --- Step 2: model + active ---
    if provider:
        result["provider_registered"] = provider.get("id")
        # Carry through any env updates the sub-flow made.
        for k in (
            "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY",
            "DEEPSEEK_API_KEY", "MISTRAL_API_KEY", "COHERE_API_KEY",
            "GOOGLE_API_KEY", "PERPLEXITY_API_KEY", "HF_TOKEN",
            "XAI_API_KEY", "OPENROUTER_API_KEY",
        ):
            if k in os.environ and os.environ[k]:
                result["env_updates"][k] = os.environ[k]

        _hr()
        _step(2, 3, "Pick the default model")
        model = _wizard_model_picker(provider)
        if model:
            # Activate it.
            from .providers import ACTIVE_MODEL
            ACTIVE_MODEL.set(provider["id"], model)
            # Also save default_model into the provider entry so
            # `hrant provider list` shows it.
            from .providers import _load_providers, _save_providers
            providers_list = _load_providers()
            for p in providers_list:
                if p.get("id") == provider["id"]:
                    p["default_model"] = model
                    break
            _save_providers(providers_list)
            result["active_model"] = model
            _ok(f"active model: {provider.get('name')} / {model}")

    # --- Step 3: optional services ---
    _hr()
    _step(3, 3, "Optional services")
    print("  Each of these is OFF by default — say yes only if you want it now.")
    print("  Everything is editable later in the WebUI Settings tabs.")

    # Voice + embeddings are stubs at 14A — Phase 14B will fill them
    # with real Edge TTS / faster-whisper / sentence-transformers
    # backends. For now offer a quick yes/no that gets persisted as
    # a "configure later" reminder; nothing destructive.
    if _ask_yes_no(
        "Set up voice (Whisper STT + TTS for Telegram voice replies)?",
        default=False,
    ):
        print(_dim("    (Voice setup is enhanced in the next release — for now"))
        print(_dim("     you can configure Whisper + Piper URLs in WebUI Settings → Voice.)"))
        result["voice_enabled"] = True

    if _ask_yes_no(
        "Set up semantic memory (embeddings for fuzzy recall)?",
        default=False,
    ):
        print(_dim("    (Default: Ollama nomic-embed-text. Configure in WebUI Settings → Memory.)"))

    # Telegram bot.
    if _wizard_telegram():
        result["telegram_enabled"] = True

    # Tailscale.
    if _wizard_tailscale():
        # _wizard_tailscale already wrote TAILSCALE_HOST into os.environ
        # and ran discovery; capture the host for the .env merge.
        result["env_updates"]["TAILSCALE_HOST"] = os.environ.get("TAILSCALE_HOST", "")
        result["tailscale_host"] = os.environ.get("TAILSCALE_HOST", "")

    return result


def print_final_summary(result: dict, data_dir, engine_root) -> None:
    """Closing screen. Tells the user what's now true and the next
    one or two commands to run."""
    _hr()
    print(_bold("  All set!"))
    print()
    print(f"  Engine:    {engine_root}")
    print(f"  Data:      {data_dir}")
    if result.get("provider_registered"):
        print(f"  Provider:  {result['provider_registered']}")
    if result.get("active_model"):
        print(f"  Model:     {result['active_model']}")
    if result.get("telegram_enabled"):
        print(_dim("  Telegram:  configured (auto-starts when you run `hrant run`)"))
    if result.get("tailscale_host"):
        print(_dim(f"  Tailscale: {result['tailscale_host']} (services auto-discovered)"))
    print()
    print(_bold("  Next:"))
    print("    hrant run                # start the server")
    print("    open http://127.0.0.1:8000")
    if not result.get("provider_registered"):
        print()
        print(_dim("  Add a provider any time:"))
        print(_dim("    hrant provider login <type>     # claude / openai / ollama / ..."))
    print()
