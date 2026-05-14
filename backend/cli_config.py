"""`hrant config` — surface the few configs that matter for noob users.

Modelled on `openclaw config <get|set|...>` plus a no-args interactive
wizard. Keeps the surface tiny on purpose: only the configs a first-
time user will actually want to change. The advanced knobs (engine
sliders, autonomic levers, knowledge caps, full provider/channel
tables) stay in the WebUI Settings tabs where they have proper UIs.

Storage map — every key here is backed by a real file under the
data_dir, NOT a new "hrant.json". That means `hrant config set` is
just a friendlier face on the same .env / JSON files the WebUI and
init wizard already write.

Key                              Backing file
─────────────────────────────────────────────────────────────────────
anthropic.api_key                .env:ANTHROPIC_API_KEY
openai.api_key                   .env:OPENAI_API_KEY
telegram.bot_token               .env:TELEGRAM_BOT_TOKEN
tailscale.host                   .env:TAILSCALE_HOST
whisper.url                      knowledge/transcriber_config.json:local_whisper.url
tts.backend                      knowledge/tts_config.json:backend
tts.piper_url                    knowledge/tts_config.json:local_piper.url
tts.edge_voice                   knowledge/tts_config.json:edge_tts.voice
tts.edge_voice_ru                knowledge/tts_config.json:edge_tts.voice_ru
autonomic.heartbeat_seconds      knowledge/autonomic_settings.json:tick_interval_seconds
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .cli_colors import c, g


# ─── Registry ─────────────────────────────────────────────────────────


@dataclass
class Key:
    """One configurable knob. Source string mini-DSL:

        env:NAME                 → row in .env
        json:<file>:<a.b.c>      → dotted path in a JSON file under
                                    knowledge_dir
    """
    key: str
    source: str
    label: str
    group: str
    secret: bool = False
    type: str = "str"                   # "str" | "int" | "float" | "choice"
    choices: Optional[list[str]] = None
    help: str = ""
    min: Optional[float] = None
    max: Optional[float] = None
    # Optional side-effect after write. Used by autonomic heartbeat so
    # the live scheduler picks up the new interval without a restart.
    after_write: Optional[Callable[[Any], None]] = field(default=None, repr=False)


def _heartbeat_after_write(value: Any) -> None:
    """Push the new tick interval into the live scheduler."""
    try:
        from .autonomic.scheduler import set_interval  # type: ignore[attr-defined]
    except Exception:
        return
    try:
        set_interval(float(value))
    except Exception:
        pass


REGISTRY: list[Key] = [
    # API keys -------------------------------------------------------
    Key("anthropic.api_key", "env:ANTHROPIC_API_KEY",
        label="Anthropic API key",
        group="API keys",
        secret=True,
        help="Claude API key. Get one at https://console.anthropic.com/keys"),
    Key("openai.api_key", "env:OPENAI_API_KEY",
        label="OpenAI API key",
        group="API keys",
        secret=True,
        help="OpenAI key. Get one at https://platform.openai.com/api-keys"),

    # Voice ----------------------------------------------------------
    Key("whisper.url", "json:transcriber_config.json:local_whisper.url",
        label="Whisper STT URL",
        group="Voice",
        help="HTTP URL of the Whisper transcription server (no trailing /)."),
    Key("tts.backend", "json:tts_config.json:backend",
        label="TTS backend",
        group="Voice",
        type="choice",
        choices=["auto", "edge_tts", "local_piper", "openai_tts", "disabled"],
        help="Which TTS backend the agent should prefer."),
    Key("tts.piper_url", "json:tts_config.json:local_piper.url",
        label="Piper TTS URL",
        group="Voice",
        help="HTTP URL of the Piper TTS server (used when backend=local_piper)."),
    Key("tts.edge_voice", "json:tts_config.json:edge_tts.voice",
        label="Edge TTS voice (default)",
        group="Voice",
        help="Voice name for Edge TTS. `python -m edge_tts --list-voices` to browse."),
    Key("tts.edge_voice_ru", "json:tts_config.json:edge_tts.voice_ru",
        label="Edge TTS voice (Russian)",
        group="Voice",
        help="Russian voice override for Edge TTS."),

    # Telegram -------------------------------------------------------
    Key("telegram.bot_token", "env:TELEGRAM_BOT_TOKEN",
        label="Telegram bot token",
        group="Telegram",
        secret=True,
        help="Bot token from @BotFather. After setting, restart with "
             "`hrant gateway restart`."),

    # Discovery ------------------------------------------------------
    Key("tailscale.host", "env:TAILSCALE_HOST",
        label="Tailscale host",
        group="Discovery",
        help="Default host for `hrant discover` (e.g. 100.64.0.5)."),

    # Autonomic ------------------------------------------------------
    Key("autonomic.heartbeat_seconds", "json:autonomic_settings.json:tick_interval_seconds",
        label="Autonomic heartbeat (seconds)",
        group="Autonomic",
        type="float",
        min=1, max=3600,
        after_write=_heartbeat_after_write,
        help="How often the autonomic loop ticks. 30s-3600s typical."),
]


# Keyed-by-name index for quick lookups.
_BY_KEY: dict[str, Key] = {k.key: k for k in REGISTRY}


def all_keys() -> list[Key]:
    return list(REGISTRY)


def find_key(name: str) -> Optional[Key]:
    return _BY_KEY.get(name)


# ─── .env read/write ──────────────────────────────────────────────────


def _env_path() -> Path:
    from . import paths
    return paths.env_path()


def _read_env() -> dict[str, str]:
    p = _env_path()
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def _write_env(env: dict[str, str]) -> None:
    p = _env_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{k}={v}" for k, v in env.items() if v]
    p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# ─── JSON file read/write ─────────────────────────────────────────────


def _json_file_path(filename: str) -> Path:
    from . import paths
    return paths.knowledge_dir() / filename


def _read_json(filename: str) -> dict:
    p = _json_file_path(filename)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _write_json(filename: str, data: dict) -> None:
    p = _json_file_path(filename)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _dotted_get(obj: dict, path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _dotted_set(obj: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _dotted_delete(obj: dict, path: str) -> None:
    parts = path.split(".")
    cur = obj
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            return
        cur = nxt
    cur.pop(parts[-1], None)


# ─── Key value read/write ─────────────────────────────────────────────


def read_value(k: Key) -> Any:
    if k.source.startswith("env:"):
        env = _read_env()
        return env.get(k.source[len("env:"):]) or None
    if k.source.startswith("json:"):
        _, filename, dotted = k.source.split(":", 2)
        return _dotted_get(_read_json(filename), dotted)
    raise ValueError(f"unknown source: {k.source}")


def write_value(k: Key, raw: str) -> Any:
    """Coerce `raw` to the right type, persist, run any after_write
    hook. Returns the stored value."""
    value: Any
    if k.type == "int":
        value = int(raw)
    elif k.type == "float":
        value = float(raw)
    elif k.type == "choice":
        if k.choices and raw not in k.choices:
            raise ValueError(
                f"'{raw}' not in choices {k.choices}"
            )
        value = raw
    else:
        value = raw
    if k.type in ("int", "float"):
        if k.min is not None and value < k.min:
            raise ValueError(f"{k.key} must be >= {k.min}")
        if k.max is not None and value > k.max:
            raise ValueError(f"{k.key} must be <= {k.max}")

    if k.source.startswith("env:"):
        env = _read_env()
        env[k.source[len("env:"):]] = str(value)
        _write_env(env)
    elif k.source.startswith("json:"):
        _, filename, dotted = k.source.split(":", 2)
        data = _read_json(filename)
        _dotted_set(data, dotted, value)
        _write_json(filename, data)
    else:
        raise ValueError(f"unknown source: {k.source}")

    if k.after_write is not None:
        try:
            k.after_write(value)
        except Exception:
            pass
    return value


def delete_value(k: Key) -> None:
    if k.source.startswith("env:"):
        env = _read_env()
        env.pop(k.source[len("env:"):], None)
        _write_env(env)
    elif k.source.startswith("json:"):
        _, filename, dotted = k.source.split(":", 2)
        data = _read_json(filename)
        _dotted_delete(data, dotted)
        _write_json(filename, data)


# ─── Display ──────────────────────────────────────────────────────────


def _redact(value: str) -> str:
    """Mask everything except the last 4 chars (when long enough)."""
    if not value:
        return ""
    if len(value) <= 6:
        return "*" * len(value)
    return value[:4] + g.ellipsis + value[-4:]


def _format_value(k: Key, value: Any) -> str:
    if value is None or value == "":
        return c.muted("(not set)")
    s = str(value)
    if k.secret:
        return c.success(_redact(s))
    return c.success(s)


def print_list() -> None:
    """`hrant config list` — print all keys grouped, colored, secrets
    redacted. The format is meant to be glanceable: one row per key,
    `(not set)` in muted gray for empties, redacted secrets in green
    with the last 4 chars visible so users can confirm which key is
    loaded."""
    from . import paths
    print()
    print(c.heading("  Hrant configuration"))
    print(f"  {c.muted('data:')} {paths.data_dir(require=False)}")
    print()
    last_group = ""
    # Compute the widest key name once for alignment.
    width = max(len(k.key) for k in REGISTRY) + 2
    for k in REGISTRY:
        if k.group != last_group:
            if last_group:
                print()
            print(f"  {c.accent_bright(k.group)}")
            last_group = k.group
        val = read_value(k)
        marker = c.success(g.check) if val else c.muted(g.bullet)
        print(f"    {marker} {k.key.ljust(width)} {_format_value(k, val)}")
    print()
    print(f"  {c.muted('change a value:')} hrant config set {c.accent('<key>')} {c.accent('<value>')}")
    print(f"  {c.muted('interactive menu:')} hrant config")
    print()


def print_files() -> None:
    """`hrant config files` — show where each backing file lives. For
    users who'd rather edit the raw files."""
    from . import paths
    data_dir = paths.data_dir(require=False)
    print()
    print(c.heading("  Hrant config files"))
    print()
    print(f"  {c.muted('.env')}                         {paths.env_path()}")
    print(f"  {c.muted('config.yaml')}                  {paths.config_yaml_path()}")
    print(f"  {c.muted('runtime_overrides.json')}       {paths.knowledge_dir() / 'runtime_overrides.json'}")
    print(f"  {c.muted('autonomic_settings.json')}      {paths.knowledge_dir() / 'autonomic_settings.json'}")
    print(f"  {c.muted('tts_config.json')}              {paths.knowledge_dir() / 'tts_config.json'}")
    print(f"  {c.muted('transcriber_config.json')}      {paths.knowledge_dir() / 'transcriber_config.json'}")
    print(f"  {c.muted('providers.json')}               {paths.knowledge_dir() / 'providers.json'}")
    print(f"  {c.muted('channels.json')}                {paths.knowledge_dir() / 'channels.json'}")
    print()
    print(f"  {c.muted('data root:')}                   {data_dir}")
    print()


def print_get(key_name: str) -> int:
    k = find_key(key_name)
    if k is None:
        print(c.error(f"  unknown key: {key_name}"))
        print(c.muted(f"  list known keys: hrant config list"))
        return 1
    val = read_value(k)
    if val is None or val == "":
        print(c.muted("(not set)"))
        return 0
    if k.secret:
        print(_redact(str(val)))
    else:
        print(str(val))
    return 0


def cmd_set(key_name: str, raw_value: str) -> int:
    k = find_key(key_name)
    if k is None:
        print(c.error(f"  unknown key: {key_name}"))
        print(c.muted(f"  list known keys: hrant config list"))
        return 1
    try:
        stored = write_value(k, raw_value)
    except ValueError as e:
        print(c.error(f"  {e}"))
        return 1
    shown = _redact(str(stored)) if k.secret else str(stored)
    print(f"  {c.success(g.check)} set {c.accent(k.key)} = {c.success(shown)}")
    return 0


def cmd_unset(key_name: str) -> int:
    k = find_key(key_name)
    if k is None:
        print(c.error(f"  unknown key: {key_name}"))
        return 1
    delete_value(k)
    print(f"  {c.success(g.check)} unset {c.accent(k.key)}")
    return 0


def cmd_edit() -> int:
    """Open $EDITOR on .env. Falls back to nano/notepad if unset."""
    from . import paths
    editor = os.environ.get("EDITOR") or (
        "notepad" if sys.platform == "win32" else "nano"
    )
    target = paths.env_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text("", encoding="utf-8")
    print(f"  {c.muted('opening')} {target} {c.muted('with')} {editor}")
    import subprocess
    try:
        return subprocess.call([editor, str(target)])
    except FileNotFoundError:
        print(c.error(f"  editor '{editor}' not found; set $EDITOR"))
        return 1


# ─── Interactive menu ─────────────────────────────────────────────────


def _ask_str(prompt: str, default: str = "", *, secret: bool = False) -> str:
    if not sys.stdin.isatty():
        return default
    hint = f" [{c.muted(default)}]" if default else ""
    try:
        if secret:
            import getpass
            v = getpass.getpass(f"  {prompt}{hint}: ").strip()
        else:
            v = input(f"  {prompt}{hint}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return default
    return v or default


def _ask_choice(prompt: str, options: list[tuple[str, str]], default_idx: int = 0) -> int:
    """Numbered menu, same shape as init_wizard._ask_choice. Returns
    the chosen index (0-based). Non-TTY returns default."""
    print(f"  {prompt}")
    print()
    for i, (label, desc) in enumerate(options, start=1):
        marker = c.accent_bright(g.arrow) if (i - 1) == default_idx else " "
        line = f"    {marker} {i}) {c.bold(label)}"
        if desc:
            line += f"  {c.muted(desc)}"
        print(line)
    print()
    if not sys.stdin.isatty():
        return default_idx
    while True:
        try:
            raw = input(f"  {c.muted('Choice')} [{c.accent(str(default_idx + 1))}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return default_idx
        if not raw:
            return default_idx
        try:
            idx = int(raw) - 1
        except ValueError:
            print(c.warn(f"  please enter a number 1..{len(options)}"))
            continue
        if 0 <= idx < len(options):
            return idx
        print(c.warn("  out of range; try again"))


def _edit_key(k: Key) -> None:
    """Single-key edit flow used by the interactive menu."""
    cur = read_value(k)
    print()
    print(f"  {c.heading(k.label)}")
    print(f"  {c.muted(k.key)}")
    if k.help:
        print(f"  {c.muted(k.help)}")
    print()
    if cur is not None and cur != "":
        shown = _redact(str(cur)) if k.secret else str(cur)
        print(f"  current: {c.success(shown)}")
    else:
        print(f"  current: {c.muted('(not set)')}")
    print()

    if k.type == "choice" and k.choices:
        opts = [(ch, "") for ch in k.choices]
        default_idx = (
            k.choices.index(str(cur)) if str(cur) in k.choices else 0
        )
        idx = _ask_choice(f"Pick a value for {c.accent(k.key)}:", opts, default_idx)
        new_raw = k.choices[idx]
    else:
        new_raw = _ask_str(
            f"New value (Enter to keep current)",
            default=str(cur) if cur is not None else "",
            secret=k.secret,
        )

    if cur is not None and str(cur) == new_raw:
        print(f"  {c.muted('unchanged')}")
        return
    try:
        stored = write_value(k, new_raw)
    except ValueError as e:
        print(c.error(f"  {e}"))
        return
    shown = _redact(str(stored)) if k.secret else str(stored)
    print(f"  {c.success(g.check)} saved {c.accent(k.key)} = {c.success(shown)}")


def _group_menu(group: str) -> None:
    """Submenu: show every key in this group; let user pick one to
    edit, repeating until they choose 'back'."""
    while True:
        keys = [k for k in REGISTRY if k.group == group]
        print()
        print(f"  {c.heading(group)}")
        print()
        rows: list[tuple[str, str]] = []
        for k in keys:
            cur = read_value(k)
            if cur is None or cur == "":
                shown = c.muted("(not set)")
            elif k.secret:
                shown = c.success(_redact(str(cur)))
            else:
                shown = c.success(str(cur))
            rows.append((k.label, f"{c.muted(k.key)}  {g.arrow}  {shown}"))
        rows.append(("Back", "return to the main menu"))
        idx = _ask_choice("Pick a setting to edit:", rows, default_idx=len(rows) - 1)
        if idx == len(rows) - 1:
            return
        _edit_key(keys[idx])


def run_menu() -> int:
    """Top-level interactive menu — what `hrant config` runs when
    given no subcommand. Designed for someone who's never touched the
    CLI before: groups are clearly labelled, current state is shown
    inline, every option says what'll happen."""
    if not sys.stdin.isatty():
        # Non-TTY (cron / scripts) — just print the list and exit.
        print_list()
        return 0

    groups: list[str] = []
    for k in REGISTRY:
        if k.group not in groups:
            groups.append(k.group)

    while True:
        print()
        print(c.heading("  Hrant configuration"))
        print(f"  {c.muted('Walk through your settings. Press Ctrl-C any time to exit.')}")
        print()
        # Per-group status row in the main menu.
        opts: list[tuple[str, str]] = []
        for g in groups:
            keys = [k for k in REGISTRY if k.group == g]
            set_count = sum(1 for k in keys if read_value(k))
            total = len(keys)
            marker = (
                c.success(f"{set_count}/{total} set")
                if set_count == total else
                c.warn(f"{set_count}/{total} set")
                if set_count > 0 else
                c.muted(f"0/{total} set")
            )
            opts.append((g, marker))
        opts.append(("Show all values", c.muted("`hrant config list` — print every key")))
        opts.append(("Show config files", c.muted("`hrant config files` — where things live on disk")))
        opts.append(("Exit", c.muted("done — leave the wizard")))
        idx = _ask_choice("Main menu:", opts, default_idx=len(opts) - 1)
        if idx < len(groups):
            _group_menu(groups[idx])
            continue
        choice = opts[idx][0]
        if choice == "Show all values":
            print_list()
            continue
        if choice == "Show config files":
            print_files()
            continue
        # Exit
        print()
        print(c.muted("  bye."))
        return 0
