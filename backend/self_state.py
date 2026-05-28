"""Agent self-state snapshot.

The agent needs to KNOW itself before it can answer task-shaped
requests properly. Pre-fix, when the user said "change your voice
to male" the agent had no idea:
  - what voice it's CURRENTLY using
  - that `tts_config.json` exists and is editable
  - that `terminal_exec` / `write_note` / `run_python` can mutate it
So it replied "Understood, I will use a male voice" and saved a preference
fact — without changing anything. The user got the same female
voice on the next reply.

This module exports `current_self_state()` returning a dict the
caller can inject into the agent's system prompt under a
`# AGENT STATE SNAPSHOT` heading. The agent then has line-of-sight
to every config it controls, the file path that owns each config,
and the tool it would use to mutate it. That lets the LLM choose
"call set_setting" instead of "ack and forget".

Design constraints:
  - Cheap (no LLM, no network) — runs on every turn.
  - Defensive (never raises) — failure to read a config file
    becomes `"unknown"` in the snapshot, not an exception.
  - Compact (<2KB serialised) — sits inside the system prompt of
    every turn, so it can't bloat the context budget.
  - Truthful — only reports what's actually wired up. A config
    that doesn't have a working `set_*` path in code is reported
    as `mutable=False` so the LLM doesn't promise edits we can't
    deliver.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from . import paths


log = logging.getLogger(__name__)


# Registry of user-mutable settings the agent should know about.
# Each entry tells `current_self_state()` how to read the current
# value, where the config lives on disk, and which tool the agent
# would use to mutate it. New settings get added here as we wire
# up `set_setting` keys (Phase 2).
#
# Shape:
#   key:        dotted identifier the LLM uses in `set_setting`
#   file:       repo-relative path to the underlying file (string)
#   getter:     callable -> current value (or None on failure)
#   mutable:    True when there's a path in code to apply the
#               change; False = "I can show you the current value
#               but cannot edit it for you"
#   description: short line the LLM sees in the snapshot
SETTINGS_REGISTRY: list[dict[str, Any]] = []


def _safe(fn, default):
    """Run a getter, swallow anything. The snapshot must never
    crash a turn — bookkeeping is best-effort."""
    try:
        return fn()
    except Exception:
        return default


def _tts_status() -> dict[str, Any]:
    """Live status from the TTS singleton + raw config file.

    Voice resolution prefers DISK config over the in-memory singleton
    cache. Reason: `Synthesizer._voice` is cached on first synthesis
    and stays stale until `tts.reset()` runs — so an agent that JUST
    wrote a new voice to `tts_config.json` would still see the old
    voice in the snapshot if we trusted the cache. Reading disk
    first means the snapshot reflects ground truth on the very next
    turn, before the singleton catches up."""
    try:
        from .tts import SYNTHESIZER, load_config
        st = SYNTHESIZER.status()
        cfg = load_config()
        config_path = paths.knowledge_dir() / "tts_config.json"
        # Pull a voice out of the on-disk config first; fall back to
        # the singleton's cached value, then to "default".
        disk_voice = ""
        if isinstance(cfg, dict):
            backend_cfg = cfg.get(cfg.get("backend") or "edge_tts") or {}
            if isinstance(backend_cfg, dict):
                disk_voice = backend_cfg.get("voice") or ""
            # Some configs put voice at top level too — try that.
            if not disk_voice:
                disk_voice = cfg.get("voice") or ""
        voice = disk_voice or st.get("voice") or "default"
        return {
            "backend": (cfg or {}).get("backend") or st.get("backend") or "unknown",
            "voice": voice,
            "voice_gender": _voice_gender_hint(voice),
            "config_path": str(config_path),
            "config_exists": config_path.exists(),
            "config": cfg,
            "last_error": st.get("last_error"),
            "in_memory_voice": st.get("voice") or "",  # for debugging
        }
    except Exception as e:
        log.debug("self_state TTS failed: %s", e)
        return {"backend": "unknown", "voice": "unknown"}


def _voice_gender_hint(voice_name: str) -> str:
    """Best-effort gender hint from an edge-tts / piper voice name.

    Edge TTS voices follow `<lang>-<region>-<Name>Neural` where the
    Name often telegraphs gender (Aria/Jenny/Sarah/Anna = female;
    Guy/Davis/Roger = male). We don't ship an exhaustive list —
    just enough so the snapshot can say `voice_gender: female` for
    the typical defaults. Unknown → empty string."""
    n = (voice_name or "").lower()
    female_hints = (
        "aria", "jenny", "sarah", "anna", "elena", "svetlana",
        "natasha", "polina", "maria", "ramona", "tatiana", "irina",
        "aigul",
    )
    male_hints = (
        "guy", "davis", "roger", "tony", "andrew", "brian",
        "dmitry", "michael", "alex", "william", "henri",
    )
    if any(h in n for h in female_hints):
        return "female"
    if any(h in n for h in male_hints):
        return "male"
    return ""


def _active_model() -> dict[str, Any]:
    """Currently-selected LLM (active model override, or default A)."""
    try:
        from .providers import ACTIVE_MODEL
        cfg = ACTIVE_MODEL.resolve_llm_config()
        if cfg is None:
            from .config import CONFIG
            return {
                "provider": "default",
                "model": CONFIG.model_a.get("model") if CONFIG.model_a else "claude",
                "mode": CONFIG.mode,
            }
        return {
            "provider": cfg.get("provider_id") or cfg.get("provider"),
            "model": cfg.get("model"),
            "mode": "active_override",
        }
    except Exception as e:
        log.debug("self_state active model failed: %s", e)
        return {"provider": "unknown", "model": "unknown"}


def _response_language(speaker_id: str | None) -> str:
    """Pinned response language for the speaker (from user_profile.md
    `## Язык общения` section). Empty = no pin → mirrors the user's
    input language."""
    try:
        from .identity import IDENTITY
        profile = IDENTITY.user_profile(speaker_id=speaker_id)
        body = IDENTITY._extract_language_section(profile)
        return body.strip().splitlines()[0] if body else ""
    except Exception:
        return ""


def _tool_names() -> list[str]:
    """Tools currently registered for any agent run. The LLM has
    this same list as JSON schemas — but the names alone in the
    snapshot are a fast scan for the model to remember `oh I have
    terminal_exec`."""
    try:
        from .tool_registry import get_registry
        return sorted(get_registry().tools.keys())
    except Exception:
        return []


def _settings_keys() -> list[dict[str, Any]]:
    """List of mutable settings the agent can apply via `set_setting`.
    Pulled live from the SETTINGS router so the snapshot stays in
    sync with the registry (new keys auto-appear in the prompt)."""
    try:
        from .settings import SETTINGS
        return SETTINGS.list_settings()
    except Exception:
        return []


def current_self_state(*, speaker_id: str | None = None) -> dict[str, Any]:
    """Cheap, never-raises snapshot of the agent's current settings.

    Used by `agent.py` to inject `# AGENT STATE SNAPSHOT` into every
    turn's system prompt. The LLM reads this block and answers
    questions like "what voice are you using?" / "switch to male
    voice" with reference to actual current state, not made-up
    answers."""
    tts = _safe(_tts_status, {})
    model = _safe(_active_model, {})
    lang = _safe(lambda: _response_language(speaker_id), "")
    tools = _safe(_tool_names, [])
    settings = _safe(_settings_keys, [])
    data_dir = _safe(lambda: str(paths.data_dir(require=False)), "")
    repo_root = _safe(lambda: str(paths.repo_root()), "")
    return {
        "data_dir": data_dir,
        "repo_root": repo_root,
        "active_model": model,
        "active_tts": tts,
        "response_language": lang,
        "tools_available": tools,
        "mutable_settings": settings,
    }


def render_snapshot(state: dict[str, Any]) -> str:
    """Format the snapshot as a compact prompt block. Keep it under
    ~1.5KB rendered so it doesn't dominate the context budget.

    Output shape (real example):

        # AGENT STATE SNAPSHOT
        active_model:  anthropic / claude-sonnet-4-5
        active_tts:    edge_tts / en-US-AriaNeural (female)
                       config: /home/hrant/.hrant/data/knowledge/tts_config.json (missing)
        response_lang: Respond to user in Russian.
        tools:         calc, delegate, fetch_url, locate_symbol, read_file,
                       run_python, save_to_workspace, schedule_message,
                       terminal_exec, web_search
        data_dir:      /home/hrant/.hrant/data
        repo_root:     /home/hrant/hrant

        # RULES
        - When the user asks you to CHANGE / SET / SWITCH something that's
          in this snapshot, USE TOOLS to apply the change — don't just
          acknowledge.
        - When a config is `missing`, you can CREATE it via terminal_exec
          / run_python — don't pretend it's already correct.
        - Cite a value from this snapshot when answering "what voice / model
          / language are you using?" instead of guessing.
    """
    model = state.get("active_model") or {}
    tts = state.get("active_tts") or {}
    voice = tts.get("voice") or "default"
    gender = tts.get("voice_gender") or ""
    cfg_path = tts.get("config_path") or ""
    cfg_exists = " (missing)" if cfg_path and not tts.get("config_exists") else ""
    tools = state.get("tools_available") or []
    # Wrap the tool list at ~70 chars per line so the snapshot stays
    # narrow in the prompt.
    tool_lines: list[str] = []
    cur = ""
    for name in tools:
        if not cur:
            cur = name
        elif len(cur) + len(name) + 2 <= 70:
            cur = f"{cur}, {name}"
        else:
            tool_lines.append(cur)
            cur = name
    if cur:
        tool_lines.append(cur)
    tools_str = "\n".join("               " + ln for ln in tool_lines).lstrip(" ")

    lines = [
        "# AGENT STATE SNAPSHOT",
        f"active_model:  {model.get('provider', '?')} / {model.get('model', '?')}",
        f"active_tts:    {tts.get('backend', '?')} / {voice}"
        + (f" ({gender})" if gender else ""),
    ]
    if cfg_path:
        lines.append(f"               config: {cfg_path}{cfg_exists}")
    lang = (state.get("response_language") or "").strip()
    if lang:
        lines.append(f"response_lang: {lang}")
    lines.append(f"tools:         {tools_str}")
    if state.get("data_dir"):
        lines.append(f"data_dir:      {state['data_dir']}")
    if state.get("repo_root"):
        lines.append(f"repo_root:     {state['repo_root']}")

    # Mutable settings: enumerate so the LLM knows EXACTLY which
    # keys to pass to `set_setting`. Pre-Phase-2 the agent
    # hand-edited tts_config.json via terminal_exec — 4-6 tool
    # calls per request. With this list visible, the LLM picks
    # `set_setting('tts.voice_gender', 'male')` in one shot.
    settings = state.get("mutable_settings") or []
    if settings:
        lines.append("")
        lines.append("# MUTABLE SETTINGS (apply via set_setting(key, value))")
        for s in settings:
            cur = s.get("current")
            cur_str = "(unset)" if cur is None else str(cur)
            if len(cur_str) > 50:
                cur_str = cur_str[:47] + "…"
            line = f"  {s['key']:24} = {cur_str}"
            if s.get("choices"):
                line += f"  choices: {', '.join(s['choices'])}"
            lines.append(line)

    lines.append("")
    lines.append("# RULES")
    lines.append(
        "- When the user asks you to CHANGE / SET / SWITCH something in "
        "MUTABLE SETTINGS, call `set_setting(key, value)` — ONE tool "
        "call, not 4-6 of hand-editing JSON."
    )
    lines.append(
        "- When a config file is `missing`, set_setting will create it "
        "with sensible defaults — don't pretend it's already correct."
    )
    lines.append(
        "- Cite a value from this snapshot when answering 'what voice "
        "/ model / language are you using?' instead of guessing."
    )
    lines.append(
        "- NEVER claim tools are unavailable, disabled, or refused. The "
        "tools listed above ARE available this turn. If a previous tool "
        "call failed, try a DIFFERENT tool — don't surrender with "
        "\"tools disabled\" / \"can't apply\" / \"tools off\". "
        "Refusal is only valid for non-existent settings (not in the "
        "MUTABLE SETTINGS list AND not patchable via terminal_exec / "
        "run_python). Even then: explain what's missing and offer a "
        "concrete next step, don't just say \"I can't\"."
    )
    lines.append(
        "- USER ASKED YOU TO DO X → DO X. Don't explain how to do X to "
        "the user; the user wants the change applied. After applying, "
        "report a one-sentence confirmation of WHAT changed and WHERE."
    )
    return "\n".join(lines)
