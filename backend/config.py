"""Загрузчик config.yaml + .env + система MODE-пресетов.

Режимы работы агента (устанавливается `mode:` в config.yaml):

  1. local_full     — локальный Qwen через Ollama + локальный fine-tune на GPU
  2. cloud_finetune — локальный Qwen через Ollama + fine-tune на арендованной cloud GPU
  3. local_cpu      — маленький локальный Qwen через CPU (1.5B/3B), тренировка экспериментальна
  4. claude_only    — только Claude API, без локальных моделей, только сбор данных в finetune_queue

Пресет задаёт разумные дефолты; любой ключ можно переопределить явно в yaml.
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml
from dotenv import load_dotenv

from . import paths

# Load .env from data_dir if it exists. Pre-init (no data dir yet)
# `paths.env_path()` still returns a valid path (under the would-be
# data_dir); load_dotenv is a no-op when the file is missing.
try:
    load_dotenv(paths.env_path())
except paths.DataDirMissing:
    pass

# Backwards-compat exports: legacy code imports `ROOT` and
# `CONFIG_PATH` directly. ROOT still means the repo root (engine);
# CONFIG_PATH resolves via paths.config_yaml_path() so it points at
# the user's config when one exists.
ROOT = paths.repo_root()
try:
    CONFIG_PATH = paths.config_yaml_path()
except paths.DataDirMissing:
    # Boot before `hrant init`: callers handle the "no config yet"
    # case by constructing Config() with default baked-in values.
    CONFIG_PATH = paths.repo_root() / "config.yaml"  # sentinel; won't exist


# -------- пресеты режимов --------
_COMMON_MODEL_A = {
    "provider": "anthropic",
    "model": "claude-sonnet-4-5",
    "api_key_env": "ANTHROPIC_API_KEY",
    "max_tokens": 2000,
    "temperature": 0.3,
    "tasks": ["task_analysis", "learning", "complex_solving", "verification", "note_creation"],
}

MODE_PRESETS: dict[str, dict] = {
    # ───────── 1. Локальный полный стек (GPU обязателен) ─────────
    "local_full": {
        "model_a": _COMMON_MODEL_A,
        "model_b": {
            "provider": "ollama",
            "model": "qwen2.5:7b-instruct",
            "base_url": "http://localhost:11434",
            "max_tokens": 2000,
            "temperature": 0.3,
            "tasks": ["simple_lookup", "keyword_extraction", "note_search", "quick_answer", "classification"],
        },
        "router": {
            "fallback_to_local": True,
            "auto_shift_after_finetune": True,
            "daily_api_budget_usd": 5.0,
            "estimated_cost_per_call_usd": 0.01,
            "api_ping_cache_seconds": 60,
            "shift_schedule": {
                "v0": {"model_a_pct": 90, "model_b_pct": 10},
                "v1": {"model_a_pct": 60, "model_b_pct": 40},
                "v2": {"model_a_pct": 40, "model_b_pct": 60},
                "v3": {"model_a_pct": 20, "model_b_pct": 80},
                "v5": {"model_a_pct": 10, "model_b_pct": 90},
            },
        },
        "finetune": {
            "enabled": True,
            "training_location": "local",
            "provider": "ollama",
            "base_model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            "inference_model": "qwen2.5:7b-instruct",
            "output_prefix": "qwen-agent",
            "confidence_threshold": 85,
            "training": {
                "epochs": 3,
                "learning_rate": 2.0e-5,
                "batch_size": 4,
                "lora_rank": 16,
                "lora_alpha": 32,
                "warmup_steps": 10,
                "max_seq_length": 2048,
            },
            "validation": {"split": 0.1, "min_examples": 50, "max_examples": 5000},
        },
    },

    # ───────── 2. Cloud fine-tune (нет локального GPU) ─────────
    "cloud_finetune": {
        "model_a": _COMMON_MODEL_A,
        "model_b": {
            "provider": "ollama",
            "model": "qwen2.5:7b-instruct",
            "base_url": "http://localhost:11434",
            "max_tokens": 2000,
            "temperature": 0.3,
            "tasks": ["simple_lookup", "keyword_extraction", "note_search", "quick_answer", "classification"],
        },
        "router": {
            "fallback_to_local": True,
            "auto_shift_after_finetune": True,
            "daily_api_budget_usd": 5.0,
            "estimated_cost_per_call_usd": 0.01,
            "api_ping_cache_seconds": 60,
            "shift_schedule": {
                "v0": {"model_a_pct": 90, "model_b_pct": 10},
                "v1": {"model_a_pct": 60, "model_b_pct": 40},
                "v2": {"model_a_pct": 40, "model_b_pct": 60},
                "v3": {"model_a_pct": 20, "model_b_pct": 80},
                "v5": {"model_a_pct": 10, "model_b_pct": 90},
            },
        },
        "finetune": {
            "enabled": True,
            "training_location": "cloud",
            "provider": "ollama",
            "base_model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
            "inference_model": "qwen2.5:7b-instruct",
            "output_prefix": "qwen-agent",
            "confidence_threshold": 85,
            "training": {
                "epochs": 3,
                "learning_rate": 2.0e-5,
                "batch_size": 4,
                "lora_rank": 16,
                "lora_alpha": 32,
                "warmup_steps": 10,
                "max_seq_length": 2048,
            },
            "validation": {"split": 0.1, "min_examples": 50, "max_examples": 5000},
            "cloud_recipe": {
                "notes": "Запусти train_script.py на арендованной GPU (RunPod/Vast.ai/Colab Pro).",
                "gpu_recommended": "RTX 3090 / A10 / A100",
                "vram_min_gb": 12,
            },
        },
    },

    # ───────── 3. Маленькая локальная модель на CPU ─────────
    "local_cpu": {
        "model_a": _COMMON_MODEL_A,
        "model_b": {
            "provider": "ollama",
            "model": "qwen2.5:1.5b-instruct",  # маленькая модель для CPU
            "base_url": "http://localhost:11434",
            "max_tokens": 1500,
            "temperature": 0.3,
            "tasks": ["simple_lookup", "keyword_extraction", "note_search", "quick_answer", "classification"],
        },
        "router": {
            "fallback_to_local": True,
            # для CPU-модели shift отключён: 1.5B слабовата для сложных задач
            "auto_shift_after_finetune": False,
            "daily_api_budget_usd": 5.0,
            "estimated_cost_per_call_usd": 0.01,
            "api_ping_cache_seconds": 60,
            "shift_schedule": {},
        },
        "finetune": {
            "enabled": True,
            "training_location": "local_cpu",   # экспериментально / медленно
            "provider": "ollama",
            "base_model": "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
            "inference_model": "qwen2.5:1.5b-instruct",
            "output_prefix": "qwen-cpu-agent",
            "confidence_threshold": 85,
            "training": {
                "epochs": 2,
                "learning_rate": 3.0e-5,
                "batch_size": 1,
                "lora_rank": 8,
                "lora_alpha": 16,
                "warmup_steps": 5,
                "max_seq_length": 1024,
            },
            "validation": {"split": 0.1, "min_examples": 50, "max_examples": 2000},
            "cpu_warning": "CPU-тренировка 1.5B может занять 10+ часов на 50 примерах. "
                           "Рассмотри mode: cloud_finetune.",
        },
    },

    # ───────── 4. Только Claude API, без локальной модели ─────────
    "claude_only": {
        "model_a": _COMMON_MODEL_A,
        "model_b": None,   # локальная модель выключена
        "router": {
            "fallback_to_local": False,
            "auto_shift_after_finetune": False,
            "daily_api_budget_usd": 5.0,
            "estimated_cost_per_call_usd": 0.01,
            "api_ping_cache_seconds": 60,
            "shift_schedule": {},
        },
        "finetune": {
            # Данные собираем даже в этом режиме — пригодятся, если перейдёшь на другой mode
            "enabled": False,
            "training_location": "disabled",
            "confidence_threshold": 85,
            "output_prefix": "claude-data",
        },
    },
}


_COMMON_OTHER = {
    "verification": {
        "enabled": True,
        "min_confidence": 70,
        "require_sources": True,
        "always_use_model_a": True,
        "critic_threshold": 50,
        "critic_max_retries": 2,
        # Stop retrying once the request has already consumed this many
        # total tokens (input + output) across all stages. Default keeps
        # a typical self-review w/ one retry under control while letting
        # short tasks retry up to max_retries without tripping.
        "critic_retry_token_budget": 60000,
    },
    "router": {
        # max_tokens granted to the forced tool-less synthesis call
        # that fires when complete_with_tools hits max_iterations.
        # Tool result re-feeding now has per-tool caps (see
        # _compact_tool_result_for_llm), so the synthesis call doesn't
        # need to be as generous as before — but we still want enough
        # room for a real review-style answer. 4000 covers the longest
        # answers we've shipped without truncating.
        "tool_synth_max_tokens": 4000,
        # Hard budget for the WHOLE tool-loop's accumulated input
        # tokens (sum across iterations). Acts as a runaway-guard,
        # NOT the primary token-saving knob (that's the curated
        # synthesis payload — see `_curate_synth_messages_*`). 300k
        # leaves room for legit deep self-reviews while still
        # catching pathological loops before they cost real money.
        "tool_loop_input_budget": 300000,
    },
    "knowledge": {
        "base_dir": "./knowledge",
        "core_memory_max_tokens": 4000,
        "auto_promote_threshold": 10,
        "finetune_min_examples": 50,
        "note_max_tokens": 1500,
    },
    "tts": {
        # Voice output policy. By default the agent replies with a
        # synthesized voice ONLY when the incoming message was a
        # voice message (mirrors human conversational habit — voice
        # in, voice out). Set `enabled_always` to true to make every
        # answer go out as both text + voice.
        "enabled_on_voice_input": True,
        "enabled_always": False,
        # Cap on synthesised text length per turn — protect against
        # 8000-char review-style answers being rendered as 4-minute
        # audio. The voice reply gets the first N chars; user reads
        # the rest in the text bubble.
        "max_chars": 1000,
    },
    "workspace": {
        # Real on-disk tree the agent reads from / writes to. Uploaded
        # files are mirrored from the sha-keyed attachment store into
        # `inbox/` under their original filename so `read_file` works
        # on the path the model already saw in the prompt. Outbox and
        # notes are agent-driven (via the `save_to_workspace` tool).
        "root": "./workspace",
        # Retention sweep, in days. 0 = never auto-delete that subtree.
        # Inbox defaults to 90 days because user uploads accumulate
        # quickly; outbox/notes default to 0 so the agent's own work
        # doesn't vanish on a timer. Turn records (P1 TurnWorkspace
        # persistence — one JSON per Agent.run) accumulate fastest of
        # all, so default 30 days; bump for longer post-mortems.
        "inbox_retention_days": 90,
        "outbox_retention_days": 0,
        "notes_retention_days": 0,
        "turns_retention_days": 30,
    },
    "search": {
        "method": "keyword",
        "fuzzy_threshold": 0.6,
    },
    "server": {
        "host": "0.0.0.0",
        "port": 8000,
    },
    "model_versions": {
        "rollback_enabled": True,
        "eval_on_upgrade": True,
    },
    # Список MCP-серверов. По умолчанию пуст. Пример:
    #   mcp_servers:
    #     - name: filesystem
    #       command: npx
    #       args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/expose"]
    #       enabled: true
    "mcp_servers": [],
}


def _deep_merge(base: dict, overlay: dict) -> dict:
    """Рекурсивно объединяет overlay в base. Overlay всегда побеждает."""
    out = dict(base)
    for k, v in (overlay or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


class Config:
    def __init__(self, path: Path | None = None):
        # Allow tests to pass an explicit path; production callers
        # let paths.config_yaml_path() resolve the right file (data
        # dir > repo root > sensible default).
        if path is None:
            path = paths.config_yaml_path()
        if not Path(path).exists():
            # Fresh install — no config.yaml yet anywhere. Boot with
            # baked-in defaults (claude_only preset + _COMMON_OTHER).
            # `hrant init` writes a real file later.
            raw: dict[str, Any] = {"mode": "claude_only"}
        else:
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

        mode = raw.get("mode", "local_full")
        if mode not in MODE_PRESETS:
            raise ValueError(
                f"Неизвестный mode '{mode}'. "
                f"Доступны: {', '.join(MODE_PRESETS.keys())}"
            )
        self._mode = mode

        # Применяем: common → preset → user
        merged = _deep_merge(_COMMON_OTHER, MODE_PRESETS[mode])
        merged = _deep_merge(merged, {k: v for k, v in raw.items() if k != "mode"})
        self._data = merged

        # Resolve knowledge base dir. Default points the data_dir's
        # knowledge/ subdir (so a split install never accidentally
        # writes to the engine repo). Absolute paths in yaml override.
        kb_raw = self._data["knowledge"].get("base_dir")
        if kb_raw and kb_raw not in ("./knowledge", "knowledge"):
            kb = Path(kb_raw)
            if not kb.is_absolute():
                # Honour the user's relative path against data_dir,
                # which is the user-facing tree, not the engine repo.
                kb = (paths.data_dir(require=False) / kb).resolve()
        else:
            kb = paths.knowledge_dir()
        self._data["knowledge"]["base_dir"] = str(kb)

        # Resolve workspace root the same way.
        ws_raw = (self._data.get("workspace") or {}).get("root")
        if ws_raw and ws_raw not in ("./workspace", "workspace"):
            ws = Path(ws_raw)
            if not ws.is_absolute():
                ws = (paths.data_dir(require=False) / ws).resolve()
        else:
            ws = paths.workspace_dir()
        self._data.setdefault("workspace", {})["root"] = str(ws)

    # ---- dict-like ----
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # ---- режим ----
    @property
    def mode(self) -> str:
        return self._mode

    def is_mode(self, *modes: str) -> bool:
        return self._mode in modes

    # ---- секции ----
    @property
    def model_a(self) -> dict:
        return self._data.get("model_a") or {}

    @property
    def model_b(self) -> dict | None:
        """Возвращает None если локальная модель выключена (claude_only)."""
        cfg = self._data.get("model_b")
        return cfg if cfg else None

    @property
    def router(self) -> dict:
        return self._data.get("router") or {}

    @property
    def knowledge(self) -> dict:
        return self._data["knowledge"]

    @property
    def workspace(self) -> dict:
        return self._data.get("workspace") or {}

    @property
    def tts(self) -> dict:
        return self._data.get("tts") or {}

    @property
    def verification(self) -> dict:
        return self._data.get("verification", {})

    @property
    def search(self) -> dict:
        return self._data.get("search", {})

    @property
    def server(self) -> dict:
        return self._data.get("server", {})

    @property
    def finetune(self) -> dict:
        return self._data.get("finetune") or {}

    @property
    def mcp_servers(self) -> list[dict]:
        """Список конфигов MCP-серверов из config.yaml. Может быть пустым."""
        return list(self._data.get("mcp_servers") or [])

    @property
    def finetune_enabled(self) -> bool:
        return bool(self.finetune.get("enabled", True))

    @property
    def training_location(self) -> str:
        return self.finetune.get("training_location", "local")

    def api_key(self, model_key: str = "model_a") -> str | None:
        cfg = self._data.get(model_key) or {}
        env_name = cfg.get("api_key_env")
        return os.getenv(env_name) if env_name else None


CONFIG = Config()
