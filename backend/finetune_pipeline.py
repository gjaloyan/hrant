"""FineTunePipeline: prepare_dataset → train (Unsloth) → register (Ollama).

Требует для реального обучения: unsloth, trl, transformers, datasets, ollama CLI.
Все импорты — только внутри методов, чтобы модуль был импортируем без этих зависимостей.
"""
from __future__ import annotations
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Callable

from .config import CONFIG
from .finetune import store as finetune_store
from .finetune_curator import FinetuneDataCurator
from .model_versions import VERSIONS

ProgressCB = Callable[[str, str], None]  # (stage, message)


def _default_config() -> dict:
    return CONFIG.finetune or {
        "provider": "ollama",
        "base_model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "inference_model": "qwen2.5:7b-instruct",
        "output_prefix": "qwen-agent",
        "training": {
            "epochs": 3,
            "learning_rate": 2e-5,
            "batch_size": 4,
            "lora_rank": 16,
            "lora_alpha": 32,
            "warmup_steps": 10,
            "max_seq_length": 2048,
        },
        "validation": {"split": 0.1, "min_examples": 50, "max_examples": 5000},
    }


def _output_prefix(cfg: dict) -> str:
    """output_prefix — новое имя; output_model — legacy с суффиксом '-v1'."""
    if cfg.get("output_prefix"):
        return cfg["output_prefix"]
    legacy = cfg.get("output_model", "qwen-agent")
    return legacy.rsplit("-v", 1)[0] if "-v" in legacy else legacy


class FineTunePipeline:
    def __init__(self, progress: ProgressCB | None = None):
        self.config = _default_config()
        self.store = finetune_store()
        self.curator = FinetuneDataCurator()
        self.progress = progress or (lambda s, m: None)

        from .knowledge_manager import KM
        self.models_dir = KM.base.parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # -------- step 1 --------
    def prepare_dataset(self) -> tuple[Path, Path, int]:
        """Курация + split train/val. Возвращает (train_path, val_path, n_total)."""
        self.progress("prepare", "читаю очередь")
        examples = self.store.list_all()
        if not examples:
            raise RuntimeError("finetune_queue пуст — нечего обучать")

        self.progress("curate", f"курирую {len(examples)} примеров")
        good = self.curator.curate(examples)
        if not good:
            raise RuntimeError("после курации 0 пригодных примеров")

        min_ex = self.config["validation"]["min_examples"]
        if len(good) < min_ex:
            raise RuntimeError(
                f"нужно минимум {min_ex} пригодных примеров, сейчас {len(good)}"
            )

        self.progress("boost", "boosting важных примеров")
        boosted = self.curator.apply_boosting(good)

        split = self.config["validation"]["split"]
        split_idx = int(len(boosted) * (1.0 - split))
        train = boosted[:split_idx]
        val = boosted[split_idx:] or boosted[-1:]

        train_path = self.models_dir / "train.jsonl"
        val_path = self.models_dir / "val.jsonl"
        for path, data in [(train_path, train), (val_path, val)]:
            with path.open("w", encoding="utf-8") as f:
                for p in data:
                    clean = {"messages": [m.model_dump() for m in p.messages]}
                    f.write(json.dumps(clean, ensure_ascii=False) + "\n")

        self.progress(
            "prepared",
            f"train={len(train)} val={len(val)} (boosted из {len(good)} уникальных)",
        )
        return train_path, val_path, len(good)

    # -------- step 2: unsloth --------
    def _write_train_script(self, train_path: Path, val_path: Path) -> Path:
        tr = self.config["training"]
        out = _output_prefix(self.config)  # e.g. "qwen-agent"
        script = f'''"""Auto-generated Unsloth LoRA fine-tune script."""
import json
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name={self.config["base_model"]!r},
    max_seq_length={tr["max_seq_length"]},
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r={tr["lora_rank"]},
    lora_alpha={tr["lora_alpha"]},
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)


def load_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


train_data = load_jsonl({str(train_path)!r})
val_data = load_jsonl({str(val_path)!r})


def fmt(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False,
    )
    return {{"text": text}}


train_ds = Dataset.from_list(train_data).map(fmt)
val_ds = Dataset.from_list(val_data).map(fmt)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    dataset_text_field="text",
    max_seq_length={tr["max_seq_length"]},
    args=TrainingArguments(
        output_dir={str(self.models_dir / "checkpoints")!r},
        num_train_epochs={tr["epochs"]},
        per_device_train_batch_size={tr["batch_size"]},
        learning_rate={tr["learning_rate"]},
        warmup_steps={tr["warmup_steps"]},
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
)
trainer.train()

model.save_pretrained({str(self.models_dir / out)!r})
tokenizer.save_pretrained({str(self.models_dir / out)!r})

model.save_pretrained_gguf(
    {str(self.models_dir / (out + "-gguf"))!r},
    tokenizer,
    quantization_method="q4_k_m",
)
'''
        path = self.models_dir / "train_script.py"
        path.write_text(script, encoding="utf-8")
        return path

    def train_with_unsloth(self, train_path: Path, val_path: Path) -> Path:
        self.progress("train", "генерирую train_script.py")
        script = self._write_train_script(train_path, val_path)

        self.progress("train", "запускаю python train_script.py (может занять часы)")
        result = subprocess.run(
            ["python", str(script)], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"training failed: {result.stderr[-2000:]}")
        self.progress("train", "обучение завершено")
        return self.models_dir / f"{_output_prefix(self.config)}-gguf"

    # -------- step 3: ollama --------
    def register_with_ollama(self, gguf_dir: Path, tag: str) -> str:
        prefix = _output_prefix(self.config)
        versioned = f"{prefix}-{tag}"  # qwen-agent-v1, qwen-agent-v2, ...
        modelfile = (
            f"FROM {gguf_dir}/unsloth.Q4_K_M.gguf\n"
            "SYSTEM You are an expert automation engineer with deep knowledge in "
            "industrial electronics, PLCs, sensors, fieldbus protocols (RS-485, Modbus, "
            "CAN), and control systems. Answer precisely, cite real-world experience. "
            "If unsure, say so explicitly.\n"
            "PARAMETER temperature 0.3\n"
            "PARAMETER num_ctx 4096\n"
        )
        modelfile_path = self.models_dir / "Modelfile"
        modelfile_path.write_text(modelfile, encoding="utf-8")

        self.progress("register", f"ollama create {versioned}")
        subprocess.run(
            ["ollama", "create", versioned, "-f", str(modelfile_path)],
            check=True,
        )

        count = self.store.count()
        VERSIONS.register(
            tag=tag,
            model_id=versioned,
            examples_count=count,
            notes=f"fine-tune from {self.config['base_model']}",
        )
        self.progress("register", f"зарегистрирована {tag}={versioned}")
        return versioned

    # -------- orchestrator --------
    def run_full_pipeline(self) -> str:
        loc = CONFIG.training_location
        if not CONFIG.finetune_enabled:
            raise RuntimeError(
                f"fine-tune выключен в режиме mode: {CONFIG.mode}. "
                "Используй 'finetune export' для выгрузки данных."
            )
        if loc == "disabled":
            raise RuntimeError(
                f"training_location=disabled в mode: {CONFIG.mode}. "
                "Данные собираются, но обучение не запускается."
            )
        if loc == "cloud":
            raise RuntimeError(
                "training_location=cloud — запусти 'finetune export-cloud' "
                "для подготовки пакета и загрузи на арендованную GPU. "
                "После обучения: 'finetune import-gguf <путь_к_gguf>'."
            )
        if loc == "local_cpu":
            warn = self.config.get("cpu_warning", "")
            self.progress("warning", f"CPU training запущен. {warn}")

        train_path, val_path, _ = self.prepare_dataset()
        gguf = self.train_with_unsloth(train_path, val_path)
        tag = VERSIONS.next_tag()
        model_name = self.register_with_ollama(gguf, tag)
        self.progress("done", f"модель {model_name} готова, тег {tag}")
        return model_name

    # -------- cloud export --------
    def export_for_cloud(self) -> Path:
        """
        Готовит пакет для запуска fine-tune на арендованной GPU:
        train.jsonl, val.jsonl, train_script.py, config.json, README_CLOUD.md.
        Возвращает путь к директории пакета.
        """
        train_path, val_path, n_total = self.prepare_dataset()
        tag = VERSIONS.next_tag()
        pkg = self.models_dir / f"cloud_export_{tag}"
        pkg.mkdir(parents=True, exist_ok=True)

        # копируем train/val
        for src in (train_path, val_path):
            (pkg / src.name).write_bytes(src.read_bytes())

        # готовим train_script (ref-версия, runnable на cloud GPU)
        script_src = self._write_train_script(
            pkg / "train.jsonl", pkg / "val.jsonl"
        )
        (pkg / "train_script.py").write_text(
            script_src.read_text(encoding="utf-8"), encoding="utf-8"
        )

        # config
        (pkg / "config.json").write_text(
            json.dumps(self.config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # инструкция
        recipe = self.config.get("cloud_recipe", {}) or {}
        prefix = _output_prefix(self.config)
        readme = f"""# Cloud Fine-Tune Package ({tag})

Всего примеров: **{n_total}**
Base model: `{self.config["base_model"]}`
Target name: `{prefix}-{tag}`

## Требования
- GPU: {recipe.get("gpu_recommended", "RTX 3090 / A10 / A100")}
- VRAM: ≥ {recipe.get("vram_min_gb", 12)} GB
- pip: `unsloth trl transformers datasets`

## Запуск (на RunPod / Vast.ai / Colab Pro)

```bash
pip install unsloth trl transformers datasets
python train_script.py
```

После завершения появится `{prefix}-gguf/unsloth.Q4_K_M.gguf` — скачай этот файл.

## Импорт обратно (локально)

```bash
# скопируй gguf в локальную папку и запусти REPL-команду:
hrant chat 'finetune import-gguf /path/to/unsloth.Q4_K_M.gguf --tag {tag}'
```

Это зарегистрирует модель в Ollama как `{prefix}-{tag}`.

## Заметки
{recipe.get("notes", "—")}
"""
        (pkg / "README_CLOUD.md").write_text(readme, encoding="utf-8")

        self.progress("export", f"пакет готов: {pkg}")
        return pkg

    # -------- gguf import --------
    def import_gguf(self, gguf_path: Path | str, *, tag: str | None = None) -> str:
        """Импортирует уже обученный gguf (из облака) в локальную Ollama."""
        src = Path(gguf_path)
        if not src.exists():
            raise FileNotFoundError(f"gguf не найден: {gguf_path}")

        # Если передан файл — используем его директорию; если директория — ищем gguf внутри
        gguf_dir = src if src.is_dir() else src.parent

        if tag is None:
            tag = VERSIONS.next_tag()

        prefix = _output_prefix(self.config)
        versioned = f"{prefix}-{tag}"

        modelfile = (
            f"FROM {src}\n"
            "SYSTEM You are an expert automation engineer with deep knowledge in "
            "industrial electronics, PLCs, sensors, fieldbus protocols (RS-485, Modbus, "
            "CAN), and control systems.\n"
            "PARAMETER temperature 0.3\n"
            "PARAMETER num_ctx 4096\n"
        )
        modelfile_path = self.models_dir / f"Modelfile.{tag}"
        modelfile_path.write_text(modelfile, encoding="utf-8")

        self.progress("import", f"ollama create {versioned}")
        subprocess.run(
            ["ollama", "create", versioned, "-f", str(modelfile_path)],
            check=True,
        )

        VERSIONS.register(
            tag=tag,
            model_id=versioned,
            examples_count=self.store.count(),
            notes=f"imported from {src}",
        )
        self.progress("import", f"зарегистрирована {tag}={versioned}")
        return versioned

    # -------- openai export --------
    def export_for_openai(self) -> Path:
        train_path, _, _ = self.prepare_dataset()
        self.progress(
            "export",
            "готов train.jsonl — загрузи через openai api fine_tunes.create -t train.jsonl -m gpt-4o-mini",
        )
        return train_path
