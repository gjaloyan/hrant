"""FineTunePipeline: prepare_dataset → train (Unsloth) → register (Ollama).

Requires for real training: unsloth, trl, transformers, datasets, ollama CLI.
All imports are inside methods so the module is importable without these dependencies.
"""
from __future__ import annotations
import json
import subprocess
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
    """output_prefix — new name; output_model — legacy with '-v1' suffix."""
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
        """Curation + split train/val. Returns (train_path, val_path, n_total)."""
        self.progress("prepare", "reading queue")
        examples = self.store.list_all()
        if not examples:
            raise RuntimeError("finetune_queue is empty — nothing to train on")

        self.progress("curate", f"curating {len(examples)} examples")
        good = self.curator.curate(examples)
        if not good:
            raise RuntimeError("0 usable examples after curation")

        min_ex = self.config["validation"]["min_examples"]
        if len(good) < min_ex:
            raise RuntimeError(
                f"need at least {min_ex} usable examples, currently {len(good)}"
            )

        self.progress("boost", "boosting important examples")
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
            f"train={len(train)} val={len(val)} (boosted from {len(good)} unique)",
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
        self.progress("train", "generating train_script.py")
        script = self._write_train_script(train_path, val_path)

        self.progress("train", "running python train_script.py (may take hours)")
        result = subprocess.run(
            ["python", str(script)], capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"training failed: {result.stderr[-2000:]}")
        self.progress("train", "training complete")
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
        self.progress("register", f"registered {tag}={versioned}")
        return versioned

    # -------- orchestrator --------
    def run_full_pipeline(self) -> str:
        loc = CONFIG.training_location
        if not CONFIG.finetune_enabled:
            raise RuntimeError(
                f"fine-tune is disabled in mode: {CONFIG.mode}. "
                "Use 'finetune export' to export the data."
            )
        if loc == "disabled":
            raise RuntimeError(
                f"training_location=disabled in mode: {CONFIG.mode}. "
                "Data is being collected, but training will not start."
            )
        if loc == "cloud":
            raise RuntimeError(
                "training_location=cloud — run 'finetune export-cloud' "
                "to prepare the package and upload to a rented GPU. "
                "After training: 'finetune import-gguf <path_to_gguf>'."
            )
        if loc == "local_cpu":
            warn = self.config.get("cpu_warning", "")
            self.progress("warning", f"CPU training started. {warn}")

        train_path, val_path, _ = self.prepare_dataset()
        gguf = self.train_with_unsloth(train_path, val_path)
        tag = VERSIONS.next_tag()
        model_name = self.register_with_ollama(gguf, tag)
        self.progress("done", f"model {model_name} is ready, tag {tag}")
        return model_name

    # -------- cloud export --------
    def export_for_cloud(self) -> Path:
        """
        Prepares a package for running fine-tune on a rented GPU:
        train.jsonl, val.jsonl, train_script.py, config.json, README_CLOUD.md.
        Returns the path to the package directory.
        """
        train_path, val_path, n_total = self.prepare_dataset()
        tag = VERSIONS.next_tag()
        pkg = self.models_dir / f"cloud_export_{tag}"
        pkg.mkdir(parents=True, exist_ok=True)

        # copy train/val
        for src in (train_path, val_path):
            (pkg / src.name).write_bytes(src.read_bytes())

        # prepare train_script (reference version, runnable on cloud GPU)
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

        # instructions
        recipe = self.config.get("cloud_recipe", {}) or {}
        prefix = _output_prefix(self.config)
        readme = f"""# Cloud Fine-Tune Package ({tag})

Total examples: **{n_total}**
Base model: `{self.config["base_model"]}`
Target name: `{prefix}-{tag}`

## Requirements
- GPU: {recipe.get("gpu_recommended", "RTX 3090 / A10 / A100")}
- VRAM: ≥ {recipe.get("vram_min_gb", 12)} GB
- pip: `unsloth trl transformers datasets`

## Running (on RunPod / Vast.ai / Colab Pro)

```bash
pip install unsloth trl transformers datasets
python train_script.py
```

After completion `{prefix}-gguf/unsloth.Q4_K_M.gguf` will appear — download that file.

## Importing back (locally)

```bash
# copy the gguf to a local folder and run the REPL command:
hrant chat 'finetune import-gguf /path/to/unsloth.Q4_K_M.gguf --tag {tag}'
```

This will register the model in Ollama as `{prefix}-{tag}`.

## Notes
{recipe.get("notes", "—")}
"""
        (pkg / "README_CLOUD.md").write_text(readme, encoding="utf-8")

        self.progress("export", f"package ready: {pkg}")
        return pkg

    # -------- gguf import --------
    def import_gguf(self, gguf_path: Path | str, *, tag: str | None = None) -> str:
        """Imports an already-trained gguf (from the cloud) into local Ollama."""
        src = Path(gguf_path)
        if not src.exists():
            raise FileNotFoundError(f"gguf not found: {gguf_path}")

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
        self.progress("import", f"registered {tag}={versioned}")
        return versioned

    # -------- openai export --------
    def export_for_openai(self) -> Path:
        train_path, _, _ = self.prepare_dataset()
        self.progress(
            "export",
            "train.jsonl is ready — upload via openai api fine_tunes.create -t train.jsonl -m gpt-4o-mini",
        )
        return train_path
