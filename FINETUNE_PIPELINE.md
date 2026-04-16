# SELF-LEARNING: Fine-Tuning Pipeline
# Дополнение к CLAUDE_CODE_PROMPT.md — Уровень 3: обучение в веса модели

---

## OVERVIEW

Три этапа превращения опыта в знание модели:

```
РАБОТА НАД ПРОЕКТОМ
        │
        ▼
┌──────────────────┐
│ 1. СБОР ДАННЫХ   │  Агент автоматически сохраняет хорошие Q&A пары
│    (автоматически)│  в finetune_queue.jsonl при каждом верифицированном
│                  │  ответе с confidence > 85%
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 2. КУРАЦИЯ       │  Пользователь или агент проверяет и фильтрует
│    (полуавтомат.) │  собранные примеры перед обучением
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 3. FINE-TUNE     │  Запуск дообучения на открытой модели
│    (по команде)  │  Знания "впечатываются" в веса
└──────────────────┘
```

---

## ЭТАП 1: АВТОМАТИЧЕСКИЙ СБОР ДАННЫХ

### Когда сохранять в finetune queue

Агент автоматически добавляет Q&A пару когда ВСЕ условия выполнены:
- Ответ прошёл самоверификацию
- Confidence >= 85%
- Ответ основан на конспектах (не выдуман)
- Пользователь не отклонил ответ (не сказал "неправильно")

### Когда НЕ сохранять
- Confidence < 85%
- Пользователь поправил ответ
- Ответ содержит неверифицированные утверждения
- Вопрос слишком общий ("что такое электричество")

### Формат записи (finetune_queue.jsonl)

Каждая строка — отдельный JSON-объект:

```jsonl
{"messages":[{"role":"system","content":"You are an expert automation engineer. Answer precisely based on your knowledge. If unsure, say so."},{"role":"user","content":"How to connect MAX485 to Arduino Uno for RS-485 communication?"},{"role":"assistant","content":"Connect DI to TX (pin 1), RO to RX (pin 0), join DE+RE to digital pin D2. VCC to 5V, GND to GND. Set D2 HIGH before sending, call Serial.flush(), then set D2 LOW to return to receive mode. Add 120 Ohm termination resistors at both ends of the RS-485 bus."}],"metadata":{"source_notes":["rs-485","max485","arduino uno"],"confidence":94,"project":"factory_automation","timestamp":"2025-01-15T14:30:00","verified":true,"category":"profession"}}
```

### Типы данных для сбора

```python
FINETUNE_CATEGORIES = {
    "factual_qa": {
        # Вопрос → точный ответ из конспектов
        "description": "Фактические вопросы с проверенными ответами",
        "example_q": "What is the max cable length for RS-485?",
        "example_a": "Up to 1200 meters at lower baud rates.",
        "priority": "high",
    },
    "troubleshooting": {
        # Проблема → диагностика → решение
        "description": "Реальные проблемы и их решения из опыта",
        "example_q": "RS-485 communication drops after 100 meters",
        "example_a": "Check: 1) Termination resistors 120 Ohm at both ends, 2) Bias resistors if bus idle, 3) Cable shielding and grounding, 4) Baud rate - reduce for longer distances",
        "priority": "highest",  # Самый ценный тип — реальный опыт
    },
    "procedure": {
        # Как сделать X → пошаговая инструкция
        "description": "Пошаговые процедуры",
        "example_q": "How to calibrate a 4-20mA pressure sensor?",
        "example_a": "1) Apply 0 pressure → adjust zero to 4.000mA, 2) Apply full scale → adjust span to 20.000mA, 3) Check linearity at 25%, 50%, 75%",
        "priority": "high",
    },
    "decision": {
        # Почему выбрали X а не Y
        "description": "Инженерные решения с обоснованием",
        "example_q": "Why use Modbus RTU over Modbus TCP for this project?",
        "example_a": "Chosen RTU because: existing RS-485 wiring in the plant, PLC supports only serial, fewer network components needed, proven reliability in this environment",
        "priority": "medium",
    },
    "correction": {
        # Пользователь поправил агента → учимся на ошибке
        "description": "Исправления от пользователя — учимся на ошибках",
        "example_q": "What voltage does the sensor X need?",
        "example_a": "Sensor X requires 24VDC (NOT 12VDC as commonly mistaken). The 12V version was discontinued in 2023.",
        "priority": "highest",  # Ошибки — самый ценный учебный материал
    },
}
```

### Автосбор в agent.py

Добавить в конец agent loop (шаг 6):

```python
# After verification passes
if verification["confidence"] >= 85 and verification["is_verified"]:
    finetune.add_example(
        system_prompt="You are an expert automation engineer...",
        user_message=original_task,
        assistant_response=answer,
        metadata={
            "source_notes": list(loaded_knowledge.keys()),
            "confidence": verification["confidence"],
            "project": current_project,
            "category": detect_category(original_task, answer),
            "timestamp": datetime.now().isoformat(),
        }
    )

# ALSO: when user corrects the agent
if user_says_wrong:
    finetune.add_example(
        system_prompt="You are an expert automation engineer...",
        user_message=original_task,
        assistant_response=corrected_answer,  # The CORRECTED version
        metadata={
            "category": "correction",
            "original_wrong_answer": wrong_answer,  # Save mistake too
            "confidence": 100,  # User-verified = 100%
        }
    )
```

---

## ЭТАП 2: КУРАЦИЯ ДАННЫХ

### Автоматическая фильтрация

```python
class FinetuneDataCurator:
    """Автоматическая проверка качества данных перед обучением."""

    def curate(self, examples: list[dict]) -> list[dict]:
        good = []
        for ex in examples:
            score = self.quality_score(ex)
            if score >= 0.7:
                good.append(ex)
        return good

    def quality_score(self, example: dict) -> float:
        score = 0.0
        msg = example["messages"]
        answer = msg[2]["content"]  # assistant response
        meta = example.get("metadata", {})

        # Length check: not too short, not too long
        if 50 < len(answer) < 2000:
            score += 0.2

        # Has confidence metadata
        if meta.get("confidence", 0) >= 85:
            score += 0.2

        # Has source notes (grounded in knowledge)
        if meta.get("source_notes"):
            score += 0.2

        # Category bonus
        if meta.get("category") in ["correction", "troubleshooting"]:
            score += 0.3  # Most valuable
        elif meta.get("category") in ["factual_qa", "procedure"]:
            score += 0.2

        # Dedup check: not too similar to existing examples
        if not self.is_duplicate(example):
            score += 0.1

        return min(score, 1.0)

    def is_duplicate(self, example: dict) -> bool:
        """Check if similar Q&A already exists in queue."""
        # Simple: check if question is >80% similar to existing
        # Advanced: use embeddings for semantic similarity
        pass
```

### Ручная курация (UI)

В веб-интерфейсе — страница `/finetune`:

```
┌─────────────────────────────────────────────┐
│  Fine-Tune Data Manager                     │
│                                             │
│  Total examples: 73                         │
│  Quality filtered: 61                       │
│  Ready for training: ✅ YES (>50)           │
│                                             │
│  ┌─────────────────────────────────────┐    │
│  │ [✓] Q: How to connect MAX485?       │    │
│  │     A: Connect DI to TX...          │    │
│  │     Confidence: 94% | Source: 3     │    │
│  │     Category: procedure             │    │
│  │     [Edit] [Remove] [Boost]         │    │
│  ├─────────────────────────────────────┤    │
│  │ [✓] Q: Why RS-485 drops at 100m?   │    │
│  │     A: Check termination...         │    │
│  │     Confidence: 91% | Source: 2     │    │
│  │     Category: troubleshooting       │    │
│  │     [Edit] [Remove] [Boost]         │    │
│  └─────────────────────────────────────┘    │
│                                             │
│  [Export JSONL]  [Start Fine-Tune]          │
└─────────────────────────────────────────────┘
```

- **Edit** — исправить ответ перед обучением
- **Remove** — убрать некачественный пример
- **Boost** — пометить как особо важный (будет повторён в датасете 2-3 раза)

---

## ЭТАП 3: FINE-TUNING

### Поддерживаемые провайдеры

```yaml
# config.yaml — fine-tune section
finetune:
  provider: "unsloth"          # unsloth (recommended) | huggingface | openai
  base_model: "unsloth/Qwen2.5-7B-Instruct"  # Qwen 2.5 7B
  output_model: "my-agent-v1" # Имя дообученной модели
  
  # Параметры обучения
  training:
    epochs: 3
    learning_rate: 2e-5
    batch_size: 4
    lora_rank: 16              # LoRA — не трогаем все веса, только адаптеры
    lora_alpha: 32
    warmup_steps: 10
    max_seq_length: 2048
  
  # Валидация
  validation:
    split: 0.1                 # 10% данных на валидацию
    min_examples: 50
    max_examples: 5000
```

### Pipeline для Ollama + LoRA

```python
# backend/finetune_pipeline.py

import subprocess
import json
from pathlib import Path

class FineTunePipeline:
    """Пайплайн дообучения через Unsloth/HuggingFace + экспорт в Ollama."""

    def __init__(self, config: dict):
        self.config = config
        self.data_path = Path("knowledge/finetune_queue.jsonl")
        self.output_dir = Path("models/")
        self.output_dir.mkdir(exist_ok=True)

    def prepare_dataset(self) -> tuple[Path, Path]:
        """Подготовка данных: фильтрация + разделение train/val."""
        examples = []
        with open(self.data_path) as f:
            for line in f:
                ex = json.loads(line)
                examples.append(ex)

        # Курация
        curator = FinetuneDataCurator()
        good = curator.curate(examples)

        # Boost important examples (repeat 2-3x in dataset)
        boosted = []
        for ex in good:
            boosted.append(ex)
            if ex.get("metadata", {}).get("category") in ["correction", "troubleshooting"]:
                boosted.append(ex)  # Repeat important ones
                if ex.get("metadata", {}).get("boosted"):
                    boosted.append(ex)  # Triple if manually boosted

        # Split train/val
        split_idx = int(len(boosted) * 0.9)
        train = boosted[:split_idx]
        val = boosted[split_idx:]

        train_path = self.output_dir / "train.jsonl"
        val_path = self.output_dir / "val.jsonl"

        for path, data in [(train_path, train), (val_path, val)]:
            with open(path, "w") as f:
                for ex in data:
                    # Strip metadata for training, keep only messages
                    clean = {"messages": ex["messages"]}
                    f.write(json.dumps(clean, ensure_ascii=False) + "\n")

        return train_path, val_path

    def train_with_unsloth(self, train_path: Path, val_path: Path) -> Path:
        """
        Fine-tune с Unsloth (быстрый LoRA fine-tuning).
        Требует: pip install unsloth
        """
        script = f'''
import json
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="{self.config['base_model']}",
    max_seq_length={self.config['training']['max_seq_length']},
    load_in_4bit=True,
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r={self.config['training']['lora_rank']},
    lora_alpha={self.config['training']['lora_alpha']},
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
)

# Load dataset
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]

train_data = load_jsonl("{train_path}")
val_data = load_jsonl("{val_path}")

def format_chat(example):
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {{"text": text}}

train_dataset = Dataset.from_list(train_data).map(format_chat)
val_dataset = Dataset.from_list(val_data).map(format_chat)

# Train
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    max_seq_length={self.config['training']['max_seq_length']},
    args=TrainingArguments(
        output_dir="./models/checkpoints",
        num_train_epochs={self.config['training']['epochs']},
        per_device_train_batch_size={self.config['training']['batch_size']},
        learning_rate={self.config['training']['learning_rate']},
        warmup_steps={self.config['training']['warmup_steps']},
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    ),
)

trainer.train()

# Save
model.save_pretrained("./models/{self.config['output_model']}")
tokenizer.save_pretrained("./models/{self.config['output_model']}")

# Export to GGUF for Ollama
model.save_pretrained_gguf(
    "./models/{self.config['output_model']}-gguf",
    tokenizer,
    quantization_method="q4_k_m"
)
'''
        script_path = self.output_dir / "train_script.py"
        script_path.write_text(script)

        result = subprocess.run(
            ["python", str(script_path)],
            capture_output=True, text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Training failed: {result.stderr}")

        return self.output_dir / f"{self.config['output_model']}-gguf"

    def register_with_ollama(self, gguf_path: Path) -> str:
        """Регистрирует дообученную модель в Ollama."""
        model_name = self.config["output_model"]

        modelfile = f"""FROM {gguf_path}/unsloth.Q4_K_M.gguf
SYSTEM You are an expert engineer with deep knowledge in automation, electronics, and industrial systems. Answer precisely and cite your experience when relevant.
PARAMETER temperature 0.3
PARAMETER num_ctx 4096
"""
        modelfile_path = self.output_dir / "Modelfile"
        modelfile_path.write_text(modelfile)

        subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            check=True
        )

        return model_name

    def run_full_pipeline(self) -> str:
        """Полный пайплайн: подготовка → обучение → регистрация."""
        print("📊 Preparing dataset...")
        train_path, val_path = self.prepare_dataset()

        print("🔥 Training model with LoRA...")
        gguf_path = self.train_with_unsloth(train_path, val_path)

        print("📦 Registering with Ollama...")
        model_name = self.register_with_ollama(gguf_path)

        print(f"✅ Model '{model_name}' ready!")
        print(f"   Switch in config.yaml: model: '{model_name}'")

        return model_name
```

### Pipeline для OpenAI-compatible fine-tuning

```python
def export_for_openai(self) -> Path:
    """Экспорт в формат OpenAI fine-tuning API."""
    train_path, _ = self.prepare_dataset()
    
    # OpenAI format is same JSONL with messages
    # Just upload via API:
    # openai api fine_tunes.create -t train.jsonl -m gpt-4o-mini
    
    print(f"Exported to {train_path}")
    print("Upload with: openai api fine_tunes.create -t train.jsonl -m gpt-4o-mini")
    return train_path
```

---

## AUTO-EVOLUTION: Модель которая улучшает сама себя

### Цикл саморазвития

```
Месяц 1: Агент работает на проекте
          ├── Собирает Q&A пары (автоматически)
          ├── Копит troubleshooting кейсы
          └── Записывает corrections от пользователя

Месяц 2: Накопилось 100+ примеров
          ├── Автоматическая курация (убирает мусор)
          ├── Пользователь просматривает и одобряет
          └── Запуск fine-tuning → модель v1

Месяц 3: Агент работает на v1 (уже умнее)
          ├── Новые Q&A пары — более сложные
          ├── Модель ошибается реже → corrections реже
          └── Качество данных растёт

Месяц 4: Ещё 100+ примеров
          └── Fine-tune v1 → v2 (ещё умнее)

...и так далее. Каждая версия лучше предыдущей.
```

### Версионирование модели

```yaml
# config.yaml
model_versions:
  v0: "mistral:7b"                    # Базовая модель
  v1: "my-agent-v1"                   # +100 примеров по автоматизации
  v2: "my-agent-v2"                   # +200 примеров, включая ошибки
  current: "v2"
  
  # Rollback если новая версия хуже
  rollback_enabled: true
  eval_on_upgrade: true               # Прогнать тесты перед переключением
```

### Автоматическая оценка перед переключением

```python
class ModelEvaluator:
    """Прогоняет тестовые вопросы на старой и новой модели, сравнивает."""

    def __init__(self):
        # Тестовые вопросы из реального опыта
        self.test_questions = self._load_test_set()

    def compare_models(self, old_model: str, new_model: str) -> dict:
        old_scores = []
        new_scores = []

        for q in self.test_questions:
            old_answer = call_llm(old_model, q["question"])
            new_answer = call_llm(new_model, q["question"])

            old_score = self._evaluate(old_answer, q["expected"])
            new_score = self._evaluate(new_answer, q["expected"])

            old_scores.append(old_score)
            new_scores.append(new_score)

        old_avg = sum(old_scores) / len(old_scores)
        new_avg = sum(new_scores) / len(new_scores)

        return {
            "old_model": old_model,
            "new_model": new_model,
            "old_score": old_avg,
            "new_score": new_avg,
            "improvement": new_avg - old_avg,
            "should_upgrade": new_avg > old_avg,
            "details": list(zip(
                [q["question"] for q in self.test_questions],
                old_scores,
                new_scores
            )),
        }
```

---

## КОМАНДЫ ПОЛЬЗОВАТЕЛЯ

Добавить в agent.py обработку:

```
"finetune status"        → показать сколько примеров собрано, готовность
"finetune review"        → открыть UI курации данных
"finetune start"         → запустить дообучение
"finetune compare"       → сравнить старую и новую модель
"finetune switch v2"     → переключиться на версию v2
"finetune rollback"      → откатиться к предыдущей версии
"learn this to model"    → принудительно добавить текущий Q&A в finetune queue
```

---

## API ENDPOINTS (добавить к основному промпту)

```
GET  /api/finetune/status          # Статус: кол-во примеров, готовность
GET  /api/finetune/examples        # Список собранных примеров для курации
PUT  /api/finetune/examples/{id}   # Редактировать пример
DELETE /api/finetune/examples/{id} # Удалить пример
POST /api/finetune/examples/{id}/boost  # Пометить как важный
POST /api/finetune/start           # Запустить дообучение
GET  /api/finetune/progress        # Прогресс обучения (SSE stream)
POST /api/finetune/compare         # Сравнить модели
POST /api/finetune/switch          # Переключить на новую модель
POST /api/finetune/rollback        # Откат
GET  /api/finetune/export          # Скачать JSONL
GET  /api/model/versions           # Список всех версий модели
```

---

## HARDWARE REQUIREMENTS

```
Minimum (Claude API + Qwen inference only, no fine-tuning):
- CPU: 4 cores
- RAM: 8 GB
- GPU: не нужен (Qwen 7B Q4 работает на CPU, медленно но работает)
- Disk: 10 GB
- Setup: Anthropic API key + ollama pull qwen2.5:7b

Recommended (Claude API + Qwen inference + fine-tuning):
- CPU: 8+ cores  
- RAM: 32 GB
- GPU: NVIDIA RTX 3060 12GB (minimum for Qwen 2.5 7B LoRA fine-tuning)
- Disk: 50 GB
- Setup: Anthropic API key + Ollama + Unsloth
- Fine-tune time: ~30-60 min for 100 examples

Optimal (fast inference + fast fine-tuning):
- CPU: 16+ cores
- RAM: 64 GB
- GPU: NVIDIA RTX 4090 24GB
- Disk: 100 GB
- Qwen 2.5 7B runs at full speed, fine-tuning in ~15 min
- Can also run Qwen 2.5 14B for even better local reasoning

Budget option (API only, no local model):
- Any computer with internet
- Claude Sonnet API for everything
- No fine-tuning, but core_memory + knowledge notes still work
- Cost: ~$5-15/day depending on usage
```
