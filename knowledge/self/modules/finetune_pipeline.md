---
module: backend/finetune_pipeline.py
category: self
kind: module
updated: 2026-05-06T05:50:26.630172+00:00
source_mtime: 2026-04-07T08:12:21.824085+00:00
loc: 372
truncated: false
---

# backend/finetune_pipeline.py

## Purpose
Модуль реализует пайплайн fine-tuning: подготавливает и курирует датасет из очереди, генерирует и запускает Unsloth/TRL скрипт обучения, экспортирует пакеты для облачного обучения, импортирует готовые GGUF-файлы и регистрирует полученные модели в Ollama с версионированием.

## Public interface
- `ProgressCB` (constant) - Тип callback-функции для сообщений прогресса вида (stage, message).
- `FineTunePipeline` (class) - Оркестратор подготовки данных, обучения, экспорта, импорта и регистрации fine-tuned моделей.
- `FineTunePipeline.prepare_dataset` (function) - Курирует примеры, применяет boosting, делит их на train/val и пишет JSONL-файлы.
- `FineTunePipeline.train_with_unsloth` (function) - Генерирует train_script.py и запускает локальное обучение через Python/Unsloth.
- `FineTunePipeline.register_with_ollama` (function) - Создает Modelfile, регистрирует GGUF-модель в Ollama и записывает версию.
- `FineTunePipeline.run_full_pipeline` (function) - Запускает полный локальный пайплайн prepare_dataset → train_with_unsloth → register_with_ollama.
- `FineTunePipeline.export_for_cloud` (function) - Создает пакет для облачного обучения с train/val JSONL, скриптом, config.json и README.
- `FineTunePipeline.import_gguf` (function) - Импортирует готовый GGUF-файл или директорию и регистрирует модель в локальной Ollama.
- `FineTunePipeline.export_for_openai` (function) - Готовит train.jsonl для последующей загрузки в OpenAI fine-tuning API.

## Dependencies
- backend.config
- backend.finetune
- backend.finetune_curator
- backend.model_versions
- backend.knowledge_manager

## Notes
Тяжелые ML-зависимости не импортируются при загрузке модуля: они попадают только в сгенерированный train_script.py. Пайплайн активно использует файловую систему и внешние команды python и ollama, а ошибки обучения передаются через RuntimeError с хвостом stderr. Для запуска полного обучения учитываются CONFIG.finetune_enabled и CONFIG.training_location.
