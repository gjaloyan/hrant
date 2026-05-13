# Fine-Tuning Pipeline

Три этапа превращения опыта в знание модели.

## Этап 1 — автосбор
Агент сохраняет Q&A в `finetune_queue.jsonl` после каждого верифицированного ответа при:
- `confidence ≥ 85%` (порог в `config.yaml → finetune.confidence_threshold`);
- есть `source_notes` (ответ основан на заметках);
- `verified=True` (нет противоречий).

Формат записи — OpenAI chat-style с метаданными:
```jsonl
{"id":"a1b2c3...","messages":[{"role":"system","content":"..."},{"role":"user","content":"Q"},{"role":"assistant","content":"A"}],"metadata":{"source_notes":["RS-485"],"confidence":94,"project":null,"timestamp":"...","verified":true,"category":"procedure","boosted":false}}
```

Категории (автодетект): `factual_qa`, `troubleshooting`, `procedure`, `decision`, `correction`, `other`.

## Этап 2 — курация
- `FinetuneDataCurator` ([backend/finetune_curator.py](../backend/finetune_curator.py)) фильтрует по quality score (0..1): длина, confidence, sources, категория, boost.
- Дедуп по fuzz.token_set_ratio ≥ 80%.
- Boosting: `correction`/`troubleshooting` повторяются 2-3 раза в датасете.
- Ручная курация — вкладка **Fine-Tune** в UI или команда `finetune review` в CLI.

## Этап 3 — обучение
`FineTunePipeline` ([backend/finetune_pipeline.py](../backend/finetune_pipeline.py)): prepare_dataset → train_with_unsloth (LoRA + GGUF) → register_with_ollama → регистрация новой версии `v1/v2/...` в `knowledge/model_versions.json`.

Требует: `pip install unsloth trl transformers datasets` + GPU (минимум RTX 3060 12GB для 7B-моделей) + Ollama CLI.

## Corrections — учимся на ошибках
Если пользователь поправил ответ, пара сохраняется с `category=correction`, `confidence=100`, `original_wrong_answer=...`. Это самые ценные примеры (priority highest).

## Команды CLI
```
finetune status             — счётчики + готовность
finetune review             — список с quality score
finetune start              — запустить полный пайплайн
finetune compare            — сравнить две последние версии
finetune switch <tag>       — переключить версию (v0/v1/...)
finetune rollback           — откат
finetune export             — экспорт jsonl
model versions              — список версий
learn this to model         — добавить последний Q&A в очередь
неправильно, правильно: ... — записать correction
```

## Версионирование и Auto-Evolution
`ModelVersionRegistry` ведёт реестр в `knowledge/model_versions.json`. `ModelEvaluator` сравнивает ответы старой и новой модели на тестовом наборе `knowledge/eval_set.json` (формат: `[{"question":"...","expected":"..."}]`) и даёт рекомендацию upgrade/rollback.
