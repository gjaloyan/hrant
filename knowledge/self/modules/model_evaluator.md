---
module: backend/model_evaluator.py
category: self
kind: module
updated: 2026-05-06T17:14:40.033917+00:00
source_mtime: 2026-04-07T07:44:14.309347+00:00
loc: 102
truncated: false
---

# backend/model_evaluator.py

## Purpose
Модуль оценивает и сравнивает качество ответов двух LLM-моделей на JSON-наборе тестов с вопросами и ожидаемыми ответами. Он загружает тестовый набор, прогоняет каждый вопрос через старую и новую модель, считает similarity-score через rapidfuzz.token_set_ratio, усредняет результаты и возвращает EvaluationResult с деталями и рекомендацией по обновлению модели.

## Public interface
- `ModelEvaluator` (class) - Загружает/сохраняет eval-набор и сравнивает две модели по fuzzy-score ответов.

## Dependencies
- backend.config
- backend.knowledge_manager
- backend.llm
- backend.models

## Notes
Если eval_set.json отсутствует, не читается или пуст, сравнение возвращает нулевые оценки и предупреждение в details. Выбор LLM завязан на имени модели: идентификаторы, начинающиеся с 'claude', используют AnthropicLLM с настройками model_a, остальные — OllamaLLM с настройками model_b. Ошибки вызова моделей не прерывают оценку, а превращаются в текстовые ответы вида '[err: ...]'.
