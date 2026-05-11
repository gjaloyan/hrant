---
topic: previous_response_id
category: profession
created: 2026-04-29 14:35
updated: 2026-04-29 14:35
keywords: previous_response_id, Responses API, conversation state, stateful responses, response.id, multi-turn conversations, function calling, token usage, Azure OpenAI, threaded conversation
source: https://developers.openai.com/api/docs/guides/conversation-state; https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses; https://github.com/openai/codex/issues/4047
confidence: partial
access_count: 18
---

# previous_response_id

## Что это
Параметр Responses API для передачи контекста из предыдущего ответа: связывает текущий запрос с ранее созданным `response` и формирует цепочку/тред диалога.

## Ключевые параметры
- Имя параметра: `previous_response_id`
- Значение: ID предыдущего ответа (`response.id`)
- Назначение: управление состоянием диалога без ручной передачи всей истории
- Поддержка: OpenAI Responses API; Azure OpenAI Responses API поддерживает stateful responses: create, retrieve, delete, streaming, tools

## Практические заметки
- Использовать для multi-turn conversations, когда нужно продолжить контекст предыдущего ответа.
- Сохранять `response.id` после каждого ответа и передавать его в следующем запросе.
- Полезно для снижения повторной передачи контекста и потенциального уменьшения token usage.
- В сценариях с tools/function calling может улучшать качество/стабильность вызовов за счёт сохранённого состояния.

## Частые ошибки
- Не сохранить `response.id` -> невозможно связать следующий запрос с предыдущим.
- Передать неверный/устаревший ID -> контекст не продолжится.
- Ожидать работу во всех клиентах одинаково -> некоторые обёртки/CLI могут игнорировать `previous_response_id`; проверить поддержку конкретного SDK/инструмента.
- Дублировать всю историю вручную вместе с `previous_response_id` -> лишние токены и риск рассинхронизации контекста.

## Причинно-следственные связи
- `previous_response_id` causes текущий ответ to inherit context from previous response
- сохранение `response.id` enables построение multi-turn conversation
- игнорирование параметра клиентом causes потерю stateful-поведения
- ручная передача всей истории causes higher token usage
- stateful responses enable retrieve/delete lifecycle management

## Связанные темы
- [[Responses API]]
- [[Conversation state]]
- [[Stateful responses]]
- [[Function calling]]
- [[Token usage]]
- [[Azure OpenAI Responses API]]
