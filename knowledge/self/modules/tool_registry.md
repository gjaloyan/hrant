---
module: backend/tool_registry.py
category: self
kind: module
updated: 2026-05-07T05:03:20.148577+00:00
source_mtime: 2026-04-08T05:36:04.163362+00:00
loc: 110
truncated: false
---

# backend/tool_registry.py

## Purpose
Модуль реализует реестр инструментов для Anthropic Tool Use API: хранит пары из JSON Schema-описания инструмента и Python-обработчика, позволяет регистрировать и удалять инструменты, экспортировать их определения в формате Anthropic и выполнять вызовы инструментов с преобразованием результата в текстовый tool_result.

## Public interface
- `Tool` (class) - Dataclass, описывающий инструмент: имя, описание, input_schema, обработчик и origin.
- `ToolRegistry` (class) - Реестр инструментов с методами регистрации, удаления, перечисления, экспорта в Anthropic-формат и выполнения.
- `REGISTRY` (constant) - Глобальный экземпляр ToolRegistry, используемый как реестр по умолчанию.
- `get_registry` (function) - Возвращает текущий глобальный реестр инструментов.
- `reset_registry` (function) - Сбрасывает глобальный реестр, создавая новый экземпляр ToolRegistry.

## Dependencies
(none)

## Notes
ToolRegistry.register запрещает повторную регистрацию инструмента с тем же именем и выбрасывает ValueError. ToolRegistry.execute не пробрасывает ошибки обработчиков наружу: неизвестный инструмент, TypeError аргументов и любые runtime-исключения возвращаются как текст с is_error=true. Результаты dict и list сериализуются в JSON с ensure_ascii=false и fallback default=str.
