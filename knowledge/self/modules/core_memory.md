---
module: backend/core_memory.py
category: self
kind: module
updated: 2026-05-05T20:51:17.435381+00:00
source_mtime: 2026-04-07T06:42:09.425202+00:00
loc: 56
truncated: false
---

# backend/core_memory.py

## Purpose
Модуль управляет core memory агента: читает файл с ключевыми фактами, оценивает его размер в токенах, добавляет и удаляет факты с учетом лимита, а также предлагает кандидатов для продвижения в core memory на основе часто используемых тем из knowledge manager.

## Public interface
- `CoreMemory` (class) - Класс для чтения, изменения и контроля размера core memory.
- `CORE` (constant) - Глобальный экземпляр CoreMemory для использования модулем как singleton-сервис.

## Dependencies
- backend.config
- backend.knowledge_manager

## Notes
Оценка токенов приблизительная: один токен считается равным примерно четырем символам. Добавление фактов блокируется при превышении configured лимита, а удаление работает построчным поиском только по строкам, начинающимся с маркера '-'.
