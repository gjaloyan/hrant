---
module: backend/model_versions.py
category: self
kind: module
updated: 2026-05-06T17:14:45.506244+00:00
source_mtime: 2026-04-28T05:07:25.518277+00:00
loc: 113
truncated: false
---

# backend/model_versions.py

## Purpose
Модуль ведёт локальный JSON-реестр версий fine-tune модели в файле knowledge/model_versions.json. Он инициализирует базовую версию v0 при первом запуске, читает и записывает состояние версий, позволяет регистрировать новые версии, переключать текущую, выполнять rollback на предыдущую и вычислять следующий тег версии.

## Public interface
- `ModelVersionRegistry` (class) - Класс для управления реестром версий модели, включая чтение, запись, регистрацию, переключение и rollback.
- `VERSIONS` (constant) - Глобальный экземпляр ModelVersionRegistry для доступа к реестру версий модели.

## Dependencies
- backend.config
- backend.knowledge_manager
- backend.models

## Notes
При повреждении или ошибке чтения JSON реестр молча возвращает пустое ModelVersionsState. Порядок rollback зависит от текущего порядка элементов в списке versions, а register заменяет существующую версию с тем же tag и добавляет новую запись в конец.
