---
module: backend/searcher.py
category: self
kind: module
updated: 2026-05-06T19:47:01.755106+00:00
source_mtime: 2026-04-07T06:42:44.643891+00:00
loc: 57
truncated: false
---

# backend/searcher.py

## Purpose
Модуль реализует поиск по базе знаний с комбинированным ранжированием: точные и частичные совпадения по slug темы, совпадения по ключевым словам и fuzzy-сравнение через rapidfuzz. Он обходит индекс тем из менеджера знаний, фильтрует результаты по настраиваемому порогу и возвращает отсортированные совпадения с оценкой релевантности.

## Public interface
- `SearchHit` (class) - Dataclass-контейнер для результата поиска: запись индекса и числовой score.
- `Searcher` (class) - Сервис поиска по темам базы знаний с keyword, slug и fuzzy matching.
- `SEARCHER` (constant) - Глобальный экземпляр Searcher для переиспользования в приложении.

## Dependencies
- backend.config
- backend.knowledge_manager
- backend.models

## Notes
Порог fuzzy-поиска берётся из CONFIG.search["fuzzy_threshold"] при создании Searcher. Точные совпадения slug получают максимальный score, затем идут частичные slug-совпадения и точные keyword-совпадения, после чего используется rapidfuzz. Метод exists проверяет наличие заметки через KM.get_note, а find_best возвращает лучшую IndexEntry или null при отсутствии результатов.
