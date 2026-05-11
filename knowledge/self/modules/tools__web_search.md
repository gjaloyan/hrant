---
module: backend/tools/web_search.py
category: self
kind: module
updated: 2026-05-07T09:05:42.183535+00:00
source_mtime: 2026-04-08T10:45:08.111719+00:00
loc: 122
truncated: false
---

# backend/tools/web_search.py

## Purpose
Модуль предоставляет простой веб-поиск и извлечение текста по URL: сначала пытается использовать Tavily API при наличии переменной окружения TAVILY_API_KEY, а при отсутствии ключа или ошибке переключается на HTML-парсинг результатов DuckDuckGo. Также содержит утилиту для загрузки страницы и грубой очистки HTML до plain text.

## Public interface
- `WebResult` (class) - Dataclass результата веб-поиска с заголовком, URL и сниппетом.
- `web_search` (function) - Выполняет поиск через Tavily или fallback через DuckDuckGo HTML и возвращает список WebResult.
- `fetch_url` (function) - Загружает URL, удаляет скрипты, стили и HTML-теги, возвращая очищенный текст ограниченной длины.

## Dependencies
(none)

## Notes
Внутренние ошибки сетевых запросов подавляются: поиск возвращает пустой список или fallback-результаты, а fetch_url возвращает строку с описанием ошибки. DuckDuckGo fallback основан на regex-парсинге HTML, поэтому чувствителен к изменениям разметки. Для ссылок DuckDuckGo вида /l/?uddg=... выполняется разворачивание до конечного URL.
