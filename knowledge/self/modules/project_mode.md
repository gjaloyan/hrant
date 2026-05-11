---
module: backend/project_mode.py
category: self
kind: module
updated: 2026-05-06T18:12:26.813650+00:00
source_mtime: 2026-04-07T06:45:21.441275+00:00
loc: 98
truncated: false
---

# backend/project_mode.py

## Purpose
Модуль управляет файловым хранилищем проектов в каталоге knowledge/projects: создаёт структуру проекта, хранит имя текущего активного проекта, добавляет контекст, решения и проблемы в Markdown-файлы, завершает активный проект, перечисляет проекты и читает overview.

## Public interface
- `ProjectManager` (class) - Менеджер проектов, работающий с директориями и Markdown-файлами проекта.
- `PROJECTS` (constant) - Глобальный экземпляр ProjectManager для использования как общий менеджер проектов.

## Dependencies
- .knowledge_manager

## Notes
Текущий проект хранится в файле .current внутри каталога projects. Имена директорий проектов нормализуются через _slug, но в .current сохраняется исходное имя проекта. Методы добавления записей возвращают текстовые статусы и ничего не добавляют, если активный проект не выбран.
