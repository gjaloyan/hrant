---
module: backend/api/knowledge.py
category: self
kind: module
updated: 2026-04-29T10:36:25.230055+00:00
source_mtime: 2026-04-28T05:15:26.077413+00:00
loc: 103
truncated: false
---

# backend/api/knowledge.py

## Purpose
Defines a FastAPI router that exposes API endpoints for knowledge notes, core memory, knowledge gaps, capabilities, and quick notes. It lists, retrieves, learns, deletes, and creates knowledge entries through the knowledge manager, manages core memory facts, reports open and closed gaps, and returns a generated capabilities block.

## Public interface
- `router` (constant) - FastAPI APIRouter containing all knowledge, memory, gaps, capabilities, and quick-note routes.
- `list_knowledge` (function) - Returns all knowledge topics and topics grouped by category.
- `get_knowledge` (function) - Returns a single knowledge note by topic or raises 404 if it is missing.
- `api_learn` (function) - Creates or updates a learned topic note using the note creator and current project context.
- `delete_knowledge` (function) - Deletes a knowledge note by topic and returns whether deletion succeeded.
- `get_core` (function) - Returns core memory content, current token count, and maximum token limit.
- `add_core` (function) - Adds a fact to core memory from a request payload.
- `delete_core` (function) - Removes a matching fact from core memory.
- `get_gaps` (function) - Returns hot knowledge gaps split into open and closed groups.
- `get_capabilities` (function) - Returns the agent capabilities block.
- `QuickNoteRequest` (class) - Pydantic request model for creating a quick note from plain text.
- `quick_note` (function) - Saves a user quick note as a personal verified knowledge note.

## Dependencies
- backend.agent
- backend.core_memory
- backend.knowledge_manager
- backend.models
- backend.note_creator
- backend.project_mode

## Notes
The route handlers are thin wrappers around shared singletons such as KM, CORE, and PROJECTS. The quick-note endpoint derives the topic from the first 40 characters and the first keyword from the first whitespace-separated word, producing no keywords for blank text.
