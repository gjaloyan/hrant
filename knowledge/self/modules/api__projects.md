---
module: backend/api/projects.py
category: self
kind: module
updated: 2026-04-29T10:36:38.066325+00:00
source_mtime: 2026-04-28T05:15:38.518924+00:00
loc: 69
truncated: false
---

# backend/api/projects.py

## Purpose
Defines a FastAPI router for project lifecycle operations, including listing projects, creating a project, reading project overview details, ending the active project, and recording project context, decisions, and issues. The endpoints delegate all project state and persistence behavior to the shared PROJECTS object from the project_mode module and enforce that mutating per-project actions only apply to the currently active project.

## Public interface
- `router` (constant) - FastAPI APIRouter containing all project lifecycle routes.
- `list_projects` (function) - Returns the current project name and the list of all projects.
- `create_project` (function) - Starts a new project from a non-empty name supplied in the request body.
- `project_detail` (function) - Returns the stored overview for the named project.
- `end_project` (function) - Ends the named project if it is currently active.
- `ProjectContextRequest` (class) - Pydantic request model for adding freeform context text to a project.
- `add_project_context` (function) - Adds context text to the active project.
- `ProjectDecisionRequest` (class) - Pydantic request model for recording a project decision and its rationale.
- `add_project_decision` (function) - Adds a decision entry to the active project.
- `ProjectIssueRequest` (class) - Pydantic request model for recording a project issue and its fix.
- `add_project_issue` (function) - Adds an issue entry to the active project.

## Dependencies
- backend.project_mode

## Notes
Mutating routes for a named project check that the requested name matches PROJECTS.current and otherwise return HTTP 400. create_project accepts a raw dict rather than a Pydantic model, so it manually validates that the name field is non-empty after stripping whitespace.
