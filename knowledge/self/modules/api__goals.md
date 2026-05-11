---
module: backend/api/goals.py
category: self
kind: module
updated: 2026-04-29T05:42:01.141312+00:00
source_mtime: 2026-04-28T05:17:15.464220+00:00
loc: 84
truncated: false
---

# backend/api/goals.py

## Purpose
Defines a FastAPI router for goal management endpoints, covering listing goals with stats, creating goals, deleting goals, and lifecycle operations such as complete, pause, resume, fail, and priority updates. The endpoints delegate all storage and state changes to the shared GOALS manager and return simple JSON responses or 404 errors when a goal is not found.

## Public interface
- `router` (constant) - FastAPI APIRouter containing all goals API routes.
- `GoalRequest` (class) - Pydantic request model for creating a goal.
- `list_goals` (function) - Returns all goals and aggregate goal statistics.
- `add_goal` (function) - Creates a new user-sourced goal from a request body.
- `complete_goal` (function) - Marks a goal complete or raises 404 if it does not exist.
- `pause_goal` (function) - Pauses a goal or raises 404 if it does not exist.
- `resume_goal` (function) - Resumes a goal or raises 404 if it does not exist.
- `fail_goal` (function) - Marks a goal failed or raises 404 if it does not exist.
- `delete_goal` (function) - Deletes a goal or raises 404 if it does not exist.
- `PriorityUpdate` (class) - Pydantic request model for updating a goal priority.
- `update_goal_priority` (function) - Updates a goal priority or raises 404 if it does not exist.

## Dependencies
- backend.goals

## Notes
The module is a thin HTTP layer over the GOALS manager and contains no goal lifecycle logic itself. GoalRequest uses a mutable list default for subtasks, though Pydantic models generally handle field defaults safely depending on version.
