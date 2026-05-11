---
module: backend/main.py
category: self
kind: module
updated: 2026-05-06T11:40:43.035100+00:00
source_mtime: 2026-04-30T10:15:52.695179+00:00
loc: 124
truncated: false
---

# backend/main.py

## Purpose
Defines the FastAPI application entry point for the backend: configures logging and CORS, manages application startup and shutdown via a lifespan handler, auto-starts configured channels, initializes and stops the autonomic scheduler bundle, mounts all API routers, optionally serves the built frontend SPA, and provides a uvicorn-based serve function.

## Public interface
- `lifespan` (function) - FastAPI lifespan context manager that starts channels and autonomic services on startup and stops them on shutdown.
- `app` (constant) - Configured FastAPI application instance with middleware, routers, lifespan, and optional static frontend serving.
- `serve_spa` (function) - Catch-all route serving frontend files or index.html when the frontend build directory exists.
- `serve` (function) - Runs the FastAPI app with uvicorn using host and port from CONFIG.

## Dependencies
- backend.config
- backend.channels
- backend.autonomic.startup
- backend.api.chat
- backend.api.knowledge
- backend.api.projects
- backend.api.finetune
- backend.api.status
- backend.api.identity
- backend.api.intel
- backend.api.goals
- backend.api.sessions
- backend.api.providers
- backend.api.channels
- backend.api.attachments
- backend.autonomic.api

## Notes
The frontend catch-all route is only registered if frontend/dist exists at import time. Startup stores multiple autonomic bundle components on application.state for use by other request handlers. Channel auto-start errors are logged and do not prevent server startup, while scheduler startup is awaited after bundle construction.
