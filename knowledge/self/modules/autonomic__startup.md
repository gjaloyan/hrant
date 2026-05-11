---
module: backend/autonomic/startup.py
category: self
kind: module
updated: 2026-05-04T12:43:40.138814+00:00
source_mtime: 2026-04-20T21:21:05.969624+00:00
loc: 112
truncated: false
---

# backend/autonomic/startup.py

## Purpose
Glue layer that wires together the autonomic scheduler subsystem for FastAPI lifespan management. Reads environment variables to configure paths and intervals, instantiates all autonomic components (safety gate, executor, state builder, Layer0 engine, event bus), registers default levers, and packages everything into a SchedulerBundle that the FastAPI app can store and access.

## Public interface
- `SchedulerBundle` (class) - Dataclass bundling scheduler, gate, executor, builder, registry, kill_switch, and log paths for app.state storage
- `build_scheduler` (function) - Factory that reads env vars, wires all autonomic components, and returns a SchedulerBundle
- `start_autonomic_scheduler` (function) - Async lifespan hook that starts the scheduler from a bundle
- `stop_autonomic_scheduler` (function) - Async lifespan hook that stops the scheduler from a bundle

## Dependencies
- backend.autonomic.events
- backend.autonomic.executor
- backend.autonomic.kill_switch
- backend.autonomic.layer0
- backend.autonomic.levers
- backend.autonomic.safety
- backend.autonomic.scheduler
- backend.autonomic.state
- backend.autonomic.tick

## Notes
Environment variables control all file paths and tick interval (AUTONOMIC_ENABLED_PATH, AUTONOMIC_TICK_SECONDS, AUTONOMIC_KNOWLEDGE_ROOT, etc.). The module clears and re-registers default levers on every build_scheduler call, which may be surprising if called multiple times. Error handling in start/stop is defensive but logs at different levels (error vs warning).
