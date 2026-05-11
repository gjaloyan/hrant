---
module: backend/autonomic/levers/service_repair.py
category: self
kind: module
updated: 2026-05-04T12:41:18.093867+00:00
source_mtime: 2026-04-17T04:58:39.641775+00:00
loc: 112
truncated: false
---

# backend/autonomic/levers/service_repair.py

## Purpose
FIRE_SERVICE_REPAIR is a whitelist-gated lever that attempts to restart systemd services on Linux platforms. It executes 'systemctl restart' for services in a hardcoded whitelist (ollama, docker, mcp, tmp_cleanup), verifies the service is active, and retries up to max_attempts times. Returns SUCCESS if the service becomes active, ESCALATED if repair fails, or BLOCKED_BY_SAFETY for non-whitelisted services.

## Public interface
- `FIRE_SERVICE_REPAIR` (class) - Lever subclass that restarts and verifies whitelisted systemd services
- `SERVICE_WHITELIST` (constant) - Set of service names allowed to be restarted: ollama, docker, mcp, tmp_cleanup
- `_PLATFORM_SUPPORTED` (constant) - Boolean flag indicating if the platform is Linux

## Dependencies
- backend.lever
- backend.types

## Notes
The lever uses subprocess to invoke systemctl commands with timeouts (30s for restart, 10s for status). It silently catches all exceptions during systemctl operations and logs warnings. The verification logic checks for the literal string 'active (running)' in stdout. Journal tail is truncated to last 500 characters in the outcome.
