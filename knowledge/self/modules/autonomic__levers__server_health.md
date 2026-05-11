---
module: backend/autonomic/levers/server_health.py
category: self
kind: module
updated: 2026-05-03T14:20:02.590634+00:00
source_mtime: 2026-04-17T04:58:39.640839+00:00
loc: 79
truncated: false
---

# backend/autonomic/levers/server_health.py

## Purpose
A health-check lever that monitors server resource usage (disk space, memory, CPU load) against configurable thresholds and reports any violations. Part of the IMMUNE category for system self-monitoring.

## Public interface
- `FIRE_SERVER_HEALTH` (class) - Lever that checks disk/memory/CPU against thresholds and returns health status
- `DEFAULT_DISK_MIN_GB` (constant) - Default minimum free disk space threshold (1.0 GB)
- `DEFAULT_MEMORY_MIN_GB` (constant) - Default minimum free memory threshold (0.5 GB)
- `DEFAULT_CPU_MAX_LOAD` (constant) - Default maximum CPU load threshold (4.0)

## Dependencies
- backend.lever
- backend.types

## Notes
Uses psutil for system metrics with fallback logic for CPU load on platforms without getloadavg(). Reads from StateSnapshot if available, otherwise queries system directly. Always returns SUCCESS status; issues are reported in outcome dict rather than failure status.
