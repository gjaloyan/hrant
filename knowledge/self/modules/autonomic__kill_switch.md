---
module: backend/autonomic/kill_switch.py
category: self
kind: module
updated: 2026-04-27T11:46:08.200389+00:00
source_mtime: 2026-04-16T20:09:53.456990+00:00
loc: 31
truncated: false
---

# backend/autonomic/kill_switch.py

## Purpose
This module implements a file-based kill switch for the autonomic subsystem, allowing the system to be enabled or disabled based on the content of a specific file.

## Public interface
- `KillSwitch` (class) - Manages a file-based kill switch to enable or disable the autonomic subsystem.
- `is_enabled` (function) - Checks if the kill switch is enabled by reading the flag file.
- `enable` (function) - Enables the kill switch by writing 'true' to the flag file.
- `disable` (function) - Disables the kill switch by writing 'false' to the flag file.
- `DEFAULT_PATH` (constant) - Default path for the kill switch flag file.

## Dependencies
(none)

## Notes
The module uses a simple file-based mechanism to control the state of the autonomic subsystem. It defaults to a fail-safe mode where the subsystem is disabled if the file is missing or contains unrecognized content. This ensures robustness in the absence of the flag file.
