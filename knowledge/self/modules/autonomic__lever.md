---
module: backend/autonomic/lever.py
category: self
kind: module
updated: 2026-04-27T11:46:17.451044+00:00
source_mtime: 2026-04-16T20:09:53.458723+00:00
loc: 57
truncated: false
---

# backend/autonomic/lever.py

## Purpose
This module defines an abstract base class 'Lever' for autonomic levers, which are components that can be executed under certain conditions and may have associated costs and safety considerations. Subclasses must define specific attributes and implement abstract methods to ensure proper functionality.

## Public interface
- `Lever` (class) - Abstract base class for autonomic levers with required attributes and methods.
- `preconditions` (function) - Abstract method to check if the lever can run in the given state.
- `run` (function) - Abstract method to execute the lever and return a LeverReport.
- `rollback` (function) - Optional method to rollback the lever's actions, defaults to no operation.

## Dependencies
- types

## Notes
The module enforces strict subclassing requirements by checking for the presence of specific class attributes at instantiation, raising a TypeError if any are missing. This ensures that all levers conform to a standard interface, which is critical for their correct operation within the system.
