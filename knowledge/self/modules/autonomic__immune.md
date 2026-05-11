---
module: backend/autonomic/immune.py
category: self
kind: module
updated: 2026-04-27T11:17:54.253979+00:00
source_mtime: 2026-04-17T04:58:39.634914+00:00
loc: 99
truncated: false
---

# backend/autonomic/immune.py

## Purpose
This module manages an immune signature store that matches error entries to known fix recipes, allowing for the identification and application of solutions to recurring errors.

## Public interface
- `ImmuneSignature` (class) - Represents an immune signature with pattern matching and fix details.
- `SignatureStore` (class) - Handles loading, matching, and updating immune signatures from a file.
- `DEFAULT_SIGNATURES_PATH` (constant) - Default file path for storing immune signatures.

## Dependencies
(none)

## Notes
The module uses JSONL format for storing signatures, allowing for easy line-by-line processing. It handles malformed entries gracefully by logging warnings and skipping them. The regex matching in the `match` method is a potential point of failure if the regex patterns are incorrect, which is mitigated by logging any regex errors.
