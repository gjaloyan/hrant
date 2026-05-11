---
module: backend/autonomic/levers/tool_install.py
category: self
kind: module
updated: 2026-05-04T12:42:29.564365+00:00
source_mtime: 2026-04-20T21:21:05.969624+00:00
loc: 184
truncated: false
---

# backend/autonomic/levers/tool_install.py

## Purpose
FIRE_TOOL_INSTALL is a yellow-safety lever that enables the agent to install Python packages via pip, pull Ollama models, and download llama.cpp GGUF model files from HTTPS URLs. It provides controlled external dependency installation with timeouts, validation, and detailed outcome reporting.

## Public interface
- `FIRE_TOOL_INSTALL` (class) - Lever for installing pip packages, pulling Ollama models, and downloading llama.cpp GGUF files
- `DEFAULT_LLAMA_CPP_DIR` (constant) - Default directory path for storing downloaded llama.cpp models
- `ALLOWED_COMMANDS` (constant) - Set of permitted installation commands: pip_install, ollama_pull, llama_cpp_pull

## Dependencies
- backend.lever
- backend.types

## Notes
The lever enforces strict safety constraints: only HTTPS URLs for GGUF downloads, filename validation to prevent path traversal, 5-30 minute timeouts per operation, and graceful handling when optional tools like Ollama are missing. Each command returns detailed outcome metadata including return codes, output tails, file sizes, and destination paths for debugging and auditing.
