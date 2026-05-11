---
module: backend/claims.py
category: self
kind: module
updated: 2026-05-07T06:13:38.357833+00:00
source_mtime: 2026-05-07T05:46:55.350354+00:00
loc: 360
truncated: false
---

# backend/claims.py

## Purpose
Provides a pure data-transformation layer for constructing user-facing claims and evidence from verifier output, thinking-trace tool calls, the user message, and optionally a structured solver-emitted claims tail. It supports a fallback path that derives claims directly from verifier buckets and an enhanced path that parses a trailing `---CLAIMS---` JSON block, strips it from the visible answer, and binds solver claims to specific tool evidence items while preserving verifier-controlled status and risk.

## Public interface
- `SOLVER_CLAIMS_MARKER` (constant) - Strict marker string used to identify the solver's trailing claims JSON block.
- `SOLVER_CLAIMS_DIRECTIVE` (constant) - Instruction text telling the solver how to append a structured claims-and-evidence JSON tail.
- `extract_solver_claims_block` (function) - Strips and parses a trailing solver claims block, returning cleaned answer text and parsed claims or None.
- `build_claims_and_evidence` (function) - Builds Claim and EvidenceItem lists from verification results, thinking trace, user message, and optional solver claims.

## Dependencies
- backend.models

## Notes
Malformed or missing solver claims blocks are handled silently by returning cleaned answer text and falling back to verifier-derived claims. Solver-provided claim status is never trusted; status is matched against verifier buckets and defaults to unverified when unmatched. Evidence quotes are capped to 800 characters, and failed tool calls produce evidence with zero confidence.
