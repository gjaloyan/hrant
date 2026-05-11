---
module: backend/verifier.py
category: self
kind: module
updated: 2026-05-07T10:32:58.113922+00:00
source_mtime: 2026-05-07T05:57:41.228708+00:00
loc: 577
truncated: false
---

# backend/verifier.py

## Purpose
This module verifies an agent's answer against loaded notes and tool outputs by prompting a verifier LLM with the question, answer, evidence, extracted identifiers, and optional structured solver claims. It classifies answer claims as verified, unverified, or contradictory, computes a deterministic confidence score, and adds a deterministic safeguard for false absence hallucinations where the answer proposes adding or says something is missing even though tool output shows the identifier already exists.

## Public interface
- `VERIFIER_SYSTEM` (constant) - System prompt instructing the verifier LLM how to classify claims and return strict JSON.
- `detect_false_absence_contradictions` (function) - Detects answer claims that an identifier is missing or should be added when that identifier is already present in extracted code identifiers.
- `verify` (function) - Verifies an answer against notes and tool context, calls the verifier LLM, applies deterministic contradiction checks, and returns a VerificationResult.

## Dependencies
- backend.llm
- backend.models

## Notes
Most helper functions are private prompt-building, tool-reference formatting, identifier extraction, context compression, and confidence calculation utilities. The verifier treats tool outputs as primary evidence and notes as secondary evidence, and confidence is computed deterministically from claim counts rather than trusted from the LLM. Structured solver claims, when provided, are rendered with cited tool outputs so the verifier can rule on each claim independently.
