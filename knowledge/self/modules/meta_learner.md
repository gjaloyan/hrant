---
module: backend/meta_learner.py
category: self
kind: module
updated: 2026-05-06T11:55:52.428710+00:00
source_mtime: 2026-05-05T10:53:22.272732+00:00
loc: 351
truncated: false
---

# backend/meta_learner.py

## Purpose
This module implements a meta-learning component that records low-confidence verified answers as failures, asks an LLM to classify their root causes, creates corrective goals from the analysis, periodically extracts recurring error patterns, and exposes recent failure history and aggregate statistics. It persists an append-only failure log and aggregated pattern data under the configured knowledge base directory.

## Public interface
- `META_ANALYSIS_SYSTEM` (constant) - System prompt used to analyze a single agent failure and return structured JSON.
- `PATTERN_EXTRACTION_SYSTEM` (constant) - System prompt used to extract recurring patterns from recent analyzed failures.
- `MetaLearner` (class) - Analyzes failures, logs them, creates corrective goals, extracts recurring patterns, and reports stats.
- `META_LEARNER` (constant) - Default singleton instance of MetaLearner.

## Dependencies
- backend.config
- backend.goals
- backend.llm
- backend.models
- backend.self_modifier

## Notes
Failures are only analyzed when verification confidence is below 60; LLM and filesystem errors are swallowed in several places to keep the feedback loop best-effort. Pattern extraction runs automatically every fifth analyzed failure in the current process, and high-priority patterns create goals. The self-modifier bridge only proposes changes and relies on explicit approval elsewhere before applying them.
