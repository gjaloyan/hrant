---
module: backend/analogy_engine.py
category: self
kind: module
updated: 2026-04-27T11:05:24.941049+00:00
source_mtime: 2026-04-13T22:12:00.524214+00:00
loc: 254
truncated: false
---

# backend/analogy_engine.py

## Purpose
This module implements an analogy engine that identifies structural similarities across different domains using a knowledge graph. It extracts reusable patterns from solved problems and applies them to new problems by finding analogous situations in other domains.

## Public interface
- `Pattern` (class) - Represents an abstract reusable pattern extracted from a solved problem.
- `AnalogyEngine` (class) - Finds and applies cross-domain analogies by managing patterns and searching for applicable analogies.
- `extract_pattern` (function) - Extracts an abstract pattern from a high-confidence answer.
- `find_analogies` (function) - Searches for applicable patterns from other domains for a given problem.
- `context_block` (function) - Builds an analogy context block for injection into solver prompts.
- `all_patterns` (function) - Returns a list of all stored patterns.
- `stats` (function) - Provides statistics about the stored patterns, such as total count and distribution by domain.
- `ANALOGIES` (constant) - An instance of the AnalogyEngine class.

## Dependencies
- config
- knowledge_graph
- knowledge_manager
- llm

## Notes
The module relies on a knowledge graph and local models for pattern extraction and analogy finding, which may involve complex interactions with external systems. The analogy search process includes filtering and ranking based on relevance, which requires careful handling of data to ensure accurate results.
