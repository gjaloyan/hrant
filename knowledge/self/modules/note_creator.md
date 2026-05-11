---
module: backend/note_creator.py
category: self
kind: module
updated: 2026-05-06T18:12:20.797580+00:00
source_mtime: 2026-04-13T22:11:16.675278+00:00
loc: 183
truncated: false
---

# backend/note_creator.py

## Purpose
Generates structured Russian-language technical notes about a topic by searching the web, fetching source pages, prompting an LLM with a fixed note template, saving the resulting note through the knowledge manager, and best-effort indexing extracted relations, causal edges, and keywords into the knowledge graph.

## Public interface
- `NOTE_SYSTEM` (constant) - System prompt instructing the LLM to produce concise structured technical notes in Russian.
- `NOTE_TEMPLATE` (constant) - Markdown template defining the required sections for generated notes.
- `learn_topic` (function) - Searches web sources for a topic, generates a structured note via LLM, saves it, and indexes relations in the knowledge graph.

## Dependencies
- backend.knowledge_graph
- backend.knowledge_manager
- backend.llm
- backend.models
- backend.tools.web_search

## Notes
Graph indexing is explicitly best-effort: all exceptions during relation extraction or graph updates are swallowed. The LLM is instructed to append a keywords section, which is stripped before saving and used for note metadata and keyword graph edges.
