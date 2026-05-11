---
module: backend/finetune_curator.py
category: self
kind: module
updated: 2026-05-06T05:15:39.936815+00:00
source_mtime: 2026-04-07T07:04:05.001384+00:00
loc: 82
truncated: false
---

# backend/finetune_curator.py

## Purpose
Provides automatic curation for finetune training examples by assigning a heuristic quality score, filtering out low-scoring examples, removing near-duplicates based on user text similarity, and optionally boosting important curated examples by repeating them.

## Public interface
- `ScoredPair` (class) - Dataclass pairing a FinetunePair with its computed quality score.
- `FinetuneDataCurator` (class) - Scores, filters, deduplicates, and boosts finetune examples.

## Dependencies
- .models

## Notes
Quality scoring is heuristic and based on assistant answer length, metadata confidence, source notes, category, and boosted flag. Deduplication compares only the user text against already accepted examples using rapidfuzz token_set_ratio with a fixed threshold.
