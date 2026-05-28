"""Model version registry: knowledge/model_versions.json."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import CONFIG
from .knowledge_manager import KM
from .models import ModelVersion, ModelVersionsState


class ModelVersionRegistry:
    """
    Registry of fine-tune model versions (local Qwen chain).
    Claude remains the orchestrator in CONFIG.llm — it is NOT tracked here.
    v0 — base inference model (qwen2.5:7b-instruct), v1+ — fine-tuned versions.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or (KM.base / "model_versions.json")
        if not self.path.exists():
            ft = CONFIG.finetune or {}
            # CONFIG.model_b is None in claude_only mode — defensively coalesce.
            model_b_cfg = CONFIG.model_b or {}
            base_id = (
                ft.get("inference_model")
                or ft.get("base_model")
                or model_b_cfg.get("model", "unknown")
            )
            base = ModelVersion(
                tag="v0",
                model_id=base_id,
                created=datetime.now().isoformat(timespec="seconds"),
                examples_count=0,
                notes="base model (before fine-tune)",
            )
            state = ModelVersionsState(current="v0", versions=[base])
            self._write(state)

    def _read(self) -> ModelVersionsState:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return ModelVersionsState(**raw)
        except Exception:
            return ModelVersionsState()

    def _write(self, state: ModelVersionsState) -> None:
        self.path.write_text(
            json.dumps(state.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ---- public ----
    def list(self) -> ModelVersionsState:
        return self._read()

    def current(self) -> ModelVersion | None:
        state = self._read()
        for v in state.versions:
            if v.tag == state.current:
                return v
        return None

    def register(self, tag: str, model_id: str, examples_count: int, notes: str = "") -> ModelVersion:
        state = self._read()
        # replace if already present
        state.versions = [v for v in state.versions if v.tag != tag]
        version = ModelVersion(
            tag=tag,
            model_id=model_id,
            created=datetime.now().isoformat(timespec="seconds"),
            examples_count=examples_count,
            notes=notes,
        )
        state.versions.append(version)
        self._write(state)
        return version

    def switch(self, tag: str) -> str:
        state = self._read()
        if not any(v.tag == tag for v in state.versions):
            return f"✗ version '{tag}' not found"
        state.current = tag
        self._write(state)
        return f"✓ switched to {tag}"

    def rollback(self) -> str:
        state = self._read()
        if not state.rollback_enabled:
            return "✗ rollback disabled"
        tags = [v.tag for v in state.versions]
        if state.current not in tags or len(tags) < 2:
            return "✗ no previous version"
        idx = tags.index(state.current)
        if idx == 0:
            return "✗ already on the earliest version"
        prev = tags[idx - 1]
        state.current = prev
        self._write(state)
        return f"✓ rolled back to {prev}"

    def next_tag(self) -> str:
        state = self._read()
        nums: list[int] = []
        for v in state.versions:
            if v.tag.startswith("v") and v.tag[1:].isdigit():
                nums.append(int(v.tag[1:]))
        n = (max(nums) + 1) if nums else 1
        return f"v{n}"


VERSIONS = ModelVersionRegistry()
