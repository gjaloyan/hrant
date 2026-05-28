"""Fine-tune queue, examples, training pipeline, model versions."""
from __future__ import annotations
import asyncio
import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..finetune import store as finetune_store
from ..finetune_curator import FinetuneDataCurator
from ..model_versions import VERSIONS
from ..models import CorrectionRequest, FinetuneEdit, ImportGgufRequest
from ._auth import require_owner_for_writes

router = APIRouter()


@router.get("/api/finetune/status")
def finetune_status():
    ft = finetune_store()
    examples = ft.list_all()
    curated = FinetuneDataCurator().curate(examples)
    return {
        "total": len(examples),
        "curated": len(curated),
        "ready": ft.ready(),
        "min_required": ft.min_required,
        "by_category": ft.count_by_category(),
    }


@router.get("/api/finetune/examples")
def finetune_examples():
    ft = finetune_store()
    curator = FinetuneDataCurator()
    items = []
    for sp in curator.score_all(ft.list_all()):
        items.append({
            "id": sp.pair.id,
            "score": sp.score,
            "user": sp.pair.user_text(),
            "assistant": sp.pair.assistant_text(),
            "metadata": sp.pair.metadata.model_dump(),
        })
    return {"items": items}


@router.put("/api/finetune/examples/{pair_id}")
def finetune_edit(pair_id: str, body: FinetuneEdit):
    require_owner_for_writes(action="editing a finetune example")
    ok = finetune_store().edit(pair_id, assistant=body.assistant, boosted=body.boosted)
    if not ok:
        raise HTTPException(404, "not found")
    return {"ok": True}


@router.delete("/api/finetune/examples/{pair_id}")
def finetune_delete(pair_id: str):
    require_owner_for_writes(action="deleting a finetune example")
    ok = finetune_store().delete(pair_id)
    if not ok:
        raise HTTPException(404, "not found")
    return {"ok": True}


@router.post("/api/finetune/examples/{pair_id}/boost")
def finetune_boost(pair_id: str):
    require_owner_for_writes(action="boosting a finetune example")
    ok = finetune_store().boost(pair_id)
    if not ok:
        raise HTTPException(404, "not found")
    return {"ok": True}


@router.post("/api/finetune/correction")
def finetune_correction(body: CorrectionRequest):
    require_owner_for_writes(action="adding a correction")
    pair = finetune_store().add_correction(
        question=body.question,
        wrong_answer=body.wrong_answer,
        corrected_answer=body.corrected_answer,
        project=body.project,
    )
    return {"ok": True, "id": pair.id}


@router.post("/api/finetune/start")
def finetune_start():
    """Start the pipeline with a progress stream."""
    require_owner_for_writes(action="starting fine-tune pipeline")
    from ..finetune_pipeline import FineTunePipeline

    queue: asyncio.Queue = asyncio.Queue()

    def progress(stage: str, msg: str) -> None:
        queue.put_nowait({"type": "progress", "stage": stage, "message": msg})

    pipe = FineTunePipeline(progress=progress)

    async def runner():
        try:
            name = await asyncio.to_thread(pipe.run_full_pipeline)
            queue.put_nowait({"type": "done", "model": name})
        except Exception as e:
            queue.put_nowait({"type": "error", "message": str(e)})
        finally:
            queue.put_nowait(None)

    async def stream() -> AsyncIterator[dict]:
        task = asyncio.create_task(runner())
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield {"data": json.dumps(item, ensure_ascii=False)}
        finally:
            await task

    return EventSourceResponse(stream())


@router.post("/api/finetune/switch")
def finetune_switch(body: dict):
    require_owner_for_writes(action="switching the active model")
    tag = body.get("tag", "").strip()
    if not tag:
        raise HTTPException(400, "tag required")
    return {"message": VERSIONS.switch(tag)}


@router.post("/api/finetune/rollback")
def finetune_rollback():
    require_owner_for_writes(action="rolling back the model")
    return {"message": VERSIONS.rollback()}


@router.get("/api/finetune/export")
def finetune_export():
    return {"jsonl": finetune_store().export_jsonl()}


@router.get("/api/model/versions")
def model_versions():
    return VERSIONS.list().model_dump()


@router.post("/api/finetune/export-cloud")
def finetune_export_cloud():
    require_owner_for_writes(action="exporting finetune data for cloud training")
    from ..finetune_pipeline import FineTunePipeline
    try:
        pipe = FineTunePipeline()
        pkg = pipe.export_for_cloud()
        files = sorted(p.name for p in pkg.iterdir() if p.is_file())
        return {
            "package_dir": str(pkg),
            "files": files,
            "tag": pkg.name.replace("cloud_export_", ""),
            "instructions": f"Download {pkg.name}/, upload to GPU, run train_script.py",
        }
    except Exception as e:
        raise HTTPException(400, str(e))


@router.post("/api/finetune/import-gguf")
def finetune_import_gguf(body: ImportGgufRequest):
    # Owner gate + path traversal mitigation: combined with the gate,
    # only the trusted owner can pass an arbitrary path. Without the
    # gate, this endpoint reads any file the agent process can read
    # — a remote arbitrary-file-read primitive.
    require_owner_for_writes(action="importing a GGUF model")
    from ..finetune_pipeline import FineTunePipeline
    try:
        pipe = FineTunePipeline()
        name = pipe.import_gguf(body.path, tag=body.tag)
        return {"ok": True, "model": name}
    except Exception as e:
        raise HTTPException(400, str(e))


class AddToFinetuneRequest(BaseModel):
    question: str
    answer: str
    used_topics: list[str] = []
    confidence: int = 100
    project: str | None = None


@router.post("/api/finetune/add-from-chat")
def add_from_chat(body: AddToFinetuneRequest):
    require_owner_for_writes(action="adding a chat example to finetune")
    finetune_store().add(
        question=body.question,
        answer=body.answer,
        source_notes=body.used_topics,
        confidence=body.confidence,
        project=body.project,
        verified=True,
    )
    return {"ok": True}


@router.post("/api/finetune/compare")
def finetune_compare():
    require_owner_for_writes(action="comparing model versions")
    from ..model_evaluator import ModelEvaluator
    state = VERSIONS.list()
    if len(state.versions) < 2:
        raise HTTPException(400, "need at least 2 model versions")
    old = state.versions[-2].model_id
    new = state.versions[-1].model_id
    return ModelEvaluator().compare(old, new).model_dump()
