"""analyze_image — ask the multimodal LLM about a stored image.

The video-overlay-removal workflow (and any future task that needs
to identify what's in a frame, where it sits, what colour it is)
was previously stuck writing OpenCV pixel-classifier code by hand.
That used 4-6 tool iterations per question and still got it wrong
on the first try.

This tool lets the agent ask a natural-language question about an
image-attachment sha256: "where is the watermark? give x,y,w,h in
the original frame", "is the logo still visible in this clip?",
"what's the foreground colour at the centre?" — the multimodal
LLM that's already routing prompts handles it in one call.

Implementation note: we don't pick a specific provider. We build
a minimal system+user payload, pass `attachments=[sha]`, and let
the active LLM class's `_build_user_content` do the inlining. The
router decides which provider answers; Codex / Anthropic / OpenAI
Chat all already know how to embed images in the right shape (see
backend/llm.py `_build_user_content` per class).
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


_ANALYZE_SYSTEM = (
    "You are an image inspection assistant. The user will ask a "
    "specific question about a single image attached to this turn. "
    "Answer concisely, in the same language as the question. Quote "
    "exact pixel coordinates when asked, in the form `x=<int> "
    "y=<int> w=<int> h=<int>` measured in the IMAGE's own pixel "
    "frame (top-left origin). If you can't see the requested "
    "content, say so plainly — don't guess."
)


def analyze_image(
    sha256: str,
    question: str,
    *,
    max_tokens: int = 400,
) -> str:
    """Ask the multimodal LLM `question` about the image at `sha256`.

    Returns the LLM's text answer, or a short error description on
    failure. Never raises — failures are signalled in the returned
    text so the tool-loop doesn't trip on them.

    Args:
        sha256: a sha256 already in the AttachmentStore (kind="image").
        question: free-form natural-language question.
        max_tokens: output cap; defaults to 400 — most overlay-detection
            answers are 1-3 sentences.
    """
    if not sha256 or not isinstance(sha256, str):
        return "[analyze_image error: sha256 is required]"
    if not question or not isinstance(question, str):
        return "[analyze_image error: question is required]"
    try:
        from ..attachments import ATTACHMENTS
    except Exception as e:
        return f"[analyze_image error: attachment store unavailable ({e})]"
    meta = ATTACHMENTS.get_meta(sha256)
    if meta is None:
        return f"[analyze_image error: sha256 {sha256[:12]}... not in attachment store]"
    if meta.kind not in ("image", "video"):
        # Allow video for callers that pass a video sha (the body
        # itself isn't sent — but downstream `_resolve_attachments`
        # would expand its frame_shas... actually no, vision needs
        # an image. Reject loud.)
        return f"[analyze_image error: attachment kind={meta.kind!r}, need image]"
    if meta.kind == "video":
        return (
            "[analyze_image error: pass a frame sha256, not the "
            "video sha. Use video_processor.preprocess_video first "
            "to extract frames, then call analyze_image on a frame.]"
        )

    try:
        from ..llm import router, TaskType
    except Exception as e:
        return f"[analyze_image error: router unavailable ({e})]"

    try:
        out = router().call(
            TaskType.CLASSIFICATION,
            system=_ANALYZE_SYSTEM,
            user=question.strip(),
            max_tokens=max_tokens,
            temperature=0.1,
            attachments=[sha256],
        )
    except Exception as e:
        log.warning("analyze_image LLM call failed: %s", e)
        return f"[analyze_image error: LLM call failed: {type(e).__name__}: {e}]"
    return (out or "").strip() or "[analyze_image: empty response from LLM]"
