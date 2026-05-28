"""Read PDF / DOCX / a wide range of text formats / images (with a
redirect to `analyze_image` for visual questions).

Whitelist of text-like extensions intentionally broad — the audit
on "what file types can Hrant handle?" found .csv / .html / .css /
.sh / .c / .cpp / .rs / .go / .java / .h / .toml / .ini / .xml all
silently rejected as "unsupported format" even though they're
just plain text. Same fix here: read as utf-8.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional


# Text-like extensions read directly as UTF-8.
_TEXT_LIKE_EXTS = frozenset({
    # Plain prose / docs
    ".txt", ".md", ".rst", ".log",
    # Structured config
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
    ".env",
    # Web / markup
    ".html", ".htm", ".xml", ".svg", ".css", ".scss", ".less",
    # Tabular text
    ".csv", ".tsv",
    # Languages
    ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
    ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hh",
    ".rs", ".go", ".java", ".kt", ".kts", ".scala",
    ".rb", ".php", ".pl", ".lua", ".sh", ".bash", ".zsh", ".fish",
    ".ps1", ".bat", ".cmd",
    ".sql", ".graphql", ".gql", ".proto",
    ".swift", ".m", ".mm", ".dart", ".ex", ".exs",
    # Build / project
    ".dockerfile", ".gitignore", ".gitattributes", ".editorconfig",
    ".diff", ".patch",
})

_IMAGE_EXTS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".heic", ".avif",
})


def _resolve_alias(stem: str, suffix: str) -> str:
    """Some text-like files have no extension (Dockerfile, Makefile,
    LICENSE). Match on stem when suffix didn't help."""
    s = stem.lower()
    if s in ("dockerfile", "makefile", "rakefile", "license", "licence",
             "readme", "changelog", "notice", "authors", "contributors",
             "todo", ".env"):
        return ".txt"
    return suffix


def read_file(
    path: str | Path,
    max_chars: int = 20000,
    *,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
) -> str:
    """Read a file. For text formats supports line-range slicing via
    `start_line` / `end_line` (1-based, inclusive on both ends) so a
    large file can be inspected in chunks without re-reading the whole
    body each time. The returned text is prefixed with the actual line
    numbers so the model can quote them back accurately.

    `max_chars` still applies AFTER line slicing — caller can request
    a wide range and trust the cap.
    """
    p = Path(path)
    if not p.exists():
        return f"[file not found: {path}]"
    suffix = p.suffix.lower()
    if not suffix:
        # Extension-less files — match on stem (Dockerfile / Makefile
        # / LICENSE / README without a suffix).
        suffix = _resolve_alias(p.stem, suffix)

    if suffix in _TEXT_LIKE_EXTS:
        text = p.read_text(encoding="utf-8", errors="replace")
        if start_line is not None or end_line is not None:
            lines = text.splitlines()
            total = len(lines)
            s = max(1, int(start_line)) if start_line is not None else 1
            e = min(total, int(end_line)) if end_line is not None else total
            if s > e or s > total:
                return f"[range out of file: requested {s}-{e}, file has {total} lines]"
            sliced = lines[s - 1 : e]
            # Prefix each line with its number so quotes are unambiguous.
            numbered = "\n".join(f"{s + i:>5}│ {ln}" for i, ln in enumerate(sliced))
            header = f"[lines {s}-{e} of {total} from {p.name}]\n"
            return (header + numbered)[:max_chars]
        return text[:max_chars]

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return "[pypdf not installed]"
        try:
            reader = PdfReader(str(p))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            return text[:max_chars]
        except Exception as e:
            return f"[pdf read error: {e}]"

    if suffix == ".docx":
        try:
            import docx  # python-docx
        except ImportError:
            return "[python-docx not installed]"
        try:
            doc = docx.Document(str(p))
            return "\n".join(par.text for par in doc.paragraphs)[:max_chars]
        except Exception as e:
            return f"[docx read error: {e}]"

    if suffix in _IMAGE_EXTS:
        # Auto-ingest the image into AttachmentStore and tell the caller
        # to ask `analyze_image` for whatever they actually wanted to
        # know about it. This closes the "image arrived as Document,
        # read_file gave up" gap from the file-type audit.
        try:
            from ..attachments import ATTACHMENTS
            data = p.read_bytes()
            ext_to_mime = {
                ".png": "image/png", ".webp": "image/webp",
                ".gif": "image/gif", ".bmp": "image/bmp",
                ".heic": "image/heic", ".avif": "image/avif",
            }
            mime = ext_to_mime.get(suffix, "image/jpeg")
            rec = ATTACHMENTS.save(data, mime, filename=p.name, kind="image")
            return (
                f"[image saved to AttachmentStore as sha256={rec.sha256[:16]}…"
                f" ({mime}, {len(data)} bytes). "
                f"To inspect contents call `analyze_image("
                f"sha256={rec.sha256!r}, question=...)` — that's the "
                f"vision tool. read_file does NOT OCR images itself.]"
            )
        except Exception as e:
            return (
                f"[image {p.name} on disk but could not be ingested: {e}. "
                f"Read the bytes manually via run_python + Pillow if needed.]"
            )

    return f"[unsupported format: {suffix!r}. read_file handles text-like + .pdf + .docx + images (via analyze_image redirect). For other binary content, use run_python with the appropriate library.]"
