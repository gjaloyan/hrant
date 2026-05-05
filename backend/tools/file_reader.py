"""Чтение PDF/DOCX/TXT/изображений (OCR — не реализован, только метаданные)."""
from __future__ import annotations
from pathlib import Path
from typing import Optional


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
        return f"[файл не найден: {path}]"
    suffix = p.suffix.lower()

    if suffix in (".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml"):
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
            return "[pypdf не установлен]"
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
            return "[python-docx не установлен]"
        try:
            doc = docx.Document(str(p))
            return "\n".join(par.text for par in doc.paragraphs)[:max_chars]
        except Exception as e:
            return f"[docx read error: {e}]"

    if suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        return f"[изображение {p.name}, OCR не реализован]"

    return f"[неподдерживаемый формат: {suffix}]"
