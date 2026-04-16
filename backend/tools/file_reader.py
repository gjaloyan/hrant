"""Чтение PDF/DOCX/TXT/изображений (OCR — не реализован, только метаданные)."""
from __future__ import annotations
from pathlib import Path


def read_file(path: str | Path, max_chars: int = 20000) -> str:
    p = Path(path)
    if not p.exists():
        return f"[файл не найден: {path}]"
    suffix = p.suffix.lower()

    if suffix in (".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml"):
        return p.read_text(encoding="utf-8", errors="replace")[:max_chars]

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
