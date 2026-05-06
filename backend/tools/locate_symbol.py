"""AST-based symbol locator.

The agent's self-analysis pass routinely needs to look at a specific
function or class inside a 2k-line source file. Without this, the
options are:
  - read the whole file (16k cap = 30-40% of self-review's input bill)
  - grep first, then read_file with a hand-picked range (two tool
    round-trips, two dev captures, ~10k extra input each iteration)

`locate_symbol` collapses that into one cheap call: parse the file
once, return every match's line range, the agent then `read_file`s
with `start_line`/`end_line` that actually fits the symbol body.

Falls back to a regex scan for non-Python text formats — close enough
for markdown headings, JS/TS exports, and config keys.
"""
from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class SymbolHit:
    name: str
    kind: str            # "function" | "class" | "method" | "var" | "heading" | "match"
    start_line: int      # 1-based, inclusive
    end_line: int        # 1-based, inclusive
    qualified_name: str  # e.g. `Agent.run`, `module.MyClass`


def locate_symbol(
    path: str | Path,
    name: str,
    *,
    kinds: Iterable[str] | None = None,
    max_hits: int = 20,
) -> list[SymbolHit]:
    """Find every definition of `name` in `path` and return its line range.

    For .py files, walks the AST so we get reliable end-line info even
    on functions with decorators and multi-line signatures. For other
    text formats falls back to a tolerant regex scan.

    `kinds` filters the result; pass `["function", "method"]` to skip
    class hits when you're hunting a callable. Default = all kinds.
    """
    p = Path(path)
    if not p.exists():
        return []
    suffix = p.suffix.lower()
    name_clean = (name or "").strip()
    if not name_clean:
        return []
    accepted = set(kinds) if kinds else None

    if suffix == ".py":
        hits = _locate_python(p, name_clean)
    elif suffix in (".md", ".markdown"):
        hits = _locate_markdown(p, name_clean)
    else:
        hits = _locate_textual(p, name_clean)

    if accepted is not None:
        hits = [h for h in hits if h.kind in accepted]
    return hits[:max_hits]


def _locate_python(p: Path, name: str) -> list[SymbolHit]:
    src = p.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src, filename=str(p))
    except SyntaxError:
        # Malformed source — fall back to text scan so we still help.
        return _locate_textual(p, name)
    hits: list[SymbolHit] = []
    total_lines = len(src.splitlines())

    def _walk(node: ast.AST, parent: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if isinstance(node, ast.ClassDef) else "function"
                qual = f"{parent}.{child.name}" if parent else child.name
                if child.name == name:
                    hits.append(SymbolHit(
                        name=child.name,
                        kind=kind,
                        start_line=_def_start_line(child),
                        end_line=getattr(child, "end_lineno", child.lineno) or child.lineno,
                        qualified_name=qual,
                    ))
                _walk(child, qual)
            elif isinstance(child, ast.ClassDef):
                qual = f"{parent}.{child.name}" if parent else child.name
                if child.name == name:
                    hits.append(SymbolHit(
                        name=child.name,
                        kind="class",
                        start_line=_def_start_line(child),
                        end_line=getattr(child, "end_lineno", child.lineno) or child.lineno,
                        qualified_name=qual,
                    ))
                _walk(child, qual)
            elif isinstance(child, ast.Assign):
                # Module-level constants: NAME = ...
                if parent == "":
                    for tgt in child.targets:
                        if isinstance(tgt, ast.Name) and tgt.id == name:
                            hits.append(SymbolHit(
                                name=name,
                                kind="var",
                                start_line=child.lineno,
                                end_line=getattr(child, "end_lineno", child.lineno) or child.lineno,
                                qualified_name=name,
                            ))
            else:
                _walk(child, parent)

    _walk(tree, "")
    # Clamp to the actual file length — defensive against weird AST quirks.
    for h in hits:
        if h.end_line > total_lines:
            h.end_line = total_lines
    return hits


def _def_start_line(node: ast.AST) -> int:
    """Definitions can be preceded by decorators; prefer the topmost
    decorator's line so the returned range covers them too."""
    decos = getattr(node, "decorator_list", []) or []
    if decos:
        first = min(d.lineno for d in decos)
        return min(first, node.lineno)
    return node.lineno


def _locate_markdown(p: Path, name: str) -> list[SymbolHit]:
    """Find a heading whose text matches `name` (case-insensitive
    substring). End line = the line before the next heading of equal
    or higher level (or EOF)."""
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    needle = name.lower()
    headings: list[tuple[int, int, str]] = []  # (line_idx, level, text)
    for i, ln in enumerate(lines):
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))
    hits: list[SymbolHit] = []
    for idx, (line_idx, level, text) in enumerate(headings):
        if needle not in text.lower():
            continue
        # End at the next heading with level <= this one.
        end_line = len(lines)
        for nxt_idx, nxt_level, _ in headings[idx + 1:]:
            if nxt_level <= level:
                end_line = nxt_idx
                break
        hits.append(SymbolHit(
            name=text,
            kind="heading",
            start_line=line_idx + 1,
            end_line=end_line,  # 1-based inclusive bound at next-heading-line-1+1
            qualified_name=text,
        ))
    return hits


def _locate_textual(p: Path, name: str) -> list[SymbolHit]:
    """Word-boundary regex scan as a last resort. Returns each match
    on its own line as a 1-line range — caller can widen with
    `read_file(start_line=hit.start_line - 5, end_line=hit.start_line + 30)`
    if they want surrounding context."""
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []
    pattern = re.compile(r"\b" + re.escape(name) + r"\b")
    hits: list[SymbolHit] = []
    for i, ln in enumerate(lines):
        if pattern.search(ln):
            hits.append(SymbolHit(
                name=name,
                kind="match",
                start_line=i + 1,
                end_line=i + 1,
                qualified_name=name,
            ))
    return hits
