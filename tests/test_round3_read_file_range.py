"""Round 3 / #3: read_file accepts start_line / end_line for slicing
large files without re-reading the whole body. Output is line-numbered
so the model can quote unambiguously."""
from __future__ import annotations

from backend.tools.file_reader import read_file


def _write(tmp_path, name, lines):
    p = tmp_path / name
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def test_full_read_unchanged_when_no_range(tmp_path):
    p = _write(tmp_path, "x.py", [f"line {i}" for i in range(1, 6)])
    out = read_file(p)
    assert out == "line 1\nline 2\nline 3\nline 4\nline 5"


def test_range_returns_only_requested_lines(tmp_path):
    p = _write(tmp_path, "x.py", [f"line {i}" for i in range(1, 11)])
    out = read_file(p, start_line=3, end_line=5)
    # Header tells the model what slice they got.
    assert "lines 3-5 of 10" in out
    assert "line 3" in out
    assert "line 5" in out
    assert "line 6" not in out
    assert "line 2" not in out


def test_line_numbers_are_prefixed(tmp_path):
    p = _write(tmp_path, "x.py", [f"hello{i}" for i in range(1, 11)])
    out = read_file(p, start_line=3, end_line=4)
    # Each rendered line carries its own number — quotes are unambiguous.
    assert "    3│ hello3" in out
    assert "    4│ hello4" in out


def test_open_ended_start_or_end(tmp_path):
    p = _write(tmp_path, "x.py", [f"l{i}" for i in range(1, 6)])
    # Only end_line — start defaults to 1
    out_head = read_file(p, end_line=2)
    assert "l1" in out_head and "l2" in out_head
    assert "l3" not in out_head
    # Only start_line — end defaults to last
    out_tail = read_file(p, start_line=4)
    assert "l4" in out_tail and "l5" in out_tail
    assert "l3" not in out_tail


def test_out_of_range_returns_clean_error(tmp_path):
    p = _write(tmp_path, "x.py", [f"l{i}" for i in range(1, 4)])
    out = read_file(p, start_line=100, end_line=200)
    assert "range out of file" in out


def test_max_chars_still_caps_after_range(tmp_path):
    p = _write(tmp_path, "x.py", [f"line {i:03d} with extra padding text" for i in range(1, 1000)])
    out = read_file(p, max_chars=200, start_line=100, end_line=999)
    assert len(out) <= 220  # 200 + small header overhead


def test_handler_passes_range_through(tmp_path, monkeypatch):
    """The tool handler must thread start_line/end_line through to
    the underlying reader — otherwise the schema is decorative."""
    p = _write(tmp_path, "x.py", [f"l{i}" for i in range(1, 11)])
    from backend.builtin_tools import _read_file_handler, FILE_CACHE
    FILE_CACHE.clear()
    out = _read_file_handler(str(p), max_chars=2000, start_line=3, end_line=5)
    assert "lines 3-5" in out
    assert "l3" in out
    assert "l6" not in out


def test_handler_caches_per_range(tmp_path):
    """Different (start_line, end_line) pairs must NOT collide in the
    cache — otherwise asking for lines 1-10 and then 11-20 would
    return the first body. Cache key includes the range params."""
    # Use sentinel tokens that don't share prefixes so substring tests
    # below don't false-match (e.g. 'l1' vs 'l15').
    sentinels = [f"<<row{i:02d}>>" for i in range(1, 21)]
    p = _write(tmp_path, "x.py", sentinels)
    from backend.builtin_tools import _read_file_handler, FILE_CACHE
    FILE_CACHE.clear()
    head = _read_file_handler(str(p), start_line=1, end_line=5)
    tail = _read_file_handler(str(p), start_line=15, end_line=20)
    assert "<<row01>>" in head and "<<row05>>" in head
    assert "<<row15>>" not in head
    assert "<<row15>>" in tail and "<<row20>>" in tail
    assert "<<row01>>" not in tail
