"""Tests for the Markdown TOC Generator task (Python + C++)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


BASE = Path(__file__).resolve().parents[1]
TASK_DIR = BASE / "tasks" / "task2_toc_generator"


def _make_sample_docs(tmp_path: Path) -> Path:
    """Write sample markdown with nested headings."""
    docs_dir = tmp_path / "sample_docs"
    docs_dir.mkdir()

    content = """\
# Main Title

Some intro text.

## Section 1

### Subsection 1.1

More text here.

### Subsection 1.2

Even more.

## Section 2

Simple section.

### Subsection 2.1

#### Deep level

Very deep.

## Section 3

Final section.
"""
    docs_file = docs_dir / "example.md"
    docs_file.write_text(content)
    return docs_file


class TestPythonTOCGenerator:
    @pytest.fixture
    def script(self) -> str:
        return str(TASK_DIR / "python" / "solve.py")

    @pytest.fixture
    def docs_file(self, tmp_path: Path) -> Path:
        return _make_sample_docs(tmp_path)

    def test_basic_toc(self, script, docs_file, tmp_path):
        """Basic TOC extraction."""
        result = subprocess.run(
            ["python3", script, str(docs_file), "--output",
             str(tmp_path / "toc.json")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        toc = json.loads((tmp_path / "toc.json").read_text())
        assert len(toc) == 3  # three H2 headings
        assert toc[0]["title"] == "Section 1"
        assert toc[0]["level"] == 2
        assert toc[0]["children"][0]["title"] == "Subsection 1.1"

    def test_nested_toc(self, script, docs_file, tmp_path):
        """Check nested children structure."""
        result = subprocess.run(
            ["python3", script, str(docs_file), "--output",
             str(tmp_path / "toc_nested.json")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        toc = json.loads((tmp_path / "toc_nested.json").read_text())
        sec1 = toc[0]
        assert sec1["title"] == "Section 1"
        assert len(sec1["children"]) == 2
        sec2_1 = toc[1]["children"][0]
        assert sec2_1["title"] == "Subsection 2.1"
        assert len(sec2_1["children"]) == 1

    def test_max_depth(self, script, docs_file, tmp_path):
        """--max-depth 2 should flatten children."""
        result = subprocess.run(
            ["python3", script, str(docs_file), "--max-depth", "2",
             "--output", str(tmp_path / "toc_shallow.json")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        toc = json.loads((tmp_path / "toc_shallow.json").read_text())
        for sec in toc:
            assert len(sec["children"]) == 0

    def test_html_output(self, script, docs_file, tmp_path):
        """HTML format output."""
        result = subprocess.run(
            ["python3", script, str(docs_file), "--format", "html",
             "--output", str(tmp_path / "toc.html")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        html = (tmp_path / "toc.html").read_text()
        assert "<ul>" in html
        assert "Section 1" in html
        assert "<li>" in html

    def test_invalid_file(self, script):
        """Non-existent file should fail."""
        result = subprocess.run(
            ["python3", script, "/nonexistent/file.md"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0


class TestCppTOCGenerator:
    @pytest.fixture
    def binary(self) -> Path:
        bin_path = TASK_DIR / "cpp" / "solve"
        if bin_path.exists():
            return bin_path
        src = TASK_DIR / "cpp" / "solve.cpp"
        if src.exists():
            subprocess.run(
                ["g++", "-std=c++17", "-O2", "-o", str(bin_path), str(src)],
                check=True, timeout=60,
            )
        pytest.skip("C++ toc_generator binary not found")

    @pytest.fixture
    def docs_file(self, tmp_path: Path) -> Path:
        return _make_sample_docs(tmp_path)

    def test_basic_toc(self, binary, docs_file, tmp_path):
        result = subprocess.run(
            [str(binary), str(docs_file), "--output",
             str(tmp_path / "toc_cpp.json")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        toc = json.loads((tmp_path / "toc_cpp.json").read_text())
        assert len(toc) == 3

    def test_matches_python(self, binary, docs_file, tmp_path):
        py_script = str(TASK_DIR / "python" / "solve.py")
        r1 = subprocess.run(
            ["python3", py_script, str(docs_file), "--output",
             str(tmp_path / "toc_py.json")],
            capture_output=True, text=True, timeout=30,
        )
        r2 = subprocess.run(
            [str(binary), str(docs_file), "--output",
             str(tmp_path / "toc_cpp.json")],
            capture_output=True, text=True, timeout=30,
        )
        assert r1.returncode == 0 and r2.returncode == 0, \
            f"py: {r1.stderr} cpp: {r2.stderr}"
        o1 = json.loads((tmp_path / "toc_py.json").read_text())
        o2 = json.loads((tmp_path / "toc_cpp.json").read_text())
        assert len(o1) == len(o2)
