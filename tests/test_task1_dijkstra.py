"""Tests for the Dijkstra shortest-path task (Python + C++)."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import pytest


BASE = Path(__file__).resolve().parents[1]
TASK_DIR = BASE / "tasks" / "task1_dijkstra"


def _make_graph_file(tmp_path: Path, edges: list[tuple[str, str, float]]) -> Path:
    """Write a CSV graph file with header: src,dst,weight."""
    graph_file = tmp_path / "graph.csv"
    with open(graph_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "weight"])
        for src, dst, wgt in edges:
            w.writerow([src, dst, wgt])
    return graph_file


class TestPythonDijkstra:
    @pytest.fixture
    def py_script(self) -> str:
        return str(TASK_DIR / "python" / "solve.py")

    @pytest.fixture
    def graph_file(self, tmp_path: Path) -> Path:
        edges = [
            ("A", "B", 1.0), ("A", "C", 4.0),
            ("B", "C", 2.0), ("B", "D", 5.0),
            ("C", "D", 1.0), ("D", "E", 3.0),
        ]
        return _make_graph_file(tmp_path, edges)

    def test_basic_run(self, py_script, graph_file):
        """Basic shortest-path from A to E."""
        result = subprocess.run(
            ["python3", py_script, str(graph_file), "A", "E"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output = json.loads(result.stdout)
        assert output["source"] == "A"
        assert output["destination"] == "E"
        assert output["distance"] == pytest.approx(7.0, rel=1e-6)
        assert "path" in output

    def test_undirected_option(self, py_script, graph_file):
        """Undirected: same shortest path 7 (A->B->C->D->E)."""
        result = subprocess.run(
            ["python3", py_script, str(graph_file), "A", "E", "--undirected"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output = json.loads(result.stdout)
        assert output["distance"] == pytest.approx(7.0, rel=1e-6)

    def test_unreachable_node(self, py_script, graph_file):
        """Z is unreachable; should report inf distance."""
        result = subprocess.run(
            ["python3", py_script, str(graph_file), "A", "Z"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output = json.loads(result.stdout)
        assert output["distance"] == float("inf")
        assert output["path"] is None

    def test_single_node(self, py_script, tmp_path: Path):
        """Start==End with empty graph."""
        graph = tmp_path / "single.csv"
        graph.write_text("src,dst,weight\n")
        result = subprocess.run(
            ["python3", py_script, str(graph), "A", "A"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output = json.loads(result.stdout)
        assert output["distance"] == 0.0
        assert output["path"] == ["A"]

    def test_negative_weight(self, py_script, tmp_path: Path):
        """Dijkstra should still run (may not be correct with negatives)."""
        edges = [("A", "B", -2.0), ("B", "C", 1.0), ("A", "C", 10.0)]
        graph = _make_graph_file(tmp_path, edges)
        result = subprocess.run(
            ["python3", py_script, str(graph), "A", "C"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output = json.loads(result.stdout)
        assert "distance" in output


class TestCppDijkstra:
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
        pytest.skip("C++ dijkstra binary not found")

    @pytest.fixture
    def graph_file(self, tmp_path: Path) -> Path:
        edges = [
            ("A", "B", 1.0), ("A", "C", 4.0),
            ("B", "C", 2.0), ("B", "D", 5.0),
            ("C", "D", 1.0), ("D", "E", 3.0),
        ]
        return _make_graph_file(tmp_path, edges)

    def test_basic_run(self, binary, graph_file):
        result = subprocess.run(
            [str(binary), str(graph_file), "A", "E"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output = json.loads(result.stdout)
        assert output["source"] == "A"
        assert output["destination"] == "E"
        assert output["distance"] == pytest.approx(7.0, rel=1e-6)

    def test_undirected(self, binary, graph_file):
        result = subprocess.run(
            [str(binary), str(graph_file), "A", "E", "--undirected"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output = json.loads(result.stdout)
        assert output["distance"] == pytest.approx(7.0, rel=1e-6)

    def test_unreachable(self, binary, graph_file):
        result = subprocess.run(
            [str(binary), str(graph_file), "A", "Z"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        output = json.loads(result.stdout)
        # json.loads parses unquoted Infinity as float('inf')
        assert output["distance"] == float("inf")

    def test_matches_python(self, binary, graph_file):
        """Python and C++ should produce matching results."""
        py_script = str(TASK_DIR / "python" / "solve.py")
        r1 = subprocess.run(
            ["python3", py_script, str(graph_file), "A", "E"],
            capture_output=True, text=True, timeout=30,
        )
        r2 = subprocess.run(
            [str(binary), str(graph_file), "A", "E"],
            capture_output=True, text=True, timeout=30,
        )
        assert r1.returncode == 0 and r2.returncode == 0, \
            f"py: {r1.stderr} cpp: {r2.stderr}"
        o1, o2 = json.loads(r1.stdout), json.loads(r2.stdout)
        assert o1["distance"] == pytest.approx(o2["distance"], rel=1e-6)
        assert o1["path"] == o2["path"]
