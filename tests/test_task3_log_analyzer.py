"""Tests for the JSON-lines Log Analyzer task (Python + C++)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


BASE = Path(__file__).resolve().parents[1]
TASK_DIR = BASE / "tasks" / "task3_log_analyzer"


def _make_log_file(tmp_path: Path, n_lines: int = 100) -> Path:
    """Generate a JSON-lines log file with realistic entries."""
    log_file = tmp_path / "app.log.jsonl"
    levels = ["INFO", "WARN", "ERROR", "DEBUG"]
    services = ["auth", "api", "db", "cache"]

    entries = []
    for i in range(n_lines):
        entry = {
            "timestamp": f"2026-05-{1 + (i % 28):02d}T{i % 24:02d}:{i % 60:02d}:00Z",
            "level": levels[i % len(levels)],
            "service": services[i % len(services)],
            "message": f"Log entry {i}",
            "duration_ms": (i * 7) % 500,
        }
        entries.append(entry)

    with open(log_file, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    return log_file


class TestPythonLogAnalyzer:
    @pytest.fixture
    def script(self) -> str:
        return str(TASK_DIR / "python" / "solve.py")

    @pytest.fixture
    def log_file(self, tmp_path: Path) -> Path:
        return _make_log_file(tmp_path, n_lines=100)

    def test_basic_analysis(self, script, log_file, tmp_path):
        result = subprocess.run(
            ["python3", script, str(log_file), "--output",
             str(tmp_path / "stats.json")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        stats = json.loads((tmp_path / "stats.json").read_text())
        assert stats["total_entries"] == 100
        assert "level_counts" in stats
        assert "service_counts" in stats
        assert stats["level_counts"]["INFO"] > 0
        assert stats["level_counts"]["ERROR"] > 0

    def test_filter_by_level(self, script, log_file, tmp_path):
        result = subprocess.run(
            ["python3", script, str(log_file), "--level", "ERROR",
             "--output", str(tmp_path / "error_stats.json")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        stats = json.loads((tmp_path / "error_stats.json").read_text())
        assert stats["total_entries"] == 100
        error_services = stats["per_service_error_counts"]
        assert "auth" in error_services or any(error_services.values())

    def test_top_slowest(self, script, log_file, tmp_path):
        result = subprocess.run(
            ["python3", script, str(log_file), "--top-n", "5",
             "--output", str(tmp_path / "slow.json")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        stats = json.loads((tmp_path / "slow.json").read_text())
        assert "slowest_entries" in stats
        assert len(stats["slowest_entries"]) == 5
        durations = [e["duration_ms"] for e in stats["slowest_entries"]]
        assert durations == sorted(durations, reverse=True)

    def test_json_output(self, script, log_file, tmp_path):
        result = subprocess.run(
            ["python3", script, str(log_file), "--format", "json",
             "--output", str(tmp_path / "out.json")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"

    def test_invalid_file(self, script):
        result = subprocess.run(
            ["python3", script, "/nonexistent/file.log.jsonl"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0


class TestCppLogAnalyzer:
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
        pytest.skip("C++ log_analyzer binary not found")

    @pytest.fixture
    def log_file(self, tmp_path: Path) -> Path:
        return _make_log_file(tmp_path, n_lines=100)

    def test_basic_analysis(self, binary, log_file, tmp_path):
        result = subprocess.run(
            [str(binary), str(log_file), "--output",
             str(tmp_path / "cpp_stats.json")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        stats = json.loads((tmp_path / "cpp_stats.json").read_text())
        assert stats["total_entries"] == 100

    def test_matches_python(self, binary, log_file, tmp_path):
        py_script = str(TASK_DIR / "python" / "solve.py")
        r1 = subprocess.run(
            ["python3", py_script, str(log_file), "--output",
             str(tmp_path / "py_stats.json")],
            capture_output=True, text=True, timeout=30,
        )
        r2 = subprocess.run(
            [str(binary), str(log_file), "--output",
             str(tmp_path / "cpp_stats.json")],
            capture_output=True, text=True, timeout=30,
        )
        assert r1.returncode == 0 and r2.returncode == 0, \
            f"py: {r1.stderr} cpp: {r2.stderr}"
        o1 = json.loads((tmp_path / "py_stats.json").read_text())
        o2 = json.loads((tmp_path / "cpp_stats.json").read_text())
        assert o1["total_entries"] == o2["total_entries"]
        assert o1["level_counts"] == o2["level_counts"]
