"""Tests for lib/model.py — decomposition, standardization, ratios."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

import pytest

from lib.harness import MeasurementSession, TraceNode
from lib.model import (
    analyse,
    apply_network_correction,
    ai_cpu_ratio,
    decompose,
    standardize_to_cpu_equivalent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_flat_session() -> MeasurementSession:
  """Build a tiny measurement session with 1 CPU node + 1 LLM node."""
  session = MeasurementSession(name="test")
  # Add a CPU child to the root
  cpu_node = TraceNode(name="cpu_task", node_type="cpu", category="compute",
                       user_cpu=1.5, system_cpu=0.3, wall_clock=2.0)
  cpu_node.total_cpu = cpu_node.user_cpu + cpu_node.system_cpu
  session._root.children.append(cpu_node)
  cpu_node._parent = session._root
  # Add an LLM child to the CPU node (use a known model name for FLOP estimation)
  llm_node = TraceNode(name="gpt-4o", node_type="llm", model="gpt-4o",
                       wall_clock=3.0, prompt_tokens=500, completion_tokens=200,
                       latency=3.0)
  llm_node.total_tokens = llm_node.prompt_tokens + llm_node.completion_tokens
  cpu_node.children.append(llm_node)
  llm_node._parent = cpu_node
  # Set session wall clock
  session._root.wall_clock = 5.0
  return session


def _make_dict_trace() -> Dict[str, Any]:
    """Build a plain-dict trace matching the format from session.to_dict()."""
    return {
        "name": "root",
        "node_type": "session",
        "wall_clock": 5.0,
        "children": [
            {
                "name": "cpu_task",
                "node_type": "cpu",
                "wall_clock": 2.0,
                "user_cpu": 1.5,
                "system_cpu": 0.3,
                "total_cpu": 1.8,
                "children": [
                    {
                        "name": "test-model-7b",
                        "node_type": "llm",
                        "wall_clock": 3.0,
                        "model": "test-model-7b",
                        "prompt_tokens": 500,
                        "completion_tokens": 200,
                        "total_tokens": 700,
                        "latency": 3.0,
                        "children": [],
                    }
                ],
            }
        ],
    }


# ---------------------------------------------------------------------------
# decompose()
# ---------------------------------------------------------------------------


class TestDecompose:
    def test_cpu_work_from_trace(self):
        session = _build_flat_session()
        result = decompose(session)
        assert result["cpu_work"] == pytest.approx(1.8, rel=1e-3)

    def test_ai_work_from_trace(self):
        session = _build_flat_session()
        result = decompose(session)
        # No network correction => ai_work == wall_clock of LLM node
        assert result["ai_work"] == pytest.approx(3.0, rel=1e-3)

    def test_wait_time_from_trace(self):
        session = _build_flat_session()
        result = decompose(session)
        # total is 5.0, cpu_work=1.8, ai_work=3.0 => wait=0.2
        assert result["wait_time"] == pytest.approx(0.2, rel=1e-3)

    def test_total_wall_clock(self):
        session = _build_flat_session()
        result = decompose(session)
        assert result["total_wall_clock"] == pytest.approx(5.0, rel=1e-3)

    def test_dict_trace_decomposition(self):
        result = decompose(_make_dict_trace())
        assert result["cpu_work"] == pytest.approx(1.8, rel=1e-3)
        assert result["ai_work"] == pytest.approx(3.0, rel=1e-3)

    def test_network_correction(self):
        session = _build_flat_session()
        result = decompose(session, rtt_estimate=0.4)
        # AI compute = 3.0 - 0.2 = 2.8
        assert result["ai_work"] == pytest.approx(2.8, rel=1e-3)

    def test_llm_nodes_breakdown(self):
        session = _build_flat_session()
        result = decompose(session)
        assert len(result["llm_nodes"]) == 1
        assert result["llm_nodes"][0]["name"] == "gpt-4o"
        assert result["llm_nodes"][0]["model"] == "gpt-4o"
        assert result["llm_nodes"][0]["total_tokens"] == 700

    def test_cpu_nodes_breakdown(self):
        session = _build_flat_session()
        result = decompose(session)
        assert len(result["cpu_nodes"]) == 1
        assert result["cpu_nodes"][0]["name"] == "cpu_task"
        assert result["cpu_nodes"][0]["total_cpu"] == pytest.approx(1.8, rel=1e-3)

    def test_invalid_trace_type(self):
        with pytest.raises(TypeError):
            decompose(None)

    def test_empty_session(self):
        session = MeasurementSession(name="empty")
        session._root.wall_clock = 0.0
        result = decompose(session)
        assert result["cpu_work"] == 0.0
        assert result["ai_work"] == 0.0
        assert result["total_wall_clock"] == 0.0


# ---------------------------------------------------------------------------
# standardize_to_cpu_equivalent()
# ---------------------------------------------------------------------------


class TestStandardize:
    def test_basic_standardization(self):
        decomp = decompose(_build_flat_session())
        result = standardize_to_cpu_equivalent(decomp)
        assert result["standard_cpu_seconds"] > 0
        assert result["baseline_key"] == "float_base"
        # fractions should sum to ~1
        frac_sum = result["cpu_work_fraction"] + result["ai_work_fraction"]
        assert frac_sum == pytest.approx(1.0, abs=0.01)

    def test_zero_cpu_work(self):
        decomp = {"cpu_work": 0.0, "ai_compute_flops": 1e14}
        result = standardize_to_cpu_equivalent(decomp)
        assert result["standard_cpu_seconds"] > 0
        assert result["cpu_work_fraction"] == 0.0

    def test_zero_ai_work(self):
        decomp = {"cpu_work": 5.0, "ai_compute_flops": 0.0}
        result = standardize_to_cpu_equivalent(decomp)
        assert result["standard_cpu_seconds"] == pytest.approx(5.0, rel=1e-3)
        assert result["ai_work_fraction"] == 0.0

    def test_zero_both(self):
        decomp = {"cpu_work": 0.0, "ai_compute_flops": 0.0}
        result = standardize_to_cpu_equivalent(decomp)
        assert result["standard_cpu_seconds"] == 0.0
        assert result["cpu_work_fraction"] == 0.0
        assert result["ai_work_fraction"] == 0.0

    def test_different_baseline(self):
        decomp = decompose(_build_flat_session())
        result = standardize_to_cpu_equivalent(decomp, baseline_key="int_base")
        assert result["baseline_key"] == "int_base"
        assert result["baseline_gflops"] > 0


# ---------------------------------------------------------------------------
# ai_cpu_ratio()
# ---------------------------------------------------------------------------


class TestAICPURatio:
    def test_balanced_ratio(self):
        decomp = {"cpu_work": 1.0, "ai_work": 1.0}
        result = ai_cpu_ratio(decomp)
        assert result["ratio"] == pytest.approx(1.0, rel=1e-3)
        assert result["interpretation"] == "balanced"

    def test_ai_heavy(self):
        decomp = {"cpu_work": 1.0, "ai_work": 10.0}
        result = ai_cpu_ratio(decomp)
        assert result["interpretation"] == "ai_heavy"
        assert result["ratio"] > 1.5

    def test_cpu_heavy(self):
        decomp = {"cpu_work": 10.0, "ai_work": 1.0}
        result = ai_cpu_ratio(decomp)
        assert result["interpretation"] == "cpu_heavy"
        assert result["ratio"] < 0.5

    def test_zero_cpu_nonzero_ai(self):
        decomp = {"cpu_work": 0.0, "ai_work": 5.0}
        result = ai_cpu_ratio(decomp)
        assert result["ratio"] == float("inf")
        assert result["interpretation"] == "ai_heavy"

    def test_zero_both(self):
        decomp = {"cpu_work": 0.0, "ai_work": 0.0}
        result = ai_cpu_ratio(decomp)
        assert result["ratio"] == 0.0
        assert result["interpretation"] == "balanced"


# ---------------------------------------------------------------------------
# apply_network_correction()
# ---------------------------------------------------------------------------


class TestNetworkCorrection:
    def test_basic(self):
        assert apply_network_correction(3.0, 0.4) == pytest.approx(2.8, rel=1e-3)

    def test_rtt_larger_than_latency(self):
        assert apply_network_correction(0.1, 1.0) == 0.0

    def test_no_correction(self):
        assert apply_network_correction(3.0, 0.0) == pytest.approx(3.0, rel=1e-3)


# ---------------------------------------------------------------------------
# analyse() — full pipeline
# ---------------------------------------------------------------------------


class TestAnalyse:
    def test_full_pipeline(self):
        session = _build_flat_session()
        result = analyse(session)
        assert "cpu_work" in result
        assert "ai_work" in result
        assert "ratio" in result
        assert "interpretation" in result
        assert "standard_cpu_seconds" in result

    def test_analysis_with_network_correction(self):
        session = _build_flat_session()
        result = analyse(session, rtt_estimate=0.4)
        assert result["ai_work"] == pytest.approx(2.8, rel=1e-3)

    def test_analysis_with_different_baseline(self):
        session = _build_flat_session()
        result = analyse(session, baseline_key="int_base")
        assert result["baseline_key"] == "int_base"
        assert result["ai_work_fraction"] > 0

    def test_dict_input(self):
        result = analyse(_make_dict_trace())
        assert result["cpu_work"] == pytest.approx(1.8, rel=1e-3)
