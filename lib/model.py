"""Composition model: decompose total wall-clock into CPU vs AI work.

This module provides functions that take a measurement trace (produced by
``lib.harness``) and decompose the total wall-clock time into three
buckets:

    total_wall_clock = cpu_work + ai_work + wait_time

where:
- ``cpu_work``    = sum of actual CPU time spent on the agent's own code
- ``ai_work``     = LLM inference time, corrected for network latency
- ``wait_time``   = remainder (I/O wait, network latency, human pauses)

The model also supports:
- **Standardized work units** — convert everything to CPU-equivalent
  seconds using SPEC CPU 2017 baselines.
- **AI-to-CPU ratio** with interpretation ("ai_heavy", "cpu_heavy",
  "balanced").
- **Network latency correction** given a round-trip time estimate.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from lib.harness import TraceNode, GaugeSession
from lib.baselines import get_baseline, flops_per_token_for_model, SPEC_CPU_2017


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Thresholds for ratio interpretation (AI compute time / CPU work):
# - Below this: cpu_heavy
# - Above this: ai_heavy
# - In between: balanced
_RATIO_CPU_HEAVY = 1.0
_RATIO_AI_HEAVY = 1.0
_RATIO_BOUNCED = 0.5  # tolerance band around 1.0


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------

def decompose(
    trace: GaugeSession | Dict[str, Any],
    rtt_estimate: Optional[float] = None,
) -> Dict[str, Any]:
    """Decompose a measurement trace into cpu_work, ai_work, and wait_time.

    Args:
        trace: A GaugeSession or a plain dict (from
            ``session.to_dict()``).
        rtt_estimate: Optional round-trip time to the LLM API server
            in seconds.  If given, the AI compute time for each LLM node
            is reduced by ``rtt_estimate / 2``.  Defaults to ``None``
            (no network correction).

    Returns:
        A dict with keys:

        - ``cpu_work`` (float): sum of total_cpu across non-LLM nodes.
        - ``ai_work`` (float): net AI compute time after network correction.
        - ``wait_time`` (float): total_wall_clock - cpu_work - ai_work.
        - ``total_wall_clock`` (float): total wall-clock seconds.
        - ``ai_compute_flops`` (float): estimated total FLOPs from LLM calls.
        - ``llm_nodes`` (list): per-node breakdown of LLM contribution.
        - ``cpu_nodes`` (list): per-node breakdown of CPU contribution.
    """
    if isinstance(trace, GaugeSession):
        nodes = trace.flattened()
        root = trace._root
    elif isinstance(trace, dict):
        nodes = _flat_from_dict(trace)
        root = trace
    else:
        raise TypeError("trace must be a GaugeSession or a dict")

    # Find root to get total session wall-clock.
    total_wall_clock = 0.0
    if isinstance(root, dict):
        total_wall_clock = root.get("wall_clock", 0.0)
    elif hasattr(root, "wall_clock"):
        total_wall_clock = root.wall_clock

    cpu_work = 0.0
    ai_work = 0.0
    ai_compute_flops = 0.0
    llm_node_breakdown: List[Dict[str, Any]] = []
    cpu_node_breakdown: List[Dict[str, Any]] = []

    for node in nodes:
        # Skip the root session node — it is not a tool call.
        if isinstance(node, TraceNode) and node.node_type == "session":
            continue
        if isinstance(node, dict) and node.get("node_type") == "session":
            continue

        if isinstance(node, TraceNode):
            is_llm = node.node_type == "llm"
        elif isinstance(node, dict):
            is_llm = node.get("node_type") == "llm"
        else:
            continue

        if is_llm:
            llm_wall = node.wall_clock if isinstance(node, TraceNode) else node.get("wall_clock", 0.0)
            # Network latency correction
            net_latency = 0.0
            if rtt_estimate is not None and rtt_estimate > 0:
                net_latency = rtt_estimate / 2.0
            ai_compute = max(0.0, llm_wall - net_latency)
            ai_work += ai_compute

            # FLOP estimate for this node
            model_name = None
            if isinstance(node, TraceNode):
                model_name = node.model
            else:
                model_name = node.get("model", "unknown")
            prompt_t = node.prompt_tokens if isinstance(node, TraceNode) else node.get("prompt_tokens", 0)
            comp_t = node.completion_tokens if isinstance(node, TraceNode) else node.get("completion_tokens", 0)
            total_t = prompt_t + comp_t
            fpt = 0.0
            if model_name and total_t > 0:
                try:
                    fpt = flops_per_token_for_model(model_name, seq_length=4096)
                    ai_compute_flops += total_t * fpt
                except KeyError:
                    pass  # unknown model, skip FLOP estimate

            llm_node_breakdown.append({
                "name": node.name if isinstance(node, TraceNode) else node.get("name", ""),
                "model": model_name,
                "latency": llm_wall,
                "net_latency_correction": net_latency,
                "ai_compute_time": ai_compute,
                "total_tokens": total_t,
                "estimated_flops": total_t * fpt if model_name else 0.0,
            })

        else:
            # Non-LLM node: count CPU work.
            node_cpu = node.total_cpu if isinstance(node, TraceNode) else node.get("total_cpu", 0.0)
            cpu_work += node_cpu

            cpu_node_breakdown.append({
                "name": node.name if isinstance(node, TraceNode) else node.get("name", ""),
                "total_cpu": node_cpu,
                "wall_clock": node.wall_clock if isinstance(node, TraceNode) else node.get("wall_clock", 0.0),
            })

    wait_time = max(0.0, total_wall_clock - cpu_work - ai_work)

    return {
        "cpu_work": cpu_work,
        "ai_work": ai_work,
        "wait_time": wait_time,
        "total_wall_clock": total_wall_clock,
        "ai_compute_flops": ai_compute_flops,
        "llm_nodes": llm_node_breakdown,
        "cpu_nodes": cpu_node_breakdown,
    }


# ---------------------------------------------------------------------------
# CPU-equivalent standardization
# ---------------------------------------------------------------------------

def standardize_to_cpu_equivalent(
    decomposition: Dict[str, Any],
    baseline_key: str = "float_base",
) -> Dict[str, Any]:
    """Convert decomposed work to standard CPU-equivalent seconds.

    Uses the SPEC CPU 2017 baseline GFLOPS rate to express AI compute
    work as an equivalent amount of CPU time.

    Args:
        decomposition: Output of :func:`decompose`.
        baseline_key: Key into SPEC_CPU_2017 (default ``"float_base"``).

    Returns:
        A dict augmented with:

        - ``standard_cpu_seconds``: cpu_work + (ai_compute_flops / baseline_gflops * 1e9)
        - ``baseline_gflops``: the GFLOPS rate used.
        - ``cpu_work_fraction``: cpu_work / standard_cpu_seconds
        - ``ai_work_fraction``: ai_compute_flops / (baseline_gflops * 1e9) / standard_cpu_seconds
    """
    baseline = get_baseline(baseline_key)
    gflops = baseline["gflops"]
    baseline_flops_per_sec = gflops * 1e9

    cpu_work = decomposition["cpu_work"]
    ai_compute_flops = decomposition["ai_compute_flops"]

    ai_as_cpu_seconds = ai_compute_flops / baseline_flops_per_sec if baseline_flops_per_sec > 0 else 0.0
    std_cpu = cpu_work + ai_as_cpu_seconds

    return {
        "cpu_work": cpu_work,
        "ai_work_cpu_equivalent": ai_as_cpu_seconds,
        "standard_cpu_seconds": std_cpu,
        "baseline_key": baseline_key,
        "baseline_gflops": gflops,
        "cpu_work_fraction": cpu_work / std_cpu if std_cpu > 0 else 0.0,
        "ai_work_fraction": ai_as_cpu_seconds / std_cpu if std_cpu > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# AI-to-CPU ratio
# ---------------------------------------------------------------------------

def ai_cpu_ratio(decomposition: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the AI-to-CPU work ratio and interpretation.

    Args:
        decomposition: Output of :func:`decompose`.

    Returns:
        A dict with:

        - ``ratio``: ai_work / cpu_work
        - ``interpretation``: one of ``"ai_heavy"``, ``"cpu_heavy"``,
          ``"balanced"``
        - ``ai_work``: ai_work from decomposition
        - ``cpu_work``: cpu_work from decomposition
    """
    cpu_work = decomposition["cpu_work"]
    ai_work = decomposition["ai_work"]

    if cpu_work == 0:
        if ai_work > 0:
            ratio = float("inf")
            interpretation = "ai_heavy"
        else:
            ratio = 0.0
            interpretation = "balanced"
    else:
        ratio = ai_work / cpu_work
        if ratio < _RATIO_CPU_HEAVY - _RATIO_BOUNCED:
            interpretation = "cpu_heavy"
        elif ratio > _RATIO_AI_HEAVY + _RATIO_BOUNCED:
            interpretation = "ai_heavy"
        else:
            interpretation = "balanced"

    return {
        "ratio": ratio,
        "interpretation": interpretation,
        "ai_work": ai_work,
        "cpu_work": cpu_work,
    }


# ---------------------------------------------------------------------------
# Network latency correction (standalone)
# ---------------------------------------------------------------------------

def apply_network_correction(
    latency: float,
    rtt: float,
) -> float:
    """Subtract half the round-trip time from a measured latency.

    Args:
        latency: Measured wall-clock latency in seconds.
        rtt: Estimated round-trip time to the API server in seconds.

    Returns:
        Net compute time (never negative).
    """
    correction = rtt / 2.0
    return max(0.0, latency - correction)


# ---------------------------------------------------------------------------
# Convenience: single-call convenience function
# ---------------------------------------------------------------------------

def analyse(
    trace: GaugeSession | Dict[str, Any],
    rtt_estimate: Optional[float] = None,
    baseline_key: str = "float_base",
) -> Dict[str, Any]:
    """Full analysis pipeline: decompose + standardize + ratio.

    Args:
        trace: GaugeSession or dict.
        rtt_estimate: RTT in seconds.
        baseline_key: SPEC baseline key.

    Returns:
        A dict combining results from :func:`decompose`,
        :func:`standardize_to_cpu_equivalent`, and :func:`ai_cpu_ratio`.
    """
    decomp = decompose(trace, rtt_estimate=rtt_estimate)
    std = standardize_to_cpu_equivalent(decomp, baseline_key=baseline_key)
    ratio = ai_cpu_ratio(decomp)

    return {
        **decomp,
        **std,
        **ratio,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _flat_from_dict(d: Dict[str, Any]) -> List[TraceNode]:
    """Flatten a dict trace tree into a list of TraceNode objects."""
    result: List[TraceNode] = []

    def _walk(node_dict: Dict[str, Any]) -> TraceNode:
        from lib.harness import TraceNode
        node = TraceNode(
            name=node_dict.get("name", "unknown"),
            node_type=node_dict.get("node_type", "cpu"),
        )
        for attr in ("category", "wall_clock", "user_cpu", "system_cpu",
                      "total_cpu", "max_rss", "io_read_bytes",
                      "io_write_bytes", "prompt_tokens", "completion_tokens",
                      "total_tokens", "latency", "model", "cost_estimate"):
            val = node_dict.get(attr)
            if val is not None:
                setattr(node, attr, val)
        result.append(node)
        for child_dict in node_dict.get("children", []):
            _walk(child_dict)
        return node

    _walk(d)
    return result
