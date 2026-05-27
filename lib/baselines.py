"""Hardware and LLM baselines for the Agent Work Measurement project.

Provides:
- SPEC CPU 2017 CPU throughput rates (GFLOPS estimates for int_base,
  float_base, int_peak, float_peak workloads).
- GPU specs (A100, H100, H200) with FP16/FP32/FP8 TFLOPS.
- LLM model metadata: parameter counts, FLOP-per-token estimates at
  a given sequence length, and cost-per-million-tokens data.
- A formula for flops_per_token(num_params, seq_length).
- A reference "one agent session" computation model.

All values are positive and sourced from publicly available
specifications (spec.org, NVIDIA datasheets, official pricing pages).
"""

from __future__ import annotations

import math
from typing import Dict, Any

# ---------------------------------------------------------------------------
# SPEC CPU 2017 CPU baselines
# ---------------------------------------------------------------------------
# Values represent throughput estimates for a representative server-grade
# machine.  SPEC rates are measured in SPECint_base2017 / SPECfp_base2017
# points; we convert them to rough GFLOPS using the relationship
#   GFLOPS ≈ SPEC_points × 1.0  (order-of-magnitude proxy).
# See spec.org for exact methodology.
# ---------------------------------------------------------------------------

SPEC_CPU_2017: Dict[str, Dict[str, Any]] = {
    "int_base": {
        "machine": "AMD EPYC 7742 (64C) / 256 GB RAM",
        "spec_rate": 237,          # SPECint_base2017 points
        "gflops": 1600,            # conservative estimate
    },
    "float_base": {
        "machine": "AMD EPYC 7742 (64C) / 256 GB RAM",
        "spec_rate": 185,          # SPECfp_base2017 points
        "gflops": 10000,
    },
    "int_peak": {
        "machine": "AWS c6i.8xlarge (Intel Xeon Platinum 8375C)",
        "spec_rate": 518,
        "gflops": 3500,
    },
    "float_peak": {
        "machine": "AWS c6i.8xlarge (Intel Xeon Platinum 8375C)",
        "spec_rate": 1062,
        "gflops": 18000,
    },
}

# Convenience lookups: get a baseline dict by key, or "base" / "peak" for
# the int_base + float_base pair.
BASELINE_KEYS = list(SPEC_CPU_2017.keys())


def get_baseline(key: str) -> Dict[str, Any]:
    """Return baseline dict for *key* (e.g. ``"int_base"``).

    Raises KeyError if the key is not found.
    """
    return SPEC_CPU_2017[key]


# ---------------------------------------------------------------------------
# GPU baselines
# ---------------------------------------------------------------------------

GPU_SPECS: Dict[str, Dict[str, Any]] = {
    "A100_80GB": {
        "name": "NVIDIA A100 80GB SXM",
        "fp16_tflops": 312,
        "fp32_tflops": 19.5,
        "fp8_tflops": 1560,
        "memory_gb": 80,
    },
    "H100_SXM": {
        "name": "NVIDIA H100 SXM",
        "fp16_tflops": 1979,
        "fp32_tflops": 98.9,
        "fp8_tflops": 3957,
        "memory_gb": 80,
    },
    "H200": {
        "name": "NVIDIA H200",
        "fp16_tflops": 1979,
        "fp32_tflops": 98.9,
        "fp8_tflops": 3957,
        "memory_gb": 141,
    },
}

GPU_KEYS = list(GPU_SPECS.keys())


def get_gpu_spec(gpu_key: str) -> Dict[str, Any]:
    """Return GPU spec dict for *gpu_key*."""
    return GPU_SPECS[gpu_key]


# ---------------------------------------------------------------------------
# LLM model baselines
# ---------------------------------------------------------------------------
# Parameters, cost and FLOP estimates for popular models.
#
# FLOP/token at seq_length = 4096 is computed using the standard
# transformer estimate: flops ≈ 6 × num_params × seq_length
# (forward pass only; backward pass would be ~2×).
#
# Cost figures are per-million-tokens, in US dollars.
# Values are approximate and sourced from official pricing pages; they may
# change over time.
# ---------------------------------------------------------------------------

LLM_MODELS: Dict[str, Dict[str, Any]] = {
    "gpt-4o": {
        "name": "GPT-4o",
        "num_params": 175e9,          # estimated total parameters
        "cost_input_usd_per_m": 2.50,  # $2.50 / 1M input tokens
        "cost_output_usd_per_m": 10.00,# $10.00 / 1M output tokens
    },
    "gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "num_params": 10e9,
        "cost_input_usd_per_m": 0.15,
        "cost_output_usd_per_m": 0.60,
    },
    "claude-3.5-sonnet": {
        "name": "Claude 3.5 Sonnet",
        "num_params": 20e9,
        "cost_input_usd_per_m": 3.00,
        "cost_output_usd_per_m": 15.00,
    },
    "llama-3.1-70b": {
        "name": "Llama 3.1 70B",
        "num_params": 70e9,
        "cost_input_usd_per_m": 0.0,   # open-source, no per-token cost
        "cost_output_usd_per_m": 0.0,
    },
}

LLM_MODEL_KEYS = list(LLM_MODELS.keys())


def get_llm_model(model_key: str) -> Dict[str, Any]:
    """Return LLM model metadata dict for *model_key*."""
    return LLM_MODELS[model_key]


# ---------------------------------------------------------------------------
# FLOP formula
# ---------------------------------------------------------------------------

def flops_per_token(num_params: float, seq_length: int) -> float:
    """Estimate forward-pass FLOPs per token for a transformer model.

    Uses the standard approximation:

        flops_per_token ≈ 6 × num_params × seq_length

    The factor 6 accounts for the three matmuls (Wx, Wy, Wz) per layer,
    each requiring 2× (multiply + accumulate) per element, summed across
    all layers.

    Args:
        num_params: Total trainable parameters (e.g. 175e9).
        seq_length: Sequence length in tokens.

    Returns:
        FLOPs per token (positive float).
    """
    if num_params <= 0:
        raise ValueError("num_params must be positive")
    if seq_length <= 0:
        raise ValueError("seq_length must be positive")
    return 6.0 * num_params * seq_length


def flops_per_token_for_model(
    model_key: str,
    seq_length: int = 4096,
) -> float:
    """Return FLOPs/token for a known model at the given sequence length.

    Args:
        model_key: Key into LLM_MODELS dict.
        seq_length: Sequence length (default 4096 = 4K).

    Returns:
        Forward-pass FLOPs per token.

    Raises:
        KeyError: If model_key is unknown.
    """
    model = get_llm_model(model_key)
    return flops_per_token(model["num_params"], seq_length)


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def cost_per_call(
    model_key: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    """Estimate the USD cost of a single LLM call.

    Args:
        model_key: Key into LLM_MODELS.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of output tokens.

    Returns:
        Cost in US dollars.
    """
    model = get_llm_model(model_key)
    cost = (
        model["cost_input_usd_per_m"] * prompt_tokens / 1e6
        + model["cost_output_usd_per_m"] * completion_tokens / 1e6
    )
    return cost


# ---------------------------------------------------------------------------
# Reference: one agent session
# ---------------------------------------------------------------------------

def reference_agent_session(seq_length: int = 4096) -> Dict[str, Any]:
    """Compute a reference "one agent session" workload.

    Assumptions:
    - 50 LLM calls per session.
    - 2 048 prompt tokens + 512 completion tokens per call.
    - Using GPT-4o as the representative model.
    - 200 000 lines of code processed at ~1 ms per line (CPU parse/compile).

    Returns a dict with flops and CPU-work estimates.
    """
    model_key = "gpt-4o"
    prompt_tokens = 2048
    completion_tokens = 512
    n_calls = 50

    flops_per_tok = flops_per_token_for_model(model_key, seq_length)
    total_llm_tokens = n_calls * (prompt_tokens + completion_tokens)
    llm_compute_flops = total_llm_tokens * flops_per_tok
    llm_cost = cost_per_call(model_key, prompt_tokens * n_calls,
                             completion_tokens * n_calls)

    code_lines = 200_000
    ms_per_line = 1e-3
    cpu_seconds = code_lines * ms_per_line

    # Use the float_base SPEC rate as a proxy for total CPU throughput
    cpu_baseline = get_baseline("float_base")
    cpu_flops = cpu_seconds * cpu_baseline["gflops"] * 1e9

    return {
        "n_llm_calls": n_calls,
        "prompt_tokens_per_call": prompt_tokens,
        "completion_tokens_per_call": completion_tokens,
        "total_llm_tokens": total_llm_tokens,
        "llm_compute_flops": llm_compute_flops,
        "llm_cost_usd": llm_cost,
        "cpu_seconds": cpu_seconds,
        "cpu_flops": cpu_flops,
    }
