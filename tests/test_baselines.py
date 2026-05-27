"""Tests for lib/baselines.py - hardware and LLM baseline values."""

from lib.baselines import (
    SPEC_CPU_2017,
    GPU_SPECS,
    LLM_MODELS,
    flops_per_token,
    cost_per_call,
    get_baseline,
)


def test_spec_values_positive():
    """All SPEC CPU 2017 entries must have positive spec_rate and gflops."""
    for key, entry in SPEC_CPU_2017.items():
        assert entry["spec_rate"] > 0, f"SPEC rate for {key} must be > 0"
        assert entry["gflops"] > 0, f"GFLOPS for {key} must be > 0"
        assert entry["machine"] != ""


def test_gpu_specs_positive():
    """All GPU TFLOPS values must be positive."""
    for gpu, specs in GPU_SPECS.items():
        assert specs["fp16_tflops"] > 0, f"{gpu} fp16 must be > 0"
        assert specs["fp32_tflops"] > 0, f"{gpu} fp32 must be > 0"
        assert specs["name"] != ""


def test_llm_flops_monotonic():
    """Bigger models (by param count) should have more FLOP/token."""
    params_order = sorted(
        LLM_MODELS.keys(),
        key=lambda m: LLM_MODELS[m]["num_params"],
    )
    flops = []
    for model in params_order:
        fpt = flops_per_token(LLM_MODELS[model]["num_params"], 4096)
        flops.append(fpt)
    for i in range(1, len(flops)):
        assert flops[i] > flops[i - 1], (
            f"{params_order[i]} has more params but fewer FLOPs/token"
        )


def test_guessed_params_reasonable():
    """Estimated model parameter counts should be in a reasonable range."""
    for name, info in LLM_MODELS.items():
        params = info["num_params"]
        assert params > 0, f"{name} params must be > 0"
        assert params <= 1_000_000_000_000, (
            f"{name} params ({params}) should be <= 1T"
        )
        # Cost values must be non-negative
        assert info["cost_input_usd_per_m"] >= 0
        assert info["cost_output_usd_per_m"] >= 0


def test_flops_formula_consistency():
    """flops_per_token(2B, 1024) should equal flops_per_token(4B, 512)."""
    f1 = flops_per_token(2_000_000_000, 1024)
    f2 = flops_per_token(4_000_000_000, 512)
    assert abs(f1 - f2) < 1, (
        "Formula should be consistent: 6 * params * seq_length"
    )


def test_cost_per_call():
    """cost_per_call should return positive value for known models."""
    cost = cost_per_call("gemini-1.5-pro", 100, 50)
    assert cost > 0, "Cost should be positive"
    cost_zero = cost_per_call("gemini-1.5-pro", 0, 0)
    assert cost_zero == 0.0


def test_get_baseline():
    """get_baseline should return valid entries and raise on bad keys."""
    entry = get_baseline("int_base")
    assert entry["spec_rate"] > 0
    try:
        get_baseline("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass


def test_reference_session():
    """Reference 'one agent session' should produce positive values."""
    from lib.baselines import reference_agent_session
    result = reference_agent_session()
    assert result["llm_compute_flops"] > 0
    assert result["cpu_seconds"] > 0
    assert result["cpu_flops"] > 0
    assert result["total_llm_tokens"] > 0
    assert result["llm_cost_usd"] > 0
