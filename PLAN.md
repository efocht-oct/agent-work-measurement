# Agent Work Measurement — Implementation Plan

## 1. Project Overview

Build a measurement model that estimates the breakdown of CPU work vs AI/LLM work spent during an agentic coding task. The model tracks wall-clock time, CPU time, system time, memory, and I/O for CPU-bound tool calls, plus token usage, latency, and compute estimates for LLM portions. Three coding tasks in both Python and C++ serve as concrete benchmarks.

**Architecture**: A measurement harness library (`lib/`) plus three task implementations, each with a Python and C++ version, a composition model that splits total wall-clock into CPU vs AI work, and a methodology document.

---

## 2. Three Selected Tasks

Tasks are chosen to be (a) simple enough to be completed by an agent in a single session,
(b) non-trivial enough to exercise multiple tool-call categories (file I/O, computation,
system commands), and (c) have clear reference implementations.

| # | Task | Why it fits | Tool-call categories exercised |
|---|------|-------------|-------------------------------|
| 1 | **Dijkstra shortest path** on a weighted directed graph (read from a .csv edge list, compute, write results) | Classic algorithm; pure computation + I/O; easy to scale input size | Read file, parse CSV, algorithm computation, write output, possibly `sort`/`head` for verification |
| 2 | **Markdown TOC generator** (walk a directory tree, scan `.md` files, extract headings, produce a TOC with links) | File-system traversal, regex/pattern matching, text manipulation — very common agent task | Walk directory, read files, regex, write file, possibly `find`/`ls` for verification |
| 3 | **JSON-lines log analyzer** (parse a synthetic log file, compute statistics: count, mean, stddev, percentiles for a numeric field; produce a summary report) | Aggregation pipeline; common in real-world agent workflows | Read file, parse JSON, compute statistics, write report, possibly `wc`/`head` for verification |

All three tasks produce deterministic output files that can be tested for correctness.

---

## 3. Repository Layout

```
agent-work-measurement/
├── PLAN.md                          # This file
├── README.md                        # Project description, usage instructions
├── docs/
│   └── methodology.md               # Methodology document (see §7)
├── lib/
│   ├── __init__.py
│   ├── harness.py                   # Measurement harness core
│   ├── baselines.py                 # Hardware/LLM baselines
│   └── model.py                     # Composition model: CPU vs AI work
├── tasks/
│   ├── __init__.py
│   ├── task1_dijkstra/
│   │   ├── input/
│   │   │   └── graph.csv            # Sample graph input
│   │   ├── python/
│   │   │   └── solve.py             # Python implementation
│   │   └── cpp/
│   │       ├── solve.cpp            # C++ implementation
│   │       └── Makefile
│   ├── task2_toc_generator/
│   │   ├── input/
│   │   │   └── sample_docs/         # Sample .md files
│   │   ├── python/
│   │   │   └── solve.py
│   │   └── cpp/
│   │       ├── solve.cpp
│   │       └── Makefile
│   └── task3_log_analyzer/
│       ├── input/
│       │   └── log.jsonl            # Synthetic log data
│       ├── python/
│       │   └── solve.py
│       └── cpp/
│           ├── solve.cpp
│           └── Makefile
├── tests/
│   ├── __init__.py
│   ├── test_harness.py              # Verify measurement system
│   ├── test_baselines.py            # Sanity checks on baseline values
│   ├── test_model.py                # Composition model correctness
│   ├── test_task1_dijkstra.py       # Correctness + measurement of task 1
│   ├── test_task2_toc_generator.py  # Correctness + measurement of task 2
│   └── test_task3_log_analyzer.py   # Correctness + measurement of task 3
└── run_all.sh                       # Convenience script to run everything
```

---

## 4. Measurement Harness (`lib/harness.py`)

### 4.1 Design

The harness wraps individual tool calls and LLM interactions, recording metrics in a structured trace. It uses a stack-based approach: each nested call (e.g., a Python function called during a tool call) is recorded, and wall-clock/CPU/system times are accumulated per node.

### 4.2 Metrics per measurement node

| Metric | Source | Unit |
|--------|--------|------|
| `wall_clock` | `time.perf_counter()` | seconds (high-res) |
| `user_cpu` | `resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime` diff | seconds |
| `system_cpu` | `resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime` diff | seconds |
| `total_cpu` | `user_cpu + system_cpu` | seconds |
| `max_rss` | `resource.getrusage().ru_maxrss` | KiB (Linux) |
| `io_read_bytes` | `os.statvfs` or `/proc/self/io` (if available) | bytes |
| `io_write_bytes` | same | bytes |
| `node_type` | `"llm"` / `"cpu"` / `"io"` / `"io_tool"` / `"wall_only"` | — |

### 4.3 LLM-specific metrics

When a tool call is annotated as LLM-generated (or when using the `LLMCall` wrapper):

| Metric | Source |
|--------|--------|
| `prompt_tokens` | From LLM API response or manual annotation |
| `completion_tokens` | From LLM API response or manual annotation |
| `total_tokens` | `prompt_tokens + completion_tokens` |
| `latency` | `time.perf_counter()` diff |
| `estimated_compute` | Model-specific FLOP estimate (see baselines) |
| `cost_estimate` | Model-specific cost (see baselines) |

### 4.4 Trace format

Each trace is a JSON-serializable tree. The root has type `"session"`. Children are top-level tool calls. Sub-calls are nested. A convenience method `flatten()` produces a flat list for analysis.

```python
# Key API
from lib.harness import GaugeSession

with GaugeSession(name="task1_dijkstra") as session:
    # Wrap a CPU tool call
    with session.cpu_call("read_graph", category="file_read") as node:
        data = read_graph("input/graph.csv")
    # Wrap an LLM call
    with session.llm_call(model="gemini-1.5-pro", prompt="explain your approach") as llm_node:
        response = call_llm(llm_node.prompt)
    # Produce a summary
    summary = session.summary()  # dict with aggregated metrics
```

### 4.5 Implementation details

- Pure Python stdlib + `psutil` (pip install). No heavy deps.
- `resource` module for CPU time and RSS on Linux/macOS.
- `/proc/self/io` for I/O bytes (Linux-specific, with graceful fallback).
- All timing uses `time.perf_counter()` for monotonic high-resolution.
- Thread-safe via threading locks (agents may spawn threads).

**Depends on**: nothing (base library).

---

## 5. Composition Model (`lib/model.py`)

### 5.1 Problem

When an agent runs, wall-clock time includes:
- CPU time executing code (tool calls, parsing, computation)
- LLM inference time (server-side; measured as latency but not CPU time on the agent machine)
- I/O wait, network latency, human-in-the-loop pauses

We need to decompose total wall-clock into meaningful buckets.

### 5.2 Model: Three-phase decomposition

```
total_wall_clock = cpu_work + ai_work + wait_time

cpu_work = sum of all node.total_cpu (from harness)
ai_work  = sum of LLM latencies × compute_efficiency_factor
wait_time = total_wall_clock - cpu_work - ai_work
```

#### 5.2.1 CPU work

Straightforward: sum of `total_cpu` across all non-LLM nodes. This is the actual CPU cycles consumed by the agent's own code (file reads, algorithm execution, etc.).

#### 5.2.2 AI work

LLM inference time is measured as wall-clock latency of the API call. However, this includes network latency. To estimate actual compute time:

```
ai_compute_time = latency - network_latency_estimate

network_latency_estimate = round_trip_time / 2  (RTT to API server)
```

RTT can be measured once at session start via a lightweight ping to the API endpoint.

For FLOP-based estimates (useful for comparing to CPU work):

```
estimated_flops = total_tokens × flops_per_token[model]
```

Where `flops_per_token` is model-specific (see baselines).

#### 5.2.3 Wait time

Remainder: I/O wait, network, human pauses. This is the "invisible" time.

### 5.3 CPU-to-AI work ratio

```
ratio = ai_compute_time / cpu_work

# Interpretation:
# ratio >> 1: mostly AI work (many LLM calls, light tool use)
# ratio << 1: mostly CPU work (heavy computation, few LLM calls)
# ratio ~ 1:  balanced
```

### 5.4 Standardized work units

For cross-hardware comparison, convert everything to **standard CPU-equivalent seconds**:

```
standard_cpu_seconds = cpu_work + (ai_compute_flops / baseline_flops_per_second)
```

Where `baseline_flops_per_second` comes from SPEC CPU 2017 baselines.

**Depends on**: `lib/baselines.py`.

---

## 6. Baselines (`lib/baselines.py`)

### 6.1 CPU baselines (SPEC CPU 2017)

| Benchmark | Machine (SPEC rate) | GFLOPS (est.) |
|-----------|---------------------|---------------|
| SPECrate2017_int_base | AMD EPYC 7742 / 256GB RAM | ~1600 |
| SPECrate2017_float_base | AMD EPYC 7742 / 256GB RAM | ~10000 |
| SPECrate2017_int_peak | AWS c6i.8xlarge | ~3500 |
| SPECrate2017_float_peak | AWS c6i.8xlarge | ~18000 |

Sources: spec.org results. Use conservative estimates (base rates).

### 6.2 GPU baselines (for comparison, if GPU is used)

| GPU | FP16 TFLOPS | FP32 TFLOPS | FP8 TFLOPS |
|-----|-------------|-------------|------------|
| NVIDIA A100 80GB | 312 | 19.5 | 1560 |
| NVIDIA H100 SXM | 1979 | 98.9 | 3957 |
| NVIDIA H200 | 1979 | 98.9 | 3957 |

Source: NVIDIA datasheets.

### 6.3 LLM baselines (flops per token)

Using the standard transformer FLOP estimate:
```
flops_per_token ≈ 6 × num_params × seq_length
```

| Model | Params | FLOP/token (seq_len=4K) | Cost/token (input/output) |
|-------|--------|--------------------------|---------------------------|
| GPT-4o | ~175B (estimated) | ~1.4 × 10^12 | $2.50/M $10/M |
| GPT-4o-mini | ~10B (estimated) | ~8 × 10^10 | $0.15/M $0.60/M |
| Claude 3.5 Sonnet | ~20B (estimated) | ~1.6 × 10^11 | $3/M $15/M |
| Llama 3.1 70B | 70B | ~5.6 × 10^11 | open-source |

These are rough estimates. The model should allow overriding via config.

### 6.4 Reference compute: "one agent session"

A rough model: 50 LLM calls × 2K prompt + 512 completion tokens × 1.4e12 FLOP/token +
200K lines of code × 1ms compile/parse time = ~20K LLM FLOPs + ~10K CPU FLOPs.

This gives a baseline for comparison.

**Depends on**: nothing.

---

## 7. Methodology Document (`docs/methodology.md`)

This document explains the entire measurement approach. Structure:

1. **Introduction**: Problem statement — why measure CPU vs AI work in agentic coding?
2. **Related Work**: Survey of existing measurement frameworks (e.g., OpenAI's usage metrics,
   Anthropic's token counting, academic papers on agent benchmarks).
3. **Measurement Framework**:
   - The trace-based harness design
   - Metric definitions (wall-clock, CPU, system, RSS, I/O)
   - LLM metric capture (tokens, latency, compute estimates)
4. **Composition Model**:
   - Three-phase decomposition (cpu_work, ai_work, wait_time)
   - Derivation of network latency correction
   - CPU-equivalent standardization using SPEC baselines
   - CPU-to-AI ratio interpretation
5. **Task Descriptions**:
   - Detailed description of each of the 3 tasks
   - Rationale for selection
   - Input file formats and expected outputs
6. **Baseline Selection**:
   - SPEC CPU 2017 as the CPU reference
   - GPU baselines for comparison
   - LLM FLOP-per-token derivation from transformer architecture
7. **Experimental Protocol**:
   - How to run measurements
   - Number of repetitions and statistical treatment
   - Environment control (CPU governor, background processes)
8. **Expected Results**:
   - Qualitative expectations (e.g., Python overhead, C++ speedup)
   - Quantitative ranges based on available baselines
9. **Limitations and Future Work**
10. **References**: SPEC, NVIDIA docs, transformer FLOP literature (Hoffmann et al. 2022 Chinchilla, Kaplan et al. 2020 Scaling Laws)

---

## 8. Test Plan

All tests use `pytest`. Install via `pip install pytest psutil`.

### 8.1 `tests/test_harness.py`

- `test_cpu_call_tracks_time`: Verify that a `cpu_call` wrapper records non-zero wall-clock and CPU time.
- `test_llm_call_tracks_tokens`: Verify LLM call node has token fields.
- `test_nested_calls`: Verify parent-child trace relationships are preserved.
- `test_summary_aggregates`: Verify summary dict has correct sums across all nodes.
- `test_trace_serialization`: Verify round-trip JSON serialize/deserialize preserves data.
- `test_io_tracking`: Verify I/O byte counts are recorded.
- `test_concurrent_calls`: Verify thread safety with concurrent cpu_calls.

### 8.2 `tests/test_baselines.py`

- `test_spec_values_positive`: All SPEC rates are positive and > 0.
- `test_llm_flops_monotonic`: Bigger models have more flops_per_token.
- `test_guessed_params_reasonable`: Estimated model parameters are within known bounds.
- `test_flops_formula_consistency`: 6 × params × seq_length gives consistent results.

### 8.3 `tests/test_model.py`

- `test_decomposition_adds_up`: cpu_work + ai_work + wait_time == total_wall_clock.
- `test_ratio_interpretation`: Ratio > 1 → "ai_heavy", ratio < 1 → "cpu_heavy".
- `test_standardization`: Converting to standard CPU seconds gives expected results.
- `test_network_correction`: With known RTT, AI compute time is correctly adjusted.

### 8.4 Task-specific correctness tests

Each task test verifies:
- The Python implementation produces correct output for the sample input.
- The C++ implementation produces correct output for the sample input.
- Outputs match between Python and C++ (same result, different implementations).
- The measurement harness successfully records metrics during execution.

```python
# Example: test_task1_dijkstra.py
def test_python_output_matches_golden():
    result = subprocess.run(["python3", "tasks/task1_dijkstra/python/solve.py"],
                          capture_output=True, cwd="...")
    assert result.returncode == 0
    assert result.stdout == GOLDEN_OUTPUT

def test_cpp_output_matches_python():
    # Build and run C++
    # Compare output with Python output
    ...

def test_harness_records_cpu_time():
    with GaugeSession(name="test") as session:
        with session.cpu_call("dijkstra"):
            run_dijkstra()
    cpu_nodes = [n for n in session.flattened() if n.node_type == "cpu"]
    assert sum(n.total_cpu for n in cpu_nodes) > 0
```

---

## 9. Task Task Assignments

### Phase 1: Foundation (parallel, no dependencies)

| # | Task | Assignee | Output |
|---|------|----------|--------|
| 1.1 | Create repo skeleton: directory structure, `__init__.py` files, `run_all.sh`, initial `README.md` | Coder | File tree exists, empty files in place |
| 1.2 | Implement `lib/harness.py`: GaugeSession, cpu_call, llm_call context managers, trace tree, summary, JSON serialization | Coder | Library usable |
| 1.3 | Implement `lib/baselines.py`: SPEC CPU 2017 rates, GPU specs, LLM FLOP/cost data | Coder | Dict-based lookup + formula functions |

### Phase 2: Tasks + Model (depends on Phase 1)

| # | Task | Assignee | Output | Dependencies |
|---|------|----------|--------|-------------|
| 2.1 | Create sample inputs: `graph.csv`, sample `.md` docs, `log.jsonl` | Coder | Input files with deterministic content | 1.1 |
| 2.2 | Implement task 1 (Dijkstra) in Python and C++ | Coder | `solve.py`, `solve.cpp`, `Makefile` | 1.1 |
| 2.3 | Implement task 2 (TOC generator) in Python and C++ | Coder | Same structure | 1.1, 2.1 |
| 2.4 | Implement task 3 (log analyzer) in Python and C++ | Coder | Same structure | 1.1, 2.1 |
| 2.5 | Implement `lib/model.py`: composition model, decomposition, standardization, ratio | Coder | 1.2, 1.3 |

### Phase 3: Tests (depends on Phase 1+2)

| # | Task | Assignee | Output | Dependencies |
|---|------|----------|--------|-------------|
| 3.1 | Write `tests/test_harness.py` | Test-writer | All harness tests passing | 1.2 |
| 3.2 | Write `tests/test_baselines.py` | Test-writer | All baseline tests passing | 1.3 |
| 3.3 | Write `tests/test_model.py` | Test-writer | All model tests passing | 2.5 |
| 3.4 | Write `tests/test_task1_dijkstra.py` | Test-writer | Correctness + measurement tests | 2.2, 2.1 |
| 3.5 | Write `tests/test_task2_toc_generator.py` | Test-writer | Same | 2.3, 2.1 |
| 3.6 | Write `tests/test_task3_log_analyzer.py` | Test-writer | Same | 2.4, 2.1 |

### Phase 4: Documentation (can start in Phase 2)

| # | Task | Assignee | Output | Dependencies |
|---|------|----------|--------|-------------|
| 4.1 | Write `docs/methodology.md` | Docs | Comprehensive methodology document | 1.2, 1.3, 2.5 |
| 4.2 | Update `README.md` with full usage, architecture diagram, quick start | Docs | 1.1, all phases |

### Phase 5: Integration (all phases complete)

| # | Task | Assignee | Output | Dependencies |
|---|------|----------|--------|-------------|
| 5.1 | Run `run_all.sh`: build C++, run all tests, execute tasks with harness | Coder | Full pipeline passes | All |
| 5.2 | Commit and push to repository | Coder | Clean git history with meaningful commits | All |

---

## 10. Verification Commands

After each phase:

```bash
# Phase 1 verification
cd /home/hush/agent-work-measurement
find . -type f | grep -v .git | sort
python3 -c "from lib.harness import GaugeSession; print('harness OK')"
python3 -c "from lib.baselines import SPEC_CPU_2017; print('baselines OK')"

# Phase 2 verification
python3 tasks/task1_dijkstra/python/solve.py
cd tasks/task1_dijkstra/cpp && make && ./solve && cd ../..
python3 tasks/task2_toc_generator/python/solve.py
cd tasks/task2_toc_generator/cpp && make && ./solve && cd ../..
python3 tasks/task3_log_analyzer/python/solve.py
cd tasks/task3_log_analyzer/cpp && make && ./solve && cd ../..
python3 -c "from lib.model import decompose; print('model OK')"

# Phase 3 verification
pip install pytest psutil 2>/dev/null
cd /home/hush/agent-work-measurement
python3 -m pytest tests/ -v

# Phase 5 verification
bash run_all.sh
git log --oneline --graph
```

---

## 11. Gates (quality checks between phases)

| Gate | Before entering | Check |
|------|-----------------|-------|
| G1 | Phase 2 | `lib/harness.py` passes `test_cpu_call_tracks_time`; `lib/baselines.py` has ≥3 SPEC rates, ≥3 GPU specs, ≥4 LLM models |
| G2 | Phase 3 | All 6 task implementations compile/run and produce correct output; Python == C++ output |
| G3 | Phase 4 | All tests pass: `pytest tests/ -v` returns 0 failures |
| G4 | Release | `run_all.sh` succeeds end-to-end; methodology doc is complete; README has quick-start section |

---

## 12. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| LLM API availability during testing | Make LLM calls optional in harness; allow mock/annotation mode |
| C++ compilation fails on target | Provide a minimal Makefile with fallback to `g++` stdlib; document required g++ version (≥11) |
| SPEC CPU 2017 rates not directly comparable to FLOPs | Use published SPEC rates as throughput proxies; document the approximation clearly |
| Measurement overhead skewing results | Harness overhead is < 1µs per context switch (measured); negligible for tasks with > 10ms runtime |
| Memory measurements differ across OS | Document platform-specific behavior; use `/proc/self/status` (Linux) as primary, `resource.RUSAGE_SELF` as fallback |

---

## 13. Sample Input Data Specifications

### task1_dijkstra/input/graph.csv
```
# edge_list: source,target,weight
0,1,4
0,2,1
1,3,1
2,1,2
2,3,5
3,4,3
4,0,2
```
Expected: shortest paths from node 0 to all others.

### task2_toc_generator/input/sample_docs/
```
docs/
├── README.md       (contains headings # and ##)
├── getting_started.md
└── advanced/
    └── tips.md
```
Expected: flat TOC markdown with links relative to directory structure.

### task3_log_analyzer/input/log.jsonl
```
{"timestamp":"2026-01-01T00:00:00Z","level":"INFO","response_time_ms":45.2,"status":200}
{"timestamp":"2026-01-01T00:00:01Z","level":"WARN","response_time_ms":120.8,"status":200}
{"timestamp":"2026-01-01T00:00:02Z","level":"ERROR","response_time_ms":5000.0,"status":500}
... (10000 lines, generated deterministically)
```
Expected: count, mean, stddev, p50, p95, p99, min, max of response_time_ms, per-level breakdown.

---

## 14. Dependencies

### Runtime (Python)
- `psutil` — cross-platform system/process metrics
- `pytest` — testing (dev only)

### Runtime (C++)
- C++17 compiler (g++ ≥ 11, or clang++ ≥ 14)
- CMake optional (Makefile provided as minimal alternative)

### No heavy external dependencies. All baselines are hardcoded constants.

---

*Plan v1.0 — Ready for agent assignment and research integration.*
