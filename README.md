# AgentGauge: Agentic Work Profiler & FLOP Estimator

An analytical framework and instrumented harness for estimating, measuring, and standardizing the breakdown of **non-AI CPU work** versus **AI/LLM work** spent during an agentic coding session.

---

## 💡 Simulation vs. Actual Agentic Work: What is AgentGauge?

**AgentGauge is not an autonomous coding agent.** It does not autonomously write code, search the web, or run git commands. 

Instead, AgentGauge is a **measurement harness and analytical model** designed to be integrated into agentic systems (such as Claude Code, Codex, or custom developer loops). It profiles, logs, and analyzes where resources are actually spent during an agent session. 

### Why do we need this?
When an agent performs a software engineering task, it spends:
1. **Local/Non-AI CPU work**: Compiling code, running tests, traversing file systems, executing graph algorithms, parsing logs, and parsing syntax.
2. **Remote AI/LLM work**: Forward passes through massive transformer architectures hosted on high-performance GPU/TPU clusters.

Standard profiling tools (like `cProfile` or `time`) only see the local Python/C++ process. They treat remote LLM calls as idle network waiting time. 

**AgentGauge solves this** by capturing both local hardware-level metrics (via `psutil` and CPU/system clocks) and LLM call parameters (token counts, latency, and models). It then uses **algebraic hardware baselines** (SPEC CPU 2017 rates and GPU TFLOPS ratings) to project remote LLM work into a standardized, equivalent "CPU-seconds" metric. This allows a true, apples-to-apples comparison of the computational work done by the AI versus the local system.

---

## 🚀 Quick Start

Ensure you have the required system dependencies:

```bash
pip install psutil
```

You can see a full evaluation pipeline in action by running the included `run_all_samples.py` script. This script executes a simulated agentic session across multiple reference tasks and prints a comprehensive work measurement breakdown:

```bash
python run_all_samples.py
```

## 🔌 Instrumenting Your Own Agent

To integrate AgentGauge into your own custom agent developer loop, wrap your agent's execution using `GaugeSession` context managers. Use `session.cpu_call()` around local tool executions (like running a compiler or `git` command) and `session.llm_call()` around API requests to your model providers.

**Caveat on Hardware-Agnostic Profiling:** The framework currently uses theoretical FLOP projections (specifically, `6 * params * tokens`) to estimate the computational work of remote LLM calls. It does not use direct GPU hardware profiling (e.g., via NVML). This approach ensures that the measurement remains hardware-agnostic, though the metric is theoretical rather than empirical.

---

## 📂 Project Layout

```text
agent-gauge/
├── lib/
│   ├── harness.py       # GaugeSession: instrumented context managers
│   ├── baselines.py     # Hardware baselines (SPEC CPU 2017, GPU FLOPS, LLM parameters)
│   └── model.py         # Analytical model (decompose, standardize_to_cpu_equivalent)
├── tasks/               # Three non-trivial reference tasks implemented in Python & C++
│   ├── task1_dijkstra/  # Graph shortest-path with string nodes & undirected flags
│   ├── task2_toc_generator/ # Markdown heading extractor & nested list builder
│   └── task3_log_analyzer/  # High-throughput JSONL log analytics & statistics
├── tests/               # 66-test verification suite (pytest)
└── docs/                # Methodology, math, and baseline derivation details
```

The core libraries are written in pure Python 3 with minimal external dependencies (`psutil`). The reference tasks contain both Python and optimized, native C++17 implementations to demonstrate the profiling of different local performance tiers.

---

## 🏗️ Architecture

AgentGauge operates by capturing an execution trace tree during an agent session and decomposing it into distinct resource domains.

```text
       +-----------------------------------------------------------+
       |                  Agentic Coding Session                   |
       |                                                           |
       |  [Tool Call]    [LLM Call]    [Tool Call]    [LLM Call]   |
       +-------+-------------+-------------+-------------+---------+
               |             |             |             |
               v             v             v             v
       +-----------------------------------------------------------+
       |               GaugeSession (harness.py)             |
       |                                                           |
       | Tracks: wall_clock, user_cpu, system_cpu, RSS, I/O,       |
       |         latency, model, prompt_tokens, completion_tokens  |
       +-----------------------------+-----------------------------+
                                     |
                                     v
                       +---------------------------+
                       |    Nested Trace Tree      |
                       +-------------+-------------+
                                     |
                                     v
                       +---------------------------+
                       |   decompose() (model.py)  |
                       +------+-------------+------+
                              |             |
           +------------------+             +------------------+
           |                                                   |
           v                                                   v
+------------------------+                          +------------------------+
|   Local CPU Work       |                          |   Remote AI/LLM Work   |
|                        |                          |                        |
| Sum of user + system   |                          | Estimated total FLOPs  |
| execution times across |                          | (forward pass:         |
| all local tool calls.  |                          |  6 * params * tokens)  |
+----------+-------------+                          +----------+-------------+
           |                                                   |
           |                                                   | [Projected using]
           |                                                   | [SPEC CPU 2017 vs]
           |                                                   | [GPU FLOPS ratio]
           |                                                   v
           |                                        +------------------------+
           |                                        |  CPU-Equivalent Work   |
           |                                        |                        |
           |                                        | Standardized AI time   |
           |                                        | projected onto target  |
           |                                        | baseline CPU.          |
           +------------------+             +------------------+
                              |             |
                              v             v
                       +---------------------------+
                       |      analyse() Summary     |
                       |                           |
                       |  - CPU Work (Seconds)     |
                       |  - AI Work (Equivalent S) |
                       |  - AI-to-CPU Ratio        |
                       |  - Residual Wait Time     |
                       +---------------------------+
```

---

## 📊 Data Flow & Analysis Example

The following example demonstrates how a mixed local/remote agent trace is decomposed and normalized:

```python
from lib.harness import GaugeSession
from lib.model import analyse

# 1. Profile the session
with GaugeSession(name="agent-session") as session:
    # Measure a local CPU tool call
    with session.cpu_call("compile_and_test", category="subprocess"):
        # Local system compilations, tests, and file system tasks are tracked here
        import time
        time.sleep(0.1) # Simulate tool execution

    # Measure a remote LLM generative task
    with session.llm_call(model="gemini-1.5-pro", prompt_tokens=500, completion_tokens=200) as node:
        # Remote LLM parameters, network latency, and tokens are tracked here
        node.latency = 2.5 # Simulate remote API round-trip

# 2. Analyze using an Intel Xeon Gold 6430 baseline ("int_base")
result = analyse(session, rtt_estimate=0.08, baseline_key="int_base")

print(f"Total Wall-Clock: {result['total_wall_clock']:.2f}s")
print(f"Local CPU Work:   {result['cpu_work']:.4f}s")
print(f"Projected AI Work: {result['ai_work']:.2f} equivalent-seconds")
print(f"Residual Wait:    {result['wait_time']:.2f}s (network RTT/idle)")
print(f"Interpretation:   {result['interpretation']}") # e.g., "ai_heavy"
```

---

## 🛠️ Measurement Specifications

| Metric | Captured Via | Scope | Target Dimension |
| :--- | :--- | :--- | :--- |
| **Wall Clock** | `time.perf_counter()` | Per-Node / Total | Elapsed temporal latency (seconds) |
| **CPU User/Sys** | `resource.getrusage()` / `psutil` | Local Tool Nodes | Operating System process scheduling (seconds) |
| **Memory RSS** | `/proc/self/status` / `psutil` | Local Tool Nodes | Peak physical memory utilization (KiB) |
| **I/O Read/Write**| `/proc/self/io` | Local Tool Nodes | Disk block and file system throughput (bytes) |
| **Token Counts** | Explicit Input / API response | LLM Call Nodes | LLM throughput (integer tokens) |
| **FLOP Projection**| 6 $\times$ parameters $\times$ tokens | LLM Call Nodes | Quantitative remote mathematical operations |

---

## 🧪 Testing

The library includes a thorough, high-coverage testing suite verifying harness safety, concurrent execution, trace serialization, model calculations, and the reference tasks.

To run the full test suite:

```bash
python -m pytest tests/ -v
```
