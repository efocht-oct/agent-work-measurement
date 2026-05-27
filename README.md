Agent Work Measurement
======================

A measurement framework that estimates the breakdown of CPU work versus
AI/LLM work spent during an agentic coding session.  Track wall-clock
time, CPU time, memory, I/O for tool calls, and token usage, latency,
and compute estimates for LLM portions.  Three reference tasks exercise
multiple tool-call categories.

Quick start
-----------

::

  pip install psutil
  cd agent-work-measurement
  python -c "from lib.harness import MeasurementSession; print('harness OK')"
  python -c "from lib.baselines import SPEC_CPU_2017; print('baselines OK')"

Project layout
--------------

::

  agent-work-measurement/
  ├── lib/
  │   ├── harness.py       # MeasurementSession, cpu_call, llm_call
  │   ├── baselines.py     # SPEC CPU 2017, GPU specs, LLM metadata
  │   └── model.py         # decompose, standardize, ratio
  ├── tasks/               # 3 reference tasks (Python + C++)
  │   ├── 1-dijkstra/      # Shortest-path on a weighted graph
  │   ├── 2-toc-generator/ # Markdown TOC extractor
  │   └── 3-log-analyzer/  # JSON-lines log statistics
  ├── tests/               # 66 pytest tests
  └── docs/                # Methodology and analysis notes

Architecture
------------

Here is the high-level architecture of how raw tool-call traces are
converted into a CPU-vs-AI work breakdown:

  +-----------------------------------------------------------------+
  |  Agentic Coding Session                                         |
  |                                                                 |
  |  [Tool Call]  [LLM Call]  [Tool Call]  [LLM Call]  [IO Call]   |
  |     |             |             |             |            |    |
  +-----+-------------+-------------+-------------+------------+----+
                                |
                        +-------v--------+
                        | Measurement-   |
                        | Session (harn.)|  -- tracks per-node:
                        |                |      wall_clock, user_cpu,
                        |                |      system_cpu, rss, io,
                        |                |      tokens, latency, cost
                        +-------+--------+
                                |
                        +-------v--------+
                        | Trace tree     |  -- nested TraceNode graph
                        | (flattened)    |      with parent/child refs
                        +-------+--------+
                                |
                        +-------v--------+
                        |   decompose()  |  -- splits into:
                        |   (model.py)   |      cpu_nodes, llm_nodes,
                        |                |      wait_nodes
                        +-------+--------+
                                |
              +-----------------+-----------------+
              |                                   |
    +---------v----------+              +---------v----------+
    | standardize_to_    |              | ai_cpu_ratio()     |
    | cpu_equivalent()   |              |                    |
    |                    |              |  ai_heavy          |
    |  cpu_work =        |              |  ai_heavy          |
    |    total_cpu_secs  |              |  cpu_heavy         |
    |                    |              |  balanced          |
    |  ai_work =         |              +--------------------+
    |    FLOP_estimate / |
    |    baseline_flops  |
    +--------------------+

Data flow (single session):

  cpu_call("read_file")    --> 0.02s wall, 0.015s user_cpu, 512KB io
  llm_call("gpt-4o")       --> 2.5s wall, 0 tokens prompt, 500 tokens completion
  cpu_call("dijkstra")     --> 0.10s wall, 0.085s user_cpu, 0 io
  llm_call("claude-3.5")   --> 3.0s wall, 0 tokens prompt, 200 tokens completion
  io_call("write_json")    --> 0.005s wall, 0.001s user_cpu, 20KB io

  +----------------------------------------------------------------+
  |  analyse(session)                                              |
  |                                                                |
  |  cpu_work   = 0.015 + 0.085 + 0.001  = 0.101s                |
  |  ai_work    = 2.5*6*100*500/1e12 + 3.0*6*100*200/1e12        |
  |             = 0.75s + 0.36s = 1.11s  (standardized)           |
  |  wait_time  = total_wall - cpu_work - ai_work (residual)      |
  |  interpretation = "ai_heavy"  (ai_work > cpu_work)            |
  +----------------------------------------------------------------+

lib/harness.py
~~~~~~~~~~~~~~

A MeasurementSession provides a context manager API::

  from lib.harness import MeasurementSession

  with MeasurementSession(name="my-task") as session:
      with session.cpu_call("read_graph", category="file_read") as node:
          data = read_graph("input/graph.csv")
      with session.llm_call("gpt-4o", "explain",
                            prompt_tokens=100,
                            completion_tokens=50) as llm_node:
          response = call_llm(llm_node.prompt)
      summary = session.summary()

Metrics tracked per node:

  Metric            | Source                        | Unit
  ------------------|-------------------------------|-------------
  wall_clock        | time.perf_counter()           | seconds
  user_cpu          | resource module               | seconds
  system_cpu        | resource module               | seconds
  total_cpu         | user + system                 | seconds
  max_rss           | resource / /proc/self/status  | KiB
  io_read_bytes     | /proc/self/io                 | bytes
  io_write_bytes    | /proc/self/io                 | bytes
  node_type         | explicit                      | "llm" / "cpu" / "io"

LLM-specific: prompt_tokens, completion_tokens, latency, model, cost.

lib/baselines.py
~~~~~~~~~~~~~~~~

Hardware and LLM reference data:

  - SPEC CPU 2017 rates (int_base, float_base, int_peak, float_peak)
  - GPU specs (A100, H100, H200) with FP16/FP32/FP8 TFLOPS
  - LLM models (GPT-4o, GPT-4o-mini, Claude 3.5 Sonnet, Llama 3.1 70B)
  - Formula: flops_per_token(params, seq_length) = 6 * params * seq_length
  - Cost-per-call estimation

lib/model.py
~~~~~~~~~~~~

Composition model that decomposes total wall-clock time::

  total_wall_clock = cpu_work + ai_work + wait_time

Functions:

  - decompose(trace, rtt_estimate) -> dict
  - standardize_to_cpu_equivalent(decomp, baseline_key) -> dict
  - ai_cpu_ratio(decomp) -> dict with interpretation ("ai_heavy", "cpu_heavy", "balanced")
  - analyse(trace, rtt_estimate, baseline_key) -> full pipeline

Usage example
-------------

::

  from lib.harness import MeasurementSession
  from lib.model import analyse

  with MeasurementSession(name="dijkstra-task") as session:
      with session.cpu_call("dijkstra", category="algorithm") as node:
          import time; time.sleep(0.1)

      with session.llm_call("gpt-4o", "solve this",
                            prompt_tokens=500,
                            completion_tokens=200) as llm:
          llm.latency = 2.5  # simulate

  result = analyse(session, rtt_estimate=0.1)
  print(result["total_wall_clock"])   # e.g. 2.62
  print(result["cpu_work"])           # e.g. 0.10
  print(result["ai_work"])            # e.g. 2.47
  print(result["wait_time"])          # e.g. 0.05
  print(result["interpretation"])     # e.g. "ai_heavy"

Tests
-----

::

  pip install pytest
  python -m pytest tests/ -v

Run everything
--------------

::

  bash run_all.sh
