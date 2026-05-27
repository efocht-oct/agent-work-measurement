# Example: Complex Agent Loop Simulation

The `example_agent_loop.py` script demonstrates how AgentGauge profiles a reasonably complex, multi-turn autonomous agent workflow. 

Unlike measuring the final generated code (the artifact), this script measures the **process**—the computational labor the agent and the local environment expend to arrive at a solution.

## The 6-Phase Agentic Workflow

The simulation runs through a heavy, realistic developer loop:

1. **Planning**: The agent ingests a large issue description and codebase context (simulating 15,000 prompt tokens).
2. **Research**: The local environment simulates a Vector DB search (`cpu_call`), and the agent synthesizes the retrieved snippets (simulating 25,000 prompt tokens).
3. **Implementation**: The agent writes the initial code. The local environment runs a linter and the test suite, which simulates an initial test failure.
4. **Debugging (Turn 1)**: The agent analyzes a 500-line stack trace (simulating 12,000 prompt tokens). The environment runs tests again, hitting an edge-case failure.
5. **Deep Debugging (Turn 2)**: The environment executes a local debugger script to inspect memory. The agent finds the root cause, applies a final patch, and tests finally pass.
6. **Documentation**: The agent updates the README and writes docstrings (simulating 20,000 prompt tokens), and the environment runs an `mkdocs` build.

## Execution Results

When executed, AgentGauge captures and normalizes the metrics across both the local CPU and the remote LLM calls (using the `gemini-1.5-pro` hardware baseline).

**Summary Output:**
```text
Total Wall-Clock Time:  34.27s
Total Local CPU Work:   0.4680s
Equivalent AI Work:     32.50s (Normalized to float_base)
Interpretation:         ai_heavy
AI/CPU Ratio:           69.45
```

## What This Tells Us

This simulation perfectly illustrates the core problem AgentGauge solves: 

If you only used standard local profiling tools (`cProfile` or `time`), this 34-second session would look like it only took **0.47 seconds** of actual work, with over 33 seconds of "idle network waiting." 

By tracking token counts and projecting them into standardized FLOP baselines, AgentGauge reveals the true computational footprint: the vast majority of the labor (**32.50 equivalent CPU-seconds**) was performed by the remote AI model, meaning the AI did ~69 times more computational work than the local machine.
