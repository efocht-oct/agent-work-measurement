# Critic / Reviewer Report: AgentGauge Usability & Documentation

## Verdict
REQUEST_CHANGES

## Scope Reviewed
- Spec/plan: User request to evaluate usefulness for gauging arbitrary agentic tasks and executing the 3 sample tasks.
- Files/artifacts: `README.md`, `tasks/`, repository structure.

## Blocking Issues
| ID | Issue | Evidence | Required Fix |
|----|-------|----------|--------------|
| B1 | Missing execution entrypoint | `README.md` references `bash run_all.sh` which does not exist in the repository. | Create a runnable example script (e.g., `run_all_samples.py`) that actually demonstrates the harness in action, and update the docs to reference it. |
| B2 | Ambiguous instrumentation instructions | The README provides a small snippet but does not explain the practical steps to instrument an *arbitrary* agent (e.g., wrapping the main agent loop, wrapping local subprocess/tool executions, and wrapping network requests). | Add a dedicated "How to Instrument Your Own Agent" section in the README with clear, step-by-step instructions. |
| B3 | Sample tasks are uninstrumented | The scripts in `tasks/` (e.g., `tasks/task1_dijkstra/python/solve.py`) are raw algorithm implementations. They do not import `GaugeSession`. There is no explanation of how a user is supposed to "evaluate" them. | Create a wrapper script that simulates an agent session by calling these task scripts using `subprocess` inside a `session.cpu_call()`, paired with simulated `session.llm_call()`s, to generate a realistic `analyse()` report. |

## Important Issues
| ID | Issue | Evidence | Suggested Fix |
|----|-------|----------|---------------|
| I1 | GPU tracking caveat | The framework uses theoretical FLOP projections (`6 * params * tokens`) rather than direct GPU profiling (e.g., NVML). | Explicitly document this in the "How to Instrument" section, clarifying that this is a hardware-agnostic theoretical projection rather than raw GPU utilization. |

## Final Recommendation
The framework is conceptually sound for standardizing local vs. remote compute, but currently lacks a cohesive "glue" example. A developer downloading this repository has no clear way to run a demonstration. 

**Next Steps**: 
1. Implement a `run_all_samples.py` script that uses the harness to measure the three existing tasks.
2. Update the `README.md` to explain how to run this script and how to instrument a real-world agent loop.
