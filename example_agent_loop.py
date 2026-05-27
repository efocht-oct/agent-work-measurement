#!/usr/bin/env python3
"""
Complex Agentic Developer Loop Simulation.

This script demonstrates AgentGauge profiling a reasonably complex, 
multi-turn agent workload. It simulates a heavy autonomous workflow:
1. Planning (Context ingestion)
2. Research (Vector DB search + synthesis)
3. Implementation (Initial code generation)
4. Debugging - Turn 1 (Stack trace analysis)
5. Deep Debugging - Turn 2 (Tool execution & root cause analysis)
6. Documentation (Writing READMEs & docstrings)

This highlights how the harness tracks both massive AI compute (LLM calls)
and local CPU constraints (linters, test suites, documentation builds).
"""

import time
import pprint
import sys
import os

from lib.harness import GaugeSession
from lib.model import analyse


def main():
    print("==================================================")
    print("Starting COMPLEX simulated agentic workflow...")
    print("==================================================")
    
    with GaugeSession(name="complex-agent-workflow") as session:
        
        # --- Phase 1: Planning ---
        print("\n[Phase 1: Planning]")
        print("  [Agent] Ingesting large issue description and codebase context...")
        with session.llm_call("gemini-1.5-pro", prompt_tokens=15000, completion_tokens=800):
            time.sleep(4.5)  # Simulate network/inference latency
            
        # --- Phase 2: Research ---
        print("\n[Phase 2: Research]")
        print("  [Environment] Searching vector DB and retrieving file snippets...")
        with session.cpu_call("vector_search", category="io"):
            time.sleep(0.5) # Simulate IO/CPU time for search
        
        print("  [Agent] Synthesizing research and establishing architecture...")
        with session.llm_call("gemini-1.5-pro", prompt_tokens=25000, completion_tokens=1200):
            time.sleep(6.0)

        # --- Phase 3: Implementation ---
        print("\n[Phase 3: Implementation]")
        print("  [Agent] Writing initial implementation...")
        with session.llm_call("gemini-1.5-pro", prompt_tokens=8000, completion_tokens=2500):
            time.sleep(8.0)
        
        print("  [Environment] Running compiler/linter...")
        with session.cpu_call("run_linter", category="compute"):
            for _ in range(2000000): pass # Simulated CPU burn
            
        print("  [Environment] Running test suite (simulating failure)...")
        with session.cpu_call("run_pytest_1", category="compute"):
            for _ in range(8000000): pass 

        # --- Phase 4: Debugging (Iteration 1) ---
        print("\n[Phase 4: Debugging - Turn 1]")
        print("  [Agent] Analyzing 500-line stack trace...")
        with session.llm_call("gemini-1.5-pro", prompt_tokens=12000, completion_tokens=600):
            time.sleep(3.5)
            
        print("  [Environment] Running test suite (simulating edge-case failure)...")
        with session.cpu_call("run_pytest_2", category="compute"):
            for _ in range(8000000): pass

        # --- Phase 5: Deep Debugging (Iteration 2) ---
        print("\n[Phase 5: Deep Debugging - Turn 2]")
        print("  [Environment] Agent executes a debugger script to inspect memory...")
        with session.cpu_call("run_debugger", category="compute"):
            for _ in range(3000000): pass
        
        print("  [Agent] Finding root cause and applying final patch...")
        with session.llm_call("gemini-1.5-pro", prompt_tokens=15000, completion_tokens=400):
            time.sleep(3.0)

        print("  [Environment] Running test suite (simulating success)...")
        with session.cpu_call("run_pytest_3", category="compute"):
            for _ in range(8000000): pass

        # --- Phase 6: Documentation Writing ---
        print("\n[Phase 6: Documentation]")
        print("  [Agent] Updating README and writing docstrings...")
        with session.llm_call("gemini-1.5-pro", prompt_tokens=20000, completion_tokens=1500):
            time.sleep(7.5)
            
        print("  [Environment] Running mkdocs build to verify docs...")
        with session.cpu_call("build_docs", category="compute"):
            time.sleep(0.8)

    print("\n==================================================")
    print("Agentic Loop Measurement Results:")
    print("==================================================")
    results = analyse(session)
    
    print(f"Total Wall-Clock Time:  {results['total_wall_clock']:.2f}s")
    print(f"Total Local CPU Work:   {results['cpu_work']:.4f}s")
    print(f"Equivalent AI Work:     {results['ai_work']:.2f}s (Normalized to {results['baseline_key']})")
    print(f"Interpretation:         {results['interpretation']}")
    print("\nFull Analysis Dict:")
    pprint.pprint(results)

if __name__ == "__main__":
    main()
