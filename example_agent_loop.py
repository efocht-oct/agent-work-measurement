#!/usr/bin/env python3
"""
Agentic Developer Loop Simulation.

This script demonstrates how to use AgentGauge to measure the *process* of
an agent writing code, rather than just measuring the final code itself.

It simulates a standard agentic workflow:
1. Plan & Code (LLM)
2. Run Tests - Fail (CPU)
3. Debug & Fix (LLM)
4. Run Tests - Pass (CPU)
"""

import time
import pprint
import sys
import os

from lib.harness import GaugeSession
from lib.model import analyse


def main():
    print("==================================================")
    print("Starting simulated agentic developer loop...")
    print("==================================================")
    
    with GaugeSession(name="agent-dev-loop") as session:
        
        # 1. Agent reads the prompt, plans, and writes initial code
        print("  [Agent] Reading issue, thinking, and writing initial code...")
        with session.llm_call("gemini-1.5-pro", prompt_tokens=2500, completion_tokens=800):
            time.sleep(3.5)  # Simulate network/inference latency
            
        # 2. Agent executes the test suite in the local environment
        print("  [Environment] Running test suite (simulating failure)...")
        with session.cpu_call("run_pytest_attempt_1", category="testing"):
            # Simulate CPU work for running tests (e.g., parsing AST, running asserts)
            # We use a simple sleep here, but in reality this would be subprocess.run(["pytest"])
            for _ in range(5000000): pass 
            
        # 3. Agent reads the traceback and generates a fix
        print("  [Agent] Analyzing traceback and generating fix...")
        with session.llm_call("gemini-1.5-pro", prompt_tokens=3500, completion_tokens=200):
            time.sleep(2.0)
            
        # 4. Agent runs tests again
        print("  [Environment] Running test suite (simulating success)...")
        with session.cpu_call("run_pytest_attempt_2", category="testing"):
            for _ in range(5000000): pass

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
