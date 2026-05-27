import subprocess
import time
import pprint
from lib.harness import GaugeSession
from lib.model import analyse

def run_task(session, task_name, command_args):
    """Executes a given python command as a subprocess and wraps it in a cpu_call."""
    with session.cpu_call(task_name) as node:
        subprocess.run(command_args, check=True)

def main():
    print("Starting agentic workflow measurement...")
    with GaugeSession(name="run_all_samples") as session:
        # Task 1
        print("Running task 1...")
        run_task(session, "task1_dijkstra", ["python3", "tasks/task1_dijkstra/python/solve.py", "tasks/task1_dijkstra/input/graph.csv", "0", "4"])
        
        print("Simulating LLM call 1...")
        with session.llm_call("gemini-1.5-pro", prompt="Analyze task 1", prompt_tokens=100, completion_tokens=50) as node:
            time.sleep(1) # simulate inference and network
            
        # Task 2
        print("Running task 2...")
        run_task(session, "task2_toc_generator", ["python3", "tasks/task2_toc_generator/python/solve.py", "tasks/task2_toc_generator/input/sample_docs/README.md"])
        
        print("Simulating LLM call 2...")
        with session.llm_call("gemini-1.5-pro", prompt="Analyze task 2", prompt_tokens=150, completion_tokens=75) as node:
            time.sleep(1.5)
            
        # Task 3
        print("Running task 3...")
        run_task(session, "task3_log_analyzer", ["python3", "tasks/task3_log_analyzer/python/solve.py", "tasks/task3_log_analyzer/input/log.jsonl"])
        
        print("Simulating LLM call 3...")
        with session.llm_call("gemini-1.5-pro", prompt="Analyze task 3", prompt_tokens=200, completion_tokens=100) as node:
            time.sleep(2.0)
            
    print("\nMeasurement complete. Analyzing results...\n")
    results = analyse(session)
    pprint.pprint(results)

if __name__ == "__main__":
    main()
