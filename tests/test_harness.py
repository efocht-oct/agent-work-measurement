"""Tests for lib/harness.py - the GaugeSession harness."""

import json
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEST_DATA = ROOT / "tasks" / "task1_dijkstra" / "input" / "graph.csv"


def _dummy_work():
    """Busy-wait for ~50ms to ensure measurable CPU time."""
    target = time.perf_counter() + 0.05
    while time.perf_counter() < target:
        _ = sum(range(1000))


def test_cpu_call_tracks_time():
    from lib.harness import GaugeSession

    with GaugeSession(name="cpu_track") as session:
        with session.cpu_call("work", category="compute") as node:
            _dummy_work()

    cpu_nodes = [n for n in session.flattened() if n.node_type == "cpu"]
    assert len(cpu_nodes) == 1
    node = cpu_nodes[0]
    assert node.wall_clock > 0, "wall_clock should be positive"
    assert node.total_cpu > 0, "total_cpu should be positive"


def test_llm_call_tracks_tokens():
    from lib.harness import GaugeSession

    with GaugeSession(name="llm_track") as session:
        with session.llm_call(
            model="gemini-1.5-pro",
            prompt="test prompt",
            prompt_tokens=10,
            completion_tokens=5,
        ) as node:
            node.prompt_tokens = 10
            node.completion_tokens = 5
            time.sleep(0.01)

    llm_nodes = [n for n in session.flattened() if n.node_type == "llm"]
    assert len(llm_nodes) == 1
    node = llm_nodes[0]
    assert node.prompt_tokens == 10
    assert node.completion_tokens == 5
    assert node.total_tokens == 15
    assert node.latency > 0
    assert node.model == "gemini-1.5-pro"
    assert node.cost_estimate > 0


def test_nested_calls():
    from lib.harness import GaugeSession

    with GaugeSession(name="nested") as session:
        with session.cpu_call("outer", category="io") as outer_node:
            with session.cpu_call("inner", category="compute") as inner_node:
                _dummy_work()
            with session.llm_call(
                "claude-3.5-sonnet", "think",
                prompt_tokens=100, completion_tokens=200,
            ):
                time.sleep(0.01)

    # outer should be parent of inner
    assert inner_node.parent is outer_node, (
        f"inner parent={inner_node.parent.name}, expected outer"
    )

    # Tree structure: root -> outer -> inner, llm
    root = session._root
    assert len(root.children) == 1, f"root children: {[c.name for c in root.children]}"
    assert len(outer_node.children) == 2, (
        f"outer children: {[c.name for c in outer_node.children]}"
    )


def test_summary_aggregates():
    from lib.harness import GaugeSession

    with GaugeSession(name="agg") as session:
        with session.cpu_call("a", category="x"):
            time.sleep(0.01)
        with session.cpu_call("b", category="y"):
            time.sleep(0.01)
        with session.llm_call(
            "gemini-1.5-pro", "p", prompt_tokens=10, completion_tokens=20,
        ):
            time.sleep(0.01)

    s = session.summary()
    assert s["name"] == "agg"
    assert s["n_nodes"] == 3
    assert s["n_llm_nodes"] == 1
    assert s["n_cpu_nodes"] == 2
    assert s["total_llm_latency"] > 0
    # llm_call sets prompt/comp tokens from constructor args
    assert s["total_prompt_tokens"] == 10
    assert s["total_completion_tokens"] == 20
    assert s["total_tokens"] == 30
    assert len(s["llm_nodes"]) == 1
    assert len(s["cpu_nodes"]) == 2
    assert s["total_cost_estimate"] > 0


def test_trace_serialization():
    from lib.harness import GaugeSession

    with GaugeSession(name="serialize") as session:
        with session.cpu_call("read_file", category="file_read"):
            time.sleep(0.01)
        with session.llm_call(
            "gemini-1.5-pro", "plan", prompt_tokens=50, completion_tokens=30,
        ):
            time.sleep(0.01)
        with session.cpu_call("write_file", category="file_write"):
            time.sleep(0.01)

    # Serialize
    json_str = session.to_json()
    data = json.loads(json_str)
    assert data["name"] == "serialize"
    assert "children" in data

    # Deserialize
    restored = GaugeSession.from_dict(data)
    restored_nodes = restored.flattened()
    assert restored._root.name == "serialize"
    assert len([n for n in restored_nodes if n.node_type == "cpu"]) == 2
    assert len([n for n in restored_nodes if n.node_type == "llm"]) == 1

    # Round-trip through JSON string too
    restored2 = GaugeSession.from_json(json_str)
    assert restored2._root.name == "serialize"


def test_io_tracking():
    from lib.harness import GaugeSession

    with GaugeSession(name="io_track") as session:
        with session.cpu_call("read_graph", category="file_read"):
            with open(TEST_DATA, "r") as f:
                _ = f.read()

    cpu_nodes = [n for n in session.flattened() if n.node_type == "cpu"]
    assert len(cpu_nodes) == 1
    node = cpu_nodes[0]
    assert node.io_read_bytes >= 0
    assert node.io_write_bytes >= 0


def test_concurrent_calls():
    from lib.harness import GaugeSession

    with GaugeSession(name="concurrent") as session:
        threads = []
        for i in range(4):
            t = threading.Thread(
                target=_run_cpu_call,
                args=(session, f"thread_{i}"),
            )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    cpu_nodes = [n for n in session.flattened() if n.node_type == "cpu"]
    assert len(cpu_nodes) == 4


def _run_cpu_call(session, name):
    with session.cpu_call(name, category="compute"):
        _dummy_work()


def test_session_context_manager():
    from lib.harness import GaugeSession

    with GaugeSession(name="session_cm") as session:
        pass

    assert session._root.wall_clock > 0
