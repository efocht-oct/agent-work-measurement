"""Measurement harness for the Agent Work Measurement project.

Provides a GaugeSession class that acts as a context manager.
Inside a session, individual tool calls and LLM interactions are wrapped
with :meth:`cpu_call` and :meth:`llm_call` context managers.  Each wrapped
call records wall-clock time, CPU time, RSS, I/O, and (for LLM calls)
token counts and cost estimates.

The session maintains a tree of TraceNode objects and supports JSON
serialization / deserialization for later analysis.

All timing uses :func:`time.perf_counter`.  On Linux the ``resource``
module and ``/proc/self/io`` are used where available; graceful fallbacks
are provided for other platforms.
"""

from __future__ import annotations

import copy
import json
import os
import resource
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# I/O helpers (Linux-specific with fallback)
# ---------------------------------------------------------------------------

def _read_proc_io() -> Dict[str, int]:
    """Read I/O bytes from /proc/self/io.

    Returns a dict with ``read_bytes`` and ``write_bytes`` keys.
    Returns {read_bytes: 0, write_bytes: 0} on any failure.
    """
    try:
        with open("/proc/self/io", "r") as f:
            lines = f.readlines()
        io = {}
        for line in lines:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if key in ("read_bytes", "write_bytes"):
                io[key] = int(val)
        return io
    except (OSError, IOError, ValueError):
        return {"read_bytes": 0, "write_bytes": 0}


def _get_rss_kb() -> int:
    """Return current RSS in KiB via resource module."""
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# TraceNode data class
# ---------------------------------------------------------------------------

@dataclass
class TraceNode:
    """A single measurement node in the trace tree.

    Attributes:
        name: Human-readable identifier for this node.
        node_type: One of ``"llm"``, ``"cpu"``, ``"io"``.
        category: Optional sub-category (e.g. ``"file_read"``).
        wall_clock: Total wall-clock seconds elapsed.
        user_cpu: User-space CPU seconds.
        system_cpu: Kernel-space CPU seconds.
        total_cpu: user_cpu + system_cpu.
        max_rss: Peak resident set size (KiB on Linux).
        io_read_bytes: Bytes read during this node.
        io_write_bytes: Bytes written during this node.
        prompt_tokens: Input tokens (LLM calls only).
        completion_tokens: Output tokens (LLM calls only).
        total_tokens: prompt_tokens + completion_tokens.
        latency: Wall-clock latency (LLM calls only; alias of wall_clock).
        model: Model name string (LLM calls only).
        cost_estimate: USD cost estimate (LLM calls only).
        children: Child TraceNode instances (empty list when leaf).
        start_ts: Monotonic timestamp when the node started.
    """
    name: str
    node_type: str
    category: Optional[str] = None
    wall_clock: float = 0.0
    user_cpu: float = 0.0
    system_cpu: float = 0.0
    total_cpu: float = 0.0
    max_rss: int = 0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency: float = 0.0
    model: Optional[str] = None
    cost_estimate: float = 0.0
    children: List[TraceNode] = field(default_factory=list)
    start_ts: float = 0.0

    # --- internal bookkeeping (not serialised) ---
    _parent: Optional[TraceNode] = field(default=None, repr=False)

    def __post_init__(self):
        self.total_cpu = self.user_cpu + self.system_cpu

    @property
    def parent(self) -> Optional[TraceNode]:
        return self._parent

    @parent.setter
    def parent(self, value: Optional[TraceNode]) -> None:
        self._parent = value


# ---------------------------------------------------------------------------
# GaugeSession
# ---------------------------------------------------------------------------

class GaugeSession:
    """Context-managed measurement session.

    Usage::

        with GaugeSession(name="my-task") as session:
            with session.cpu_call("read_graph", category="file_read"):
                read_graph("input/graph.csv")
            with session.llm_call("gpt-4o", "explain your approach") as node:
                node.prompt_tokens = 100
                node.completion_tokens = 50
                response = call_llm(node.prompt)
            summary = session.summary()

    Thread-safe: all modifications to the trace tree are protected by a
    single ``threading.Lock``.
    """

    def __init__(self, name: str = "session") -> None:
        self.name = name
        self._root = TraceNode(name=name, node_type="session")
        self._lock = threading.Lock()
        self._start_wall: float = 0.0
        self._end_wall: float = 0.0
        # Stack of currently active nodes (for parent tracking).
        self._stack: List[TraceNode] = []
        # Pre-computed I/O snapshot taken at session start.
        self._io_start: Dict[str, int] = {}

    # -- context manager -----------------------------------------------------

    def __enter__(self) -> "GaugeSession":
        self._start_wall = time.perf_counter()
        self._io_start = _read_proc_io()
        return self

    def __exit__(self, *args: Any) -> None:
        self._end_wall = time.perf_counter()
        # Update root wall_clock to cover the entire session.
        self._root.wall_clock = self._end_wall - self._start_wall
        self._root.user_cpu = self._root.user_cpu or self._cpu_snapshot()

    # -- internal helpers ----------------------------------------------------

    def _cpu_snapshot(self) -> Dict[str, float]:
        """Return current user/system CPU seconds."""
        try:
            ru = resource.getrusage(resource.RUSAGE_SELF)
            return {"user": ru.ru_utime, "system": ru.ru_stime}
        except Exception:
            return {"user": 0.0, "system": 0.0}

    def _cpu_diff(self, before: Dict[str, float]) -> Dict[str, float]:
        after = self._cpu_snapshot()
        return {
            "user": max(0.0, after["user"] - before["user"]),
            "system": max(0.0, after["system"] - before["system"]),
        }

    # -- public APIs ---------------------------------------------------------

    def flattened(self) -> List[TraceNode]:
        """Return a flat list of all nodes in the trace (pre-order)."""
        result: List[TraceNode] = []

        def _walk(node: TraceNode) -> None:
            result.append(node)
            for child in node.children:
                _walk(child)

        _walk(self._root)
        return result

    def summary(self) -> Dict[str, Any]:
        """Aggregate summary of all metrics across the trace.

        Returns a dict with:
        - ``name``: session name
        - ``total_wall_clock``: wall-clock seconds for the whole session
        - ``total_user_cpu``, ``total_system_cpu``, ``total_cpu``
        - ``total_max_rss``: peak RSS across all nodes
        - ``total_io_read_bytes``, ``total_io_write_bytes``
        - ``total_prompt_tokens``, ``total_completion_tokens``, ``total_tokens``
        - ``total_cost_estimate``: USD
        - ``total_llm_latency``: sum of LLM latency
        - ``n_nodes``: total number of trace nodes (excluding root)
        - ``n_llm_nodes``: number of LLM nodes
        - ``n_cpu_nodes``: number of CPU nodes
        - ``llm_nodes``: list of dicts for each LLM node
        - ``cpu_nodes``: list of dicts for each CPU node
        """
        nodes = [n for n in self.flattened() if n is not self._root]

        total_wall = 0.0
        total_user = 0.0
        total_sys = 0.0
        total_rss = 0
        total_io_r = 0
        total_io_w = 0
        total_prompt = 0
        total_comp = 0
        total_cost = 0.0
        total_llm_lat = 0.0
        n_llm = 0
        n_cpu = 0
        llm_node_list: List[Dict[str, Any]] = []
        cpu_node_list: List[Dict[str, Any]] = []

        for n in nodes:
            total_wall += n.wall_clock
            total_user += n.user_cpu
            total_sys += n.system_cpu
            if n.max_rss > total_rss:
                total_rss = n.max_rss
            total_io_r += n.io_read_bytes
            total_io_w += n.io_write_bytes
            total_prompt += n.prompt_tokens
            total_comp += n.completion_tokens
            total_cost += n.cost_estimate
            total_llm_lat += n.latency
            node_dict = {k: getattr(n, k) for k in [
                "name", "node_type", "wall_clock", "user_cpu", "system_cpu",
                "total_cpu", "max_rss", "prompt_tokens", "completion_tokens",
                "model", "cost_estimate",
            ]}
            if n.node_type == "llm":
                n_llm += 1
                llm_node_list.append(node_dict)
            elif n.node_type in ("cpu", "io", "io_tool"):
                n_cpu += 1
                cpu_node_list.append(node_dict)

        return {
            "name": self.name,
            "total_wall_clock": total_wall,
            "total_user_cpu": total_user,
            "total_system_cpu": total_sys,
            "total_cpu": total_user + total_sys,
            "total_max_rss": total_rss,
            "total_io_read_bytes": total_io_r,
            "total_io_write_bytes": total_io_w,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_comp,
            "total_tokens": total_prompt + total_comp,
            "total_cost_estimate": total_cost,
            "total_llm_latency": total_llm_lat,
            "n_nodes": len(nodes),
            "n_llm_nodes": n_llm,
            "n_cpu_nodes": n_cpu,
            "llm_nodes": llm_node_list,
            "cpu_nodes": cpu_node_list,
        }

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entire trace tree to a plain dict."""
        return _node_to_dict(self._root)

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GaugeSession":
        """Deserialise a GaugeSession from a dict."""
        session = cls(name=data.get("name", "restored"))
        session._root = _dict_to_node(data)
        return session

    @classmethod
    def from_json(cls, json_str: str) -> "GaugeSession":
        """Deserialise a GaugeSession from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def serialize(self, indent: int = 2) -> str:
        """Alias for :meth:`to_json`."""
        return self.to_json(indent=indent)

    def deserialize(self, json_str: str) -> "GaugeSession":
        """Alias for :meth:`from_json`."""
        return self.from_json(json_str)

    # -- context managers ----------------------------------------------------

    @contextmanager
    def cpu_call(
        self,
        name: str,
        category: str = "cpu",
    ) -> Any:
        """Context manager for a CPU-bound tool call.

        Records wall-clock, CPU, RSS, and I/O deltas.

        Args:
            name: Identifier for this call.
            category: Optional sub-category string.

        Yields:
            The TraceNode for the call (allowing manual attribute access).
        """
        node = TraceNode(name=name, node_type="cpu", category=category)
        before_wall = time.perf_counter()
        before_cpu = self._cpu_snapshot()
        before_rss = _get_rss_kb()
        before_io = _read_proc_io()

        # Attach to tree BEFORE yielding so children can find us on _stack
        node.start_ts = before_wall
        with self._lock:
            if self._stack:
                self._stack[-1].children.append(node)
                node._parent = self._stack[-1]
            else:
                self._root.children.append(node)
                node._parent = self._root
            self._stack.append(node)

        try:
            yield node
        finally:
            self._stack.pop()

            after_wall = time.perf_counter()
            after_cpu = self._cpu_snapshot()
            after_rss = _get_rss_kb()
            after_io = _read_proc_io()

            diff_cpu = self._cpu_diff(before_cpu)
            wall = after_wall - before_wall

            node.wall_clock = wall
            node.user_cpu = float(diff_cpu["user"])
            node.system_cpu = float(diff_cpu["system"])
            node.total_cpu = node.user_cpu + node.system_cpu
            node.max_rss = max(0, after_rss - before_rss) if before_rss else after_rss
            node.io_read_bytes = max(0, after_io.get("read_bytes", 0)
                                    - before_io.get("read_bytes", 0))
            node.io_write_bytes = max(0, after_io.get("write_bytes", 0)
                                     - before_io.get("write_bytes", 0))

    @contextmanager
    def llm_call(
        self,
        model: str,
        prompt: str = "",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        category: str = "llm",
    ) -> Any:
        """Context manager for an LLM call.

        Records wall-clock latency, tokens, and cost estimate.

        Args:
            model: Model name (e.g. ``"gpt-4o"``).
            prompt: The prompt text.
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.
            category: Optional sub-category.

        Yields:
            The TraceNode so that the caller can annotate it further
            (e.g. set actual token counts after the call returns).
        """
        node = TraceNode(
            name=model,
            node_type="llm",
            category=category,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        before_wall = time.perf_counter()

        # Attach to tree BEFORE yielding so children can find us on _stack
        node.start_ts = before_wall
        with self._lock:
            if self._stack:
                self._stack[-1].children.append(node)
                node._parent = self._stack[-1]
            else:
                self._root.children.append(node)
                node._parent = self._root
            self._stack.append(node)

        try:
            yield node
        finally:
            self._stack.pop()
            after_wall = time.perf_counter()

            node.wall_clock = after_wall - before_wall
            node.latency = node.wall_clock
            node.total_tokens = node.prompt_tokens + node.completion_tokens

            # Estimate cost
            from lib.baselines import cost_per_call
            if node.prompt_tokens > 0 or node.completion_tokens > 0:
                node.cost_estimate = cost_per_call(
                    model, node.prompt_tokens, node.completion_tokens)


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

_NODE_FIELDS = [
    "name", "node_type", "category", "wall_clock", "user_cpu",
    "system_cpu", "total_cpu", "max_rss", "io_read_bytes",
    "io_write_bytes", "prompt_tokens", "completion_tokens",
    "total_tokens", "latency", "model", "cost_estimate",
]


def _node_to_dict(node: TraceNode) -> Dict[str, Any]:
    """Convert a TraceNode tree to a serialisable dict."""
    d: Dict[str, Any] = {f: getattr(node, f) for f in _NODE_FIELDS}
    # Ensure user_cpu is a float (it may be a dict from _cpu_snapshot)
    if isinstance(d.get("user_cpu"), dict):
        d["user_cpu"] = d["user_cpu"].get("user", 0.0)
    children = [
        _node_to_dict(c) for c in node.children
    ]
    if children:
        d["children"] = children
    return d


def _dict_to_node(d: Dict[str, Any]) -> TraceNode:
    """Reconstruct a TraceNode tree from a serialisable dict."""
    kwargs: Dict[str, Any] = {f: d.get(f) for f in _NODE_FIELDS}
    kwargs.setdefault("name", "unknown")
    kwargs.setdefault("node_type", "cpu")
    kwargs["children"] = []
    node = TraceNode(**kwargs)
    for child_data in d.get("children", []):
        node.children.append(_dict_to_node(child_data))
    return node
