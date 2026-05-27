#!/usr/bin/env python3
"""Dijkstra shortest path solver.
Reads a weighted directed graph from CSV, computes shortest paths
between named nodes, and outputs JSON results.
"""
import csv
import heapq
import json
import sys


def solve(input_path: str, source: str, dest: str, directed: bool = True):
    """Run Dijkstra from *source* to *dest* on graph in *input_path*.

    CSV format: src,dst,weight  (one header row expected).
    """
    graph: dict = {}
    nodes: set = set()

    with open(input_path, "r") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)  # skip header
        except StopIteration:
            # empty file
            graph = {}
            nodes = set()

        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            src = row[0].strip()
            dst = row[1].strip()
            w = float(row[2].strip())
            nodes.add(src)
            nodes.add(dst)
            graph.setdefault(src, []).append((dst, w))
            if not directed:
                graph.setdefault(dst, []).append((src, w))

    # Add source/dest to nodes if they're not in the graph
    nodes.add(source)
    nodes.add(dest)

    # Dijkstra
    dist: dict = {n: float("inf") for n in nodes}
    prev: dict = {n: None for n in nodes}
    dist[source] = 0
    pq = [(0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == dest:
            break
        for v, w in graph.get(u, []):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))

    # Reconstruct path
    if dist[dest] == float("inf"):
        return {"source": source, "destination": dest, "distance": float("inf"), "path": None}

    path = []
    cur = dest
    while cur is not None and cur in prev:
        path.append(cur)
        cur = prev[cur]
    path.reverse()

    return {"source": source, "destination": dest, "distance": dist[dest], "path": path}


def main():
    directed = True
    args = sys.argv[1:]
    input_path = ""
    source = ""
    dest = ""

    i = 0
    while i < len(args):
        if args[i] == "--undirected":
            directed = False
        elif not args[i].startswith("--"):
            if not input_path:
                input_path = args[i]
            elif not source:
                source = args[i]
            elif not dest:
                dest = args[i]
        i += 1

    if not input_path or not source or not dest:
        print(json.dumps({"error": "Usage: solve.py <graph.csv> <source> <dest> [--undirected]"}), file=sys.stderr)
        sys.exit(1)

    result = solve(input_path, source, dest, directed)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
