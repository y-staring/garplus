from __future__ import annotations

"""A lightweight, pure-Python subgraph matcher.

This module plays the role of the pattern-matching engine in the GAR pipeline.
It is not a full VF3 reimplementation; instead, it uses DFS backtracking with:
- label-based candidate filtering
- incremental edge-feasibility checks
- optional pivot restriction for spawned patterns
"""

from typing import Dict, List, Optional

from graph_types import DataGraph, GraphInstance, GraphPattern, PatternEdge


def _candidate_data_nodes(pattern: GraphPattern, graph: DataGraph, pattern_node: int) -> List[int]:
    """Return data-graph nodes whose labels match the current pattern node."""

    label = pattern.node_labels[pattern_node]
    return [node_id for node_id, vertex in graph.vertices.items() if vertex.label == label]


def _compatible_edge(pattern_edge: PatternEdge, mapping: Dict[int, int], graph: DataGraph) -> bool:
    """Check whether a fully grounded pattern edge exists in the data graph."""

    src = mapping.get(pattern_edge.src)
    dst = mapping.get(pattern_edge.dst)
    if src is None or dst is None:
        return True
    return any(edge.dst == dst and edge.label == pattern_edge.label for edge in graph.out_neighbors(src))


def _is_feasible(pattern: GraphPattern, graph: DataGraph, mapping: Dict[int, int], next_pattern_node: int, next_data_node: int) -> bool:
    """Check whether the next node assignment keeps the partial match valid."""

    if next_data_node in mapping.values():
        return False
    if graph.vertices[next_data_node].label != pattern.node_labels[next_pattern_node]:
        return False
    test_mapping = dict(mapping)
    test_mapping[next_pattern_node] = next_data_node
    for edge in pattern.edges:
        if not _compatible_edge(edge, test_mapping, graph):
            return False
    return True


def _materialize_instance(pattern: GraphPattern, graph: DataGraph, mapping: Dict[int, int]) -> GraphInstance:
    """Convert one successful node mapping into a `GraphInstance`."""

    edge_triplets = []
    edge_bindings: Dict[int, int] = {}
    for edge_index, pattern_edge in enumerate(pattern.edges):
        src = mapping[pattern_edge.src]
        dst = mapping[pattern_edge.dst]
        matches = graph.find_edges(src, dst, pattern_edge.label)
        if not matches:
            raise ValueError(f"edge binding missing for pattern edge {pattern_edge}")
        chosen = matches[0]
        edge_triplets.append((chosen.src, chosen.dst, chosen.label))
        edge_bindings[edge_index] = chosen.edge_id
    edge_ids = tuple(sorted(edge_triplets))
    return GraphInstance(node_map=dict(mapping), edge_ids=edge_ids, pivot=mapping.get(0), edge_bindings=edge_bindings)


def find_matches_with_limit(pattern: GraphPattern, graph: DataGraph, limit: Optional[int] = None, pivot_candidates: Optional[List[int]] = None) -> List[GraphInstance]:
    """Enumerate matches of one pattern in the data graph.

    `pivot_candidates` is useful during pattern expansion: it narrows pattern node 0 to
    previously known pivots so we do not re-search the full graph from scratch.
    """

    order = sorted(range(pattern.node_count()), key=lambda idx: len(pattern.out_adj(idx)) + len(pattern.in_adj(idx)), reverse=True)
    results: List[GraphInstance] = []

    def dfs(depth: int, mapping: Dict[int, int]) -> bool:
        if limit is not None and len(results) >= limit:
            return True
        if depth == len(order):
            results.append(_materialize_instance(pattern, graph, mapping))
            return False
        pattern_node = order[depth]
        candidates = pivot_candidates if pattern_node == 0 and pivot_candidates is not None else _candidate_data_nodes(pattern, graph, pattern_node)
        for data_node in candidates:
            if _is_feasible(pattern, graph, mapping, pattern_node, data_node):
                mapping[pattern_node] = data_node
                should_stop = dfs(depth + 1, mapping)
                mapping.pop(pattern_node)
                if should_stop:
                    return True
        return False

    dfs(0, {})
    return results


def find_matches(pattern: GraphPattern, graph: DataGraph) -> List[GraphInstance]:
    """Unbounded version of `find_matches_with_limit`."""

    return find_matches_with_limit(pattern, graph, limit=None)
