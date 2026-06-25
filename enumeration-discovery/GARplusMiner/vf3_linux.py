from __future__ import annotations

"""vf3py-backed global matcher with GAR edge-id materialization."""

import os
from typing import Dict, List, Optional

import networkx as nx
import vf3py

from exact_subgraph_matcher import find_matches_with_limit as exact_find_matches_with_limit
from graph_types import DataGraph, GraphInstance, GraphPattern


def _to_networkx(pattern: GraphPattern, graph: DataGraph):
    pattern_graph = nx.DiGraph()
    for node_id, label in enumerate(pattern.node_labels):
        pattern_graph.add_node(node_id, label=label)
    for edge in pattern.edges:
        pattern_graph.add_edge(edge.src, edge.dst, label=edge.label)
    data_graph = nx.DiGraph()
    for node_id, vertex in graph.vertices.items():
        data_graph.add_node(node_id, label=vertex.label)
    for edge in graph.all_edges():
        data_graph.add_edge(edge.src, edge.dst, label=edge.label)
    return pattern_graph, data_graph


def _normalize_mapping(raw_mapping: Dict, pattern: GraphPattern) -> Optional[Dict[int, int]]:
    pattern_nodes = set(range(pattern.node_count()))
    mapping = {int(key): int(value) for key, value in raw_mapping.items()}
    if pattern_nodes.issubset(mapping):
        return {node_id: mapping[node_id] for node_id in pattern_nodes}
    if pattern_nodes.issubset(set(mapping.values())):
        inverse = {value: key for key, value in mapping.items()}
        return {node_id: inverse[node_id] for node_id in pattern_nodes}
    return None


def _append_edge_bindings(pattern, graph, node_map, results, limit) -> bool:
    choices = []
    for pattern_edge_index, edge in enumerate(pattern.edges):
        candidates = graph.find_edges(node_map[edge.src], node_map[edge.dst], edge.label)
        if not candidates:
            return False
        choices.append((pattern_edge_index, candidates))
    choices.sort(key=lambda item: len(item[1]))
    bindings = {}
    used_edge_ids = set()

    def bind(index: int) -> bool:
        if limit is not None and len(results) >= limit:
            return True
        if index == len(choices):
            edge_ids = tuple(sorted((graph.edges_by_id[eid].src, graph.edges_by_id[eid].dst, graph.edges_by_id[eid].label) for eid in bindings.values()))
            results.append(GraphInstance(node_map=dict(node_map), edge_ids=edge_ids, pivot=node_map.get(0), edge_bindings=dict(bindings)))
            return False
        pattern_edge_index, candidates = choices[index]
        for edge in candidates:
            if edge.edge_id in used_edge_ids:
                continue
            bindings[pattern_edge_index] = edge.edge_id
            used_edge_ids.add(edge.edge_id)
            should_stop = bind(index + 1)
            used_edge_ids.remove(edge.edge_id)
            bindings.pop(pattern_edge_index)
            if should_stop:
                return True
        return False

    return bind(0)


def find_matches_with_limit(
    pattern: GraphPattern,
    graph: DataGraph,
    limit: Optional[int] = None,
    pivot_candidates: Optional[List[int]] = None,
    target_edge_index: Optional[int] = None,
    max_instances_per_target_edge: Optional[int] = None,
    target_edge_undirected: bool = False,
) -> List[GraphInstance]:
    """Use vf3py's documented monomorphism API, then bind concrete GAR edges."""

    if pivot_candidates is not None or max_instances_per_target_edge is not None:
        if max_instances_per_target_edge is not None and not getattr(find_matches_with_limit, "_reported_target_cap_backend", False):
            print(
                f"[VF3Linux] max_instances_per_target_edge={max_instances_per_target_edge}; "
                "using exact matcher for target-edge capped rematch"
            )
            setattr(find_matches_with_limit, "_reported_target_cap_backend", True)
        return exact_find_matches_with_limit(
            pattern,
            graph,
            limit,
            pivot_candidates,
            target_edge_index=target_edge_index,
            max_instances_per_target_edge=max_instances_per_target_edge,
            target_edge_undirected=target_edge_undirected,
        )
    pattern_graph, data_graph = _to_networkx(pattern, graph)
    threads = max(1, int(os.environ.get("GARPLUS_VF3PY_THREADS", "1")))
    try:
        raw_mappings = vf3py.get_subgraph_monomorphisms(
            pattern_graph,
            data_graph,
            node_match=lambda pattern_attrs, data_attrs: pattern_attrs.get("label") == data_attrs.get("label"),
            edge_match=lambda pattern_attrs, data_attrs: pattern_attrs.get("label") == data_attrs.get("label"),
            variant="L",
            num_threads=threads,
        )
    except Exception as exc:
        print(f"[VF3Linux] vf3py_failed={type(exc).__name__}; using exact matcher")
        return exact_find_matches_with_limit(pattern, graph, limit, pivot_candidates)
    if not getattr(find_matches_with_limit, "_reported_backend", False):
        print(f"[VF3Linux] backend=vf3py version={getattr(vf3py, '__version__', 'unknown')} variant=L threads={threads}")
        setattr(find_matches_with_limit, "_reported_backend", True)
    results = []
    for raw_mapping in raw_mappings:
        node_map = _normalize_mapping(raw_mapping, pattern)
        if node_map is not None and _append_edge_bindings(pattern, graph, node_map, results, limit):
            break
    return results


def find_matches(pattern: GraphPattern, graph: DataGraph) -> List[GraphInstance]:
    return find_matches_with_limit(pattern, graph, limit=None)
