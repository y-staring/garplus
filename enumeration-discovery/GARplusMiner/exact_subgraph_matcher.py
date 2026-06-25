from __future__ import annotations

"""Exact non-induced, edge-aware matcher used for global GAR support."""

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from graph_types import DataGraph, GraphInstance, GraphPattern


def find_matches_with_limit(
    pattern: GraphPattern,
    graph: DataGraph,
    limit: Optional[int] = None,
    pivot_candidates: Optional[List[int]] = None,
    target_edge_index: Optional[int] = None,
    max_instances_per_target_edge: Optional[int] = None,
    target_edge_undirected: bool = False,
) -> List[GraphInstance]:
    """Enumerate directed embeddings and retain a distinct binding for each edge id."""

    by_label: Dict[object, List[int]] = defaultdict(list)
    edge_index: Dict[Tuple[int, int, object], list] = defaultdict(list)
    for node_id, vertex in graph.vertices.items():
        by_label[vertex.label].append(node_id)
    for edge in graph.all_edges():
        edge_index[(edge.src, edge.dst, edge.label)].append(edge)

    degree = {
        node_id: len(pattern.in_adj(node_id)) + len(pattern.out_adj(node_id))
        for node_id in range(pattern.node_count())
    }
    order = sorted(range(pattern.node_count()), key=lambda node_id: (-degree[node_id], node_id))
    results: List[GraphInstance] = []
    mapping: Dict[int, int] = {}
    used_nodes = set()
    target_edge_counts: Dict[Tuple[int, int, object], int] = {}

    def target_key(edge_id: int) -> Tuple[int, int, object]:
        edge = graph.edges_by_id[edge_id]
        if target_edge_undirected:
            return min(edge.src, edge.dst), max(edge.src, edge.dst), edge.label
        return edge.src, edge.dst, edge.label

    def has_grounded_edges() -> bool:
        for edge in pattern.edges:
            if edge.src in mapping and edge.dst in mapping:
                if not edge_index.get((mapping[edge.src], mapping[edge.dst], edge.label)):
                    return False
        return True

    def materialize_edges() -> None:
        choices = []
        for edge_index_in_pattern, edge in enumerate(pattern.edges):
            candidates = edge_index.get((mapping[edge.src], mapping[edge.dst], edge.label), [])
            if not candidates:
                return
            choices.append((edge_index_in_pattern, candidates))
        choices.sort(key=lambda item: len(item[1]))
        bindings: Dict[int, int] = {}
        used_edge_ids = set()

        def bind(index: int) -> bool:
            if limit is not None and len(results) >= limit:
                return True
            if index == len(choices):
                edge_bindings = dict(bindings)
                counted_target_key = None
                if target_edge_index is not None and max_instances_per_target_edge is not None:
                    target_edge_id = edge_bindings.get(target_edge_index)
                    if target_edge_id is None:
                        return False
                    counted_target_key = target_key(target_edge_id)
                    if target_edge_counts.get(counted_target_key, 0) >= max_instances_per_target_edge:
                        return False
                edge_ids = tuple(
                    sorted(
                        (graph.edges_by_id[edge_id].src, graph.edges_by_id[edge_id].dst, graph.edges_by_id[edge_id].label)
                        for edge_id in edge_bindings.values()
                    )
                )
                results.append(
                    GraphInstance(
                        node_map=dict(mapping),
                        edge_ids=edge_ids,
                        pivot=mapping.get(0),
                        edge_bindings=edge_bindings,
                    )
                )
                if counted_target_key is not None:
                    target_edge_counts[counted_target_key] = target_edge_counts.get(counted_target_key, 0) + 1
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

        bind(0)

    def search(depth: int) -> bool:
        if limit is not None and len(results) >= limit:
            return True
        if depth == len(order):
            materialize_edges()
            return limit is not None and len(results) >= limit
        pattern_node = order[depth]
        if pattern_node == 0 and pivot_candidates is not None:
            candidates = pivot_candidates
        else:
            candidates = by_label.get(pattern.node_labels[pattern_node], [])
        for data_node in candidates:
            if data_node in used_nodes:
                continue
            mapping[pattern_node] = data_node
            used_nodes.add(data_node)
            if has_grounded_edges() and search(depth + 1):
                return True
            used_nodes.remove(data_node)
            mapping.pop(pattern_node)
        return False

    search(0)
    return results


def find_matches(pattern: GraphPattern, graph: DataGraph) -> List[GraphInstance]:
    return find_matches_with_limit(pattern, graph, limit=None)
