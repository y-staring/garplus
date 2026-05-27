from __future__ import annotations

from typing import Any

import networkx as nx

from garplus_types import PatternState


def _short_id(s: str, max_len: int = 60) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def canonical_edge(u: Any, v: Any) -> tuple[str, str]:
    left, right = sorted((str(u), str(v)))
    return left, right


def graph_edges_as_set(g: nx.Graph) -> set[tuple[str, str]]:
    return {canonical_edge(u, v) for u, v in g.edges()}


def pattern_signature(g: nx.Graph) -> str:
    """
    First-version pattern signature for de-duplication.

    This is not an isomorphism-invariant hash. It simply canonicalizes the
    observed node/edge ids from sampled patterns. Later this can be replaced by
    WL-hash or exact graph isomorphism.
    """
    nodes = sorted(str(n) for n in g.nodes())
    edges = sorted(f"{u}--{v}" for u, v in (canonical_edge(u, v) for u, v in g.edges()))
    return f"nodes:[{'|'.join(nodes)}];edges:[{'|'.join(edges)}]"


def make_pattern_state(g: nx.Graph, parent_id=None, added_edge=None, bn_score=None) -> PatternState:
    sig = pattern_signature(g)
    return PatternState(
        pattern_id=sig,
        graph=g.copy(),
        edge_count=int(g.number_of_edges()),
        node_count=int(g.number_of_nodes()),
        parent_id=parent_id,
        added_edge=added_edge,
        support=None,
        bn_score=bn_score,
    )


def build_union_graph(pattern_graphs: list[tuple[int, nx.Graph]]) -> nx.Graph:
    union_graph = nx.Graph()
    for _, graph in pattern_graphs:
        union_graph.add_nodes_from(graph.nodes(data=True))
        union_graph.add_edges_from(graph.edges(data=True))
    return union_graph


def initialize_seed_patterns(pattern_graphs, max_seed_edges=1) -> list[PatternState]:
    """
    Initialize the first vertical level P_1.

    Prefer all sampled patterns with exactly one edge.
    If there are none, use the smallest-edge bucket among sampled patterns.
    De-duplicate by pattern_signature.
    """
    if not pattern_graphs:
        return []

    edge_buckets: dict[int, list[nx.Graph]] = {}
    for _, graph in pattern_graphs:
        edge_buckets.setdefault(int(graph.number_of_edges()), []).append(graph)

    if max_seed_edges in edge_buckets:
        seed_graphs = edge_buckets[max_seed_edges]
    else:
        min_edges = min(edge_buckets.keys())
        seed_graphs = edge_buckets[min_edges]

    seen = set()
    seeds: list[PatternState] = []
    for graph in seed_graphs:
        state = make_pattern_state(graph)
        if state.pattern_id in seen:
            continue
        seen.add(state.pattern_id)
        seeds.append(state)
    return seeds


def compute_pattern_match_ids(
    pattern: nx.Graph,
    pattern_graphs: list[tuple[int, nx.Graph]],
    support_mode: str = "edge_subset",
) -> list[int]:
    """
    Coarse pattern verification over sampled patterns.

    edge_subset:
        Q is considered supported by a sampled pattern if all edges of Q appear
        in that sampled pattern.

    exact_signature:
        only identical signatures count.
    """
    query_sig = pattern_signature(pattern)
    query_edges = graph_edges_as_set(pattern)
    match_ids = []

    for sampled_id, sampled_graph in pattern_graphs:
        if support_mode == "exact_signature":
            if pattern_signature(sampled_graph) == query_sig:
                match_ids.append(int(sampled_id))
            continue

        sampled_edges = graph_edges_as_set(sampled_graph)
        if query_edges.issubset(sampled_edges):
            match_ids.append(int(sampled_id))

    return match_ids


def generate_pattern_extensions(
    pattern: nx.Graph,
    union_graph: nx.Graph,
    max_new_edges: int = 1,
) -> list[tuple[nx.Graph, tuple[Any, Any]]]:
    """
    First-version vertical spawning (VSpawn): add one edge.

    Allowed:
    - add one edge between two existing nodes
    - add one edge from an existing node to one new node

    Not allowed:
    - adding one edge whose two endpoints are both new nodes, because that
      would disconnect the new pattern from the current one.
    """
    if max_new_edges != 1:
        raise ValueError("First version only supports add-one-edge expansion.")

    pattern_nodes = set(pattern.nodes())
    pattern_edges = graph_edges_as_set(pattern)
    extensions = []
    seen = set()

    for u, v in union_graph.edges():
        edge_key = canonical_edge(u, v)
        if edge_key in pattern_edges:
            continue
        if u not in pattern_nodes and v not in pattern_nodes:
            continue

        new_graph = pattern.copy()
        if u not in new_graph:
            new_graph.add_node(u)
        if v not in new_graph:
            new_graph.add_node(v)
        new_graph.add_edge(u, v)

        sig = pattern_signature(new_graph)
        if sig in seen:
            continue
        seen.add(sig)
        extensions.append((new_graph, (u, v)))

    return extensions


def extend_pattern(Q: nx.Graph, gamma: tuple[Any, Any]) -> nx.Graph:
    """
    Apply one structural extension gamma to pattern Q and return the child pattern.

    # Tree expansion:
    # parent pattern Q is expanded into child pattern Q_gamma
    # by applying one structural extension gamma.
    """
    u, v = gamma
    Q_gamma = Q.copy()
    if u not in Q_gamma:
        Q_gamma.add_node(u)
    if v not in Q_gamma:
        Q_gamma.add_node(v)
    Q_gamma.add_edge(u, v)
    return Q_gamma

