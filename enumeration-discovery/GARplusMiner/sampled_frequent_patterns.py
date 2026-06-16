from __future__ import annotations

"""Frequent structural summaries from sampled PPI/DDA/TI subgraphs.

The module can either return lightweight skeleton patterns or materialize small
2/3-edge motifs with a fast edge-driven matcher. This avoids running the generic
node-DFS matcher over an all-Protein graph.
"""

from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Tuple

from graph_types import DataGraph, Edge, FrequentPattern, GraphInstance, GraphPattern, PatternEdge

PatternSignature = Tuple[object, ...]


def _edge_label(edge: Edge) -> str:
    return str(edge.label)


def _node_label(graph: DataGraph, node_id: int) -> str:
    return str(graph.vertices[node_id].label)


def _edge_sig(graph: DataGraph, edge: Edge) -> PatternSignature:
    labels = sorted([_node_label(graph, edge.src), _node_label(graph, edge.dst)])
    return ("edge", tuple(labels), _edge_label(edge))


def _path2_sig(graph: DataGraph, e1: Edge, e2: Edge, center: int) -> PatternSignature:
    ends = []
    for edge in (e1, e2):
        other = edge.dst if edge.src == center else edge.src
        ends.append((_node_label(graph, other), _edge_label(edge)))
    ends.sort(key=str)
    return ("path2", _node_label(graph, center), tuple(ends))


def _triangle_sig(graph: DataGraph, edges: Iterable[Edge]) -> PatternSignature:
    edge_labels = sorted(_edge_label(edge) for edge in edges)
    node_labels = sorted({_node_label(graph, edge.src) for edge in edges} | {_node_label(graph, edge.dst) for edge in edges})
    return ("triangle", tuple(node_labels), tuple(edge_labels))


def mine_sampled_frequent_patterns(
    graph: DataGraph,
    min_graph_support: int = 5,
    max_patterns: int = 20,
) -> List[Tuple[PatternSignature, int]]:
    """Count small undirected motifs by sampled subgraph support."""

    signatures_by_graph: Dict[object, set] = defaultdict(set)
    edges_by_graph: Dict[object, List[Edge]] = defaultdict(list)
    for edge in graph.all_edges():
        graph_id = edge.attrs.get("sampled_graph_id")
        if graph_id in (None, -1, "-1"):
            continue
        edges_by_graph[graph_id].append(edge)
        signatures_by_graph[graph_id].add(_edge_sig(graph, edge))

    for graph_id, edges in edges_by_graph.items():
        incident: Dict[int, List[Edge]] = defaultdict(list)
        pair_to_edges: Dict[Tuple[int, int], List[Edge]] = defaultdict(list)
        for edge in edges:
            incident[edge.src].append(edge)
            incident[edge.dst].append(edge)
            pair_to_edges[tuple(sorted((edge.src, edge.dst)))].append(edge)

        for center, incident_edges in incident.items():
            for i in range(len(incident_edges)):
                for j in range(i + 1, len(incident_edges)):
                    signatures_by_graph[graph_id].add(_path2_sig(graph, incident_edges[i], incident_edges[j], center))

        nodes = sorted(incident)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                for k in range(j + 1, len(nodes)):
                    pairs = [tuple(sorted((nodes[i], nodes[j]))), tuple(sorted((nodes[i], nodes[k]))), tuple(sorted((nodes[j], nodes[k])))]
                    if all(pair in pair_to_edges for pair in pairs):
                        signatures_by_graph[graph_id].add(_triangle_sig(graph, [pair_to_edges[pair][0] for pair in pairs]))

    support = Counter()
    for signatures in signatures_by_graph.values():
        support.update(signatures)
    frequent = [(signature, count) for signature, count in support.items() if count >= min_graph_support]
    frequent.sort(key=lambda item: (-item[1], str(item[0])))
    return frequent[:max_patterns]


def edge_priors_from_frequent_patterns(frequent_patterns: Iterable[Tuple[PatternSignature, int]]) -> Dict[Tuple[str, str, str], float]:
    """Convert frequent sampled signatures into normalized edge-type priors."""

    raw = Counter()
    for signature, support in frequent_patterns:
        kind = signature[0] if signature else None
        if kind == "edge":
            _, node_labels, edge_label = signature
            left, right = sorted([str(node_labels[0]), str(node_labels[1])])
            raw[(left, right, str(edge_label))] += support
        elif kind == "path2":
            _, center_label, ends = signature
            for end_label, edge_label in ends:
                left, right = sorted([str(center_label), str(end_label)])
                raw[(left, right, str(edge_label))] += support
        elif kind == "triangle":
            _, node_labels, edge_labels = signature
            if not node_labels:
                continue
            left = str(node_labels[0])
            right = str(node_labels[-1])
            for edge_label in edge_labels:
                raw[(left, right, str(edge_label))] += support
    if not raw:
        return {}
    max_count = max(raw.values())
    return {key: count / max_count for key, count in raw.items()}


def _skeleton(pattern: GraphPattern, sampled_support: int, pattern_id: int) -> FrequentPattern:
    pattern.pattern_id = pattern_id
    return FrequentPattern(pattern=pattern, instances=[], sampled=True, total_single_support=sampled_support, total_multi_support=sampled_support)


def _edge_ok(graph: DataGraph, edge: Edge, pattern_edge: PatternEdge) -> bool:
    if edge.label != pattern_edge.label:
        return False
    return graph.vertices[edge.src].label == "Protein" and graph.vertices[edge.dst].label == "Protein"


def _candidate_edges(graph: DataGraph, pattern_edge: PatternEdge, mapping: Dict[int, int]) -> Iterable[Edge]:
    src = mapping.get(pattern_edge.src)
    dst = mapping.get(pattern_edge.dst)
    if src is not None:
        candidates = graph.out_neighbors(src)
    elif dst is not None:
        candidates = graph.in_neighbors(dst)
    else:
        candidates = graph.all_edges()
    for edge in candidates:
        if not _edge_ok(graph, edge, pattern_edge):
            continue
        if src is not None and edge.src != src:
            continue
        if dst is not None and edge.dst != dst:
            continue
        yield edge


def _materialize_fast(graph: DataGraph, pattern: GraphPattern, max_multi_support: int) -> List[GraphInstance]:
    """Edge-driven matcher for small injected directed motifs."""

    if pattern.edge_count() not in (2, 3):
        return []
    results: List[GraphInstance] = []
    ordered_edges = sorted(enumerate(pattern.edges), key=lambda item: item[0])

    def dfs(pos: int, mapping: Dict[int, int], edge_bindings: Dict[int, int]) -> bool:
        if len(results) >= max_multi_support:
            return True
        if pos >= len(ordered_edges):
            edge_ids = tuple(sorted((graph.edges_by_id[eid].src, graph.edges_by_id[eid].dst, graph.edges_by_id[eid].label) for eid in edge_bindings.values()))
            results.append(GraphInstance(node_map=dict(mapping), edge_ids=edge_ids, pivot=mapping.get(0), edge_bindings=dict(edge_bindings)))
            return len(results) >= max_multi_support
        edge_index, pattern_edge = ordered_edges[pos]
        for edge in _candidate_edges(graph, pattern_edge, mapping):
            if edge.edge_id in edge_bindings.values():
                continue
            next_mapping = dict(mapping)
            if pattern_edge.src in next_mapping and next_mapping[pattern_edge.src] != edge.src:
                continue
            if pattern_edge.dst in next_mapping and next_mapping[pattern_edge.dst] != edge.dst:
                continue
            if pattern_edge.src not in next_mapping and edge.src in next_mapping.values():
                continue
            if pattern_edge.dst not in next_mapping and edge.dst in next_mapping.values():
                continue
            next_mapping[pattern_edge.src] = edge.src
            next_mapping[pattern_edge.dst] = edge.dst
            next_bindings = dict(edge_bindings)
            next_bindings[edge_index] = edge.edge_id
            if dfs(pos + 1, next_mapping, next_bindings):
                return True
        return False

    dfs(0, {}, {})
    return results


def build_directed_frequent_patterns(
    graph: DataGraph,
    frequent_patterns: Iterable[Tuple[PatternSignature, int]],
    edge_label: str = "candidate_interaction",
    min_support: int = 5,
    max_multi_support: int = 10000,
    start_pattern_id: int = 100000,
    include_edge: bool = False,
    materialize_instances: bool = False,
) -> List[FrequentPattern]:
    """Build directed patterns from sampled frequent structures.

    When `materialize_instances` is True, instances are enumerated with a fast
    edge-driven matcher; otherwise support is sampled graph support metadata.
    """

    support_by_kind = {}
    for signature, sampled_support in frequent_patterns:
        if signature:
            support_by_kind[signature[0]] = max(support_by_kind.get(signature[0], 0), sampled_support)

    result: List[FrequentPattern] = []
    seen = set()
    next_id = start_pattern_id

    def add(pattern: GraphPattern, kind: str) -> None:
        nonlocal next_id
        sampled_support = support_by_kind.get(kind, 0)
        if sampled_support < min_support:
            return
        code = pattern.canonical_code()
        if code in seen:
            return
        seen.add(code)
        if materialize_instances:
            instances = _materialize_fast(graph, pattern, max_multi_support)
            if len(instances) < min_support:
                return
            pattern.pattern_id = next_id
            result.append(FrequentPattern(
                pattern=pattern,
                instances=instances,
                sampled=True,
                total_single_support=len({instance.pivot for instance in instances if instance.pivot is not None}) or len(instances),
                total_multi_support=len(instances),
            ))
        else:
            result.append(_skeleton(pattern, sampled_support, next_id))
        next_id += 1

    if include_edge and "edge" in support_by_kind:
        add(GraphPattern(node_labels=["Protein", "Protein"], edges=[PatternEdge(0, 1, edge_label)]), "edge")
        add(GraphPattern(node_labels=["Protein", "Protein"], edges=[PatternEdge(1, 0, edge_label)]), "edge")

    if "path2" in support_by_kind:
        add(GraphPattern(node_labels=["Protein", "Protein", "Protein"], edges=[PatternEdge(0, 1, edge_label), PatternEdge(0, 2, edge_label)]), "path2")
        add(GraphPattern(node_labels=["Protein", "Protein", "Protein"], edges=[PatternEdge(1, 0, edge_label), PatternEdge(2, 0, edge_label)]), "path2")
        add(GraphPattern(node_labels=["Protein", "Protein", "Protein"], edges=[PatternEdge(0, 1, edge_label), PatternEdge(2, 0, edge_label)]), "path2")
        add(GraphPattern(node_labels=["Protein", "Protein", "Protein"], edges=[PatternEdge(1, 0, edge_label), PatternEdge(0, 2, edge_label)]), "path2")

    if "triangle" in support_by_kind:
        add(GraphPattern(node_labels=["Protein", "Protein", "Protein"], edges=[PatternEdge(0, 1, edge_label), PatternEdge(1, 2, edge_label), PatternEdge(2, 0, edge_label)]), "triangle")
        add(GraphPattern(node_labels=["Protein", "Protein", "Protein"], edges=[PatternEdge(0, 1, edge_label), PatternEdge(0, 2, edge_label), PatternEdge(1, 2, edge_label)]), "triangle")
        add(GraphPattern(node_labels=["Protein", "Protein", "Protein"], edges=[PatternEdge(1, 0, edge_label), PatternEdge(2, 0, edge_label), PatternEdge(2, 1, edge_label)]), "triangle")

    result.sort(key=lambda item: (item.pattern.edge_count(), item.single_support(), item.multi_support()), reverse=True)
    return result
