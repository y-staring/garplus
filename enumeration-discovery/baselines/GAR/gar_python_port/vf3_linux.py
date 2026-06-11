from __future__ import annotations

"""Linux-oriented VF3 adapter that keeps the same public interface as `vf3_like.py`.

Purpose
-------
This module is intentionally added as a *new* matcher implementation and does not
replace the existing pure-Python matcher in `vf3_like.py`.

Expected environment
--------------------
- Linux / WSL
- `networkx` installed
- `vf3py` installed

Public API
----------
The exported functions mirror the current matcher module so the rest of the pipeline can
switch implementations with minimal changes:
- `find_matches(pattern, graph)`
- `find_matches_with_limit(pattern, graph, limit=None, pivot_candidates=None)`

Notes
-----
`vf3py` is a thin wrapper around a VF3 implementation, but its Python API may differ by
version. This adapter therefore:
1. converts our internal graph structures to `networkx.DiGraph`
2. tries a few likely vf3py entry points via introspection
3. falls back to a clear error if the installed `vf3py` version exposes a different API
"""

from typing import Dict, Iterable, Iterator, List, Optional

from graph_types import DataGraph, GraphInstance, GraphPattern

try:
    import networkx as nx
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "vf3_linux.py requires `networkx`. Install it in the Linux environment first."
    ) from exc

try:
    import vf3py  # type: ignore
except ImportError as exc:  # pragma: no cover - environment dependent
    raise RuntimeError(
        "vf3_linux.py requires `vf3py`. Install it in the Linux environment first."
    ) from exc


def _pattern_to_networkx(pattern: GraphPattern) -> nx.DiGraph:
    """Convert an internal `GraphPattern` into a NetworkX directed graph."""

    graph = nx.DiGraph()
    for pattern_node_id, label in enumerate(pattern.node_labels):
        graph.add_node(pattern_node_id, label=label)
    for edge_index, edge in enumerate(pattern.edges):
        graph.add_edge(edge.src, edge.dst, label=edge.label, pattern_edge_index=edge_index)
    return graph


def _data_graph_to_networkx(data_graph: DataGraph, allowed_pivots: Optional[set[int]] = None) -> nx.DiGraph:
    """Convert the large `DataGraph` into a NetworkX directed graph.

    If `allowed_pivots` is provided, it is stored as a node attribute so a custom node
    feasibility function can restrict pattern node 0.
    """

    graph = nx.DiGraph()
    for node_id, vertex in data_graph.vertices.items():
        graph.add_node(
            node_id,
            label=vertex.label,
            allowed_as_pivot=(allowed_pivots is None or node_id in allowed_pivots),
            **vertex.attrs,
        )
    for edge in data_graph.all_edges():
        graph.add_edge(edge.src, edge.dst, label=edge.label, edge_id=edge.edge_id, **edge.attrs)
    return graph


def _node_compatible(pattern_node_attrs: Dict[str, object], data_node_attrs: Dict[str, object]) -> bool:
    """Basic node-attribute compatibility used for subgraph matching.

    Current policy:
    - labels must match
    - if the pattern node is the pivot node 0 and pivot filtering is enabled,
      the target data node must be marked `allowed_as_pivot`
    """

    if pattern_node_attrs.get("label") != data_node_attrs.get("label"):
        return False
    pattern_node_id = pattern_node_attrs.get("pattern_node_id")
    if pattern_node_id == 0 and not data_node_attrs.get("allowed_as_pivot", True):
        return False
    return True


def _edge_compatible(pattern_edge_attrs: Dict[str, object], data_edge_attrs: Dict[str, object]) -> bool:
    """Basic edge compatibility: edge labels must match."""

    return pattern_edge_attrs.get("label") == data_edge_attrs.get("label")


def _prepare_pattern_graph(pattern: GraphPattern) -> nx.DiGraph:
    """Attach helper attributes expected by the compatibility callbacks."""

    graph = _pattern_to_networkx(pattern)
    for node_id in graph.nodes:
        graph.nodes[node_id]["pattern_node_id"] = node_id
    return graph


def _call_vf3py_matcher(pattern_graph: nx.DiGraph, data_graph: nx.DiGraph) -> Iterator[Dict[int, int]]:
    """Try several likely `vf3py` APIs and yield node mappings.

    Because `vf3py` versions may differ, this adapter probes a small set of possible
    entry points. If none match, we raise a detailed error.
    """

    attempted: List[str] = []

    matcher_cls = getattr(vf3py, "Matcher", None)
    if matcher_cls is not None:
        attempted.append("vf3py.Matcher(...).match()")
        matcher = matcher_cls(
            pattern_graph,
            data_graph,
            node_match=_node_compatible,
            edge_match=_edge_compatible,
        )
        if hasattr(matcher, "match"):
            result = matcher.match()
            return _normalize_match_result(result)
        if hasattr(matcher, "isomorphisms_iter"):
            return _normalize_match_result(matcher.isomorphisms_iter())

    graph_match_fn = getattr(vf3py, "graph_match", None)
    if callable(graph_match_fn):
        attempted.append("vf3py.graph_match(...)" )
        return _normalize_match_result(
            graph_match_fn(
                pattern_graph,
                data_graph,
                node_match=_node_compatible,
                edge_match=_edge_compatible,
            )
        )

    match_fn = getattr(vf3py, "match", None)
    if callable(match_fn):
        attempted.append("vf3py.match(...)" )
        return _normalize_match_result(
            match_fn(
                pattern_graph,
                data_graph,
                node_match=_node_compatible,
                edge_match=_edge_compatible,
            )
        )

    find_fn = getattr(vf3py, "find_isomorphisms", None)
    if callable(find_fn):
        attempted.append("vf3py.find_isomorphisms(...)" )
        return _normalize_match_result(
            find_fn(
                pattern_graph,
                data_graph,
                node_match=_node_compatible,
                edge_match=_edge_compatible,
            )
        )

    raise RuntimeError(
        "Unsupported vf3py API. Tried: "
        + ", ".join(attempted or ["no known entry points found"])
        + ". Inspect the installed vf3py package and adapt `_call_vf3py_matcher()`."
    )


def _normalize_match_result(result: object) -> Iterator[Dict[int, int]]:
    """Normalize different matcher return types into an iterator of node mappings."""

    if result is None:
        return iter(())
    if isinstance(result, dict):
        return iter((result,))
    if isinstance(result, list):
        return iter(result)
    if hasattr(result, "__iter__"):
        return iter(result)  # type: ignore[arg-type]
    return iter(())


def _materialize_instance(pattern: GraphPattern, graph: DataGraph, mapping: Dict[int, int]) -> GraphInstance:
    """Convert a node mapping returned by VF3 into our internal `GraphInstance`."""

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
    """Enumerate pattern matches using `vf3py` in a Linux environment."""

    allowed_pivots = set(pivot_candidates) if pivot_candidates is not None else None
    pattern_graph = _prepare_pattern_graph(pattern)
    data_graph = _data_graph_to_networkx(graph, allowed_pivots=allowed_pivots)

    instances: List[GraphInstance] = []
    for raw_mapping in _call_vf3py_matcher(pattern_graph, data_graph):
        mapping = dict(raw_mapping)
        instances.append(_materialize_instance(pattern, graph, mapping))
        if limit is not None and len(instances) >= limit:
            break
    return instances


def find_matches(pattern: GraphPattern, graph: DataGraph) -> List[GraphInstance]:
    """Unbounded version of `find_matches_with_limit`."""

    return find_matches_with_limit(pattern, graph, limit=None)
