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
3. falls back to NetworkX's VF2-style matcher if the installed `vf3py` version exposes a different API
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
    """Try several likely `vf3py` APIs and yield pattern-node -> data-node mappings.

    Because `vf3py` versions may differ, this adapter probes a small set of possible
    entry points. If none match, it falls back to NetworkX so mining can continue.
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

    for function_name in ("get_subgraph_isomorphisms", "get_subgraph_monomorphisms"):
        vf3_function = getattr(vf3py, function_name, None)
        if callable(vf3_function):
            attempted.append(f"vf3py.{function_name}(...)")
            result = _try_modern_vf3py_function(vf3_function, data_graph, pattern_graph)
            if result is not None:
                return result

    return _call_networkx_matcher(pattern_graph, data_graph, attempted)






def _try_modern_vf3py_function(vf3_function, data_graph: nx.DiGraph, pattern_graph: nx.DiGraph) -> Optional[Iterator[Dict[int, int]]]:
    """Call modern `vf3py.get_subgraph_*` APIs with several known signatures.

    Newer `vf3py` versions expose functions such as `get_subgraph_isomorphisms`
    instead of a `Matcher` class. The exact argument names vary, so we try the
    common forms and normalize the mapping direction afterward.
    """

    call_attempts = (
        lambda: vf3_function(data_graph, pattern_graph, node_label="label", edge_label="label"),
        lambda: vf3_function(data_graph, pattern_graph, node_attr="label", edge_attr="label"),
        lambda: vf3_function(data_graph, pattern_graph, node_attrs=["label"], edge_attrs=["label"]),
        lambda: vf3_function(data_graph, pattern_graph, node_labels=["label"], edge_labels=["label"]),
        lambda: vf3_function(data_graph, pattern_graph),
        lambda: vf3_function(pattern_graph, data_graph, node_label="label", edge_label="label"),
        lambda: vf3_function(pattern_graph, data_graph),
    )
    for call in call_attempts:
        try:
            result = call()
        except TypeError:
            continue
        except Exception:
            continue
        return _normalize_vf3py_mapping_result(result, data_graph, pattern_graph)
    return None


def _normalize_vf3py_mapping_result(result: object, data_graph: nx.DiGraph, pattern_graph: nx.DiGraph) -> Iterator[Dict[int, int]]:
    """Normalize modern vf3py mappings into pattern-node -> data-node mappings."""

    for mapping in _normalize_match_result(result):
        normalized = _coerce_mapping_dict(mapping)
        if not normalized:
            continue
        keys = set(normalized.keys())
        values = set(normalized.values())
        pattern_nodes = set(pattern_graph.nodes())
        data_nodes = set(data_graph.nodes())
        if pattern_nodes.issubset(keys):
            yield {pattern_node: normalized[pattern_node] for pattern_node in pattern_nodes}
        elif pattern_nodes.issubset(values):
            inverted = {pattern_node: data_node for data_node, pattern_node in normalized.items()}
            yield {pattern_node: inverted[pattern_node] for pattern_node in pattern_nodes}
        elif keys.issubset(data_nodes) and values.issubset(data_nodes):
            # Some vf3py builds return an ordered list/dict of data nodes only.
            ordered_values = list(normalized.values())
            if len(ordered_values) >= len(pattern_nodes):
                yield {pattern_node: ordered_values[index] for index, pattern_node in enumerate(sorted(pattern_nodes))}


def _coerce_mapping_dict(mapping: object) -> Dict[int, int]:
    """Coerce dict/list/tuple mapping outputs to a dictionary when possible."""

    if isinstance(mapping, dict):
        return {int(key): int(value) for key, value in mapping.items()}
    if isinstance(mapping, (list, tuple)):
        if mapping and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in mapping):
            return {int(key): int(value) for key, value in mapping}
        return {index: int(value) for index, value in enumerate(mapping)}
    return {}


def _call_networkx_matcher(pattern_graph: nx.DiGraph, data_graph: nx.DiGraph, attempted: List[str]) -> Iterator[Dict[int, int]]:
    """Fallback matcher used when the installed `vf3py` exposes no supported API.

    NetworkX's `DiGraphMatcher(data, pattern).subgraph_isomorphisms_iter()` returns
    mappings in the opposite direction: data-node -> pattern-node. We invert them
    before returning so the rest of GARplusMiner still receives pattern-node -> data-node.
    """

    attempted_msg = ", ".join(attempted or ["no known vf3py entry points found"])
    if not getattr(_call_networkx_matcher, "_warned", False):
        print(f"[VF3Linux] vf3py API unsupported ({attempted_msg}); falling back to networkx matcher")
        setattr(_call_networkx_matcher, "_warned", True)

    from networkx.algorithms import isomorphism as iso

    matcher = iso.DiGraphMatcher(
        data_graph,
        pattern_graph,
        node_match=lambda data_attrs, pattern_attrs: _node_compatible(pattern_attrs, data_attrs),
        edge_match=lambda data_attrs, pattern_attrs: _edge_compatible(pattern_attrs, data_attrs),
    )
    for data_to_pattern in matcher.subgraph_isomorphisms_iter():
        pattern_to_data = {pattern_node: data_node for data_node, pattern_node in data_to_pattern.items()}
        if len(pattern_to_data) == pattern_graph.number_of_nodes():
            yield pattern_to_data


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




def _match_single_edge_pattern(pattern: GraphPattern, graph: DataGraph, pivot_candidates: Optional[List[int]] = None, limit: Optional[int] = None) -> Optional[List[GraphInstance]]:
    """Fast path for the common 2-node/1-edge pattern.

    VSpawn first needs to validate one-edge expansions. Calling a general VF3
    backend for this case is unnecessary and some vf3py builds disagree about
    NetworkX attribute semantics. Direct edge scanning is exact here and much
    faster.
    """

    if pattern.node_count() != 2 or pattern.edge_count() != 1:
        return None
    pattern_edge = pattern.edges[0]
    src_label = pattern.node_labels[pattern_edge.src]
    dst_label = pattern.node_labels[pattern_edge.dst]
    allowed_pivots = set(pivot_candidates) if pivot_candidates is not None else None
    instances: List[GraphInstance] = []
    for edge in graph.all_edges():
        if edge.label != pattern_edge.label:
            continue
        if graph.vertices[edge.src].label != src_label or graph.vertices[edge.dst].label != dst_label:
            continue
        mapping = {pattern_edge.src: edge.src, pattern_edge.dst: edge.dst}
        pivot = mapping.get(0)
        if allowed_pivots is not None and pivot not in allowed_pivots:
            continue
        instances.append(
            GraphInstance(
                node_map=mapping,
                edge_ids=((edge.src, edge.dst, edge.label),),
                pivot=pivot,
                edge_bindings={0: edge.edge_id},
            )
        )
        if limit is not None and len(instances) >= limit:
            break
    return instances


def find_matches_with_limit(pattern: GraphPattern, graph: DataGraph, limit: Optional[int] = None, pivot_candidates: Optional[List[int]] = None) -> List[GraphInstance]:
    """Enumerate pattern matches using `vf3py` in a Linux environment."""

    fast_matches = _match_single_edge_pattern(pattern, graph, pivot_candidates=pivot_candidates, limit=limit)
    if fast_matches is not None:
        return fast_matches

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
