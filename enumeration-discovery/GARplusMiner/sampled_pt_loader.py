from __future__ import annotations

"""Load a sampled PyG `.pt` graph as the GARplusMiner mining graph.

Expected sampled file shape, based on `sampling/pick_patterns.py` output:

    (
        Data(
            x=[num_sampled_nodes, ...],
            edge_index=[2, num_sampled_edges],
            orig_node_ids=[num_sampled_nodes],
            ...
        ),
        slices_or_meta
    )

The sampled graph uses local PyG node ids in `edge_index`. This loader maps them
back to original protein ids through `orig_node_ids`, then builds the same
`DataGraph` type used by the CSV loader.

Important:
- The sampled `.pt` graph usually has topology only, not all original edge attrs.
- Therefore we optionally look up each sampled edge in `protein_protein_signed.csv`
  by `(index_A, index_B)` and copy the original CSV row columns as edge attrs.
- Protein attributes are optionally merged from `protein.csv` by original index.
"""

import csv
from collections import Counter
from typing import Dict, Iterable, Optional, Tuple

from graph_types import DataGraph, FrequentPattern, GraphInstance, GraphPattern, Vertex
from ppi_loader import _assign_degree_features, _edge_attrs_from_row, _merge_vertex, _normalize_edge_label, _normalize_key, _normalize_scalar, _protein_vertex_from_row


def _tensor_to_list(value) -> list:
    """Convert torch tensors / Python lists to plain Python lists."""

    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _load_torch_object(path: str):
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Loading sampled .pt graphs requires torch in this environment.") from exc
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_data_object(obj):
    """Support both `Data` and `(Data, slices)` style saved objects."""

    if isinstance(obj, tuple) and obj:
        return obj[0]
    return obj


def _sampled_orig_node_ids(data) -> Dict[int, int]:
    """Return `local_node_id -> original_protein_index`."""

    if hasattr(data, "orig_node_ids"):
        orig_ids = _tensor_to_list(data.orig_node_ids)
        return {local_id: int(orig_id) for local_id, orig_id in enumerate(orig_ids)}
    node_count = int(getattr(data, "num_nodes", 0) or len(getattr(data, "x", [])))
    return {local_id: local_id for local_id in range(node_count)}


def _load_protein_attrs(protein_path: Optional[str], index_column: str = "index") -> Dict[int, Vertex]:
    """Load protein.csv attributes by original protein index."""

    result: Dict[int, Vertex] = {}
    if not protein_path:
        return result
    with open(protein_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            vertex = _protein_vertex_from_row(row, index_column=index_column)
            if vertex is not None:
                result[vertex.id] = vertex
    return result


def _interaction_node_id(row: Dict[str, str], suffix: str) -> Optional[int]:
    """Read Interactor A/B id from the signed interaction CSV."""

    raw = row.get(f"index_{suffix}") or row.get(f"Entrez Gene Interactor {suffix}")
    normalized = _normalize_scalar(raw)
    return int(normalized) if normalized is not None else None


def _load_interaction_lookup(path: Optional[str], edge_label_column: str = "Experimental System") -> Dict[Tuple[int, int], Tuple[str, Dict[str, object]]]:
    """Build `(src_original_id, dst_original_id) -> (edge_label, edge_attrs)` lookup."""

    lookup: Dict[Tuple[int, int], Tuple[str, Dict[str, object]]] = {}
    if not path:
        return lookup
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            left = _interaction_node_id(row, "A")
            right = _interaction_node_id(row, "B")
            if left is None or right is None:
                continue
            edge_label = _normalize_edge_label(row.get(edge_label_column, "sampled_interaction"))
            edge_attrs = _edge_attrs_from_row(row)
            edge_attrs["source_row_id"] = row_index
            lookup[(left, right)] = (edge_label, edge_attrs)
            lookup.setdefault((right, left), (edge_label, dict(edge_attrs, direction_role="reverse_lookup")))
    return lookup


def _assign_sampled_node_features(graph: DataGraph, data, local_to_orig: Dict[int, int]) -> None:
    """Expose numeric PyG `x` columns as optional node attributes for debugging/mining."""

    if not hasattr(data, "x"):
        return
    rows = _tensor_to_list(data.x)
    for local_id, values in enumerate(rows):
        orig_id = local_to_orig.get(local_id)
        if orig_id not in graph.vertices:
            continue
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            values = [values]
        for feature_index, value in enumerate(values):
            graph.vertices[orig_id].attrs[f"sampled_x{feature_index}"] = value


def load_sampled_pt_graph(
    sampled_pt_path: str,
    interaction_path: Optional[str] = None,
    protein_path: Optional[str] = None,
    protein_index_column: str = "index",
    edge_label_column: str = "Experimental System",
    default_edge_label: str = "sampled_interaction",
    keep_sampled_x: bool = True,
) -> DataGraph:
    """Load sampled PyG graph and enrich it from the original CSV files."""

    obj = _load_torch_object(sampled_pt_path)
    data = _extract_data_object(obj)
    if not hasattr(data, "edge_index"):
        raise ValueError(f"{sampled_pt_path} does not look like a PyG Data object with edge_index")

    local_to_orig = _sampled_orig_node_ids(data)
    protein_attrs = _load_protein_attrs(protein_path, index_column=protein_index_column)
    interaction_lookup = _load_interaction_lookup(interaction_path, edge_label_column=edge_label_column)

    vertices: Dict[int, Vertex] = {}
    for orig_id in local_to_orig.values():
        vertex = Vertex(id=orig_id, label="Protein", attrs={"original_index": orig_id})
        if orig_id in protein_attrs:
            _merge_vertex(vertex, protein_attrs[orig_id])
        vertices[orig_id] = vertex

    graph = DataGraph(vertices=vertices)
    edge_index = _tensor_to_list(data.edge_index)
    src_list, dst_list = edge_index[0], edge_index[1]
    seen_edges = set()
    for sampled_edge_id, (local_src, local_dst) in enumerate(zip(src_list, dst_list)):
        orig_src = local_to_orig.get(int(local_src))
        orig_dst = local_to_orig.get(int(local_dst))
        if orig_src is None or orig_dst is None or orig_src == orig_dst:
            continue
        edge_key = (orig_src, orig_dst, sampled_edge_id)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        edge_label, edge_attrs = interaction_lookup.get((orig_src, orig_dst), (default_edge_label, {}))
        attrs = dict(edge_attrs)
        attrs.setdefault("sampled_edge_id", sampled_edge_id)
        attrs.setdefault("sampled_src_local_id", int(local_src))
        attrs.setdefault("sampled_dst_local_id", int(local_dst))
        attrs.setdefault("sampled_src_original_id", orig_src)
        attrs.setdefault("sampled_dst_original_id", orig_dst)
        if "interaction_label" not in attrs:
            attrs["interaction_label"] = "unknown"
        graph.add_edge(orig_src, orig_dst, edge_label, attrs)

    if keep_sampled_x:
        _assign_sampled_node_features(graph, data, local_to_orig)
    _assign_degree_features(graph)
    return graph


def build_sampled_seed_pattern(graph: DataGraph, node_label: str = "Protein") -> FrequentPattern:
    """Create a 1-node seed pattern for the sampled graph."""

    pattern = GraphPattern(node_labels=[node_label])
    instances = [GraphInstance(node_map={0: node_id}, edge_ids=(), pivot=node_id) for node_id, vertex in graph.vertices.items() if vertex.label == node_label]
    return FrequentPattern(pattern=pattern, instances=instances)
