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
- By default the DataGraph node id is the sampled local id, not the original id.
  This is important for batched PyG sampled subgraphs: the same original protein can
  appear in many sampled ego-graphs and should not be collapsed unless explicitly requested.
"""

import csv
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

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


def _extract_data_and_slices(obj):
    """Support both `Data` and `(Data, slices)` style saved objects."""

    if isinstance(obj, tuple) and len(obj) >= 2:
        return obj[0], obj[1]
    return obj, None




def _slice_value(slices, key: str) -> Optional[List[int]]:
    """Return PyG InMemoryDataset slices for one key as a Python list."""

    if slices is None:
        return None
    if key not in slices:
        return None
    return [int(value) for value in _tensor_to_list(slices[key])]


def _num_graphs_from_slices(slices) -> int:
    """Infer number of collated sampled subgraphs."""

    for key in ("x", "orig_node_ids", "edge_index"):
        values = _slice_value(slices, key)
        if values is not None and len(values) >= 2:
            return len(values) - 1
    return 1


def _iter_sampled_graph_records(data, slices):
    """Yield per-subgraph node/edge records from a PyG collated `(data, slices)` object.

    `pick_patterns.py` saves selected subgraphs via `raw_dataset.collate(selected_graphs)`.
    Reading `data.edge_index` directly treats local subgraph node ids as global ids in some
    PyG versions, which makes almost every node look isolated. This iterator respects the
    `slices` object and reconstructs each sampled graph separately.
    """

    edge_index = _tensor_to_list(data.edge_index)
    x_rows = _tensor_to_list(data.x) if hasattr(data, "x") else []
    orig_rows = _tensor_to_list(data.orig_node_ids) if hasattr(data, "orig_node_ids") else None
    x_slices = _slice_value(slices, "x")
    edge_slices = _slice_value(slices, "edge_index")
    orig_slices = _slice_value(slices, "orig_node_ids")
    graph_count = _num_graphs_from_slices(slices)

    if slices is None or x_slices is None or edge_slices is None:
        node_count = int(getattr(data, "num_nodes", 0) or len(x_rows))
        orig_ids = list(range(node_count)) if orig_rows is None else [int(value) for value in orig_rows]
        edges = [(int(src), int(dst)) for src, dst in zip(edge_index[0], edge_index[1])]
        yield 0, 0, orig_ids, x_rows, edges
        return

    for graph_id in range(graph_count):
        node_start = x_slices[graph_id]
        node_end = x_slices[graph_id + 1]
        edge_start = edge_slices[graph_id]
        edge_end = edge_slices[graph_id + 1]
        node_count = node_end - node_start
        graph_x = x_rows[node_start:node_end]
        if orig_rows is not None and orig_slices is not None:
            orig_start = orig_slices[graph_id]
            orig_end = orig_slices[graph_id + 1]
            orig_ids = [int(value) for value in orig_rows[orig_start:orig_end]]
        else:
            orig_ids = list(range(node_start, node_end))
        raw_edges = [(int(src), int(dst)) for src, dst in zip(edge_index[0][edge_start:edge_end], edge_index[1][edge_start:edge_end])]
        if raw_edges and max(max(src, dst) for src, dst in raw_edges) >= node_count:
            graph_edges = [(src - node_start, dst - node_start) for src, dst in raw_edges]
        else:
            graph_edges = raw_edges
        graph_edges = [(src, dst) for src, dst in graph_edges if 0 <= src < node_count and 0 <= dst < node_count]
        yield graph_id, node_start, orig_ids, graph_x, graph_edges


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


def _load_interaction_lookup(path: Optional[str], edge_label_column: str = "Experimental System", force_edge_label: Optional[str] = None) -> Dict[Tuple[int, int], Tuple[str, Dict[str, object]]]:
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
            edge_label = force_edge_label or _normalize_edge_label(row.get(edge_label_column, "sampled_interaction"))
            edge_attrs = _edge_attrs_from_row(row)
            edge_attrs["source_row_id"] = row_index
            lookup[(left, right)] = (edge_label, edge_attrs)
            lookup.setdefault((right, left), (edge_label, dict(edge_attrs, direction_role="reverse_lookup")))
    return lookup




def _iter_signed_rows(path: Optional[str], label_column: str = "interaction_label"):
    """Yield signed CSV rows with parsed endpoints and normalized edge attrs."""

    if not path:
        return
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            left = _interaction_node_id(row, "A")
            right = _interaction_node_id(row, "B")
            if left is None or right is None or left == right:
                continue
            attrs = _edge_attrs_from_row(row)
            attrs["source_row_id"] = row_index
            label = str(attrs.get(label_column, row.get(label_column, ""))).strip().lower()
            yield row_index, left, right, label, attrs, row


def _ensure_augmented_vertex(vertices: Dict[int, Vertex], orig_id: int, protein_attrs: Dict[int, Vertex], graph_node_id: int, sampled_graph_id: int) -> None:
    """Create one augmented node if it does not already exist."""

    if graph_node_id in vertices:
        return
    vertex = Vertex(
        id=graph_node_id,
        label="Protein",
        attrs={
            "original_index": orig_id,
            "sampled_graph_id": sampled_graph_id,
            "sampled_local_id": graph_node_id,
            "augmented_negative_node": "yes",
        },
    )
    if orig_id in protein_attrs:
        _merge_vertex(vertex, protein_attrs[orig_id])
    vertices[graph_node_id] = vertex


def _append_negative_edges(
    vertices: Dict[int, Vertex],
    pending_edges: List[Tuple[int, int, str, Dict[str, object]]],
    protein_attrs: Dict[int, Vertex],
    interaction_path: Optional[str],
    edge_label_column: str,
    force_edge_label: Optional[str],
    negative_edge_limit: int,
    label_column: str = "interaction_label",
    negative_value: str = "negative",
) -> int:
    """Append negative signed CSV edges into the sampled mining graph.

    This fixes the common case where `ppi_selected.pt` was sampled from an unsigned
    positive graph and therefore contains no negative edges to mine as consequents.
    """

    if negative_edge_limit <= 0:
        return 0
    existing_original_pairs = {
        (int(attrs.get("sampled_src_original_id", -1)), int(attrs.get("sampled_dst_original_id", -1)))
        for _, _, _, attrs in pending_edges
    }
    next_node_id = (max(vertices.keys()) + 1) if vertices else 0
    added = 0
    augmented_graph_id = -1
    for row_index, left, right, label, attrs, row in _iter_signed_rows(interaction_path, label_column=label_column):
        if label != negative_value:
            continue
        if (left, right) in existing_original_pairs:
            continue
        src_node = next_node_id
        dst_node = next_node_id + 1
        next_node_id += 2
        _ensure_augmented_vertex(vertices, left, protein_attrs, src_node, augmented_graph_id)
        _ensure_augmented_vertex(vertices, right, protein_attrs, dst_node, augmented_graph_id)
        edge_label = force_edge_label or _normalize_edge_label(row.get(edge_label_column, "negative_interaction"))
        edge_attrs = dict(attrs)
        edge_attrs.setdefault(label_column, negative_value)
        edge_attrs.setdefault("sampled_graph_id", augmented_graph_id)
        edge_attrs.setdefault("augmented_negative_edge", "yes")
        edge_attrs.setdefault("sampled_src_original_id", left)
        edge_attrs.setdefault("sampled_dst_original_id", right)
        edge_attrs.setdefault("sampled_src_local_id", src_node)
        edge_attrs.setdefault("sampled_dst_local_id", dst_node)
        pending_edges.append((src_node, dst_node, edge_label, edge_attrs))
        existing_original_pairs.add((left, right))
        added += 1
        if added >= negative_edge_limit:
            break
    return added


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




def _balance_edges_by_label(
    pending_edges: List[Tuple[int, int, str, Dict[str, object]]],
    label_column: str = "interaction_label",
    preferred_order: Tuple[str, ...] = ("negative", "positive", "unknown"),
) -> List[Tuple[int, int, str, Dict[str, object]]]:
    """Interleave edges by target label so early matching limits see rare labels.

    `MAX_MULTI_SUPPORT` stops matching after the first N instances. Since negative
    edges are often appended after the sampled positive graph, they may never be
    reached. This reorders edges without changing graph content.
    """

    buckets: Dict[str, List[Tuple[int, int, str, Dict[str, object]]]] = {}
    for item in pending_edges:
        label = str(item[3].get(label_column, "unknown")).strip().lower()
        buckets.setdefault(label, []).append(item)
    ordered_labels = [label for label in preferred_order if label in buckets]
    ordered_labels.extend(sorted(label for label in buckets if label not in set(ordered_labels)))
    balanced: List[Tuple[int, int, str, Dict[str, object]]] = []
    index = 0
    while True:
        added = False
        for label in ordered_labels:
            bucket = buckets[label]
            if index < len(bucket):
                balanced.append(bucket[index])
                added = True
        if not added:
            break
        index += 1
    return balanced


def load_sampled_pt_graph(
    sampled_pt_path: str,
    interaction_path: Optional[str] = None,
    protein_path: Optional[str] = None,
    protein_index_column: str = "index",
    edge_label_column: str = "Experimental System",
    default_edge_label: str = "sampled_interaction",
    keep_sampled_x: bool = True,
    use_original_ids_as_node_ids: bool = False,
    force_edge_label: Optional[str] = None,
    augment_negative_edges: bool = False,
    negative_edge_limit: int = 0,
    interaction_label_column: str = "interaction_label",
    balance_edge_labels: bool = True,
) -> DataGraph:
    """Load sampled PyG graph and enrich it from the original CSV files."""

    obj = _load_torch_object(sampled_pt_path)
    data, slices = _extract_data_and_slices(obj)
    if not hasattr(data, "edge_index"):
        raise ValueError(f"{sampled_pt_path} does not look like a PyG Data object with edge_index")

    protein_attrs = _load_protein_attrs(protein_path, index_column=protein_index_column)
    interaction_lookup = _load_interaction_lookup(interaction_path, edge_label_column=edge_label_column, force_edge_label=force_edge_label)

    vertices: Dict[int, Vertex] = {}
    pending_edges = []
    next_graph_node_id = 0
    sampled_edge_id = 0
    for graph_id, _node_start, orig_ids, graph_x, graph_edges in _iter_sampled_graph_records(data, slices):
        local_to_graph_id: Dict[int, int] = {}
        for local_id, orig_id in enumerate(orig_ids):
            graph_node_id = orig_id if use_original_ids_as_node_ids else next_graph_node_id
            next_graph_node_id += 1 if not use_original_ids_as_node_ids else 0
            local_to_graph_id[local_id] = graph_node_id
            if graph_node_id not in vertices:
                vertex = Vertex(
                    id=graph_node_id,
                    label="Protein",
                    attrs={
                        "original_index": orig_id,
                        "sampled_graph_id": graph_id,
                        "sampled_local_id": local_id,
                    },
                )
                if orig_id in protein_attrs:
                    _merge_vertex(vertex, protein_attrs[orig_id])
                vertices[graph_node_id] = vertex
            if keep_sampled_x and local_id < len(graph_x):
                values = graph_x[local_id]
                if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
                    values = [values]
                for feature_index, value in enumerate(values):
                    vertices[graph_node_id].attrs[f"sampled_x{feature_index}"] = value

        for local_src, local_dst in graph_edges:
            if local_src == local_dst:
                continue
            if local_src >= len(orig_ids) or local_dst >= len(orig_ids):
                continue
            graph_src = local_to_graph_id.get(local_src)
            graph_dst = local_to_graph_id.get(local_dst)
            if graph_src is None or graph_dst is None or graph_src == graph_dst:
                continue
            orig_src = int(orig_ids[local_src])
            orig_dst = int(orig_ids[local_dst])
            edge_label, edge_attrs = interaction_lookup.get((orig_src, orig_dst), (force_edge_label or default_edge_label, {}))
            attrs = dict(edge_attrs)
            attrs.setdefault("sampled_graph_id", graph_id)
            attrs.setdefault("sampled_edge_id", sampled_edge_id)
            attrs.setdefault("sampled_src_local_id", local_src)
            attrs.setdefault("sampled_dst_local_id", local_dst)
            attrs.setdefault("sampled_src_original_id", orig_src)
            attrs.setdefault("sampled_dst_original_id", orig_dst)
            if "interaction_label" not in attrs:
                attrs["interaction_label"] = "unknown"
            pending_edges.append((graph_src, graph_dst, edge_label, attrs))
            sampled_edge_id += 1

    if augment_negative_edges:
        added_negative = _append_negative_edges(
            vertices=vertices,
            pending_edges=pending_edges,
            protein_attrs=protein_attrs,
            interaction_path=interaction_path,
            edge_label_column=edge_label_column,
            force_edge_label=force_edge_label,
            negative_edge_limit=negative_edge_limit,
            label_column=interaction_label_column,
            negative_value="negative",
        )
        print(f"[SampledPT] augmented_negative_edges={added_negative}")

    if balance_edge_labels:
        before_counts = Counter(str(attrs.get(interaction_label_column, "unknown")).strip().lower() for _, _, _, attrs in pending_edges)
        pending_edges = _balance_edges_by_label(pending_edges, label_column=interaction_label_column)
        print(f"[SampledPT] balanced_edge_label_counts={dict(before_counts)}")

    graph = DataGraph(vertices=vertices)
    for graph_src, graph_dst, edge_label, attrs in pending_edges:
        graph.add_edge(graph_src, graph_dst, edge_label, attrs)
    _assign_degree_features(graph)
    return graph


def build_sampled_seed_pattern(graph: DataGraph, node_label: str = "Protein") -> FrequentPattern:
    """Create a 1-node seed pattern for the sampled graph."""

    pattern = GraphPattern(node_labels=[node_label])
    instances = [GraphInstance(node_map={0: node_id}, edge_ids=(), pivot=node_id) for node_id, vertex in graph.vertices.items() if vertex.label == node_label]
    return FrequentPattern(pattern=pattern, instances=instances)
