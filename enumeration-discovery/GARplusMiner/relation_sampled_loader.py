from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from graph_types import DataGraph, FrequentPattern, GraphInstance, GraphPattern, Vertex
from ppi_loader import _assign_degree_features, _edge_attrs_from_row, _merge_vertex, _normalize_edge_label, _normalize_key, _normalize_scalar
from sampled_pt_loader import _balance_edges_by_label, _extract_data_and_slices, _iter_sampled_graph_records, _load_torch_object


DISEASE_NODE_OFFSET = 1_000_000_000


@dataclass(frozen=True)
class RelationGraphConfig:
    relation_name: str
    source_label: str
    target_label: str
    source_index_column: str
    target_index_column: str
    default_edge_label: str
    edge_csv_path: str
    source_node_csv_path: Optional[str] = None
    target_node_csv_path: Optional[str] = None
    target_node_offset: int = DISEASE_NODE_OFFSET
    load_node_attributes: bool = False


def _tensor_value(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _node_kind(orig_id: int, cfg: RelationGraphConfig) -> Tuple[str, int]:
    if orig_id >= cfg.target_node_offset:
        return cfg.target_label, orig_id - cfg.target_node_offset
    return cfg.source_label, orig_id


def _load_node_attrs(path: Optional[str], label: str, offset: int = 0) -> Dict[int, Vertex]:
    result: Dict[int, Vertex] = {}
    if not path or not Path(path).exists():
        return result
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_id = _normalize_scalar(row.get("index"))
            if raw_id is None:
                continue
            node_id = int(raw_id) + offset
            attrs = {}
            for column, value in row.items():
                if column == "index":
                    continue
                normalized = _normalize_scalar(value)
                if normalized is not None:
                    attrs[_normalize_key(column)] = normalized
            result[node_id] = Vertex(id=node_id, label=label, attrs=attrs)
    return result


def _load_relation_node_attrs(cfg: RelationGraphConfig) -> Dict[int, Vertex]:
    if not cfg.load_node_attributes:
        return {}
    attrs = _load_node_attrs(cfg.source_node_csv_path, cfg.source_label, offset=0)
    attrs.update(_load_node_attrs(cfg.target_node_csv_path, cfg.target_label, offset=cfg.target_node_offset))
    return attrs


def _load_edge_lookup(path: str, cfg: RelationGraphConfig, force_edge_label: Optional[str]) -> Dict[Tuple[int, int], Tuple[str, Dict[str, object]]]:
    lookup: Dict[Tuple[int, int], Tuple[str, Dict[str, object]]] = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            src = _normalize_scalar(row.get(cfg.source_index_column))
            dst = _normalize_scalar(row.get(cfg.target_index_column))
            if src is None or dst is None:
                continue
            src_id = int(src)
            dst_id = int(dst) + cfg.target_node_offset
            attrs = _edge_attrs_from_row(row)
            attrs.setdefault("source_row_id", row_index)
            attrs.setdefault("interaction_label", str(row.get("interaction_label", "unknown")).strip().lower() or "unknown")
            edge_label = force_edge_label or _normalize_edge_label(row.get("EdgeLabel", cfg.default_edge_label))
            lookup[(src_id, dst_id)] = (edge_label, attrs)
            lookup.setdefault((dst_id, src_id), (edge_label, dict(attrs, direction_role="reverse_lookup")))
    return lookup


def _append_negative_edges(
    vertices: Dict[int, Vertex],
    pending_edges: List[Tuple[int, int, str, Dict[str, object]]],
    node_attrs: Dict[int, Vertex],
    cfg: RelationGraphConfig,
    edge_csv_path: str,
    force_edge_label: Optional[str],
    limit: int,
) -> int:
    if limit <= 0:
        return 0
    existing_pairs = {
        (int(attrs.get("sampled_src_original_id", -1)), int(attrs.get("sampled_dst_original_id", -1)))
        for _, _, _, attrs in pending_edges
    }
    next_node_id = max(vertices.keys(), default=-1) + 1
    added = 0
    with Path(edge_csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            label = str(row.get("interaction_label", "unknown")).strip().lower()
            if label != "negative":
                continue
            src_raw = _normalize_scalar(row.get(cfg.source_index_column))
            dst_raw = _normalize_scalar(row.get(cfg.target_index_column))
            if src_raw is None or dst_raw is None:
                continue
            src_orig = int(src_raw)
            dst_orig = int(dst_raw) + cfg.target_node_offset
            if (src_orig, dst_orig) in existing_pairs:
                continue
            src_node, dst_node = next_node_id, next_node_id + 1
            next_node_id += 2
            for graph_node_id, orig_id in ((src_node, src_orig), (dst_node, dst_orig)):
                label_name, source_id = _node_kind(orig_id, cfg)
                vertex = Vertex(
                    id=graph_node_id,
                    label=label_name,
                    attrs={
                        "source_node_id": source_id,
                        "original_index": orig_id,
                        "augmented_negative_node": "yes",
                    },
                )
                if orig_id in node_attrs:
                    _merge_vertex(vertex, node_attrs[orig_id])
                vertices[graph_node_id] = vertex
            attrs = _edge_attrs_from_row(row)
            attrs.setdefault("source_row_id", row_index)
            attrs.setdefault("interaction_label", "negative")
            attrs.setdefault("sampled_src_original_id", src_orig)
            attrs.setdefault("sampled_dst_original_id", dst_orig)
            attrs.setdefault("augmented_negative_edge", "yes")
            edge_label = force_edge_label or _normalize_edge_label(row.get("EdgeLabel", cfg.default_edge_label))
            pending_edges.append((src_node, dst_node, edge_label, attrs))
            existing_pairs.add((src_orig, dst_orig))
            added += 1
            if added >= limit:
                break
    return added


def load_relation_sampled_pt_graph(
    relation_config: RelationGraphConfig,
    sampled_pt_path: str,
    interaction_path: Optional[str] = None,
    protein_path: Optional[str] = None,
    protein_index_column: str = "index",
    edge_label_column: str = "EdgeLabel",
    default_edge_label: Optional[str] = None,
    keep_sampled_x: bool = True,
    use_original_ids_as_node_ids: bool = False,
    force_edge_label: Optional[str] = None,
    augment_negative_edges: bool = False,
    negative_edge_limit: int = 0,
    interaction_label_column: str = "interaction_label",
    balance_edge_labels: bool = True,
) -> DataGraph:
    edge_csv_path = interaction_path or relation_config.edge_csv_path
    obj = _load_torch_object(sampled_pt_path)
    data, slices = _extract_data_and_slices(obj)
    node_attrs = _load_relation_node_attrs(relation_config)
    edge_lookup = _load_edge_lookup(edge_csv_path, relation_config, force_edge_label)

    vertices: Dict[int, Vertex] = {}
    pending_edges: List[Tuple[int, int, str, Dict[str, object]]] = []
    next_graph_node_id = 0
    sampled_edge_id = 0
    for graph_id, _node_start, orig_ids, graph_x, graph_edges in _iter_sampled_graph_records(data, slices):
        local_to_graph_id: Dict[int, int] = {}
        for local_id, orig_id in enumerate(orig_ids):
            orig_id = int(orig_id)
            graph_node_id = orig_id if use_original_ids_as_node_ids else next_graph_node_id
            if not use_original_ids_as_node_ids:
                next_graph_node_id += 1
            local_to_graph_id[local_id] = graph_node_id
            if graph_node_id not in vertices:
                label_name, source_id = _node_kind(orig_id, relation_config)
                vertex = Vertex(
                    id=graph_node_id,
                    label=label_name,
                    attrs={
                        "original_index": orig_id,
                        "source_node_id": source_id,
                        "sampled_graph_id": graph_id,
                        "sampled_local_id": local_id,
                    },
                )
                if orig_id in node_attrs:
                    _merge_vertex(vertex, node_attrs[orig_id])
                vertices[graph_node_id] = vertex
            if keep_sampled_x and local_id < len(graph_x):
                values = graph_x[local_id]
                if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
                    values = [values]
                for feature_index, value in enumerate(values):
                    vertices[graph_node_id].attrs[f"sampled_x{feature_index}"] = _tensor_value(value)

        for local_src, local_dst in graph_edges:
            if local_src >= len(orig_ids) or local_dst >= len(orig_ids) or local_src == local_dst:
                continue
            graph_src = local_to_graph_id.get(local_src)
            graph_dst = local_to_graph_id.get(local_dst)
            if graph_src is None or graph_dst is None or graph_src == graph_dst:
                continue
            orig_src = int(orig_ids[local_src])
            orig_dst = int(orig_ids[local_dst])
            edge_label, edge_attrs = edge_lookup.get(
                (orig_src, orig_dst),
                (force_edge_label or default_edge_label or relation_config.default_edge_label, {"interaction_label": "unknown"}),
            )
            attrs = dict(edge_attrs)
            attrs.setdefault("sampled_graph_id", graph_id)
            attrs.setdefault("sampled_edge_id", sampled_edge_id)
            attrs.setdefault("sampled_src_local_id", local_src)
            attrs.setdefault("sampled_dst_local_id", local_dst)
            attrs.setdefault("sampled_src_original_id", orig_src)
            attrs.setdefault("sampled_dst_original_id", orig_dst)
            attrs.setdefault(interaction_label_column, "unknown")
            pending_edges.append((graph_src, graph_dst, edge_label, attrs))
            sampled_edge_id += 1

    if augment_negative_edges:
        added = _append_negative_edges(
            vertices,
            pending_edges,
            node_attrs,
            relation_config,
            edge_csv_path,
            force_edge_label,
            negative_edge_limit,
        )
        print(f"[SampledPT/{relation_config.relation_name}] augmented_negative_edges={added}")
    if balance_edge_labels:
        counts = Counter(str(attrs.get(interaction_label_column, "unknown")).strip().lower() for _, _, _, attrs in pending_edges)
        pending_edges = _balance_edges_by_label(pending_edges, label_column=interaction_label_column)
        print(f"[SampledPT/{relation_config.relation_name}] balanced_edge_label_counts={dict(counts)}")

    graph = DataGraph(vertices=vertices)
    for graph_src, graph_dst, edge_label, attrs in pending_edges:
        graph.add_edge(graph_src, graph_dst, edge_label, attrs)
    _assign_degree_features(graph)
    return graph


def build_source_seed_pattern(graph: DataGraph, source_label: str) -> FrequentPattern:
    pattern = GraphPattern(node_labels=[source_label])
    instances = [
        GraphInstance(node_map={0: node_id}, edge_ids=(), pivot=node_id)
        for node_id, vertex in graph.vertices.items()
        if vertex.label == source_label
    ]
    return FrequentPattern(pattern=pattern, instances=instances)
