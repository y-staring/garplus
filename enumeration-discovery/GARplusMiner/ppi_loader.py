from __future__ import annotations

"""Load PPI CSV files into the Python GAR graph format.

This loader merges two data sources:
1. `protein_protein.csv`: the interaction / edge table
2. optional `protein.csv`: the protein annotation table

Important behavior aligned with the original Go code:
- the interaction table drives graph construction
- `protein.csv` only enriches vertices that already appear in interactions
- isolated proteins from `protein.csv` are not added to the graph
"""

import csv
from collections import Counter
from typing import Dict, Optional

from graph_types import DataGraph, FrequentPattern, GraphInstance, GraphPattern, Vertex


def _normalize_key(raw: str) -> str:
    """Normalize column names into simple literal keys."""

    return raw.strip().lower().replace("#", "num_").replace("-", "_").replace(" ", "_").replace("/", "_")


def _normalize_scalar(value: object) -> object:
    """Normalize empty / placeholder values to `None`."""

    if value is None:
        return None
    text = str(value).strip()
    if text in ("", "-", "NA", "N/A"):
        return None
    return text


def _merge_attr(existing: object, incoming: object) -> object:
    """Merge repeated attributes when the same protein appears in multiple rows."""

    if existing is None:
        return incoming
    if incoming is None:
        return existing
    if existing == incoming:
        return existing
    if not isinstance(existing, list):
        values = [existing]
    else:
        values = list(existing)
    if incoming not in values:
        values.append(incoming)
    return values


def _normalize_edge_label(raw: str) -> str:
    """Turn the chosen interaction field into a graph edge label."""

    text = str(_normalize_scalar(raw) or "interacts_with")
    return text.replace(" ", "_").replace("/", "_")


def _node_id(row: Dict[str, str], suffix: str) -> int:
    """Read the interaction-table node id for Interactor A / B."""

    preferred = row.get(f"index_{suffix}") or row.get(f"Entrez Gene Interactor {suffix}")
    if preferred is None or preferred == "":
        raise ValueError(f"missing node id for Interactor {suffix}")
    return int(preferred)


def _extract_role_attrs(row: Dict[str, str], suffix: str) -> Dict[str, object]:
    """Extract only the columns that belong to one interactor side."""

    attrs: Dict[str, object] = {}
    postfix = f" {suffix}"
    for column, value in row.items():
        if not column.endswith(postfix):
            continue
        normalized_value = _normalize_scalar(value)
        if normalized_value is None:
            continue
        key = _normalize_key(column[:-len(postfix)])
        attrs[key] = normalized_value
    return attrs


def _vertex_from_row(row: Dict[str, str], suffix: str) -> Vertex:
    """Build one `Protein` vertex from one interaction-table side."""

    node_id = _node_id(row, suffix)
    return Vertex(id=node_id, label="Protein", attrs=_extract_role_attrs(row, f"Interactor {suffix}"))


def _merge_vertex(existing: Vertex, incoming: Vertex) -> None:
    for key, value in incoming.attrs.items():
        existing.attrs[key] = _merge_attr(existing.attrs.get(key), value)


def _edge_attrs_from_row(row: Dict[str, str]) -> Dict[str, object]:
    """Extract the non-interactor columns as edge attributes."""

    attrs: Dict[str, object] = {}
    for column, value in row.items():
        if column.endswith(" Interactor A") or column.endswith(" Interactor B"):
            continue
        if column.startswith("index_"):
            continue
        normalized_value = _normalize_scalar(value)
        if normalized_value is None:
            continue
        key = _normalize_key(column)
        attrs[key] = normalized_value
    return attrs


def _protein_vertex_from_row(row: Dict[str, str], index_column: str = "index") -> Optional[Vertex]:
    """Convert one `protein.csv` row into a vertex-shaped annotation container."""

    raw_id = _normalize_scalar(row.get(index_column))
    if raw_id is None:
        return None
    node_id = int(raw_id)
    attrs: Dict[str, object] = {}
    for column, value in row.items():
        if column == index_column:
            continue
        normalized_value = _normalize_scalar(value)
        if normalized_value is None:
            continue
        attrs[_normalize_key(column)] = normalized_value
    return Vertex(id=node_id, label="Protein", attrs=attrs)


def _merge_protein_annotations(vertices: Dict[int, Vertex], protein_path: str, index_column: str = "index") -> None:
    """Merge `protein.csv` attributes into already-known interaction vertices only."""

    with open(protein_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            protein_vertex = _protein_vertex_from_row(row, index_column=index_column)
            if protein_vertex is None:
                continue
            existing = vertices.get(protein_vertex.id)
            if existing is not None:
                _merge_vertex(existing, protein_vertex)


def _assign_degree_features(graph: DataGraph) -> None:
    """Derive simple node attributes so rule mining has usable targets/features."""

    degrees = Counter()
    for edge in graph.all_edges():
        degrees[edge.src] += 1
        degrees[edge.dst] += 1
    values = sorted(degrees.values())
    low_cut = values[max(0, len(values) // 3 - 1)] if values else 0
    high_cut = values[max(0, (2 * len(values)) // 3 - 1)] if values else 0
    for node_id, vertex in graph.vertices.items():
        degree = degrees.get(node_id, 0)
        if degree <= low_cut:
            bucket = "low"
        elif degree <= high_cut:
            bucket = "medium"
        else:
            bucket = "high"
        vertex.attrs["degree"] = degree
        vertex.attrs["degree_bucket"] = bucket
        vertex.attrs["high_degree"] = "yes" if bucket == "high" else "no"


def load_ppi_csv(path: str, max_rows: Optional[int] = None, undirected: bool = True, edge_label_column: str = "Experimental System", protein_path: Optional[str] = None, protein_index_column: str = "index", force_edge_label: Optional[str] = None) -> DataGraph:
    """Load the interaction CSV and optionally enrich existing vertices from `protein.csv`."""

    vertices: Dict[int, Vertex] = {}
    graph = DataGraph(vertices=vertices)
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if max_rows is not None and index >= max_rows:
                break
            left = _vertex_from_row(row, "A")
            right = _vertex_from_row(row, "B")
            if left.id in vertices:
                _merge_vertex(vertices[left.id], left)
            else:
                vertices[left.id] = left
            if right.id in vertices:
                _merge_vertex(vertices[right.id], right)
            else:
                vertices[right.id] = right
            edge_label = force_edge_label or _normalize_edge_label(row.get(edge_label_column, ""))
            edge_attrs = _edge_attrs_from_row(row)
            graph.add_edge(left.id, right.id, edge_label, edge_attrs)
            if undirected and left.id != right.id:
                reverse_attrs = dict(edge_attrs)
                reverse_attrs["direction_role"] = "reverse_copy"
                graph.add_edge(right.id, left.id, edge_label, reverse_attrs)
    if protein_path:
        _merge_protein_annotations(vertices, protein_path, index_column=protein_index_column)
    _assign_degree_features(graph)
    return graph


def build_ppi_seed_pattern(graph: DataGraph, node_label: str = "Protein") -> FrequentPattern:
    """Create the 1-node seed pattern used to start VSpawn."""

    pattern = GraphPattern(node_labels=[node_label])
    instances = [GraphInstance(node_map={0: node_id}, edge_ids=(), pivot=node_id) for node_id, vertex in graph.vertices.items() if vertex.label == node_label]
    return FrequentPattern(pattern=pattern, instances=instances)
