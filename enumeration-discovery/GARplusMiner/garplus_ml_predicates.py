from __future__ import annotations

"""Runtime ML-predicate adapters for GARplusMiner.

The expensive embedding-based similarity predicate is best generated offline by
`ml-predicate/similarity.py`. This module adds the lightweight equivalence
predicate at graph-load time so predicate mining can consume it as ordinary edge
attributes.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional
import csv

from graph_types import DataGraph


MISSING_VALUES = {"", "-", "null", "none", "nan", "na", "n/a"}


@dataclass(frozen=True)
class MLPredicateConfig:
    enabled: bool = False
    equivalence_enabled: bool = True
    equivalence_threshold: Optional[float] = None
    similarity_enabled: bool = True
    similarity_threshold: Optional[float] = None
    offline_enabled: bool = True
    offline_csv_path: Optional[str] = None
    offline_undirected: bool = True
    offline_default_value: str = "none"
    precomputed_edge_csv_path: Optional[str] = None
    precomputed_left_column: Optional[str] = None
    precomputed_right_column: Optional[str] = None
    precomputed_undirected: bool = True

def _clean(value: object, *, lower: bool = True) -> str:
    text = str(value or "").strip()
    if text.lower() in MISSING_VALUES:
        return ""
    return text.lower() if lower else text


def _split_values(value: object) -> set[str]:
    text = _clean(value)
    if not text:
        return set()
    for char in "{}[]\"":
        text = text.replace(char, "")
    pieces = []
    for sep in ("|", ";", ","):
        next_pieces = []
        for piece in pieces or [text]:
            next_pieces.extend(piece.split(sep))
        pieces = next_pieces
    return {_clean(piece) for piece in pieces if _clean(piece)}


def _values_for(obj: Mapping[str, object], fields: Iterable[str]) -> set[str]:
    values: set[str] = set()
    for field in fields:
        values.add(_clean(obj.get(field)))
        values.update(_split_values(obj.get(field)))
    values.discard("")
    return values


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _any_exact(left: Mapping[str, object], right: Mapping[str, object], pairs: Iterable[tuple[str, str]]) -> bool:
    for left_field, right_field in pairs:
        left_value = _clean(left.get(left_field))
        right_value = _clean(right.get(right_field))
        if left_value and left_value == right_value:
            return True
    return False


def _any_overlap(left: Mapping[str, object], right: Mapping[str, object], pairs: Iterable[tuple[str, str]]) -> bool:
    for left_field, right_field in pairs:
        if _values_for(left, (left_field,)) & _values_for(right, (right_field,)):
            return True
    return False


def _bucket(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.40:
        return "mid"
    return "low"


def _ppi_equivalence_score(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    exact_pairs = (
        ("original_index", "original_index"),
        ("source_node_id", "source_node_id"),
        ("entrez_gene_id", "entrez_gene_id"),
        ("biogrid_id", "biogrid_id"),
        ("official_symbol", "official_symbol"),
        ("entry", "entry"),
    )
    accession_pairs = (
        ("swiss_prot_accessions", "swiss_prot_accessions"),
        ("trembl_accessions", "trembl_accessions"),
        ("refseq_accessions", "refseq_accessions"),
        ("uniprotids", "uniprotids"),
        ("entry", "entry"),
    )
    alias_fields = (
        "official_symbol",
        "synonyms",
        "protein_names",
        "gene_names",
        "gene_names_(synonym)",
    )
    if _any_exact(left, right, exact_pairs) or _any_overlap(left, right, accession_pairs):
        return 1.0
    return _jaccard(_values_for(left, alias_fields), _values_for(right, alias_fields))


def _dda_equivalence_score(edge_attrs: Mapping[str, object]) -> float:
    drug_ids = _values_for(edge_attrs, ("node_1", "chemicalid"))
    disease_ids = _values_for(edge_attrs, ("node_2", "diseaseid"))
    if drug_ids & disease_ids:
        return 1.0
    return _jaccard(
        _values_for(edge_attrs, ("chemicalname", "synonyms", "name")),
        _values_for(edge_attrs, ("diseasename", "synonyms")),
    )


def _ti_equivalence_score(edge_attrs: Mapping[str, object]) -> float:
    gene_ids = _values_for(edge_attrs, ("node_1", "geneid"))
    disease_ids = _values_for(edge_attrs, ("node_2", "diseaseid"))
    if gene_ids & disease_ids:
        return 1.0
    return _jaccard(
        _values_for(edge_attrs, ("genesymbol", "genename", "synonyms")),
        _values_for(edge_attrs, ("diseasename", "synonyms")),
    )



def _canonical_pair(x: object, y: object, undirected: bool = True) -> Optional[tuple[int, int]]:
    try:
        left = int(float(str(x).strip()))
        right = int(float(str(y).strip()))
    except (TypeError, ValueError):
        return None
    if undirected and left > right:
        left, right = right, left
    return left, right


def _edge_original_pair(graph: DataGraph, edge, undirected: bool = True) -> Optional[tuple[int, int]]:
    attrs = edge.attrs
    key_pairs = (
        ("sampled_src_original_id", "sampled_dst_original_id"),
        ("src_original_id", "dst_original_id"),
        ("index_a", "index_b"),
        ("x", "y"),
    )
    for left_key, right_key in key_pairs:
        if left_key in attrs and right_key in attrs:
            pair = _canonical_pair(attrs[left_key], attrs[right_key], undirected=undirected)
            if pair is not None:
                return pair
    left_vertex = graph.vertices.get(edge.src)
    right_vertex = graph.vertices.get(edge.dst)
    if left_vertex is not None and right_vertex is not None:
        for left_key, right_key in (("original_index", "original_index"), ("source_node_id", "source_node_id")):
            if left_key in left_vertex.attrs and right_key in right_vertex.attrs:
                pair = _canonical_pair(left_vertex.attrs[left_key], right_vertex.attrs[right_key], undirected=undirected)
                if pair is not None:
                    return pair
    return _canonical_pair(edge.src, edge.dst, undirected=undirected)


def _load_offline_predicates(path: Optional[str], undirected: bool = True) -> dict[tuple[int, int], list[dict[str, object]]]:
    if not path:
        return {}
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    result: dict[tuple[int, int], list[dict[str, object]]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"x", "y", "predicate_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"ML predicate csv missing columns: {sorted(missing)}")
        for row in reader:
            pair = _canonical_pair(row.get("x"), row.get("y"), undirected=undirected)
            if pair is None:
                continue
            pred_name = _clean(row.get("predicate_name"), lower=False)
            if not pred_name:
                continue
            item = dict(row)
            item["predicate_name"] = pred_name
            result.setdefault(pair, []).append(item)
    return result



def _default_pair_columns(dataset_name: str) -> tuple[str, str]:
    dataset = dataset_name.upper()
    if dataset == "PPI":
        return "index_A", "index_B"
    if dataset == "DDA":
        return "chemical_index", "disease_index"
    if dataset == "TI":
        return "gene_index", "disease_index"
    return "x", "y"


def _load_precomputed_edge_rows(
    path: Optional[str],
    left_column: str,
    right_column: str,
    undirected: bool = True,
) -> dict[tuple[int, int], dict[str, object]]:
    if not path:
        return {}
    csv_path = Path(path)
    if not csv_path.exists():
        return {}
    result: dict[tuple[int, int], dict[str, object]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {left_column, right_column}
        missing = required - fields
        if missing:
            raise ValueError(f"precomputed ML predicate csv missing columns: {sorted(missing)}")
        kept = fields & {"equivalence_score", "equivalence_pred", "similarity_score", "similarity_pred"}
        if not kept:
            return {}
        for row in reader:
            pair = _canonical_pair(row.get(left_column), row.get(right_column), undirected=undirected)
            if pair is None:
                continue
            result[pair] = {key: row.get(key) for key in kept if row.get(key) not in (None, "")}
    return result


def _inject_precomputed_edge_rows(
    graph: DataGraph,
    rows_by_pair: dict[tuple[int, int], dict[str, object]],
    undirected: bool = True,
) -> dict[str, object]:
    matched = 0
    copied = Counter()
    if not rows_by_pair:
        return {"pairs": 0, "matched_edges": 0, "copied": {}}
    for edge in graph.all_edges():
        pair = _edge_original_pair(graph, edge, undirected=undirected)
        values = rows_by_pair.get(pair) if pair is not None else None
        if not values:
            continue
        matched += 1
        for key, value in values.items():
            if value not in (None, ""):
                edge.attrs[key] = value
                copied[key] += 1
    return {"pairs": len(rows_by_pair), "matched_edges": matched, "copied": dict(copied)}

def _offline_score(rows: list[dict[str, object]]) -> Optional[float]:
    scores = []
    for row in rows:
        try:
            scores.append(float(row.get("score")))
        except (TypeError, ValueError):
            continue
    if not scores:
        return None
    return max(scores)
def _threshold_for(dataset_name: str, configured: Optional[float]) -> float:
    if configured is not None:
        return configured
    return 0.80 if dataset_name.upper() == "PPI" else 0.95


def _similarity_threshold_for(dataset_name: str, configured: Optional[float]) -> float:
    if configured is not None:
        return configured
    return 0.85 if dataset_name.upper() == "PPI" else 0.80


def _score_edge(graph: DataGraph, dataset_name: str, src: int, dst: int, attrs: Mapping[str, object]) -> float:
    dataset = dataset_name.upper()
    if dataset == "PPI":
        left = graph.vertices[src].attrs if src in graph.vertices else {}
        right = graph.vertices[dst].attrs if dst in graph.vertices else {}
        return _ppi_equivalence_score(left, right)
    if dataset == "DDA":
        return _dda_equivalence_score(attrs)
    if dataset == "TI":
        return _ti_equivalence_score(attrs)
    return 0.0


def inject_ml_predicates(graph: DataGraph, dataset_name: str, config: MLPredicateConfig) -> dict[str, object]:
    if not config.enabled:
        return {"enabled": False}
    summary: dict[str, object] = {"enabled": True}
    if config.precomputed_edge_csv_path:
        left_column = config.precomputed_left_column
        right_column = config.precomputed_right_column
        if not left_column or not right_column:
            left_column, right_column = _default_pair_columns(dataset_name)
        precomputed_path = Path(config.precomputed_edge_csv_path)
        if precomputed_path.exists():
            edge_rows = _load_precomputed_edge_rows(
                config.precomputed_edge_csv_path,
                left_column,
                right_column,
                undirected=config.precomputed_undirected,
            )
            summary["precomputed_edge_csv"] = _inject_precomputed_edge_rows(
                graph,
                edge_rows,
                undirected=config.precomputed_undirected,
            )
            summary["precomputed_edge_csv"].update(
                {"csv_path": config.precomputed_edge_csv_path, "left_column": left_column, "right_column": right_column}
            )
        else:
            summary["precomputed_edge_csv"] = {"csv_path": config.precomputed_edge_csv_path, "skipped": "missing"}
    if config.offline_enabled and config.offline_csv_path:
        offline_path = Path(config.offline_csv_path)
        if offline_path.exists():
            offline_index = _load_offline_predicates(config.offline_csv_path, undirected=config.offline_undirected)
            pred_counts: Counter = Counter()
            matched_edges = 0
            for edge in graph.all_edges():
                pair = _edge_original_pair(graph, edge, undirected=config.offline_undirected)
                rows = offline_index.get(pair, []) if pair is not None else []
                names = sorted({str(row.get("predicate_name")) for row in rows if row.get("predicate_name")})
                if names:
                    matched_edges += 1
                    edge.attrs["ml_offline_predicate_name"] = "|".join(names)
                    score = _offline_score(rows)
                    if score is not None:
                        edge.attrs["ml_offline_score"] = round(score, 6)
                        edge.attrs["ml_offline_bucket"] = _bucket(score)
                else:
                    edge.attrs["ml_offline_predicate_name"] = config.offline_default_value
                for pred_name in ("ml_pred_ppi", "ml_pred_not_ppi"):
                    value = "yes" if pred_name in names else "no"
                    edge.attrs[pred_name] = value
                    pred_counts[f"{pred_name}={value}"] += 1
            summary["offline"] = {
                "csv_path": config.offline_csv_path,
                "pairs": len(offline_index),
                "matched_edges": matched_edges,
                "pred_counts": dict(pred_counts),
            }
        else:
            summary["offline"] = {"csv_path": config.offline_csv_path, "skipped": "missing"}
    if config.equivalence_enabled:
        threshold = _threshold_for(dataset_name, config.equivalence_threshold)
        pred_counts: Counter = Counter()
        bucket_counts: Counter = Counter()
        precomputed = 0
        for edge in graph.all_edges():
            raw_score = edge.attrs.get("equivalence_score")
            if raw_score is None:
                score = _score_edge(graph, dataset_name, edge.src, edge.dst, edge.attrs)
                pred = "yes" if score >= threshold else "no"
            else:
                try:
                    score = float(raw_score)
                except (TypeError, ValueError):
                    score = _score_edge(graph, dataset_name, edge.src, edge.dst, edge.attrs)
                raw_pred = edge.attrs.get("equivalence_pred")
                if raw_pred in (0, 1, "0", "1"):
                    pred = "yes" if int(raw_pred) == 1 else "no"
                    precomputed += 1
                else:
                    pred = "yes" if score >= threshold else "no"
            bucket = _bucket(score)
            edge.attrs["ml_equivalence_pred"] = pred
            edge.attrs["ml_equivalence_bucket"] = bucket
            edge.attrs["ml_equivalence_score"] = round(score, 6)
            pred_counts[pred] += 1
            bucket_counts[bucket] += 1
        summary["equivalence"] = {
            "threshold": threshold,
            "source": "precomputed_edge_attrs" if precomputed else "runtime",
            "precomputed_edges": precomputed,
            "pred_counts": dict(pred_counts),
            "bucket_counts": dict(bucket_counts),
        }
    if config.similarity_enabled:
        threshold = _similarity_threshold_for(dataset_name, config.similarity_threshold)
        pred_counts = Counter()
        bucket_counts = Counter()
        present = 0
        for edge in graph.all_edges():
            raw_score = edge.attrs.get("similarity_score")
            if raw_score is None:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            raw_pred = edge.attrs.get("similarity_pred")
            if raw_pred in (0, 1, "0", "1"):
                pred = "yes" if int(raw_pred) == 1 else "no"
            else:
                pred = "yes" if score >= threshold else "no"
            bucket = _bucket(score)
            edge.attrs["ml_similarity_pred"] = pred
            edge.attrs["ml_similarity_bucket"] = bucket
            edge.attrs["ml_similarity_score"] = round(score, 6)
            pred_counts[pred] += 1
            bucket_counts[bucket] += 1
            present += 1
        summary["similarity"] = {
            "source": "precomputed_edge_attrs",
            "present_edges": present,
            "threshold": threshold,
            "pred_counts": dict(pred_counts),
            "bucket_counts": dict(bucket_counts),
        }
    return summary







