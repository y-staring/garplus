from __future__ import annotations

"""Runtime ML-predicate adapters for GARplusMiner.

The expensive embedding-based similarity predicate is best generated offline by
`ml-predicate/similarity.py`. This module adds the lightweight equivalence
predicate at graph-load time so predicate mining can consume it as ordinary edge
attributes.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from graph_types import DataGraph


MISSING_VALUES = {"", "-", "null", "none", "nan", "na", "n/a"}


@dataclass(frozen=True)
class MLPredicateConfig:
    enabled: bool = False
    equivalence_enabled: bool = True
    equivalence_threshold: Optional[float] = None
    similarity_enabled: bool = True
    similarity_threshold: Optional[float] = None


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
    if config.equivalence_enabled:
        threshold = _threshold_for(dataset_name, config.equivalence_threshold)
        pred_counts: Counter = Counter()
        bucket_counts: Counter = Counter()
        for edge in graph.all_edges():
            score = _score_edge(graph, dataset_name, edge.src, edge.dst, edge.attrs)
            pred = "yes" if score >= threshold else "no"
            bucket = _bucket(score)
            edge.attrs["ml_equivalence_pred"] = pred
            edge.attrs["ml_equivalence_bucket"] = bucket
            edge.attrs["ml_equivalence_score"] = round(score, 6)
            pred_counts[pred] += 1
            bucket_counts[bucket] += 1
        summary["equivalence"] = {
            "threshold": threshold,
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
