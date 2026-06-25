from __future__ import annotations

"""Runtime predicate enrichment for GARplusMiner.

This module adds lightweight symbolic predicates directly to graph attributes
before predicate mining. Numeric attributes are discretized into low/medium/high
bins, so ordinary predicate selectors can mine conditions such as
``v0.length_bin=high`` or ``e0.score_bin=low``.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional

from graph_types import DataGraph


MISSING_VALUES = {"", "-", "null", "none", "nan", "na", "n/a", "inf", "-inf"}
DEFAULT_EXCLUDED_TOKENS = (
    "id",
    "index",
    "row",
    "sampled",
    "local",
    "original",
    "source_node_id",
)


@dataclass(frozen=True)
class PredicateEnrichmentConfig:
    enabled: bool = True
    numeric_bin_enabled: bool = True
    quantiles: tuple[float, float] = (0.33, 0.66)
    min_non_missing: int = 20
    max_numeric_cardinality: int = 5000
    excluded_key_tokens: tuple[str, ...] = DEFAULT_EXCLUDED_TOKENS
    node_numeric_keys: Optional[tuple[str, ...]] = None
    edge_numeric_keys: Optional[tuple[str, ...]] = None
    add_pair_equal_bins: bool = True
    max_pair_equal_bin_keys: int = 20
    inference_edge_predicates: bool = False
    inference_presence_key: Optional[str] = None


def _clean_key(key: object) -> str:
    return str(key or "").strip().lower()


def _to_float(value: object) -> Optional[float]:
    text = str(value or "").strip()
    if not text or text.lower() in MISSING_VALUES:
        return None
    try:
        val = float(text)
    except (TypeError, ValueError):
        return None
    if val != val or val in (float("inf"), float("-inf")):
        return None
    return val


def _allowed_key(key: str, excluded_tokens: Iterable[str]) -> bool:
    low = _clean_key(key)
    if low.endswith("_bin") or low.endswith("_bucket"):
        return False
    return not any(token and token in low for token in excluded_tokens)


def _candidate_numeric_keys(records: Iterable[Mapping[str, object]], cfg: PredicateEnrichmentConfig) -> dict[str, list[float]]:
    values_by_key: dict[str, list[float]] = defaultdict(list)
    for attrs in records:
        for key, value in attrs.items():
            key = _clean_key(key)
            if not _allowed_key(key, cfg.excluded_key_tokens):
                continue
            val = _to_float(value)
            if val is not None:
                values_by_key[key].append(val)
    result = {}
    for key, values in values_by_key.items():
        if len(values) < cfg.min_non_missing:
            continue
        if len(set(values)) > cfg.max_numeric_cardinality:
            continue
        result[key] = values
    return result


def _thresholds(values: list[float], quantiles: tuple[float, float]) -> tuple[float, float]:
    values = sorted(values)
    if not values:
        return 0.0, 0.0
    def pick(q: float) -> float:
        idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
        return float(values[idx])
    return pick(quantiles[0]), pick(quantiles[1])


def _bin_value(value: object, low_upper: float, high_lower: float) -> Optional[str]:
    val = _to_float(value)
    if val is None:
        return None
    if val <= low_upper:
        return "low"
    if val <= high_lower:
        return "medium"
    return "high"


def _select_keys(candidate: dict[str, list[float]], explicit: Optional[tuple[str, ...]]) -> list[str]:
    if explicit is None:
        return sorted(candidate)
    explicit_set = {_clean_key(key) for key in explicit}
    return sorted(key for key in candidate if key in explicit_set)


def enrich_numeric_bin_predicates(graph: DataGraph, cfg: PredicateEnrichmentConfig) -> dict[str, object]:
    if not cfg.enabled or not cfg.numeric_bin_enabled:
        return {"enabled": False}

    node_candidates = _candidate_numeric_keys((vertex.attrs for vertex in graph.vertices.values()), cfg)
    edge_candidates = _candidate_numeric_keys((edge.attrs for edge in graph.all_edges()), cfg)
    node_keys = _select_keys(node_candidates, cfg.node_numeric_keys)
    edge_keys = _select_keys(edge_candidates, cfg.edge_numeric_keys)

    node_thresholds = {key: _thresholds(node_candidates[key], cfg.quantiles) for key in node_keys}
    edge_thresholds = {key: _thresholds(edge_candidates[key], cfg.quantiles) for key in edge_keys}
    node_bin_counts: dict[str, Counter] = defaultdict(Counter)
    edge_bin_counts: dict[str, Counter] = defaultdict(Counter)

    for vertex in graph.vertices.values():
        for key, (low_upper, high_lower) in node_thresholds.items():
            bin_value = _bin_value(vertex.attrs.get(key), low_upper, high_lower)
            if bin_value is None:
                continue
            vertex.attrs[f"{key}_bin"] = bin_value
            node_bin_counts[key][bin_value] += 1

    for edge in graph.all_edges():
        for key, (low_upper, high_lower) in edge_thresholds.items():
            bin_value = _bin_value(edge.attrs.get(key), low_upper, high_lower)
            if bin_value is None:
                continue
            edge.attrs[f"{key}_bin"] = bin_value
            edge_bin_counts[key][bin_value] += 1

    pair_equal_counts: Counter = Counter()
    if cfg.add_pair_equal_bins:
        pair_keys = node_keys[: cfg.max_pair_equal_bin_keys]
        for edge in graph.all_edges():
            left = graph.vertices.get(edge.src)
            right = graph.vertices.get(edge.dst)
            if left is None or right is None:
                continue
            for key in pair_keys:
                left_bin = left.attrs.get(f"{key}_bin")
                right_bin = right.attrs.get(f"{key}_bin")
                if left_bin is None or right_bin is None:
                    continue
                same = "yes" if left_bin == right_bin else "no"
                edge.attrs[f"same_{key}_bin"] = same
                pair_equal_counts[f"same_{key}_bin={same}"] += 1

    inference_predicate_counts: Counter = Counter()
    if cfg.inference_edge_predicates:
        score_values = [_to_float(edge.attrs.get("inferencescore")) for edge in graph.all_edges()]
        score_low, score_high = _thresholds([value for value in score_values if value is not None], cfg.quantiles)
        for edge in graph.all_edges():
            attrs = edge.attrs
            direct = str(attrs.pop("directevidence", "")).strip().lower()
            direct_value = "inference_evidence" if not direct or direct in MISSING_VALUES else ("marker_mechanism" if direct == "marker/mechanism" else "other")
            attrs["direct_evidence_category"] = direct_value
            inference_predicate_counts[f"direct_evidence_category={direct_value}"] += 1
            if cfg.inference_presence_key:
                raw = str(attrs.pop(cfg.inference_presence_key, "")).strip().lower()
                value = "no" if not raw or raw in MISSING_VALUES else "yes"
                predicate_key = "inference_gene_present" if cfg.inference_presence_key == "inferencegenesymbol" else "inference_chemical_present"
                attrs[predicate_key] = value
                inference_predicate_counts[f"{predicate_key}={value}"] += 1
            score_bin = _bin_value(attrs.pop("inferencescore", None), score_low, score_high) or "missing"
            attrs["inference_score_bin"] = score_bin
            inference_predicate_counts[f"inference_score_bin={score_bin}"] += 1

    return {
        "enabled": True,
        "node_numeric_keys": node_keys,
        "edge_numeric_keys": edge_keys,
        "node_thresholds": {key: {"low_upper": lo, "high_lower": hi} for key, (lo, hi) in node_thresholds.items()},
        "edge_thresholds": {key: {"low_upper": lo, "high_lower": hi} for key, (lo, hi) in edge_thresholds.items()},
        "node_bin_counts": {key: dict(counts) for key, counts in node_bin_counts.items()},
        "edge_bin_counts": {key: dict(counts) for key, counts in edge_bin_counts.items()},
        "pair_equal_counts": dict(pair_equal_counts),
        "inference_edge_predicates": dict(inference_predicate_counts),
    }
