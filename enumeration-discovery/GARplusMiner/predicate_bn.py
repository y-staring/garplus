from __future__ import annotations

"""pgmpy-based Predicate Bayesian Network for GARplusMiner.

For every frequent pattern, matched instances are flattened into rows. The
configured target key, e.g. `e0.interaction_label`, becomes the BN target node.
Feature columns are filtered before fitting so sparse ID-like attributes do not
explode the discrete CPD size.
"""

import math
import os
import pickle
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from predicate_selection import Rule


@dataclass
class PredicateBNConfig:
    """Controls pgmpy Predicate BN training and feature pruning."""

    enabled: bool = True
    target_key: str = ""
    top_k_features: Optional[int] = None
    min_score: float = 0.0
    focus_target_item: Optional[str] = None
    min_keep_features: int = 4
    feature_score: str = "cpd_lift"  # cpd_lift | bic
    estimator: str = "bayesian"  # bayesian | maximum_likelihood
    equivalent_sample_size: float = 5.0
    drop_target_from_antecedent: bool = True
    max_parent_features: int = 4
    max_feature_cardinality: int = 50
    max_unique_ratio: float = 0.25
    min_non_missing_count: int = 20
    min_mode_count: int = 2
    max_cpd_cells: int = 200_000
    cache_path: Optional[str] = None
    retrain: bool = False
    excluded_key_tokens: Tuple[str, ...] = (
        "id",
        "index",
        "accession",
        "biogrid",
        "entrez",
        "uniprot",
        "refseq",
        "sequence",
        "comments",
    )


@dataclass
class FeatureStats:
    count: int
    cardinality: int
    unique_ratio: float
    mode_count: int
    reason: str = ""


class PredicateBayesianNetwork:
    """Predicate BN backed by pgmpy CPDs."""

    def __init__(self, config: Optional[PredicateBNConfig] = None) -> None:
        self.config = config or PredicateBNConfig()
        self.model = None
        self.data = None
        self.target_counts: Dict[str, int] = {}
        self.feature_columns: List[str] = []
        self.row_count = 0
        self.total_feature_rank_calls = 0
        self.total_features_seen = 0
        self.total_features_kept = 0
        self.total_tau_pruned = 0
        self.total_feature_limit_pruned = 0
        self.total_topk_pruned = 0
        self.total_min_keep_rescued = 0
        self.last_feature_snapshot: List[Tuple[float, str]] = []
        self.last_rule_count = 0
        self.last_scored_features: List[Tuple[float, str]] = []
        self.last_unranked_features: List[str] = []
        self.trained = False
        self.training_feature_count = 0
        self.target_cardinality = 0
        self.estimated_target_cpd_cells = 0
        self.filtered_feature_stats: Dict[str, FeatureStats] = {}
        self.skipped_feature_stats: Dict[str, FeatureStats] = {}
        self.skipped_feature_budget: Dict[str, int] = {}

    @staticmethod
    def literal_item(key: str, value: object) -> str:
        return f"{key}={value}"

    @staticmethod
    def item_key(item: str) -> str:
        return item.split("=", 1)[0]

    def fit_rows(self, rows: Sequence[Dict[str, object]], target_key: Optional[str] = None) -> "PredicateBayesianNetwork":
        """Train a pgmpy Predicate BN from flattened matching rows."""

        if target_key is not None:
            self.config.target_key = target_key
        y_key = self.config.target_key
        if self.config.cache_path and os.path.exists(self.config.cache_path) and not self.config.retrain:
            runtime_config = self.config
            with open(self.config.cache_path, "rb") as handle:
                cached = pickle.load(handle)
            self.__dict__.update(cached.__dict__)
            self.config = runtime_config
            return self
        pd, model_cls, estimator_cls = _load_pgmpy(self.config.estimator)
        clean_rows: List[Dict[str, str]] = []
        feature_values: Dict[str, Counter] = {}
        target_counts: Dict[str, int] = {}
        for row in rows:
            if y_key not in row:
                continue
            clean_row: Dict[str, str] = {y_key: str(row[y_key])}
            target_counts[f"{y_key}={row[y_key]}"] = target_counts.get(f"{y_key}={row[y_key]}", 0) + 1
            for key, value in row.items():
                if key == y_key:
                    continue
                clean_value = str(value)
                clean_row[key] = clean_value
                feature_values.setdefault(key, Counter())[clean_value] += 1
            clean_rows.append(clean_row)
        if not clean_rows:
            self._reset_empty()
            return self

        self.row_count = len(clean_rows)
        self.target_counts = target_counts
        self.target_cardinality = max(1, len({row[y_key] for row in clean_rows}))
        self.filtered_feature_stats, self.skipped_feature_stats = self._filter_sparse_features(feature_values)
        ranked_features = sorted(
            self.filtered_feature_stats,
            key=lambda key: (
                -self.filtered_feature_stats[key].count,
                self.filtered_feature_stats[key].cardinality,
                key,
            ),
        )
        self.feature_columns = self._select_budgeted_parent_features(ranked_features)
        selected_columns = [y_key] + self.feature_columns
        self.data = pd.DataFrame([{column: row.get(column, "__MISSING__") for column in selected_columns} for row in clean_rows]).astype(str)
        self.training_feature_count = len(self.feature_columns)
        if not self.feature_columns:
            self.model = model_cls([])
            self.trained = False
            return self
        self.model = model_cls([(feature, y_key) for feature in self.feature_columns])
        if self.config.estimator == "maximum_likelihood":
            self.model.fit(self.data)
        else:
            self.model.fit(
                self.data,
                estimator=estimator_cls,
                prior_type="BDeu",
                equivalent_sample_size=self.config.equivalent_sample_size,
            )
        self.trained = True
        self._save_cache_if_needed()
        return self

    def _reset_empty(self) -> None:
        self.model = None
        self.data = None
        self.feature_columns = []
        self.row_count = 0
        self.target_counts = {}
        self.trained = False
        self.training_feature_count = 0
        self.target_cardinality = 0
        self.estimated_target_cpd_cells = 0
        self.filtered_feature_stats = {}
        self.skipped_feature_stats = {}
        self.skipped_feature_budget = {}

    def _filter_sparse_features(self, feature_values: Dict[str, Counter]) -> Tuple[Dict[str, FeatureStats], Dict[str, FeatureStats]]:
        kept: Dict[str, FeatureStats] = {}
        skipped: Dict[str, FeatureStats] = {}
        for key, counts in feature_values.items():
            count = sum(counts.values())
            cardinality = len(counts)
            mode_count = max(counts.values()) if counts else 0
            unique_ratio = cardinality / max(count, 1)
            reason = self._skip_reason(key, count, cardinality, unique_ratio, mode_count)
            stats = FeatureStats(count=count, cardinality=cardinality, unique_ratio=unique_ratio, mode_count=mode_count, reason=reason)
            if reason:
                skipped[key] = stats
            else:
                kept[key] = stats
        return kept, skipped

    def _skip_reason(self, key: str, count: int, cardinality: int, unique_ratio: float, mode_count: int) -> str:
        key_tail = key.split(".", 1)[-1].lower()
        if any(token in key_tail for token in self.config.excluded_key_tokens):
            return "excluded_key_token"
        if count < self.config.min_non_missing_count:
            return f"non_missing<{self.config.min_non_missing_count}"
        if cardinality > self.config.max_feature_cardinality:
            return f"cardinality>{self.config.max_feature_cardinality}"
        if unique_ratio > self.config.max_unique_ratio:
            return f"unique_ratio>{self.config.max_unique_ratio}"
        if mode_count < self.config.min_mode_count:
            return f"mode_count<{self.config.min_mode_count}"
        return ""

    def _select_budgeted_parent_features(self, ranked_features: Sequence[str]) -> List[str]:
        selected: List[str] = []
        self.skipped_feature_budget = {}
        parent_state_product = 1
        max_parents = max(0, self.config.max_parent_features)
        for feature in ranked_features:
            cardinality = max(1, self.filtered_feature_stats[feature].cardinality)
            estimated_cells = self.target_cardinality * parent_state_product * cardinality
            if estimated_cells > self.config.max_cpd_cells:
                self.skipped_feature_budget[feature] = estimated_cells
                continue
            selected.append(feature)
            parent_state_product *= cardinality
            if len(selected) >= max_parents:
                break
        self.estimated_target_cpd_cells = self.target_cardinality * parent_state_product
        return selected

    def _save_cache_if_needed(self) -> None:
        if not self.config.cache_path:
            return
        cache_dir = os.path.dirname(self.config.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(self.config.cache_path, "wb") as handle:
            pickle.dump(self, handle)

    def score_item_for_target(self, item: str, target_item: Optional[str] = None) -> float:
        """Score one feature literal using pgmpy target CPD probability."""

        if self.model is None or self.row_count <= 0:
            return 0.0
        feature_key, feature_value = item.split("=", 1)
        y_key = self.config.target_key
        if feature_key not in self.feature_columns:
            return 0.0
        if target_item is None:
            if not self.target_counts:
                return 0.0
            target_item = max(self.target_counts, key=self.target_counts.get)
        target_key, target_value = target_item.split("=", 1)
        if target_key != y_key:
            return 0.0
        evidence = {feature_key: feature_value}
        conditional = _cpd_probability(self.model, y_key, target_value, evidence)
        base = self.target_counts.get(target_item, 0) / self.row_count if self.row_count else 0.0
        return conditional / base if base else 0.0

    def score_feature_key(self, rows: Sequence[Dict[str, object]], feature_key: str) -> float:
        """Score a predicate column by the configured BN feature score."""

        if feature_key not in self.feature_columns:
            return 0.0
        if self.config.feature_score == "bic":
            return self.bic_feature_score(rows, feature_key)
        best = 0.0
        target_item = self.config.focus_target_item
        for row in rows:
            if feature_key in row:
                item = self.literal_item(feature_key, row[feature_key])
                best = max(best, self.score_item_for_target(item, target_item=target_item))
        return best

    def bic_feature_score(self, rows: Sequence[Dict[str, object]], feature_key: str) -> float:
        """Return positive BIC gain for adding `feature_key -> target`.

        BIC is used here as a feature-selection score. It is not passed to
        pgmpy's `model.fit`, because BIC scores structures while CPDs still need
        a parameter estimator such as maximum likelihood or BayesianEstimator.
        """

        y_key = self.config.target_key
        pairs = [(str(row[feature_key]), str(row[y_key])) for row in rows if feature_key in row and y_key in row]
        n = len(pairs)
        if n <= 1:
            return 0.0
        y_counts = Counter(y for _, y in pairs)
        x_counts = Counter(x for x, _ in pairs)
        xy_counts = Counter(pairs)
        ll_base = 0.0
        for _, y in pairs:
            ll_base += math.log(max(y_counts[y] / n, 1e-12))
        ll_cond = 0.0
        for x, y in pairs:
            ll_cond += math.log(max(xy_counts[(x, y)] / x_counts[x], 1e-12))
        x_card = max(1, len(x_counts))
        y_card = max(1, len(y_counts))
        params = max(1, (x_card - 1) * (y_card - 1))
        bic_gain = (ll_cond - ll_base) - 0.5 * params * math.log(n)
        return max(0.0, bic_gain / n)

    def rank_feature_keys(self, rows: Sequence[Dict[str, object]], feature_keys: Iterable[str]) -> List[Tuple[float, str]]:
        """Rank and optionally prune feature columns through pgmpy CPDs."""

        original_feature_list = list(feature_keys)
        feature_list = [key for key in original_feature_list if key in self.feature_columns]
        scored = [(self.score_feature_key(rows, key), key) for key in feature_list]
        scored.sort(key=lambda item: item[0], reverse=True)
        self.last_scored_features = scored[:20]
        self.last_unranked_features = sorted(set(original_feature_list) - set(self.feature_columns))[:20]
        thresholded = [item for item in scored if item[0] >= self.config.min_score]
        tau_pruned = len(scored) - len(thresholded)
        min_keep = max(0, self.config.min_keep_features)
        min_keep_target = min(min_keep, len(scored))
        min_keep_rescued = 0
        if min_keep and len(thresholded) < min_keep_target:
            min_keep_rescued = min_keep_target - len(thresholded)
            thresholded = scored[:min_keep_target]
        before_topk = len(thresholded)
        scored = thresholded
        if self.config.top_k_features is not None:
            scored = scored[: max(0, self.config.top_k_features)]
        topk_pruned = before_topk - len(scored)
        self.total_feature_rank_calls += 1
        self.total_features_seen += len(original_feature_list)
        self.total_features_kept += len(scored)
        self.total_tau_pruned += tau_pruned
        self.total_feature_limit_pruned += max(0, len(original_feature_list) - len(feature_list))
        self.total_topk_pruned += topk_pruned
        self.total_min_keep_rescued += min_keep_rescued
        self.last_feature_snapshot = scored[:8]
        return scored

    def rank_rules(self, rules: Sequence[Rule]) -> List[Rule]:
        """Rank generated rules by pgmpy Predicate-BN score."""

        def rule_score(rule: Rule) -> Tuple[float, float, float]:
            target_item = rule.consequent
            antecedent_score = 1.0
            for item in rule.antecedent:
                antecedent_score *= max(1e-9, self.score_item_for_target(item, target_item))
            return antecedent_score, rule.confidence, rule.lift

        ranked = sorted(rules, key=rule_score, reverse=True)
        self.last_rule_count = len(ranked)
        return ranked

    def pruning_summary(self) -> Dict[str, object]:
        """Return observable Predicate-BN pruning/ranking statistics."""

        skipped_by_reason = Counter(stats.reason for stats in self.skipped_feature_stats.values())
        skipped_examples: Dict[str, List[Dict[str, object]]] = {}
        for key, stats in self.skipped_feature_stats.items():
            bucket = skipped_examples.setdefault(stats.reason, [])
            if len(bucket) < 8:
                bucket.append({
                    "key": key,
                    "count": stats.count,
                    "cardinality": stats.cardinality,
                    "unique_ratio": round(stats.unique_ratio, 4),
                    "mode_count": stats.mode_count,
                })
        return {
            "backend": "pgmpy",
            "rows": self.row_count,
            "trained": self.trained,
            "training_feature_count": self.training_feature_count,
            "target_cardinality": self.target_cardinality,
            "estimated_target_cpd_cells": self.estimated_target_cpd_cells,
            "max_cpd_cells": self.config.max_cpd_cells,
            "target_values": dict(self.target_counts),
            "focus_target_item": self.config.focus_target_item,
            "feature_score": self.config.feature_score,
            "feature_rank_calls": self.total_feature_rank_calls,
            "features_seen": self.total_features_seen,
            "features_kept": self.total_features_kept,
            "features_pruned": self.total_features_seen - self.total_features_kept,
            "tau_x": self.config.min_score,
            "tau_pruned": self.total_tau_pruned,
            "feature_limit_pruned": self.total_feature_limit_pruned,
            "topk_pruned": self.total_topk_pruned,
            "min_keep_rescued": self.total_min_keep_rescued,
            "candidate_features_after_sparse_filter": len(self.filtered_feature_stats),
            "training_features": list(self.feature_columns),
            "sparse_features_skipped": len(self.skipped_feature_stats),
            "sparse_skip_reasons": dict(skipped_by_reason),
            "sparse_skip_examples": skipped_examples,
            "last_scored_features": self.last_scored_features,
            "last_unranked_features": self.last_unranked_features,
            "skipped_cpd_budget": dict(list(self.skipped_feature_budget.items())[:8]),
            "rules_ranked": self.last_rule_count,
            "top_features": self.last_feature_snapshot,
        }


def _load_pgmpy(estimator: str):
    try:
        import pandas as pd
        try:
            from pgmpy.models import DiscreteBayesianNetwork as ModelCls
        except ImportError:
            from pgmpy.models import BayesianNetwork as ModelCls
        if estimator == "maximum_likelihood":
            from pgmpy.estimators import MaximumLikelihoodEstimator as EstimatorCls
        else:
            from pgmpy.estimators import BayesianEstimator as EstimatorCls
    except ImportError as exc:
        raise ImportError(
            "GARplusMiner Predicate BN now requires pgmpy and pandas. "
            "Install them in this environment, e.g. `pip install pgmpy pandas`."
        ) from exc
    return pd, ModelCls, EstimatorCls


def _cpd_probability(model, variable: str, state: str, evidence: Dict[str, str]) -> float:
    """Read a local CPD probability using pgmpy's own CPD accessor."""

    cpd = model.get_cpds(variable)
    if cpd is None:
        return 0.0
    try:
        kwargs = {variable: state}
        for evidence_var in cpd.variables[1:]:
            evidence_states = list(cpd.state_names.get(evidence_var, []))
            evidence_value = evidence.get(evidence_var, "__MISSING__")
            if evidence_value not in evidence_states:
                evidence_value = "__MISSING__" if "__MISSING__" in evidence_states else evidence_states[0]
            kwargs[evidence_var] = evidence_value
        return float(cpd.get_value(**kwargs))
    except Exception:
        return 0.0






