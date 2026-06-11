from __future__ import annotations

"""pgmpy-based Predicate Bayesian Network for GARplusMiner.

For every frequent pattern, matched instances are flattened into rows. The
configured target key, e.g. `e0.interaction_label`, becomes the BN target node.
All other literal columns become predicate feature nodes pointing to the target:

    feature_1 -> target <- feature_2 <- ...

The learned pgmpy CPDs are used to rank/prune predicate columns and to rank final
rules. This keeps the existing rule miners, but makes candidate selection BN-led.
"""

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
    estimator: str = "bayesian"  # bayesian | maximum_likelihood
    equivalent_sample_size: float = 5.0
    drop_target_from_antecedent: bool = True
    max_parent_features: int = 12


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
        self.last_feature_snapshot: List[Tuple[float, str]] = []
        self.last_rule_count = 0
        self.trained = False
        self.training_feature_count = 0

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
        pd, model_cls, estimator_cls = _load_pgmpy(self.config.estimator)
        clean_rows: List[Dict[str, str]] = []
        feature_counts: Dict[str, int] = {}
        target_counts: Dict[str, int] = {}
        for row in rows:
            if y_key not in row:
                continue
            clean_row: Dict[str, str] = {y_key: str(row[y_key])}
            target_counts[f"{y_key}={row[y_key]}"] = target_counts.get(f"{y_key}={row[y_key]}", 0) + 1
            for key, value in row.items():
                if key == y_key and self.config.drop_target_from_antecedent:
                    continue
                if key == y_key:
                    continue
                clean_row[key] = str(value)
                feature_counts[key] = feature_counts.get(key, 0) + 1
            clean_rows.append(clean_row)
        if not clean_rows:
            self.model = None
            self.data = None
            self.feature_columns = []
            self.row_count = 0
            self.target_counts = {}
            self.trained = False
            self.training_feature_count = 0
            return self

        ranked_features = sorted(feature_counts, key=feature_counts.get, reverse=True)
        self.feature_columns = ranked_features[: max(0, self.config.max_parent_features)]
        selected_columns = [y_key] + self.feature_columns
        self.data = pd.DataFrame([{column: row.get(column, "__MISSING__") for column in selected_columns} for row in clean_rows]).astype(str)
        self.row_count = len(self.data)
        self.target_counts = target_counts
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
        return self

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
        """Score a predicate column by its best literal-target dependency."""

        if feature_key not in self.feature_columns:
            return 0.0
        best = 0.0
        for row in rows:
            if feature_key in row:
                item = self.literal_item(feature_key, row[feature_key])
                best = max(best, self.score_item_for_target(item))
        return best

    def rank_feature_keys(self, rows: Sequence[Dict[str, object]], feature_keys: Iterable[str]) -> List[Tuple[float, str]]:
        """Rank and optionally prune feature columns through pgmpy CPDs."""

        feature_list = list(feature_keys)
        scored = [(self.score_feature_key(rows, key), key) for key in feature_list]
        scored = [item for item in scored if item[0] >= self.config.min_score]
        scored.sort(key=lambda item: item[0], reverse=True)
        if self.config.top_k_features is not None:
            scored = scored[: max(0, self.config.top_k_features)]
        self.total_feature_rank_calls += 1
        self.total_features_seen += len(feature_list)
        self.total_features_kept += len(scored)
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

        return {
            "backend": "pgmpy",
            "rows": self.row_count,
            "trained": self.trained,
            "training_feature_count": self.training_feature_count,
            "target_values": dict(self.target_counts),
            "feature_rank_calls": self.total_feature_rank_calls,
            "features_seen": self.total_features_seen,
            "features_kept": self.total_features_kept,
            "features_pruned": self.total_features_seen - self.total_features_kept,
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
