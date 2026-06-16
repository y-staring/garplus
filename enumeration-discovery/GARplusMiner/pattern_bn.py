from __future__ import annotations

"""pgmpy-based Pattern Bayesian Network for GARplusMiner.

Pattern BN is trained from data-graph edges. Each directed training case records:

    SRC_LABEL, DIRECTION, EDGE_LABEL, DST_LABEL

During VSpawn, each candidate expansion is scored by querying the learned CPDs,
mainly `P(EDGE_LABEL | SRC_LABEL, DIRECTION)` and
`P(DST_LABEL | SRC_LABEL, DIRECTION, EDGE_LABEL)`. The score is then used to
rank or prune structural expansions before expensive subgraph matching.
"""

import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from graph_types import DataGraph, GraphPattern, SpawnEdge


CandidateScore = Tuple[float, SpawnEdge]


@dataclass
class PatternBNConfig:
    """Controls pgmpy Pattern BN training and pruning."""

    enabled: bool = True
    top_k_per_spawn_node: Optional[int] = None
    min_score: float = 0.0
    min_keep_per_spawn_node: int = 1
    estimator: str = "bayesian"  # bayesian | maximum_likelihood
    equivalent_sample_size: float = 5.0
    cache_path: Optional[str] = None
    retrain: bool = False
    frequent_edge_priors: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    frequent_prior_weight: float = 0.25


class PatternBayesianNetwork:
    """Pattern BN backed by pgmpy CPDs."""

    SRC_LABEL = "src_label"
    DIRECTION = "direction"
    EDGE_LABEL = "edge_label"
    DST_LABEL = "dst_label"

    def __init__(self, config: Optional[PatternBNConfig] = None) -> None:
        self.config = config or PatternBNConfig()
        self.model = None
        self.data = None
        self.state_names: Dict[str, List[object]] = {}
        self.total_rank_calls = 0
        self.total_candidates_seen = 0
        self.total_candidates_kept = 0
        self.total_threshold_pruned = 0
        self.total_topk_pruned = 0
        self.total_min_keep_rescued = 0
        self.last_rank_snapshot: List[Tuple[float, str]] = []

    @classmethod
    def fit_graph(cls, graph: DataGraph, config: Optional[PatternBNConfig] = None) -> "PatternBayesianNetwork":
        bn = cls(config=config)
        bn.fit(graph)
        return bn

    def fit(self, graph: DataGraph) -> None:
        """Train the Pattern BN with pgmpy from directed graph-edge samples."""

        if self.config.cache_path and os.path.exists(self.config.cache_path) and not self.config.retrain:
            with open(self.config.cache_path, "rb") as handle:
                cached = pickle.load(handle)
            self.__dict__.update(cached.__dict__)
            self.config = cached.config
            return
        pd, model_cls, estimator_cls = _load_pgmpy(self.config.estimator)
        rows = []
        for edge in graph.all_edges():
            src_label = str(graph.vertices[edge.src].label)
            dst_label = str(graph.vertices[edge.dst].label)
            edge_label = str(edge.label)
            rows.append(
                {
                    self.SRC_LABEL: src_label,
                    self.DIRECTION: "out",
                    self.EDGE_LABEL: edge_label,
                    self.DST_LABEL: dst_label,
                }
            )
            rows.append(
                {
                    self.SRC_LABEL: dst_label,
                    self.DIRECTION: "in",
                    self.EDGE_LABEL: edge_label,
                    self.DST_LABEL: src_label,
                }
            )
        if not rows:
            raise ValueError("Pattern BN cannot be trained because the graph has no edges")

        self.data = pd.DataFrame(rows).astype(str)
        self.state_names = {column: sorted(self.data[column].unique().tolist()) for column in self.data.columns}
        self.model = model_cls(
            [
                (self.SRC_LABEL, self.EDGE_LABEL),
                (self.DIRECTION, self.EDGE_LABEL),
                (self.SRC_LABEL, self.DST_LABEL),
                (self.DIRECTION, self.DST_LABEL),
                (self.EDGE_LABEL, self.DST_LABEL),
            ]
        )
        if self.config.estimator == "maximum_likelihood":
            self.model.fit(self.data)
        else:
            self.model.fit(
                self.data,
                estimator=estimator_cls,
                prior_type="BDeu",
                equivalent_sample_size=self.config.equivalent_sample_size,
            )

        self._save_cache_if_needed()

    def _save_cache_if_needed(self) -> None:
        if not self.config.cache_path:
            return
        cache_dir = os.path.dirname(self.config.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(self.config.cache_path, "wb") as handle:
            pickle.dump(self, handle)

    def score_spawn_edge(self, pattern: GraphPattern, spawn_node: int, spawn_edge: SpawnEdge) -> float:
        """Score one candidate expansion from learned pgmpy CPDs."""

        if not self.config.enabled:
            return 1.0
        if self.model is None:
            return 0.0
        src_label = str(pattern.node_labels[spawn_node])
        direction = str(spawn_edge.direction)
        edge_label = str(spawn_edge.edge_label)
        dst_label = str(spawn_edge.target_label)
        edge_prob = _cpd_probability(
            self.model,
            self.EDGE_LABEL,
            edge_label,
            {self.SRC_LABEL: src_label, self.DIRECTION: direction},
        )
        dst_prob = _cpd_probability(
            self.model,
            self.DST_LABEL,
            dst_label,
            {self.SRC_LABEL: src_label, self.DIRECTION: direction, self.EDGE_LABEL: edge_label},
        )
        return edge_prob * dst_prob

    def _frequent_edge_prior(self, src_label: str, dst_label: str, edge_label: str) -> float:
        left, right = sorted([str(src_label), str(dst_label)])
        return float(self.config.frequent_edge_priors.get((left, right, str(edge_label)), 0.0))

    def rank_spawn_edges(self, pattern: GraphPattern, spawn_node: int, candidates: Iterable[SpawnEdge]) -> List[CandidateScore]:
        """Rank and optionally prune VSpawn actions with pgmpy CPDs."""

        candidate_list = list(candidates)
        scored = [(self.score_spawn_edge(pattern, spawn_node, candidate), candidate) for candidate in candidate_list]
        scored.sort(key=lambda item: item[0], reverse=True)
        thresholded = [item for item in scored if item[0] >= self.config.min_score]
        threshold_pruned = len(scored) - len(thresholded)
        min_keep = max(0, self.config.min_keep_per_spawn_node)
        min_keep_target = min(min_keep, len(scored))
        min_keep_rescued = 0
        if min_keep and len(thresholded) < min_keep_target:
            min_keep_rescued = min_keep_target - len(thresholded)
            thresholded = scored[:min_keep_target]
        before_topk = len(thresholded)
        scored = thresholded
        if self.config.top_k_per_spawn_node is not None:
            scored = scored[: max(0, self.config.top_k_per_spawn_node)]
        topk_pruned = before_topk - len(scored)
        self.total_rank_calls += 1
        self.total_candidates_seen += len(candidate_list)
        self.total_candidates_kept += len(scored)
        self.total_threshold_pruned += threshold_pruned
        self.total_topk_pruned += topk_pruned
        self.total_min_keep_rescued += min_keep_rescued
        self.last_rank_snapshot = [
            (score, f"{edge.from_node}->{edge.to_node} {edge.direction}:{edge.edge_label}->{edge.target_label}")
            for score, edge in scored[:5]
        ]
        return scored

    def pruning_summary(self) -> Dict[str, object]:
        """Return observable pruning statistics for demo/debug printing."""

        return {
            "backend": "pgmpy",
            "rank_calls": self.total_rank_calls,
            "candidates_seen": self.total_candidates_seen,
            "candidates_kept": self.total_candidates_kept,
            "candidates_pruned": self.total_candidates_seen - self.total_candidates_kept,
            "tau_p": self.config.min_score,
            "threshold_pruned": self.total_threshold_pruned,
            "topk_pruned": self.total_topk_pruned,
            "min_keep_rescued": self.total_min_keep_rescued,
            "frequent_edge_prior_count": len(self.config.frequent_edge_priors),
            "frequent_prior_weight": self.config.frequent_prior_weight,
            "top_snapshot": self.last_rank_snapshot,
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
            "GARplusMiner Pattern BN now requires pgmpy and pandas. "
            "Install them in this environment, e.g. `pip install pgmpy pandas`."
        ) from exc
    return pd, ModelCls, EstimatorCls


def _cpd_probability(model, variable: str, state: str, evidence: Dict[str, str]) -> float:
    """Read a local CPD probability with graceful zero for unseen states."""

    cpd = model.get_cpds(variable)
    if cpd is None:
        return 0.0
    try:
        variable_states = list(cpd.state_names.get(variable, []))
        if state not in variable_states:
            return 0.0
        state_index = variable_states.index(state)
        values = cpd.values
        for evidence_var in cpd.variables[1:]:
            evidence_states = list(cpd.state_names.get(evidence_var, []))
            evidence_value = evidence.get(evidence_var)
            if evidence_value not in evidence_states:
                return 0.0
            values = values.take(evidence_states.index(evidence_value), axis=1)
        return float(values[state_index])
    except Exception:
        return 0.0











