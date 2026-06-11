from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx


@dataclass(frozen=True)
class PredicateMeta:
    predicate_id: str
    kind: str
    family: str
    variables: tuple[str, ...]

    edge_label: str | None = None
    polarity: str | None = None
    expression: str | None = None

    can_be_head: bool = True
    can_be_body: bool = True


@dataclass
class PatternTreeNode:
    node_id: str
    pattern: nx.Graph
    level: int
    variables: tuple[str, ...]

    parent_id: str | None = None
    added_extension: Any | None = None
    children: list[str] = field(default_factory=list)

    canonical_id: str | None = None
    iso_group_id: str | None = None

    support: int = 0
    is_frequent: bool = False
    match_ids: tuple[str, ...] = field(default_factory=tuple)
    match_bindings: dict[str, dict[str, Any]] | None = None

    predicate_table_id: str | None = None
    available_predicates: tuple[str, ...] = field(default_factory=tuple)
    body_candidates: tuple[str, ...] = field(default_factory=tuple)
    head_candidates: tuple[str, ...] = field(default_factory=tuple)

    rule_root_ids: list[str] = field(default_factory=list)
    rule_node_ids: list[str] = field(default_factory=list)
    valid_rule_ids: list[str] = field(default_factory=list)

    bn_score: float = 1.0
    is_pruned_by_bn: bool = False
    is_pruned_by_support: bool = False


@dataclass
class RuleTreeNode:
    node_id: str
    pattern_id: str
    X: frozenset[str]
    p0: str
    level: int

    parent_id: str | None = None
    added_predicate: str | None = None
    children: list[str] = field(default_factory=list)

    support: int = 0
    X_support: int = 0
    p0_support: int = 0
    confidence: float = 0.0
    lift: float = 0.0

    contains_negative_edge: bool = False
    negative_edge_predicates: tuple[str, ...] = field(default_factory=tuple)

    is_minimal: bool = True
    is_duplicate: bool = False
    is_valid: bool = False
    is_output_rule: bool = False

    bn_score: float = 1.0
    is_pruned_by_bn: bool = False
    is_pruned_by_support: bool = False
    is_pruned_by_confidence: bool = False
    prune_reason: str | None = None


@dataclass(frozen=True)
class GARPlusRule:
    rule_id: str
    pattern_id: str

    pattern_edges: str
    pattern_nodes: str
    pattern_support: int

    X: tuple[str, ...]
    p0: str
    X_size: int

    support: int
    X_support: int
    p0_support: int
    confidence: float
    lift: float

    contains_negative_edge: bool
    negative_edge_predicates: tuple[str, ...]

    source_rule_node_id: str
    validation_stage: str = "coarse"


@dataclass(frozen=True)
class PatternState:
    pattern_id: str
    graph: nx.Graph
    edge_count: int
    node_count: int

    variables: tuple[str, ...] = field(default_factory=tuple)

    parent_id: str | None = None
    added_extension: Any | None = None

    support: int | None = None
    bn_score: float | None = None

    source: str | None = None