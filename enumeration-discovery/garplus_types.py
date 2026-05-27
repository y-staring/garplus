from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx


@dataclass
class PatternTreeNode:
    """
    Node in the Pattern Search Tree (vertical spawning).

    Each node represents one pattern Q, and parent-child relationships represent
    structural extensions Q -> extend(Q, gamma).
    """

    node_id: str
    pattern: nx.Graph
    level: int
    parent_id: str | None
    added_extension: Any | None
    support: int
    bn_score: float
    is_frequent: bool
    children: list[str]


@dataclass
class RuleTreeNode:
    """
    Node in the Rule Search Tree (horizontal spawning) under a fixed pattern Q.

    A rule node represents one GAR+ rule:
        phi = Q[x](X -> p0)

    where:
      - pattern_id identifies Q
      - body identifies X
      - head identifies p0
    """

    node_id: str
    pattern_id: str
    X: frozenset[str]
    p0: str
    level: int
    parent_id: str | None
    added_predicate: str | None
    support: int
    X_support: int
    p0_support: int
    confidence: float
    lift: float
    bn_score: float
    is_valid: bool
    is_pruned_by_bn: bool
    is_pruned_by_support: bool
    children: list[str]


@dataclass(frozen=True)
class GARPlusRule:
    """
    Final normalized GAR+ rule output.
    """

    rule_id: str
    pattern_id: str
    pattern_edges: str
    pattern_nodes: str
    X: tuple[str, ...]
    p0: str
    X_size: int
    support: int
    X_support: int
    p0_support: int
    confidence: float
    lift: float
    pattern_support: int
    source_rule_node_id: str


@dataclass(frozen=True)
class PatternState:
    """
    Lightweight internal pattern state used by seed initialization helpers.

    The main mining loop uses PatternTreeNode.
    """

    pattern_id: str
    graph: nx.Graph
    edge_count: int
    node_count: int
    parent_id: str | None = None
    added_edge: tuple[Any, Any] | None = None
    support: int | None = None
    bn_score: float | None = None
