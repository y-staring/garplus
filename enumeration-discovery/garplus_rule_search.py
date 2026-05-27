from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from garplus_types import GARPlusRule, PatternTreeNode, RuleTreeNode
from garplus_pattern_utils import graph_edges_as_set


BN_FALLBACK_SCORE = 0.1


def load_predicate_repository(repo_path: str) -> dict:
    with open(repo_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predicate_table(table_path: str) -> pd.DataFrame:
    return pd.read_csv(table_path)


def load_family_bn_edges(family_bns_dir: str) -> dict[str, dict]:
    """
    Load the learned family-wise Predicate-BNs as undirected adjacency maps.

    Expected layout:
      family_bns/
        family_<name>/
          result.json  (expects {"status": "learned"} when usable)
          edges.csv    (expects columns: source,target)
    """
    family_bns_path = Path(family_bns_dir)
    family_bn_states: dict[str, dict] = {}
    if not family_bns_path.exists():
        return family_bn_states

    for family_dir in sorted(family_bns_path.glob("family_*")):
        if not family_dir.is_dir():
            continue

        family_name = family_dir.name[len("family_") :]
        result_path = family_dir / "result.json"
        edges_path = family_dir / "edges.csv"

        status = "skipped"
        if result_path.exists():
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                status = str(result.get("status", "skipped"))
            except Exception:
                status = "skipped"

        neighbors: dict[str, set[str]] = {}
        nodes: set[str] = set()

        if status == "learned" and edges_path.exists():
            try:
                edges_df = pd.read_csv(edges_path)
                for _, row in edges_df.iterrows():
                    src = str(row["source"])
                    dst = str(row["target"])
                    nodes.add(src)
                    nodes.add(dst)
                    neighbors.setdefault(src, set()).add(dst)
                    neighbors.setdefault(dst, set()).add(src)
            except Exception:
                status = "skipped"
                neighbors = {}
                nodes = set()

        family_bn_states[family_name] = {
            "neighbors": neighbors,
            "nodes": nodes,
            "status": status,
        }

    return family_bn_states


def get_predicate_family_map(repository: dict) -> dict[str, str]:
    pred_family: dict[str, str] = {}
    for pred in repository.get("predicates", []):
        pid = pred.get("pid")
        family = pred.get("family", "unknown")
        if pid is not None:
            pred_family[str(pid)] = str(family)
    return pred_family


def get_candidate_predicates(
    repository: dict,
    table: pd.DataFrame,
    exclude_families: set[str] | None = None,
    exclude_sources: set[str] | None = None,
    min_support: int = 1,
    max_predicates: int | None = None,
) -> list[str]:
    exclude_families = set() if exclude_families is None else set(exclude_families)
    exclude_sources = set() if exclude_sources is None else set(exclude_sources)

    table_columns = set(table.columns) - {"pattern_id"}
    support_series = table.drop(columns=["pattern_id"], errors="ignore").sum().sort_values(ascending=False)

    candidates: list[tuple[str, int]] = []
    for pred in repository.get("predicates", []):
        pid = str(pred.get("pid"))
        family = str(pred.get("family", "unknown"))
        source = str(pred.get("source", "unknown"))

        if pid not in table_columns:
            continue
        if family in exclude_families:
            continue
        if source in exclude_sources:
            continue

        support = int(support_series.get(pid, 0))
        if support < min_support:
            continue
        candidates.append((pid, support))

    candidates.sort(key=lambda x: (-x[1], x[0]))
    if max_predicates is not None:
        candidates = candidates[:max_predicates]
    return [pid for pid, _ in candidates]


def compute_X_mask(table: pd.DataFrame, X: frozenset[str]) -> pd.Series:
    if not X:
        return pd.Series(True, index=table.index)
    return table[list(X)].all(axis=1)


def compute_rule_stats(table: pd.DataFrame, X: frozenset[str], p0: str) -> dict:
    num_patterns = int(table.shape[0])
    X_mask = compute_X_mask(table, X)
    X_support = int(X_mask.sum())
    p0_support = int(table[p0].sum())

    if X_support == 0:
        support = 0
        confidence = 0.0
    else:
        support = int((X_mask & (table[p0] == 1)).sum())
        confidence = float(support / X_support)

    p0_prob = float(p0_support / num_patterns) if num_patterns > 0 else 0.0
    lift = 0.0 if p0_prob <= 0 else float(confidence / p0_prob)

    return {
        "X_support": X_support,
        "support": support,
        "confidence": confidence,
        "p0_support": p0_support,
        "lift": lift,
    }


def bn_rule_score(
    X: frozenset[str],
    p0: str,
    family_bn_states: dict,
    predicate_family: dict[str, str],
) -> float:
    """
    Head-aware predicate BN score for rule node Q[x](X -> p0).

    BN pruning is heuristic; exact support/confidence still validate rules.
    """
    if not X:
        return 1.0

    p0_family = predicate_family.get(p0)
    if p0_family is None:
        return BN_FALLBACK_SCORE

    family_state = family_bn_states.get(p0_family)
    if not family_state or family_state.get("status") != "learned":
        return BN_FALLBACK_SCORE

    neighbors: dict[str, set[str]] = family_state.get("neighbors", {})
    p0_neighbors = set(neighbors.get(p0, set()))
    if not p0_neighbors:
        return BN_FALLBACK_SCORE

    X_set = set(X)
    if X_set & p0_neighbors:
        return 1.0

    for b in X_set:
        for hop in neighbors.get(b, set()):
            if p0 in neighbors.get(hop, set()):
                return 0.5

    return BN_FALLBACK_SCORE


def rank_rule_extensions_by_bn(
    X: frozenset[str],
    p0: str,
    candidates: list[str],
    family_bn_states: dict,
    predicate_family: dict[str, str],
    tau_bn: float,
    top_k: int | None,
) -> list[tuple[str, float]]:
    """
    Rank candidate X extensions by bn_rule_score(X ∪ {p}, p0).
    """
    scored: list[tuple[str, float]] = []
    for p in candidates:
        score = float(bn_rule_score(frozenset(set(X) | {p}), p0, family_bn_states, predicate_family))
        if score >= tau_bn:
            scored.append((p, score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    if top_k is not None:
        scored = scored[:top_k]
    return scored


def _rule_node_id(pattern_id: str, X: frozenset[str], p0: str) -> str:
    X_key = "&&".join(sorted(X))
    return f"{pattern_id}::[{X_key}]=>{p0}"


def predicate_tree_search_for_pattern(
    pattern_node: PatternTreeNode,
    predicate_table: pd.DataFrame,
    repository: dict,
    family_bn_states: dict,
    sigma_rule: int,
    delta: float,
    max_body_size: int,
    tau_predicate_bn: float,
    top_k_predicate_extensions: int | None,
    min_predicate_support: int,
    max_predicates: int | None,
    exclude_families: set[str] | None,
    head_families: set[str] | None = None,
) -> tuple[list[GARPlusRule], list[RuleTreeNode], dict]:
    """
    Rule Search Tree for a fixed pattern Q.

    The tree roots are Q[x](empty -> p0). Nodes expand by adding one predicate
    into the body: Q[x](X -> p0) -> Q[x](X ∪ {p} -> p0).
    """
    start_time = time.time()
    head_families = set() if head_families is None else set(head_families)

    if predicate_table.empty:
        return [], [], {
            "pattern_id": pattern_node.node_id,
            "roots": 0,
            "nodes": 0,
            "valid_rules": 0,
            "bn_pruned": 0,
            "support_pruned": 0,
            "elapsed_seconds": float(time.time() - start_time),
            "warnings": ["predicate_table_q is empty"],
            "rule_level_stats": {},
            "rule_tree_edges": [],
        }

    working_table = predicate_table.drop(columns=["pattern_id"], errors="ignore").copy()
    predicate_family = get_predicate_family_map(repository)

    candidates_all = get_candidate_predicates(
        repository=repository,
        table=predicate_table,
        exclude_families=exclude_families,
        exclude_sources=None,
        min_support=min_predicate_support,
        max_predicates=max_predicates,
    )

    warnings: list[str] = []
    head_candidates: list[str] = []
    if head_families:
        head_candidates = [p for p in candidates_all if predicate_family.get(p) in head_families]
        if not head_candidates and candidates_all:
            warnings.append("HEAD_FAMILIES matched no p0 predicates; fallback to all predicates as p0.")
            head_candidates = list(candidates_all)
    else:
        head_candidates = list(candidates_all)

    if not head_candidates:
        warnings.append("No p0 predicates available; skip rule tree search.")
        return [], [], {
            "pattern_id": pattern_node.node_id,
            "roots": 0,
            "nodes": 0,
            "valid_rules": 0,
            "bn_pruned": 0,
            "support_pruned": 0,
            "elapsed_seconds": float(time.time() - start_time),
            "warnings": warnings,
            "rule_level_stats": {},
            "rule_tree_edges": [],
        }

    pattern_edges_str = " | ".join(f"{u}--{v}" for u, v in sorted(graph_edges_as_set(pattern_node.pattern)))
    pattern_nodes_str = " | ".join(sorted(str(n) for n in pattern_node.pattern.nodes()))

    rules_out: list[GARPlusRule] = []
    rule_node_map: dict[str, RuleTreeNode] = {}
    rule_tree_edges: list[dict] = []
    rule_level_stats: dict[str, dict] = {}
    bn_pruned = 0
    support_pruned = 0

    for p0 in head_candidates:
        root_X = frozenset()
        root_id = _rule_node_id(pattern_node.node_id, root_X, p0)

        root_stats = compute_rule_stats(working_table, root_X, p0)
        root_node = RuleTreeNode(
            node_id=root_id,
            pattern_id=pattern_node.node_id,
            X=root_X,
            p0=p0,
            level=0,
            parent_id=None,
            added_predicate=None,
            support=int(root_stats["support"]),
            X_support=int(root_stats["X_support"]),
            p0_support=int(root_stats["p0_support"]),
            confidence=float(root_stats["confidence"]),
            lift=float(root_stats["lift"]),
            bn_score=1.0,
            is_valid=bool(root_stats["support"] >= sigma_rule and root_stats["confidence"] >= delta),
            is_pruned_by_bn=False,
            is_pruned_by_support=bool(root_stats["support"] < sigma_rule),
            children=[],
        )
        rule_node_map[root_id] = root_node
        rule_level_stats.setdefault("0", {"nodes": 0, "expanded": 0, "valid": 0})
        rule_level_stats["0"]["nodes"] += 1

        if root_node.is_valid:
            rule_id = f"{pattern_node.node_id}::=>{p0}"
            rules_out.append(
                GARPlusRule(
                    rule_id=rule_id,
                    pattern_id=pattern_node.node_id,
                    pattern_edges=pattern_edges_str,
                    pattern_nodes=pattern_nodes_str,
                    X=tuple(),
                    p0=p0,
                    X_size=0,
                    support=root_node.support,
                    X_support=root_node.X_support,
                    p0_support=root_node.p0_support,
                    confidence=root_node.confidence,
                    lift=root_node.lift,
                    pattern_support=pattern_node.support,
                    source_rule_node_id=root_id,
                )
            )
            rule_level_stats["0"]["valid"] += 1

        frontier: list[str] = [root_id]
        while frontier:
            cur_id = frontier.pop(0)
            cur_node = rule_node_map[cur_id]

            if cur_node.level >= max_body_size:
                continue

            # ===== Rule Tree Expansion / Horizontal Spawning =====
            # A rule node Q[x](X -> p0) is expanded by adding one
            # predicate p into the body, producing Q[x](X ∪ {p} -> p0).
            # This is the horizontal spawning step in GFD/GAR discovery.

            if cur_node.level == 0:
                cur_node.bn_score = 1.0
            else:
                cur_node.bn_score = float(bn_rule_score(cur_node.X, cur_node.p0, family_bn_states, predicate_family))
                if cur_node.bn_score < tau_predicate_bn:
                    cur_node.is_pruned_by_bn = True
                    bn_pruned += 1
                    continue

            if cur_node.support < sigma_rule:
                cur_node.is_pruned_by_support = True
                support_pruned += 1
                continue

            used = set(cur_node.X) | {cur_node.p0}
            extension_candidates = [p for p in candidates_all if p not in used]
            next_level = str(cur_node.level + 1)
            rule_level_stats.setdefault(next_level, {"nodes": 0, "expanded": 0, "valid": 0})

            ranked = rank_rule_extensions_by_bn(
                X=cur_node.X,
                p0=cur_node.p0,
                candidates=extension_candidates,
                family_bn_states=family_bn_states,
                predicate_family=predicate_family,
                tau_bn=tau_predicate_bn,
                top_k=top_k_predicate_extensions,
            )

            for added_pred, ext_score in ranked:
                child_X = frozenset(set(cur_node.X) | {added_pred})
                child_id = _rule_node_id(pattern_node.node_id, child_X, cur_node.p0)

                if child_id in rule_node_map:
                    if child_id not in cur_node.children:
                        cur_node.children.append(child_id)
                    continue

                stats = compute_rule_stats(working_table, child_X, cur_node.p0)
                child_node = RuleTreeNode(
                    node_id=child_id,
                    pattern_id=pattern_node.node_id,
                    X=child_X,
                    p0=cur_node.p0,
                    level=cur_node.level + 1,
                    parent_id=cur_node.node_id,
                    added_predicate=added_pred,
                    support=int(stats["support"]),
                    X_support=int(stats["X_support"]),
                    p0_support=int(stats["p0_support"]),
                    confidence=float(stats["confidence"]),
                    lift=float(stats["lift"]),
                    bn_score=float(ext_score),
                    is_valid=bool(stats["support"] >= sigma_rule and stats["confidence"] >= delta),
                    is_pruned_by_bn=False,
                    is_pruned_by_support=bool(stats["support"] < sigma_rule),
                    children=[],
                )

                rule_node_map[child_id] = child_node
                cur_node.children.append(child_id)
                rule_tree_edges.append(
                    {
                        "parent_id": cur_node.node_id,
                        "child_id": child_id,
                        "pattern_id": pattern_node.node_id,
                        "parent_level": int(cur_node.level),
                        "child_level": int(child_node.level),
                        "added_predicate": added_pred,
                    }
                )
                rule_level_stats[next_level]["nodes"] += 1
                rule_level_stats[next_level]["expanded"] += 1

                if child_node.is_valid:
                    X_sorted = tuple(sorted(child_X))
                    rule_id = f"{pattern_node.node_id}::{'&&'.join(X_sorted)}=>{cur_node.p0}"
                    rules_out.append(
                        GARPlusRule(
                            rule_id=rule_id,
                            pattern_id=pattern_node.node_id,
                            pattern_edges=pattern_edges_str,
                            pattern_nodes=pattern_nodes_str,
                            X=X_sorted,
                            p0=cur_node.p0,
                            X_size=len(X_sorted),
                            support=child_node.support,
                            X_support=child_node.X_support,
                            p0_support=child_node.p0_support,
                            confidence=child_node.confidence,
                            lift=child_node.lift,
                            pattern_support=pattern_node.support,
                            source_rule_node_id=child_id,
                        )
                    )
                    rule_level_stats[next_level]["valid"] += 1

                if child_node.level < max_body_size and child_node.support >= sigma_rule:
                    frontier.append(child_id)

    return rules_out, list(rule_node_map.values()), {
        "pattern_id": pattern_node.node_id,
        "roots": int(len(head_candidates)),
        "nodes": int(len(rule_node_map)),
        "valid_rules": int(len(rules_out)),
        "bn_pruned": int(bn_pruned),
        "support_pruned": int(support_pruned),
        "elapsed_seconds": float(time.time() - start_time),
        "warnings": warnings,
        "rule_level_stats": rule_level_stats,
        "rule_tree_edges": rule_tree_edges,
    }
