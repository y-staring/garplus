from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from garplus_pattern_utils import graph_edges_as_set
from garplus_types import GARPlusRule, PatternTreeNode, RuleTreeNode


def save_mining_outputs(
    output_dir: str,
    mined_rules: list[GARPlusRule],
    pattern_tree_nodes: list[PatternTreeNode],
    pattern_tree_edges: list[dict],
    rule_tree_nodes: list[RuleTreeNode],
    rule_tree_edges: list[dict],
    summary: dict,
) -> None:
    """
    Save normalized outputs under OUTPUT_PATH.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1) mined_rules.{csv,json} (deduplicated by rule_id).
    rules_before = list(mined_rules)
    dedup_map: dict[str, GARPlusRule] = {}
    for rule in rules_before:
        # Keep the best rule per id by (confidence, support, lift, smaller X).
        if rule.rule_id not in dedup_map:
            dedup_map[rule.rule_id] = rule
            continue
        prev = dedup_map[rule.rule_id]
        cur_key = (rule.confidence, rule.support, rule.lift, -rule.body_size)
        prev_key = (prev.confidence, prev.support, prev.lift, -prev.body_size)
        if cur_key > prev_key:
            dedup_map[rule.rule_id] = rule

    rules_after = list(dedup_map.values())
    rules_after.sort(key=lambda r: (-r.confidence, -r.support, -r.lift, r.body_size, r.rule_id))

    rules_rows_csv = []
    rules_rows_json = []
    for rule in rules_after:
        X_sorted = tuple(rule.X)
        rules_rows_csv.append(
            {
                "rule_id": rule.rule_id,
                "pattern_id": rule.pattern_id,
                "pattern_edges": rule.pattern_edges,
                "pattern_nodes": rule.pattern_nodes,
                "X": " && ".join(X_sorted),
                "p0": rule.p0,
                "X_size": int(rule.X_size),
                "support": int(rule.support),
                "X_support": int(rule.X_support),
                "p0_support": int(rule.p0_support),
                "confidence": float(rule.confidence),
                "lift": float(rule.lift),
                "pattern_support": int(rule.pattern_support),
                "source_rule_node_id": rule.source_rule_node_id,
            }
        )
        rules_rows_json.append(
            {
                "rule_id": rule.rule_id,
                "pattern_id": rule.pattern_id,
                "pattern_edges": rule.pattern_edges,
                "pattern_nodes": rule.pattern_nodes,
                "X": list(X_sorted),
                "p0": rule.p0,
                "X_size": int(rule.X_size),
                "support": int(rule.support),
                "X_support": int(rule.X_support),
                "p0_support": int(rule.p0_support),
                "confidence": float(rule.confidence),
                "lift": float(rule.lift),
                "pattern_support": int(rule.pattern_support),
                "source_rule_node_id": rule.source_rule_node_id,
            }
        )

    mined_rules_columns = [
        "rule_id",
        "pattern_id",
        "pattern_edges",
        "pattern_nodes",
        "X",
        "p0",
        "X_size",
        "support",
        "X_support",
        "p0_support",
        "confidence",
        "lift",
        "pattern_support",
        "source_rule_node_id",
    ]
    pd.DataFrame(rules_rows_csv, columns=mined_rules_columns).to_csv(output_path / "mined_rules.csv", index=False)
    with open(output_path / "mined_rules.json", "w", encoding="utf-8") as f:
        json.dump(rules_rows_json, f, indent=2, ensure_ascii=False)

    # 2) Pattern tree.
    pattern_rows = []
    for node in pattern_tree_nodes:
        pattern_rows.append(
            {
                "node_id": node.node_id,
                "parent_id": node.parent_id,
                "level": int(node.level),
                "edge_count": int(node.pattern.number_of_edges()),
                "node_count": int(node.pattern.number_of_nodes()),
                "support": int(node.support),
                "bn_score": float(node.bn_score),
                "is_frequent": bool(node.is_frequent),
                "edges": " | ".join(f"{u}--{v}" for u, v in sorted(graph_edges_as_set(node.pattern))),
                "nodes": " | ".join(sorted(str(n) for n in node.pattern.nodes())),
            }
        )
    pattern_nodes_columns = [
        "node_id",
        "parent_id",
        "level",
        "edge_count",
        "node_count",
        "support",
        "bn_score",
        "is_frequent",
        "edges",
        "nodes",
    ]
    pd.DataFrame(pattern_rows, columns=pattern_nodes_columns).to_csv(output_path / "pattern_tree_nodes.csv", index=False)
    pattern_edges_columns = [
        "parent_id",
        "child_id",
        "parent_level",
        "child_level",
        "added_extension",
        "bn_score",
    ]
    pd.DataFrame(pattern_tree_edges, columns=pattern_edges_columns).to_csv(output_path / "pattern_tree_edges.csv", index=False)

    # 3) Rule tree nodes/edges.
    rule_rows = []
    for node in rule_tree_nodes:
        rule_rows.append(
            {
                "node_id": node.node_id,
                "pattern_id": node.pattern_id,
                "parent_id": node.parent_id,
                "level": int(node.level),
                "added_predicate": node.added_predicate,
                "X": " && ".join(sorted(node.X)),
                "p0": node.p0,
                "support": int(node.support),
                "X_support": int(node.X_support),
                "p0_support": int(node.p0_support),
                "confidence": float(node.confidence),
                "lift": float(node.lift),
                "bn_score": float(node.bn_score),
                "is_valid": bool(node.is_valid),
                "is_pruned_by_bn": bool(node.is_pruned_by_bn),
                "is_pruned_by_support": bool(node.is_pruned_by_support),
                "children": json.dumps(node.children, ensure_ascii=False),
            }
        )
    rule_nodes_columns = [
        "node_id",
        "pattern_id",
        "parent_id",
        "level",
        "added_predicate",
        "X",
        "p0",
        "support",
        "X_support",
        "p0_support",
        "confidence",
        "lift",
        "bn_score",
        "is_valid",
        "is_pruned_by_bn",
        "is_pruned_by_support",
        "children",
    ]
    pd.DataFrame(rule_rows, columns=rule_nodes_columns).to_csv(output_path / "rule_tree_nodes.csv", index=False)
    rule_edges_columns = [
        "parent_id",
        "child_id",
        "pattern_id",
        "parent_level",
        "child_level",
        "added_predicate",
    ]
    pd.DataFrame(rule_tree_edges, columns=rule_edges_columns).to_csv(output_path / "rule_tree_edges.csv", index=False)

    # 4) Summary.
    summary = dict(summary)
    summary["num_rules_before_dedup"] = int(len(rules_before))
    summary["num_rules_after_dedup"] = int(len(rules_after))
    summary["num_pattern_tree_nodes"] = int(len(pattern_tree_nodes))
    summary["num_pattern_tree_edges"] = int(len(pattern_tree_edges))
    summary["num_rule_tree_nodes"] = int(len(rule_tree_nodes))
    summary["num_rule_tree_edges"] = int(len(rule_tree_edges))

    with open(output_path / "mining_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
