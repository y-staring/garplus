from __future__ import annotations

"""
Level-wise GAR+ miner (tree-shaped vertical + horizontal spawning).

This file keeps the main pipeline easy to read:
  - load data / BN states
  - run mining loop
  - save outputs under OUTPUT_PATH

Implementation details (data structures, tree expansion logic, BN scoring, I/O)
are split into helper modules under `enumeration-discovery/`.
"""

import importlib.util
import sys
import time
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent

# Allow running from the repo root by ensuring `enumeration-discovery` is importable.
current_dir_str = str(CURRENT_DIR)
if current_dir_str not in sys.path:
    sys.path.insert(0, current_dir_str)

from garplus_outputs import save_mining_outputs
from garplus_pattern_bn import filter_pattern_extensions_by_bn, load_pattern_bn_state
from garplus_pattern_utils import (
    _short_id,
    build_union_graph,
    compute_pattern_match_ids,
    extend_pattern,
    generate_pattern_extensions,
    initialize_seed_patterns,
    pattern_signature,
)
from garplus_rule_search import (
    load_family_bn_edges,
    load_predicate_repository,
    load_predicate_table,
    predicate_tree_search_for_pattern,
)
from garplus_types import GARPlusRule, PatternTreeNode, RuleTreeNode

# ---------------------------------------------------------------------
# Paths / Config
# ---------------------------------------------------------------------

# pipeline: pick_patterns.py -> predicate_construction.py -> build_pattern_edge_node_bn.py -> this miner
SELECTED_PATH = str(CURRENT_DIR / "processed" / "ppi" / "ppi_selected.pt")
REPO_PATH = str(CURRENT_DIR / "processed" / "ppi" / "global_predicate_repo" / "global_predicate_repository.json")
TABLE_PATH = str(CURRENT_DIR / "processed" / "ppi" / "global_predicate_repo" / "global_predicate_table_full.csv")
FAMILY_BNS_PATH = str(CURRENT_DIR / "processed" / "ppi" / "global_predicate_repo" / "family_bns")
PATTERN_BNS_PATH = str(CURRENT_DIR / "processed" / "ppi" / "pattern_multi_bn")
OUTPUT_PATH = str(CURRENT_DIR / "processed" / "ppi" / "levelwise_garplus_mining")

SIGMA_PATTERN = 5
SIGMA_RULE = 5
DELTA = 0.8
MAX_PATTERN_EDGES = 3
MAX_X_SIZE = 3
TAU_PATTERN_BN = 0.0
TAU_PREDICATE_BN = 0.0
TOP_K_PATTERN_EXTENSIONS = 50
TOP_K_PREDICATE_EXTENSIONS = 50
MIN_PREDICATE_SUPPORT = 2
MAX_PREDICATES = 300
EXCLUDE_FAMILIES = {"qualifications", "edge_label_other"}
HEAD_FAMILIES: set[str] = set()
PATTERN_SUPPORT_MODE = "edge_subset"  # "edge_subset" | "exact_signature"


def _load_pattern_bn_module():
    """
    Load the Pattern-BN module.

    In some layouts the file lives next to this miner as
    `enumeration-discovery/build_pattern_edge_node_bn.py`, while in others it
    lives under `enumeration-discovery/BNlearning/build_pattern_edge_node_bn.py`.
    """
    candidates = [
        CURRENT_DIR / "build_pattern_edge_node_bn.py",
        CURRENT_DIR / "BNlearning" / "build_pattern_edge_node_bn.py",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        spec = importlib.util.spec_from_file_location("pattern_bn_module", str(candidate))
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        return module
    raise ModuleNotFoundError(
        "Cannot find build_pattern_edge_node_bn.py in either "
        f"{candidates[0]} or {candidates[1]}"
    )


def levelwise_garplus_mine(
    pattern_graphs,  #sampled graphs
    predicate_table: pd.DataFrame,
    repository: dict,
    pattern_bn_state: dict | None,
    pattern_bn_module,
    family_bn_states: dict,
    sigma_pattern: int,
    sigma_rule: int,
    delta: float,
    max_pattern_edges: int = 3,
    max_X_size: int = 3,
    tau_pattern_bn: float = 0.0,
    tau_predicate_bn: float = 0.0,
    top_k_pattern_extensions: int | None = 50,
    top_k_predicate_extensions: int | None = 50,
    min_predicate_support: int = 2,
    max_predicates: int | None = 300,
) -> tuple[
    list[GARPlusRule],
    list[PatternTreeNode],
    list[dict],
    list[RuleTreeNode],
    list[dict],
    dict,
]:
    start_time = time.time()
    #union all subgraph to a big one
    union_graph = build_union_graph(pattern_graphs)

    # Initialize pattern tree roots from sampled patterns S_G.
    seed_states = initialize_seed_patterns(pattern_graphs, max_seed_edges=1)
    pattern_nodes: dict[str, PatternTreeNode] = {}
    pattern_tree_edges: list[dict] = []
    seen_patterns: set[str] = set()

    current_level_ids: list[str] = []
    for seed in seed_states:
        pid = seed.pattern_id
        node = PatternTreeNode(
            node_id=pid,
            pattern=seed.graph.copy(),
            level=int(seed.edge_count),
            parent_id=None,
            added_extension=None,
            support=0,
            bn_score=1.0,
            is_frequent=False,
            children=[],
        )
        pattern_nodes[pid] = node
        current_level_ids.append(pid)
        seen_patterns.add(pid)

    mined_rules: list[GARPlusRule] = []
    rule_tree_nodes: list[RuleTreeNode] = []
    rule_tree_edges: list[dict] = []

    pattern_level_stats: dict[str, dict] = {}
    rule_level_stats_global: dict[str, dict] = {}

    pruning_stats = {
        "pattern_support_pruned": 0,
        "pattern_bn_pruned": 0,
        "rule_bn_pruned": 0,
        "rule_support_pruned": 0,
    }
    #要多大就挖多少层
    for edge_level in range(1, max_pattern_edges + 1):
        if not current_level_ids:
            #TODO seed有问题，改掉
            print(f"[Pattern-Level {edge_level}] level is empty, stop early.")
            break

        print(f"[Pattern-Level {edge_level}] current_nodes={len(current_level_ids)}")
        next_level_ids: list[str] = []

        frequent = 0
        support_pruned = 0
        bn_pruned = 0
        generated_children = 0

        for pattern_id in current_level_ids:
            node = pattern_nodes[pattern_id]

            match_ids = compute_pattern_match_ids(node.pattern, pattern_graphs, support_mode=PATTERN_SUPPORT_MODE)
            #计算当前pattern的匹配在sampled graph里的次数，作为support
            node.support = int(len(match_ids))
            node.is_frequent = bool(node.support >= sigma_pattern)

            if not node.is_frequent:
                support_pruned += 1
                pruning_stats["pattern_support_pruned"] += 1
                continue

            frequent += 1

            if "pattern_id" in predicate_table.columns:
                predicate_table_q = predicate_table[predicate_table["pattern_id"].isin(match_ids)].copy()
            else:
                predicate_table_q = predicate_table.copy()

            rules_q, rule_nodes_q, rule_summary_q = predicate_tree_search_for_pattern(
                pattern_node=node,
                predicate_table=predicate_table_q,
                repository=repository,
                family_bn_states=family_bn_states,
                sigma_rule=sigma_rule,
                delta=delta,
                max_body_size=max_X_size,
                tau_predicate_bn=tau_predicate_bn,
                top_k_predicate_extensions=top_k_predicate_extensions,
                min_predicate_support=min_predicate_support,
                max_predicates=max_predicates,
                exclude_families=EXCLUDE_FAMILIES,
                head_families=HEAD_FAMILIES,
            )
            mined_rules.extend(rules_q)
            rule_tree_nodes.extend(rule_nodes_q)
            rule_tree_edges.extend(rule_summary_q.get("rule_tree_edges", []))

            pruning_stats["rule_bn_pruned"] += int(rule_summary_q.get("bn_pruned", 0))
            pruning_stats["rule_support_pruned"] += int(rule_summary_q.get("support_pruned", 0))
            rule_level_stats_global[node.node_id] = rule_summary_q.get("rule_level_stats", {})

            print(
                f"[Rule-Tree] pattern={_short_id(node.node_id)} roots={rule_summary_q.get('roots')} "
                f"nodes={rule_summary_q.get('nodes')} valid_rules={rule_summary_q.get('valid_rules')} "
                f"bn_pruned={rule_summary_q.get('bn_pruned')} support_pruned={rule_summary_q.get('support_pruned')}"
            )

            # ===== Pattern Tree Expansion / Vertical Spawning =====
            # This corresponds to the vertical spawning step in GFD/GAR discovery:
            # by adding one structural extension gamma, expand parent pattern Q
            # into child pattern Q_gamma = extend_pattern(Q, gamma).
            # Pattern BN B_P is used to prune/rank structural extensions.
            if node.pattern.number_of_edges() < max_pattern_edges:
                candidates = generate_pattern_extensions(node.pattern, union_graph)
                scored_candidates = filter_pattern_extensions_by_bn(
                    current_pattern=node.pattern,
                    candidate_extensions=candidates,
                    pattern_bn_state=pattern_bn_state or {},
                    pattern_bn_module=pattern_bn_module,
                    tau_pattern_bn=tau_pattern_bn,
                    top_k=top_k_pattern_extensions,
                )

                for _, gamma, score in scored_candidates:
                    child_pattern = extend_pattern(node.pattern, gamma)
                    child_id = pattern_signature(child_pattern)
                    if child_id in seen_patterns:
                        continue

                    child_node = PatternTreeNode(
                        node_id=child_id,
                        pattern=child_pattern,
                        level=int(child_pattern.number_of_edges()),
                        parent_id=node.node_id,
                        added_extension=gamma,
                        support=0,
                        bn_score=float(score),
                        is_frequent=False,
                        children=[],
                    )

                    pattern_nodes[child_id] = child_node
                    seen_patterns.add(child_id)
                    node.children.append(child_id)
                    next_level_ids.append(child_id)
                    generated_children += 1

                    pattern_tree_edges.append(
                        {
                            "parent_id": node.node_id,
                            "child_id": child_id,
                            "parent_level": int(node.level),
                            "child_level": int(child_node.level),
                            "added_extension": str(gamma),
                            "bn_score": float(score),
                        }
                    )

                bn_pruned += max(0, len(candidates) - len(scored_candidates))
                pruning_stats["pattern_bn_pruned"] += max(0, len(candidates) - len(scored_candidates))

        pattern_level_stats[str(edge_level)] = {
            "current_nodes": int(len(current_level_ids)),
            "frequent": int(frequent),
            "support_pruned": int(support_pruned),
            "bn_pruned": int(bn_pruned),
            "generated_children": int(generated_children),
        }

        print(
            f"[Pattern-Level {edge_level}] frequent={frequent} support_pruned={support_pruned} "
            f"bn_pruned={bn_pruned} generated_children={generated_children} cumulative_rules={len(mined_rules)}"
        )

        current_level_ids = next_level_ids

    summary = {
        "elapsed_seconds": float(time.time() - start_time),
        "sigma_pattern": int(sigma_pattern),
        "sigma_rule": int(sigma_rule),
        "delta": float(delta),
        "max_pattern_edges": int(max_pattern_edges),
        "max_X_size": int(max_X_size),
        "tau_pattern_bn": float(tau_pattern_bn),
        "tau_predicate_bn": float(tau_predicate_bn),
        "top_k_pattern_extensions": None if top_k_pattern_extensions is None else int(top_k_pattern_extensions),
        "top_k_predicate_extensions": None if top_k_predicate_extensions is None else int(top_k_predicate_extensions),
        "pattern_level_stats": pattern_level_stats,
        "rule_level_stats": rule_level_stats_global,
        "pruning_stats": pruning_stats,
        "head_families": sorted(list(HEAD_FAMILIES)),
        "notes": [
            "This is a coarse first version.",
            "Pattern support is computed over sampled patterns, not exact subgraph isomorphism over G.",
            "Predicate support/confidence is computed over global_predicate_table_full.csv.",
            "Pattern-BN and Predicate-BN are only used for soft pruning/ranking.",
            "Final exact GAR+ validation over the original graph G is left for the next stage.",
        ],
    }

    summary["num_pattern_tree_nodes"] = int(len(pattern_nodes))
    summary["num_pattern_tree_edges"] = int(len(pattern_tree_edges))
    summary["num_rule_tree_nodes"] = int(len(rule_tree_nodes))
    summary["num_rule_tree_edges"] = int(len(rule_tree_edges))
    summary["num_rules_before_dedup"] = int(len(mined_rules))
    summary["num_rules_after_dedup"] = int(len({r.rule_id for r in mined_rules}))

    return (
        mined_rules,
        list(pattern_nodes.values()),
        pattern_tree_edges,
        rule_tree_nodes,
        rule_tree_edges,
        summary,
    )


def main():
    start_time = time.time()
    #加载BN文件
    pattern_bn_module = _load_pattern_bn_module()

    pattern_graphs = pattern_bn_module.load_selected_pattern_graphs(SELECTED_PATH)
    repository = load_predicate_repository(REPO_PATH)
    predicate_table = load_predicate_table(TABLE_PATH)
    family_bn_states = load_family_bn_edges(FAMILY_BNS_PATH)
    pattern_bn_state = load_pattern_bn_state(PATTERN_BNS_PATH, pattern_bn_module)

    print(f"[Info] loaded_patterns={len(pattern_graphs)}")
    print(f"[Info] loaded_predicates={len(repository.get('predicates', []))}")
    print(
        f"[Info] predicate_table patterns={predicate_table.shape[0]} "
        f"predicates={len([c for c in predicate_table.columns if c != 'pattern_id'])}"
    )
    print(f"[Info] loaded_family_bns={sum(1 for s in family_bn_states.values() if s.get('status') == 'learned')}")
    print(f"[Info] pattern_bn_available={bool(pattern_bn_state)}")

    mined_rules, pattern_tree_nodes, pattern_tree_edges, rule_tree_nodes, rule_tree_edges, summary = levelwise_garplus_mine(
        pattern_graphs=pattern_graphs,
        predicate_table=predicate_table,
        repository=repository,
        pattern_bn_state=pattern_bn_state,
        pattern_bn_module=pattern_bn_module,
        family_bn_states=family_bn_states,
        sigma_pattern=SIGMA_PATTERN,
        sigma_rule=SIGMA_RULE,
        delta=DELTA,
        max_pattern_edges=MAX_PATTERN_EDGES,
        max_X_size=MAX_X_SIZE,
        tau_pattern_bn=TAU_PATTERN_BN,
        tau_predicate_bn=TAU_PREDICATE_BN,
        top_k_pattern_extensions=TOP_K_PATTERN_EXTENSIONS,
        top_k_predicate_extensions=TOP_K_PREDICATE_EXTENSIONS,
        min_predicate_support=MIN_PREDICATE_SUPPORT,
        max_predicates=MAX_PREDICATES,
    )

    save_mining_outputs(
        output_dir=OUTPUT_PATH,
        mined_rules=mined_rules,
        pattern_tree_nodes=pattern_tree_nodes,
        pattern_tree_edges=pattern_tree_edges,
        rule_tree_nodes=rule_tree_nodes,
        rule_tree_edges=rule_tree_edges,
        summary=summary,
    )

    elapsed = time.time() - start_time
    print(
        f"[Done] pattern_nodes={summary.get('num_pattern_tree_nodes')} "
        f"rule_nodes={summary.get('num_rule_tree_nodes')} "
        f"rules={summary.get('num_rules_after_dedup')} elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
