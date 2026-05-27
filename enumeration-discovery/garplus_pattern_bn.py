from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from garplus_pattern_utils import graph_edges_as_set, pattern_signature


PATTERN_BN_FALLBACK_SCORE = 0.1


def load_pattern_bn_state(pattern_bn_dir: str, pattern_bn_module: Any) -> dict:
    """
    Load Pattern-BN outputs from processed/ppi/pattern_multi_bn.

    We reuse the saved node_family_values and group edge files. The loaded state
    is sufficient for:
    - extracting pattern BN variables
    - scoring delta variables by direct / 2-hop connectivity
    """
    root = Path(pattern_bn_dir)
    if not root.exists():
        return {}

    node_family_path = root / "node_family_values.json"
    if not node_family_path.exists():
        return {}

    with open(node_family_path, "r", encoding="utf-8") as f:
        raw_node_family_values = json.load(f)

    node_family_values = {}
    for node, values in raw_node_family_values.items():
        try:
            node_key = int(node)
        except Exception:
            node_key = node
        node_family_values[node_key] = values

    group_states: dict[str, dict] = {}
    global_light_variables: list[str] = []

    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir():
            continue
        result_path = group_dir / "result.json"
        edges_path = group_dir / "edges.csv"
        table_path = group_dir / "table.csv"

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
        if edges_path.exists():
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
                neighbors = {}
                nodes = set()

        if group_dir.name == "global_light" and table_path.exists():
            try:
                global_light_variables = pd.read_csv(table_path).columns.tolist()
            except Exception:
                global_light_variables = []

        group_states[group_dir.name] = {
            "status": status,
            "neighbors": neighbors,
            "nodes": nodes,
        }

    # Reuse extract_multi_bn_vars from the existing pattern BN module.
    pattern_bn_module.MULTI_BN_STATE = {
        "node_family_values": node_family_values,
        "global_light_variables": global_light_variables,
    }

    return {
        "node_family_values": node_family_values,
        "group_states": group_states,
        "global_light_variables": global_light_variables,
    }


def extract_pattern_vars_for_scoring(pattern: nx.Graph, pattern_bn_state: dict, pattern_bn_module: Any) -> dict[str, set[str]]:
    """
    Reuse the multi-BN variable extraction when Pattern-BN state is available.

    If not available, fall back to simple NODE/EDGE variables.
    """
    if pattern_bn_state and pattern_bn_state.get("node_family_values"):
        try:
            return pattern_bn_module.extract_multi_bn_vars(pattern)
        except Exception:
            pass

    node_vars = {f"NODE:{node}" for node in pattern.nodes()}
    edge_vars = {f"EDGE:{u}-{v}" for u, v in sorted(graph_edges_as_set(pattern))}
    return {"fallback": node_vars | edge_vars}


def score_pattern_extension_by_pattern_bn(
    current_pattern: nx.Graph,
    candidate_pattern: nx.Graph,
    pattern_bn_state: dict,
    pattern_bn_module: Any,
) -> float:
    """
    Score Q -> Q' using Pattern-BN on delta variables only.

    We compare:
        current_vars = vars(Q)
        candidate_vars = vars(Q')
        delta_vars = vars(Q') - vars(Q)
    """
    if not pattern_bn_state or not pattern_bn_state.get("group_states"):
        return PATTERN_BN_FALLBACK_SCORE

    current_vars = extract_pattern_vars_for_scoring(current_pattern, pattern_bn_state, pattern_bn_module)
    candidate_vars = extract_pattern_vars_for_scoring(candidate_pattern, pattern_bn_state, pattern_bn_module)
    group_states = pattern_bn_state.get("group_states", {})

    best_score = PATTERN_BN_FALLBACK_SCORE
    has_any_delta = False

    for group_name, cand_group_vars in candidate_vars.items():
        cur_group_vars = current_vars.get(group_name, set())
        delta_vars = set(cand_group_vars) - set(cur_group_vars)
        if not delta_vars:
            continue
        has_any_delta = True

        group_state = group_states.get(group_name)
        if not group_state or group_state.get("status") not in {"ok", "learned"}:
            best_score = max(best_score, PATTERN_BN_FALLBACK_SCORE)
            continue

        neighbors = group_state.get("neighbors", {})
        direct_hit = False
        two_hop_hit = False

        for delta_var in delta_vars:
            delta_neighbors = set(neighbors.get(delta_var, set()))
            for cur_var in cur_group_vars:
                if cur_var in delta_neighbors:
                    direct_hit = True
                    break
                cur_neighbors = set(neighbors.get(cur_var, set()))
                for hop in cur_neighbors:
                    if delta_var in neighbors.get(hop, set()):
                        two_hop_hit = True
                if direct_hit:
                    break
            if direct_hit:
                break

        if direct_hit:
            best_score = max(best_score, 1.0)
        elif two_hop_hit:
            best_score = max(best_score, 0.5)
        else:
            best_score = max(best_score, PATTERN_BN_FALLBACK_SCORE)

    if not has_any_delta:
        return 0.0
    return best_score


def filter_pattern_extensions_by_bn(
    current_pattern: nx.Graph,
    candidate_extensions: list[tuple[nx.Graph, tuple[Any, Any]]],
    pattern_bn_state: dict,
    pattern_bn_module: Any,
    tau_pattern_bn: float,
    top_k: int | None,
) -> list[tuple[nx.Graph, tuple[Any, Any], float]]:
    scored = []
    for candidate_graph, added_edge in candidate_extensions:
        score = score_pattern_extension_by_pattern_bn(current_pattern, candidate_graph, pattern_bn_state, pattern_bn_module)
        if score >= tau_pattern_bn:
            scored.append((candidate_graph, added_edge, score))

    scored.sort(key=lambda x: (-x[2], pattern_signature(x[0])))
    if top_k is not None:
        scored = scored[:top_k]
    return scored

