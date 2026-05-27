from __future__ import annotations

"""
Pattern-BN multi-BN version.

This module replaces the single global structural BN with three coordinated BN
families built from pick_patterns selected subgraphs:
1. Family-specific BN
2. Cross-family BN
3. Light-global BN

Node labels are computed globally on the union graph of all selected patterns.
Edges are modeled structurally through family-specific pair predicates rather
than as independent edge-node variables. The learned BNs are then used to score
candidate pattern extensions.
"""

import json
import sys
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, HillClimbSearch

try:
    from pgmpy.estimators import BIC as BICScore
except ImportError:
    # Older pgmpy releases expose the same score under BicScore.
    from pgmpy.estimators import BicScore as BICScore

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent

# Ensure `enumeration-discovery` is importable when running from repo root.
base_dir_str = str(BASE_DIR)
if base_dir_str not in sys.path:
    sys.path.insert(0, base_dir_str)

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
except ImportError:
    # pgmpy 1.1.2 and some older releases expose BayesianNetwork instead.
    from pgmpy.models import BayesianNetwork as BayesianModel

from inspect_graph import DEFAULT_SELECTED_PATH, SelectedPPIDataset

OUTPUT_DIR = BASE_DIR / "processed" / "ppi" / "pattern_multi_bn"
SELECTED_PATH = Path(DEFAULT_SELECTED_PATH)

FAMILIES = ["role", "clustering", "core"]
CROSS_FAMILY_PAIRS = [
    ("role", "clustering"),
    ("role", "core"),
    ("clustering", "core"),
]

FAMILY_VALUE_ORDER = {
    "role": ["hub", "mid", "leaf"],
    "clustering": ["high", "mid", "low"],
    "core": ["high", "mid", "low"],
}

MIN_SUPPORT = 5
MAX_VARIABLES_PER_BN = None
MAX_GLOBAL_VARIABLES = 50
MAX_INDEGREE = 2
MAX_ITER = int(1e4)

FAMILY_WEIGHT = 0.4
CROSS_WEIGHT = 0.4
GLOBAL_WEIGHT = 0.2

MULTI_BN_STATE = None


def data_to_nx_graph(data):
    graph = nx.Graph()
    #TODO
    if hasattr(data, "orig_node_ids"):
        node_names = [int(v) for v in data.orig_node_ids.tolist()]
    else:
        node_names = list(range(int(data.num_nodes)))

    for local_idx, node_name in enumerate(node_names):
        attrs = {"local_idx": int(local_idx)}
        if hasattr(data, "center_id"):
            attrs["is_center"] = int(local_idx == int(data.center_id.item()))
        graph.add_node(node_name, **attrs)

    seen_edges = set()
    edge_index = data.edge_index
    for eid in range(edge_index.size(1)):
        src_local = int(edge_index[0, eid])
        dst_local = int(edge_index[1, eid])
        if src_local == dst_local:
            continue
        src_name = node_names[src_local]
        dst_name = node_names[dst_local]
        key = tuple(sorted((src_name, dst_name)))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        graph.add_edge(src_name, dst_name)

    return graph


def load_selected_pattern_graphs(selected_path=SELECTED_PATH):
    dataset = SelectedPPIDataset(str(selected_path))
    graphs = []
    for idx in range(len(dataset)):
        data = dataset.get(idx)
        graphs.append((idx, data_to_nx_graph(data)))
    return graphs


def build_union_graph(patterns):
    union_graph = nx.Graph()
    for _, graph in patterns:
        union_graph.add_nodes_from(graph.nodes(data=True))
        union_graph.add_edges_from(graph.edges(data=True))
    return union_graph


def _three_quantile_labels(values, family_name):
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return np.array([], dtype=object)

    if np.all(values == values[0]):
        return np.array(["mid"] * len(values), dtype=object)

    q1 = float(np.quantile(values, 1.0 / 3.0))
    q2 = float(np.quantile(values, 2.0 / 3.0))

    if family_name == "role":
        low_label, mid_label, high_label = "leaf", "mid", "hub"
    else:
        low_label, mid_label, high_label = "low", "mid", "high"

    labels = []
    for value in values:
        if value <= q1:
            labels.append(low_label)
        elif value >= q2:
            labels.append(high_label)
        else:
            labels.append(mid_label)
    return np.asarray(labels, dtype=object)


def compute_node_family_values(union_graph):
    """
    Compute global family labels on the union graph using quantile binning.
    """
    nodes = sorted(union_graph.nodes(), key=lambda x: str(x))
    if not nodes:
        return {}

    degree_map = dict(union_graph.degree())
    clustering_map = nx.clustering(union_graph)
    core_map = nx.core_number(union_graph) if union_graph.number_of_edges() > 0 else {n: 0 for n in nodes}

    degree_labels = _three_quantile_labels([degree_map[n] for n in nodes], "role")
    clustering_labels = _three_quantile_labels([clustering_map[n] for n in nodes], "clustering")
    core_labels = _three_quantile_labels([core_map[n] for n in nodes], "core")

    node_family_values = {}
    for i, node in enumerate(nodes):
        node_family_values[node] = {
            "role": str(degree_labels[i]),
            "clustering": str(clustering_labels[i]),
            "core": str(core_labels[i]),
            "metrics": {
                "degree": float(degree_map[node]),
                "clustering": float(clustering_map[node]),
                "core": float(core_map[node]),
            },
        }

    return node_family_values


def canonical_pair_value(value1, value2, family_name=None):
    if family_name == "role":
        order = {v: i for i, v in enumerate(FAMILY_VALUE_ORDER[family_name])}
        parts = sorted((value1, value2), key=lambda x: (order.get(x, 999), x))
    elif family_name in FAMILY_VALUE_ORDER:
        order = {v: i for i, v in enumerate(FAMILY_VALUE_ORDER[family_name])}
        parts = sorted((value1, value2), key=lambda x: (order.get(x, 999), x))
    else:
        parts = sorted((value1, value2), key=lambda x: str(x))
    return tuple(parts)


def family_nl_var(family, value):
    return f"NL:{family}:{value}"


def family_np_var(family, value1, value2):
    left, right = canonical_pair_value(value1, value2, family)
    return f"NP:{family}:{left}__{right}"


def cross_xp_var(family1, family2, value1, value2):
    return f"XP:{family1}_{family2}:{value1}__{value2}"


def extract_family_vars(pattern_graph, node_family_values, family):
    vars_found = set()

    for node in pattern_graph.nodes():
        fam_value = node_family_values[node][family]
        vars_found.add(family_nl_var(family, fam_value))

    for u, v in pattern_graph.edges():
        u_value = node_family_values[u][family]
        v_value = node_family_values[v][family]
        vars_found.add(family_np_var(family, u_value, v_value))

    return vars_found


def extract_cross_vars(pattern_graph, node_family_values, family1, family2):
    vars_found = set()

    for node in pattern_graph.nodes():
        vars_found.add(family_nl_var(family1, node_family_values[node][family1]))
        vars_found.add(family_nl_var(family2, node_family_values[node][family2]))

    for u, v in pattern_graph.edges():
        u_f1 = node_family_values[u][family1]
        v_f1 = node_family_values[v][family1]
        u_f2 = node_family_values[u][family2]
        v_f2 = node_family_values[v][family2]

        vars_found.add(cross_xp_var(family1, family2, u_f1, v_f2))
        vars_found.add(cross_xp_var(family1, family2, v_f1, u_f2))

    return vars_found


def extract_multi_bn_vars(pattern_graph, node_family_values=None):
    """
    Extract variables for all BN groups from one pattern graph.
    """
    global MULTI_BN_STATE
    if node_family_values is None:
        if MULTI_BN_STATE is None:
            raise ValueError("node_family_values not provided and no global multi-BN state is loaded")
        node_family_values = MULTI_BN_STATE["node_family_values"]

    group_vars = {}

    for family in FAMILIES:
        group_vars[f"family_{family}"] = extract_family_vars(pattern_graph, node_family_values, family)

    for family1, family2 in CROSS_FAMILY_PAIRS:
        group_vars[f"cross_{family1}_{family2}"] = extract_cross_vars(
            pattern_graph,
            node_family_values,
            family1,
            family2,
        )

    all_non_global = set().union(*group_vars.values()) if group_vars else set()
    selected_global_vars = None
    if MULTI_BN_STATE is not None:
        selected_global_vars = set(MULTI_BN_STATE.get("global_light_variables", []))
    group_vars["global_light"] = (
        all_non_global & selected_global_vars if selected_global_vars is not None else all_non_global
    )

    return group_vars


def _build_binary_table(row_var_sets):
    all_vars = sorted(set().union(*row_var_sets)) if row_var_sets else []
    if not all_vars:
        return pd.DataFrame(columns=[]), pd.Series(dtype=int)

    rows = []
    for vars_present in row_var_sets:
        rows.append({var: int(var in vars_present) for var in all_vars})

    df = pd.DataFrame(rows, columns=all_vars).fillna(0).astype(int)
    support = df.sum().astype(int)
    return df, support


def filter_and_prepare_table(df, min_support=MIN_SUPPORT, max_variables=MAX_VARIABLES_PER_BN):
    if df.empty:
        return df, pd.Series(dtype=int), {
            "constant_columns": [],
            "low_support_columns": [],
            "selected_top_columns": list(df.columns),
        }

    info = {
        "constant_columns": [],
        "low_support_columns": [],
        "selected_top_columns": [],
    }

    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        warnings.warn(f"Dropping constant columns: {constant_cols}", RuntimeWarning)
        df = df.drop(columns=constant_cols)
        info["constant_columns"] = constant_cols

    if df.empty:
        return df, pd.Series(dtype=int), info

    support = df.sum()
    low_support_cols = support[support < min_support].index.tolist()
    if low_support_cols:
        warnings.warn(
            f"Dropping low-support columns (< {min_support}): {low_support_cols}",
            RuntimeWarning,
        )
        df = df.drop(columns=low_support_cols)
        info["low_support_columns"] = low_support_cols

    if df.empty:
        return df, pd.Series(dtype=int), info

    support = df.sum().sort_values(ascending=False)
    if max_variables is not None and df.shape[1] > max_variables:
        top_vars = support.head(max_variables).index.tolist()
        df = df[top_vars]
        support = df.sum().sort_values(ascending=False)
        info["selected_top_columns"] = top_vars
    else:
        info["selected_top_columns"] = list(df.columns)

    return df.astype(int), support.astype(int), info


def learn_bayesian_network(df):
    if df.empty:
        raise ValueError("DataFrame is empty.")
    if df.shape[0] < 10:
        raise ValueError(f"Need at least 10 rows, got {df.shape[0]}")
    if df.shape[1] < 2:
        raise ValueError(f"Need at least 2 variables, got {df.shape[1]}")

    score = BICScore(df)
    search = HillClimbSearch(df)
    structure = search.estimate(
        scoring_method=score,
        max_indegree=MAX_INDEGREE,
        max_iter=MAX_ITER,
    )

    model = BayesianModel()
    model.add_nodes_from(df.columns.tolist())
    model.add_edges_from(structure.edges())
    model.fit(df, estimator=BayesianEstimator, prior_type="BDeu", equivalent_sample_size=10)
    return model


def build_family_tables(pattern_graphs, node_family_values):
    raw_tables = {}
    raw_supports = {}

    for family in FAMILIES:
        row_var_sets = [extract_family_vars(graph, node_family_values, family) for _, graph in pattern_graphs]
        raw_tables[f"family_{family}"], raw_supports[f"family_{family}"] = _build_binary_table(row_var_sets)

    for family1, family2 in CROSS_FAMILY_PAIRS:
        key = f"cross_{family1}_{family2}"
        row_var_sets = [extract_cross_vars(graph, node_family_values, family1, family2) for _, graph in pattern_graphs]
        raw_tables[key], raw_supports[key] = _build_binary_table(row_var_sets)

    return raw_tables, raw_supports


def build_global_light_table(filtered_tables, max_global_variables=MAX_GLOBAL_VARIABLES):
    support_union = defaultdict(int)
    per_pattern_rows = []

    table_names = sorted(filtered_tables.keys())
    if not table_names:
        return pd.DataFrame(), pd.Series(dtype=int), []

    num_rows = next(iter(filtered_tables.values())).shape[0] if filtered_tables else 0
    for row_idx in range(num_rows):
        vars_present = set()
        for table_name in table_names:
            df = filtered_tables[table_name]
            active_cols = set(df.columns[df.iloc[row_idx] == 1].tolist())
            vars_present.update(active_cols)
        per_pattern_rows.append(vars_present)
        for var in vars_present:
            support_union[var] += 1

    ranked = sorted(support_union.items(), key=lambda x: (-x[1], x[0]))
    selected_vars = [var for var, _ in ranked[:max_global_variables]]

    if not selected_vars:
        return pd.DataFrame(), pd.Series(dtype=int), []

    rows = []
    for vars_present in per_pattern_rows:
        rows.append({var: int(var in vars_present) for var in selected_vars})

    df = pd.DataFrame(rows, columns=selected_vars).fillna(0).astype(int)
    support = df.sum().astype(int)
    return df, support, selected_vars


def _cpd_to_json(cpd):
    state_names = getattr(cpd, "state_names", {}) or {}
    variable_states = state_names.get(cpd.variable, list(range(cpd.variable_card)))
    evidence = list(cpd.variables[1:])
    evidence_cards = list(cpd.cardinality[1:])
    evidence_states = [state_names.get(ev, list(range(card))) for ev, card in zip(evidence, evidence_cards)]

    rows = []
    values = cpd.get_values()
    if evidence:
        import itertools

        for col_idx, evidence_assignment in enumerate(itertools.product(*evidence_states)):
            assignment = {str(evidence[i]): str(evidence_assignment[i]) for i in range(len(evidence))}
            probabilities = {
                str(variable_states[row_idx]): float(values[row_idx, col_idx])
                for row_idx in range(len(variable_states))
            }
            rows.append({"evidence_states": assignment, "probabilities": probabilities})
    else:
        probabilities = {
            str(variable_states[row_idx]): float(values[row_idx, 0])
            for row_idx in range(len(variable_states))
        }
        rows.append({"evidence_states": {}, "probabilities": probabilities})

    return {
        "variable": str(cpd.variable),
        "evidence": [str(x) for x in evidence],
        "rows": rows,
    }


def save_group_result(group_name, df, support, filter_info, model, output_dir, status="ok", reason=""):
    group_dir = Path(output_dir) / group_name
    group_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(group_dir / "table.csv", index=False)
    support_df = support.reset_index()
    support_df.columns = ["variable", "support"]
    support_df.to_csv(group_dir / "support.csv", index=False)

    result = {
        "group_name": group_name,
        "status": status,
        "reason": reason,
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "filter_info": filter_info,
        "edges": [],
        "cpds": [],
    }

    if model is not None:
        directed_edges = sorted([[str(src), str(dst)] for src, dst in model.edges()], key=lambda x: (x[0], x[1]))
        result["edges"] = directed_edges
        result["cpds"] = [_cpd_to_json(cpd) for cpd in model.get_cpds()]

        pd.DataFrame(directed_edges, columns=["source", "target"]).to_csv(
            group_dir / "edges.csv", index=False
        )
    else:
        pd.DataFrame(columns=["source", "target"]).to_csv(group_dir / "edges.csv", index=False)

    with open(group_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def save_node_family_values(node_family_values, output_dir):
    payload = {}
    for node, values in sorted(node_family_values.items(), key=lambda x: str(x[0])):
        payload[str(node)] = {
            "role": values["role"],
            "clustering": values["clustering"],
            "core": values["core"],
            "metrics": values["metrics"],
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "node_family_values.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def train_multi_bns(pattern_graphs, output_dir=OUTPUT_DIR):
    # union all graphs into one
    union_graph = build_union_graph(pattern_graphs)
    node_family_values = compute_node_family_values(union_graph)

    raw_tables, raw_supports = build_family_tables(pattern_graphs, node_family_values)

    filtered_tables = {}
    filtered_supports = {}
    filter_infos = {}
    group_results = {}

    for group_name, raw_df in raw_tables.items():
        max_vars = MAX_VARIABLES_PER_BN
        filtered_df, filtered_support, filter_info = filter_and_prepare_table(
            raw_df,
            min_support=MIN_SUPPORT,
            max_variables=max_vars,
        )
        filtered_tables[group_name] = filtered_df
        filtered_supports[group_name] = filtered_support
        filter_infos[group_name] = filter_info

    global_df, global_support, global_vars = build_global_light_table(
        filtered_tables,
        max_global_variables=MAX_GLOBAL_VARIABLES,
    )
    global_df, global_support, global_filter_info = filter_and_prepare_table(
        global_df,
        min_support=MIN_SUPPORT,
        max_variables=MAX_GLOBAL_VARIABLES,
    )
    filtered_tables["global_light"] = global_df
    filtered_supports["global_light"] = global_support
    filter_infos["global_light"] = global_filter_info

    output_dir = Path(output_dir)
    save_node_family_values(node_family_values, output_dir)

    for group_name, df in filtered_tables.items():
        try:
            model = learn_bayesian_network(df)
            group_results[group_name] = {
                "status": "ok",
                "model": model,
                "table": df,
                "support": filtered_supports[group_name],
            }
            save_group_result(
                group_name,
                df,
                filtered_supports[group_name],
                filter_infos[group_name],
                model,
                output_dir,
                status="ok",
            )
        except Exception as exc:
            warnings.warn(f"Skipping {group_name}: {exc}", RuntimeWarning)
            group_results[group_name] = {
                "status": "skipped",
                "reason": str(exc),
                "model": None,
                "table": df,
                "support": filtered_supports[group_name],
            }
            save_group_result(
                group_name,
                df,
                filtered_supports[group_name],
                filter_infos[group_name],
                None,
                output_dir,
                status="skipped",
                reason=str(exc),
            )

    state = {
        "node_family_values": node_family_values,
        "group_results": group_results,
        "global_light_variables": list(global_df.columns),
    }
    return state


def _bn_connection_score(current_vars, candidate_vars, model, support_series):
    if model is None or not current_vars or not candidate_vars:
        return 0.0

    bn_nodes = set(model.nodes())
    current_in_bn = current_vars & bn_nodes
    candidate_in_bn = candidate_vars & bn_nodes
    if not current_in_bn or not candidate_in_bn:
        return 0.0

    edge_set = set(model.edges())
    direct = 0.0
    for cur in current_in_bn:
        for cand in candidate_in_bn:
            if (cur, cand) in edge_set or (cand, cur) in edge_set:
                direct += 1.0

    support_bonus = float(np.mean([support_series.get(v, 0) for v in candidate_in_bn])) if len(candidate_in_bn) else 0.0
    norm = max(len(current_in_bn) * len(candidate_in_bn), 1)
    return 0.8 * (direct / norm) + 0.2 * np.log1p(support_bonus) / 10.0


def score_extension_by_multi_bns(current_vars, candidate_vars):
    global MULTI_BN_STATE
    if MULTI_BN_STATE is None:
        raise ValueError("MULTI_BN_STATE is not initialized. Run train_multi_bns() first.")

    family_scores = []
    for family in FAMILIES:
        group_name = f"family_{family}"
        result = MULTI_BN_STATE["group_results"].get(group_name, {})
        score = _bn_connection_score(
            current_vars.get(group_name, set()),
            candidate_vars.get(group_name, set()),
            result.get("model"),
            result.get("support", pd.Series(dtype=float)),
        )
        family_scores.append(score)

    cross_scores = []
    for family1, family2 in CROSS_FAMILY_PAIRS:
        group_name = f"cross_{family1}_{family2}"
        result = MULTI_BN_STATE["group_results"].get(group_name, {})
        score = _bn_connection_score(
            current_vars.get(group_name, set()),
            candidate_vars.get(group_name, set()),
            result.get("model"),
            result.get("support", pd.Series(dtype=float)),
        )
        cross_scores.append(score)

    global_result = MULTI_BN_STATE["group_results"].get("global_light", {})
    global_score = _bn_connection_score(
        current_vars.get("global_light", set()),
        candidate_vars.get("global_light", set()),
        global_result.get("model"),
        global_result.get("support", pd.Series(dtype=float)),
    )

    family_score = float(np.mean(family_scores)) if family_scores else 0.0
    cross_score = float(np.mean(cross_scores)) if cross_scores else 0.0

    return (
        FAMILY_WEIGHT * family_score
        + CROSS_WEIGHT * cross_score
        + GLOBAL_WEIGHT * global_score
    )


def rank_extensions_by_multi_bns(current_pattern, candidate_extensions):
    current_vars = extract_multi_bn_vars(current_pattern)
    scored = []
    for name, candidate_graph in candidate_extensions:
        candidate_vars = extract_multi_bn_vars(candidate_graph)
        score = score_extension_by_multi_bns(current_vars, candidate_vars)
        scored.append((name, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def run_pattern_multi_bn_pipeline(selected_path=SELECTED_PATH, output_dir=OUTPUT_DIR):
    global MULTI_BN_STATE

    pattern_graphs = load_selected_pattern_graphs(selected_path)
    if not pattern_graphs:
        raise ValueError("No selected pattern graphs loaded.")

    MULTI_BN_STATE = train_multi_bns(pattern_graphs, output_dir=output_dir)
    return MULTI_BN_STATE


def main():
    state = run_pattern_multi_bn_pipeline(selected_path=SELECTED_PATH, output_dir=OUTPUT_DIR)
    ok_count = sum(1 for result in state["group_results"].values() if result["status"] == "ok")
    print(f"[Done] trained_groups={ok_count}/{len(state['group_results'])}")
    print(f"[Done] output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
