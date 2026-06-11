import json
import time
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pgmpy.estimators import BayesianEstimator, HillClimbSearch
from pgmpy.models import BayesianNetwork


PAIRWISE_CSV = "predicate_pairwise_table.csv"
OUTPUT_DIR = "processed/ppi/bn_output"
SCORING_METHOD = "k2score"
MAX_ITER = int(1e4)
MAX_INDEGREE = None
PRIOR_TYPE = "BDeu"
EQUIVALENT_SAMPLE_SIZE = 10


def prepare_bn_data(csv_path):
    """
    Read the endpoint-expanded pair-wise predicate table and turn it into a
    clean discrete table for BN learning.

    This block:
    1. Drops raw pair identifier columns.
    2. Fills missing values.
    3. Converts all columns to integers.
    4. Drops constant columns.
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["protein_x", "protein_y", "protein_a", "protein_b"], errors="ignore")
    df = df.fillna(0)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print("[Info] Dropping constant columns:", constant_cols)
        df = df.drop(columns=constant_cols)

    for col in df.columns:
        df[col] = df[col].astype(int)

    if df.shape[1] == 0:
        raise RuntimeError(
            "No non-constant predicate columns remain after preprocessing. "
            "Please check whether the input predicate table has variation."
        )

    print("[Info] Final BN columns:")
    print(df.columns.tolist())
    print("[Info] Data shape:", df.shape)
    return df


def learn_bn_hc(input_data, scoring_method="k2score", max_iter=int(1e4), max_indegree=None):
    """
    Learn Bayesian network structure with HillClimbSearch.
    """
    start_time = time.time()
    est = HillClimbSearch(data=input_data)
    estimated_model = est.estimate(
        scoring_method=scoring_method,
        max_indegree=max_indegree,
        max_iter=max_iter,
    )
    elapsed = time.time() - start_time

    print("[Info] Learned BN structure")
    print("Number of nodes:", estimated_model.number_of_nodes())
    print("Number of directed edges:", estimated_model.number_of_edges())
    print("Elapsed time:", elapsed)
    return estimated_model, elapsed


def fit_bn_parameters(structure_model, input_data, prior_type="BDeu", equivalent_sample_size=10):
    """
    Fit CPDs for the learned BN structure.
    """
    bn_model = BayesianNetwork()
    bn_model.add_nodes_from(input_data.columns.tolist())
    bn_model.add_edges_from(structure_model.edges())
    bn_model.fit(
        input_data,
        estimator=BayesianEstimator,
        prior_type=prior_type,
        equivalent_sample_size=equivalent_sample_size,
    )

    print("[Info] Fitted BN parameters")
    print("Number of CPDs:", len(bn_model.get_cpds()))
    return bn_model


def summarize_cpds(model):
    """
    Convert CPDs into JSON-friendly and CSV-friendly summaries.
    """
    cpds_json = []
    cpds_rows = []

    for cpd in model.get_cpds():
        variable = str(cpd.variable)
        evidence = [str(x) for x in (cpd.variables[1:] if len(cpd.variables) > 1 else [])]
        state_names = getattr(cpd, "state_names", {}) or {}
        variable_states = [str(x) for x in state_names.get(cpd.variable, list(range(cpd.variable_card)))]

        values = cpd.get_values()
        evidence_state_lists = [state_names.get(ev, list(range(card))) for ev, card in zip(evidence, cpd.cardinality[1:])]

        cpd_entry = {
            "variable": variable,
            "evidence": evidence,
            "variable_states": [str(x) for x in variable_states],
            "rows": [],
        }

        if evidence:
            import itertools

            for col_idx, evidence_states in enumerate(itertools.product(*evidence_state_lists)):
                evidence_mapping = {
                    evidence[i]: str(evidence_states[i]) for i in range(len(evidence))
                }
                probabilities = {}
                for row_idx, state in enumerate(variable_states):
                    prob = float(values[row_idx, col_idx])
                    probabilities[str(state)] = prob
                    cpds_rows.append(
                        {
                            "variable": variable,
                            "state": str(state),
                            "evidence": "|".join(evidence),
                            "evidence_states": "|".join(
                                f"{evidence[i]}={evidence_states[i]}" for i in range(len(evidence))
                            ),
                            "probability": prob,
                        }
                    )
                cpd_entry["rows"].append(
                    {
                        "evidence_states": evidence_mapping,
                        "probabilities": probabilities,
                    }
                )
        else:
            probabilities = {}
            for row_idx, state in enumerate(variable_states):
                prob = float(values[row_idx, 0])
                probabilities[str(state)] = prob
                cpds_rows.append(
                    {
                        "variable": variable,
                        "state": str(state),
                        "evidence": "",
                        "evidence_states": "",
                        "probability": prob,
                    }
                )
            cpd_entry["rows"].append(
                {
                    "evidence_states": {},
                    "probabilities": probabilities,
                }
            )

        cpds_json.append(cpd_entry)

    return cpds_json, cpds_rows


def summarize_binary_positive_cpds(cpds_json):
    """
    Build a compact table for binary variables, focusing on P(variable=1 | parents).
    """
    compact_rows = []

    for cpd in cpds_json:
        variable = cpd["variable"]
        variable_states = [str(x) for x in cpd.get("variable_states", [])]
        if "1" not in variable_states:
            continue

        for row in cpd["rows"]:
            compact_rows.append(
                {
                    "variable": variable,
                    "parents": "|".join(cpd["evidence"]),
                    "parent_states": "|".join(
                        f"{k}={v}" for k, v in sorted(row["evidence_states"].items())
                    ),
                    "probability_of_1": float(row["probabilities"].get("1", 0.0)),
                }
            )

    return compact_rows


def print_cpd_summary(cpds_json):
    print("\n[CPDs] Conditional probabilities")
    if not cpds_json:
        print("  (none)")
        return

    for cpd in cpds_json:
        variable = cpd["variable"]
        evidence = cpd["evidence"]
        print(f"\n  Variable: {variable}")
        print(f"  Parents: {evidence if evidence else []}")
        for row in cpd["rows"]:
            print(f"    Given {row['evidence_states']}: {row['probabilities']}")


def summarize_structure(model, columns):
    """
    Summarize the learned structure.

    Associated pairs:
    - variables directly connected by an edge in either direction.

    Independent pairs:
    - variables with no direct edge in the learned graph.
    """
    directed_edges = sorted((str(u), str(v)) for u, v in model.edges())
    associated_pairs = sorted(set(tuple(sorted((str(u), str(v)))) for u, v in model.edges()))

    associated_set = set(associated_pairs)
    independent_pairs = []
    for a, b in combinations(sorted(columns), 2):
        pair = tuple(sorted((str(a), str(b))))
        if pair not in associated_set:
            independent_pairs.append(pair)

    parents = {str(node): sorted(str(x) for x in model.predecessors(node)) for node in model.nodes()}
    children = {str(node): sorted(str(x) for x in model.successors(node)) for node in model.nodes()}

    return {
        "nodes": sorted(str(x) for x in model.nodes()),
        "directed_edges": directed_edges,
        "associated_pairs": associated_pairs,
        "independent_pairs": independent_pairs,
        "parents": parents,
        "children": children,
    }


def print_structure_summary(summary):
    print("\n[Structure] Directed edges")
    if summary["directed_edges"]:
        for src, dst in summary["directed_edges"]:
            print(f"  {src} -> {dst}")
    else:
        print("  (none)")

    print("\n[Structure] Associated pairs (directly connected)")
    if summary["associated_pairs"]:
        for a, b in summary["associated_pairs"]:
            print(f"  {a} -- {b}")
    else:
        print("  (none)")

    print("\n[Structure] Independent pairs (no direct edge)")
    if summary["independent_pairs"]:
        for a, b in summary["independent_pairs"]:
            print(f"  {a} ⟂ {b}")
    else:
        print("  (none)")

    print("\n[Structure] Parents of each node")
    for node, parent_list in summary["parents"].items():
        print(f"  {node}: {parent_list}")

    print("\n[Structure] Children of each node")
    for node, child_list in summary["children"].items():
        print(f"  {node}: {child_list}")


def save_structure_summary(summary, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "bn_structure_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    pd.DataFrame(summary["directed_edges"], columns=["source", "target"]).to_csv(
        output_dir / "bn_directed_edges.csv", index=False
    )
    pd.DataFrame(summary["associated_pairs"], columns=["var_a", "var_b"]).to_csv(
        output_dir / "bn_associated_pairs.csv", index=False
    )
    pd.DataFrame(summary["independent_pairs"], columns=["var_a", "var_b"]).to_csv(
        output_dir / "bn_independent_pairs.csv", index=False
    )

    print(f"[Saved] {json_path}")
    print(f"[Saved] {output_dir / 'bn_directed_edges.csv'}")
    print(f"[Saved] {output_dir / 'bn_associated_pairs.csv'}")
    print(f"[Saved] {output_dir / 'bn_independent_pairs.csv'}")


def save_cpd_summary(cpds_json, cpds_rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    cpds_json_path = output_dir / "bn_cpds.json"
    with open(cpds_json_path, "w", encoding="utf-8") as f:
        json.dump(cpds_json, f, indent=2, ensure_ascii=False)

    cpds_csv_path = output_dir / "bn_cpds.csv"
    pd.DataFrame(cpds_rows).to_csv(cpds_csv_path, index=False)

    compact_rows = summarize_binary_positive_cpds(cpds_json)
    compact_csv_path = output_dir / "bn_cpds_binary_p1.csv"
    pd.DataFrame(compact_rows).to_csv(compact_csv_path, index=False)

    print(f"[Saved] {cpds_json_path}")
    print(f"[Saved] {cpds_csv_path}")
    print(f"[Saved] {compact_csv_path}")


def visualize_structure(model, output_dir):
    """
    Visualize the learned Bayesian network as a directed graph.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / "bn_structure.png"

    graph = nx.DiGraph()
    graph.add_nodes_from(model.nodes())
    graph.add_edges_from(model.edges())

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42, k=1.2)
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color="#4c78a8",
        node_size=2400,
        edgecolors="black",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(graph, pos, font_size=9, font_color="black")
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="#7f7f7f",
        arrows=True,
        arrowsize=18,
        width=1.8,
        connectionstyle="arc3,rad=0.05",
    )
    plt.title("Learned Bayesian Network Structure", fontsize=13)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {png_path}")


def partition(pairwise_path, output_dir=None):
    """
    Full pipeline:
    1. Prepare BN input data
    2. Learn structure
    3. Summarize direct associations and no-direct-edge pairs
    4. Save results and visualization
    """
    input_data = prepare_bn_data(
        csv_path=pairwise_path,
    )

    model, elapsed = learn_bn_hc(
        input_data=input_data,
        scoring_method=SCORING_METHOD,
        max_iter=MAX_ITER,
        max_indegree=MAX_INDEGREE,
    )
    fitted_model = fit_bn_parameters(
        structure_model=model,
        input_data=input_data,
        prior_type=PRIOR_TYPE,
        equivalent_sample_size=EQUIVALENT_SAMPLE_SIZE,
    )
    cpds_json, cpds_rows = summarize_cpds(fitted_model)

    summary = summarize_structure(model, columns=input_data.columns.tolist())
    summary["elapsed_seconds"] = elapsed
    print_structure_summary(summary)
    print_cpd_summary(cpds_json)

    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / OUTPUT_DIR
    save_structure_summary(summary, output_dir)
    save_cpd_summary(cpds_json, cpds_rows, output_dir)
    visualize_structure(model, output_dir)
    return fitted_model, summary


if __name__ == "__main__":
    pairwise_csv = Path(__file__).resolve().parent / PAIRWISE_CSV
    partition(pairwise_csv)
