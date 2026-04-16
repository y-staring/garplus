import json
import time
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pgmpy.estimators import HillClimbSearch


PAIRWISE_CSV = "processed/ppi/processed/predicate_pairwise_table.csv"
OUTPUT_DIR = "processed/ppi/bn_output"
SCORING_METHOD = "k2score"
MAX_ITER = int(1e4)
MAX_INDEGREE = None


def prepare_bn_data(csv_path, keep_label=True, drop_complement=True):
    """
    Read the pair-wise table and turn it into a clean discrete table for BN learning.

    This block:
    1. Drops raw ID columns.
    2. Fills missing values.
    3. Converts all columns to integers.
    4. Drops sampled_non_interaction if it is exactly complementary to label.
    5. Drops constant columns.
    """
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["protein_a", "protein_b"], errors="ignore")
    df = df.fillna(0)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    if keep_label:
        if drop_complement and "label" in df.columns and "sampled_non_interaction" in df.columns:
            if ((df["label"] + df["sampled_non_interaction"]) == 1).all():
                print("[Info] 'sampled_non_interaction' is complementary to 'label', dropping it.")
                df = df.drop(columns=["sampled_non_interaction"])
    else:
        if "label" in df.columns:
            df = df.drop(columns=["label"])

    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        print("[Info] Dropping constant columns:", constant_cols)
        df = df.drop(columns=constant_cols)

    for col in df.columns:
        df[col] = df[col].astype(int)

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
    nx.draw_networkx_labels(graph, pos, font_size=9, font_color="white")
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
        keep_label=True,
        drop_complement=True,
    )

    model, elapsed = learn_bn_hc(
        input_data=input_data,
        scoring_method=SCORING_METHOD,
        max_iter=MAX_ITER,
        max_indegree=MAX_INDEGREE,
    )

    summary = summarize_structure(model, columns=input_data.columns.tolist())
    summary["elapsed_seconds"] = elapsed
    print_structure_summary(summary)

    output_dir = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / OUTPUT_DIR
    save_structure_summary(summary, output_dir)
    visualize_structure(model, output_dir)
    return model, summary


if __name__ == "__main__":
    pairwise_csv = Path(__file__).resolve().parent / PAIRWISE_CSV
    partition(pairwise_csv)
