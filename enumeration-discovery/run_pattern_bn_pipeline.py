from pathlib import Path

import pandas as pd

import bayesian_based_prune as bbp
import bayesian_preprocess as bp


CURRENT_DIR = Path(__file__).resolve().parent

# Input protein table used to build per-node predicates.
PROTEIN_CSV = CURRENT_DIR / "data" / "protein.csv"

# Directory produced by visualize_patterns_with_negative_edges_fixed.ipynb
# Each pattern file should look like:
#   pattern_<graph_index>_support_<count>_interactions.csv
PATTERN_INTERACTION_DIR = CURRENT_DIR / "processed" / "ppi" / "negative_pattern_interactions"

# Root output directory for the full BN pipeline.
PIPELINE_OUTPUT_DIR = CURRENT_DIR / "processed" / "ppi" / "pattern_bn_results"

# Optional limit for quick testing. Set to an integer like 3 to process only
# the first few pattern csv files.
MAX_PATTERN_FILES = None

# Negative sampling ratio for pair-wise table construction.
NEG_RATIO = 1.0


def build_pairwise_table_compat(protein_csv, interaction_csv, output_csv, neg_ratio=1.0):
    # Prefer the new reusable entrypoint when available. If the runtime still
    # has an older bayesian_preprocess.py, fall back to composing its existing
    # helper functions so the batch pipeline can still run.
    if hasattr(bp, "build_pairwise_table"):
        return bp.build_pairwise_table(
            protein_csv=protein_csv,
            interaction_csv=interaction_csv,
            output_csv=output_csv,
            neg_ratio=neg_ratio,
        )

    protein_df = pd.read_csv(protein_csv, low_memory=False)
    interaction_df = pd.read_csv(interaction_csv, low_memory=False)

    protein_map = bp.build_protein_feature_map(protein_df)
    protein_node_ids = sorted(protein_map.keys())

    edge_map = bp.build_positive_edges(interaction_df)
    positive_edge_set = set(edge_map.keys())
    num_positive = len(positive_edge_set)

    num_negative = int(round(num_positive * neg_ratio))
    negative_edges = bp.sample_negative_edges(
        node_ids=protein_node_ids,
        positive_edge_set=positive_edge_set,
        num_negatives=num_negative,
        seed=bp.SEED,
    )

    rows = bp.build_pairwise_rows(edge_map, negative_edges, protein_map)
    out_df = pd.DataFrame(rows)

    predicate_cols = [
        "label",
        "both_reviewed",
        "both_high_evidence",
        "both_membrane_related",
        "both_has_coiled_coil",
        "both_long_protein",
        "same_location",
        "same_pathway",
        "go_bp_overlap",
        "domain_match",
        "family_match",
        "length_similar",
        "physical_interaction",
        "high_conf_interaction",
        "sampled_non_interaction",
    ]
    for col in predicate_cols:
        out_df[col] = out_df[col].fillna(0).astype(int)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = out_df.sort_values(["protein_a", "protein_b"]).reset_index(drop=True)
    out_df.to_csv(output_csv, index=False)

    summary = {
        "protein_nodes": len(protein_node_ids),
        "positive_edges": num_positive,
        "negative_edges": len(negative_edges),
        "output_rows": len(out_df),
        "output_csv": str(output_csv),
    }
    return out_df, summary


def partition_compat(pairwise_csv, output_dir):
    # Prefer the newer partition(pairwise_path, output_dir=...) signature.
    # If the runtime still has the older version, temporarily redirect its
    # module-level OUTPUT_DIR and call the legacy signature.
    if "output_dir" in getattr(bbp.partition, "__code__", None).co_varnames:
        return bbp.partition(pairwise_csv, output_dir=output_dir)

    original_output_dir = getattr(bbp, "OUTPUT_DIR", "processed/ppi/bn_output")
    try:
        output_dir_path = Path(output_dir)
        try:
            relative_output = output_dir_path.relative_to(CURRENT_DIR)
            bbp.OUTPUT_DIR = str(relative_output).replace("\\", "/")
        except ValueError:
            bbp.OUTPUT_DIR = str(output_dir_path)
        return bbp.partition(pairwise_csv)
    finally:
        bbp.OUTPUT_DIR = original_output_dir


def iter_pattern_csvs(input_dir):
    files = sorted(
        path for path in input_dir.glob("pattern_*_interaction.csv")
        if path.is_file()
    )
    if MAX_PATTERN_FILES is not None:
        files = files[:MAX_PATTERN_FILES]
    return files


def run_single_pattern(pattern_csv, protein_csv, output_root):
    pattern_name = pattern_csv.stem
    pattern_output_dir = output_root / pattern_name
    pairwise_csv = pattern_output_dir / "pairwise_table.csv"
    bn_output_dir = pattern_output_dir / "bn_output"

    print(f"\n[Pattern] {pattern_name}")
    print(f"[Input] interaction_csv={pattern_csv}")

    _, preprocess_summary = build_pairwise_table_compat(
        protein_csv=protein_csv,
        interaction_csv=pattern_csv,
        output_csv=pairwise_csv,
        neg_ratio=NEG_RATIO,
    )
    print(f"[Preprocess] output_csv={pairwise_csv}")
    print(
        f"[Preprocess] protein_nodes={preprocess_summary['protein_nodes']} "
        f"positive_edges={preprocess_summary['positive_edges']} "
        f"negative_edges={preprocess_summary['negative_edges']} "
        f"output_rows={preprocess_summary['output_rows']}"
    )

    _, bn_summary = partition_compat(pairwise_csv, output_dir=bn_output_dir)
    print(f"[BN] output_dir={bn_output_dir}")
    print(
        f"[BN] nodes={len(bn_summary['nodes'])} "
        f"directed_edges={len(bn_summary['directed_edges'])} "
        f"associated_pairs={len(bn_summary['associated_pairs'])}"
    )

    return {
        "pattern_file": str(pattern_csv),
        "pattern_name": pattern_name,
        "pairwise_csv": str(pairwise_csv),
        "bn_output_dir": str(bn_output_dir),
        "protein_nodes": preprocess_summary["protein_nodes"],
        "positive_edges": preprocess_summary["positive_edges"],
        "negative_edges": preprocess_summary["negative_edges"],
        "pairwise_rows": preprocess_summary["output_rows"],
        "bn_nodes": len(bn_summary["nodes"]),
        "bn_directed_edges": len(bn_summary["directed_edges"]),
        "bn_associated_pairs": len(bn_summary["associated_pairs"]),
        "bn_independent_pairs": len(bn_summary["independent_pairs"]),
        "elapsed_seconds": bn_summary.get("elapsed_seconds", None),
    }


def main():
    if not PROTEIN_CSV.exists():
        raise FileNotFoundError(f"Protein CSV not found: {PROTEIN_CSV}")
    if not PATTERN_INTERACTION_DIR.exists():
        raise FileNotFoundError(f"Pattern interaction directory not found: {PATTERN_INTERACTION_DIR}")

    pattern_csvs = iter_pattern_csvs(PATTERN_INTERACTION_DIR)
    if not pattern_csvs:
        raise RuntimeError(f"No pattern interaction CSV files found in: {PATTERN_INTERACTION_DIR}")

    PIPELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for pattern_csv in pattern_csvs:
        try:
            summary_rows.append(
                run_single_pattern(
                    pattern_csv=pattern_csv,
                    protein_csv=PROTEIN_CSV,
                    output_root=PIPELINE_OUTPUT_DIR,
                )
            )
        except Exception as exc:
            print(f"[Error] {pattern_csv.name}: {exc}")
            summary_rows.append(
                {
                    "pattern_file": str(pattern_csv),
                    "pattern_name": pattern_csv.stem,
                    "pairwise_csv": "",
                    "bn_output_dir": "",
                    "protein_nodes": None,
                    "positive_edges": None,
                    "negative_edges": None,
                    "pairwise_rows": None,
                    "bn_nodes": None,
                    "bn_directed_edges": None,
                    "bn_associated_pairs": None,
                    "bn_independent_pairs": None,
                    "elapsed_seconds": None,
                    "error": str(exc),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = PIPELINE_OUTPUT_DIR / "pattern_bn_pipeline_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"\n[Done] processed_patterns={len(summary_df)}")
    print(f"[Done] summary_csv={summary_csv}")


if __name__ == "__main__":
    main()
