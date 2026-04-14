import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42

# Edit these paths directly if needed.
# protein.csv: protein node table
# interaction.csv: interaction edge table
# pairwise_table.csv: final pair-wise predicate table
PROTEIN_CSV = "/home/yyyy/codework/GARplus/GNN/code/DDA_test/data/去病图数据/protein.csv"
INTERACTION_CSV = "/home/yyyy/codework/GARplus/GNN/code/DDA_test/data/去病图数据/protein_protein.csv"
OUTPUT_CSV = "predicate_pairwise_table.csv"

# Number of sampled negative edges = NEG_RATIO * number of positive edges.
NEG_RATIO = 1.0


def normalize_colname(name):
    # Normalize raw column names so we can match columns more robustly.
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def build_column_map(df):
    # Build a normalized-name -> raw-name lookup table.
    return {normalize_colname(col): col for col in df.columns}


def find_first_existing(df, candidates):
    # Return the first matching raw column name from a list of candidates.
    colmap = build_column_map(df)
    for cand in candidates:
        key = normalize_colname(cand)
        if key in colmap:
            return colmap[key]
    return None


def find_id_column_protein(df):
    # Prefer biogrid_id; fallback to index.
    return find_first_existing(df, ["biogrid_id", "index"])


def find_interactor_columns(df):
    # Try to identify the A/B endpoint columns in the interaction table.
    a_candidates = [
        "index_A",
        "BioGRID ID Interactor A",
        "biogrid_id_interactor_a",
        "interactor_a",
        "protein_a",
        "proteina",
        "a",
        "src",
        "source",
    ]
    b_candidates = [
        "index_B",
        "BioGRID ID Interactor B",
        "biogrid_id_interactor_b",
        "interactor_b",
        "protein_b",
        "proteinb",
        "b",
        "dst",
        "target",
    ]

    a_col = find_first_existing(df, a_candidates)
    b_col = find_first_existing(df, b_candidates)

    if a_col and b_col:
        return a_col, b_col

    colmap = build_column_map(df)
    for norm_name, raw_name in colmap.items():
        if a_col is None and ("interactora" in norm_name or norm_name.endswith("a")):
            a_col = raw_name
        if b_col is None and ("interactorb" in norm_name or norm_name.endswith("b")):
            b_col = raw_name

    return a_col, b_col


def is_missing(value):
    # Robust missing-value check for dirty CSV content.
    if value is None:
        return True
    if pd.isna(value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "-", "--", "na", "n/a"}


def parse_bool_like(value):
    # Parse loose boolean-like strings into 0/1.
    if is_missing(value):
        return 0
    s = str(value).strip().lower()
    return int(s in {"reviewed", "yes", "true", "1", "y", "t"})


def parse_float(value, default=None):
    # Safe float parsing.
    if is_missing(value):
        return default
    try:
        return float(str(value).strip())
    except Exception:
        return default


def parse_int(value, default=None):
    # Safe integer parsing; supports numeric strings and float-like content.
    f = parse_float(value, default=None)
    if f is None:
        return default
    try:
        return int(f)
    except Exception:
        return default


def clean_text(value):
    # Normalize a text field for keyword matching.
    if is_missing(value):
        return ""
    return str(value).strip().lower()


def split_to_set(value):
    # Convert a semicolon/comma/pipe separated text field into a cleaned token set.
    if is_missing(value):
        return set()

    text = str(value).strip().lower()
    parts = re.split(r"[;,|]", text)
    cleaned = set()

    for part in parts:
        token = re.sub(r"\s+", " ", part).strip()
        if token:
            cleaned.add(token)

    return cleaned


def merge_sets(*values):
    # Merge multiple tokenized fields into one set.
    out = set()
    for value in values:
        out |= split_to_set(value)
    return out


def has_overlap(set_a, set_b):
    # Predicate helper: whether two token sets overlap.
    return int(len(set_a & set_b) > 0)


def safe_get(row, colname):
    # Read a cell safely from a pandas row.
    if colname is None or colname not in row.index:
        return None
    return row[colname]


def build_protein_feature_map(protein_df):
    # ------------------------------------------------------------------
    # Step 1. Build per-protein feature records from protein.csv
    #
    # Output:
    #   protein_map[protein_id] = {
    #       reviewed, high_evidence, membrane_related, has_coiled_coil,
    #       long_protein, length,
    #       location_set, pathway_set, go_bp_set, domain_set, family_set
    #   }
    # ------------------------------------------------------------------
    id_col = find_id_column_protein(protein_df)
    if id_col is None:
        raise ValueError("protein.csv must contain either 'biogrid_id' or 'index'.")

    reviewed_col = find_first_existing(protein_df, ["Reviewed"])
    pe_col = find_first_existing(protein_df, ["Protein existence"])
    length_col = find_first_existing(protein_df, ["Length"])
    pathway_col = find_first_existing(protein_df, ["pathway"])
    pathway1_col = find_first_existing(protein_df, ["Pathway1"])
    location_col = find_first_existing(protein_df, ["Subcellular location [CC]"])
    go_bp_col = find_first_existing(protein_df, ["Gene Ontology (biological process)"])
    go_col = find_first_existing(protein_df, ["Gene Ontology (GO)"])
    domain_col = find_first_existing(protein_df, ["domain"])
    domain_cc_col = find_first_existing(protein_df, ["Domain [CC]"])
    domain_ft_col = find_first_existing(protein_df, ["Domain [FT]"])
    family_col = find_first_existing(protein_df, ["Protein families"])
    tm_col = find_first_existing(protein_df, ["Transmembrane"])
    coiled_col = find_first_existing(protein_df, ["Coiled coil"])

    protein_map = {}

    for _, row in protein_df.iterrows():
        pid = parse_int(safe_get(row, id_col), default=None)
        if pid is None:
            continue

        reviewed = parse_bool_like(safe_get(row, reviewed_col))
        pe_text = clean_text(safe_get(row, pe_col))
        high_evidence = int(pe_text == "evidence at protein level")

        length_val = parse_int(safe_get(row, length_col), default=None)
        long_protein = int(length_val is not None and length_val >= 500)

        location_raw = safe_get(row, location_col)
        location_set = split_to_set(location_raw)

        pathway_set = merge_sets(
            safe_get(row, pathway_col),
            safe_get(row, pathway1_col),
        )

        go_bp_raw = safe_get(row, go_bp_col)
        if len(split_to_set(go_bp_raw)) == 0:
            go_bp_raw = safe_get(row, go_col)
        go_bp_set = split_to_set(go_bp_raw)

        domain_set = merge_sets(
            safe_get(row, domain_col),
            safe_get(row, domain_cc_col),
            safe_get(row, domain_ft_col),
        )

        family_set = split_to_set(safe_get(row, family_col))

        membrane_related = 0
        if not is_missing(safe_get(row, tm_col)):
            membrane_related = 1
        elif "membrane" in clean_text(location_raw):
            membrane_related = 1

        has_coiled_coil = int(not is_missing(safe_get(row, coiled_col)))

        protein_map[pid] = {
            "reviewed": int(reviewed),
            "high_evidence": int(high_evidence),
            "membrane_related": int(membrane_related),
            "has_coiled_coil": int(has_coiled_coil),
            "long_protein": int(long_protein),
            "length": length_val,
            "location_set": location_set,
            "pathway_set": pathway_set,
            "go_bp_set": go_bp_set,
            "domain_set": domain_set,
            "family_set": family_set,
        }

    return protein_map


def build_positive_edges(interaction_df):
    # ------------------------------------------------------------------
    # Step 2. Build deduplicated positive undirected edges from interaction.csv
    #
    # For each unique undirected pair (a, b), aggregate:
    #   - physical_interaction
    #   - high_conf_interaction
    #
    # Output:
    #   edge_map[(a, b)] = {
    #       physical_interaction: 0/1,
    #       high_conf_interaction: 0/1
    #   }
    # ------------------------------------------------------------------
    a_col, b_col = find_interactor_columns(interaction_df)
    if a_col is None or b_col is None:
        raise ValueError("interaction.csv must contain recognizable A/B protein ID columns.")

    exp_type_col = find_first_existing(interaction_df, ["Experimental System Type"])
    score_col = find_first_existing(interaction_df, ["Score"])

    edge_map = {}

    for _, row in interaction_df.iterrows():
        a = parse_int(safe_get(row, a_col), default=None)
        b = parse_int(safe_get(row, b_col), default=None)

        if a is None or b is None or a == b:
            continue

        x, y = sorted((a, b))

        exp_type = clean_text(safe_get(row, exp_type_col))
        physical_interaction = int("physical" in exp_type)

        score = parse_float(safe_get(row, score_col), default=None)
        high_conf_interaction = int(score is not None and score >= 0.7)

        if (x, y) not in edge_map:
            edge_map[(x, y)] = {
                "physical_interaction": 0,
                "high_conf_interaction": 0,
            }

        edge_map[(x, y)]["physical_interaction"] = max(
            edge_map[(x, y)]["physical_interaction"],
            physical_interaction,
        )
        edge_map[(x, y)]["high_conf_interaction"] = max(
            edge_map[(x, y)]["high_conf_interaction"],
            high_conf_interaction,
        )

    return edge_map


def sample_negative_edges(node_ids, positive_edge_set, num_negatives, seed=42):
    # ------------------------------------------------------------------
    # Step 3. Randomly sample negative edges
    #
    # Rules:
    #   - not a positive edge
    #   - no self-loop
    #   - undirected edge stored as sorted tuple
    # ------------------------------------------------------------------
    rng = random.Random(seed)
    node_ids = list(sorted(set(node_ids)))

    if len(node_ids) < 2:
        return []

    negative_edges = set()
    max_attempts = max(100000, num_negatives * 50)
    attempts = 0

    while len(negative_edges) < num_negatives and attempts < max_attempts:
        a = rng.choice(node_ids)
        b = rng.choice(node_ids)
        attempts += 1

        if a == b:
            continue

        edge = tuple(sorted((a, b)))
        if edge in positive_edge_set:
            continue
        if edge in negative_edges:
            continue

        negative_edges.add(edge)

    if len(negative_edges) < num_negatives:
        print(
            f"[WARN] Requested {num_negatives} negative edges, "
            f"but only sampled {len(negative_edges)}."
        )

    return sorted(negative_edges)


def default_protein_features():
    # Default fallback features when a protein is missing from protein_map.
    return {
        "reviewed": 0,
        "high_evidence": 0,
        "membrane_related": 0,
        "has_coiled_coil": 0,
        "long_protein": 0,
        "length": None,
        "location_set": set(),
        "pathway_set": set(),
        "go_bp_set": set(),
        "domain_set": set(),
        "family_set": set(),
    }


def build_pairwise_rows(edge_map, negative_edges, protein_map):
    # ------------------------------------------------------------------
    # Step 4. Convert positive + negative edges into the final pair-wise table
    #
    # Each row corresponds to one protein pair (protein_a, protein_b) and
    # contains:
    #   - label
    #   - pair-wise boolean predicates
    #   - edge-level flags from the interaction table
    #
    # Output:
    #   List[dict] -> later written to pairwise_table.csv
    # ------------------------------------------------------------------
    rows = []

    def get_feat(pid):
        return protein_map.get(pid, default_protein_features())

    all_edges = []

    for (a, b), attrs in edge_map.items():
        all_edges.append(
            {
                "protein_a": int(a),
                "protein_b": int(b),
                "label": 1,
                "physical_interaction": int(attrs["physical_interaction"]),
                "high_conf_interaction": int(attrs["high_conf_interaction"]),
                "sampled_non_interaction": 0,
            }
        )

    for a, b in negative_edges:
        all_edges.append(
            {
                "protein_a": int(a),
                "protein_b": int(b),
                "label": 0,
                "physical_interaction": 0,
                "high_conf_interaction": 0,
                "sampled_non_interaction": 1,
            }
        )

    for edge_row in all_edges:
        a = edge_row["protein_a"]
        b = edge_row["protein_b"]
        fa = get_feat(a)
        fb = get_feat(b)

        len_a = fa["length"]
        len_b = fb["length"]
        if len_a is None or len_b is None or max(len_a, len_b) == 0:
            length_similar = 0
        else:
            length_similar = int(abs(len_a - len_b) / max(len_a, len_b) <= 0.2)

        row = {
            "protein_a": a,
            "protein_b": b,
            "label": int(edge_row["label"]),
            "both_reviewed": int(fa["reviewed"] == 1 and fb["reviewed"] == 1),
            "both_high_evidence": int(fa["high_evidence"] == 1 and fb["high_evidence"] == 1),
            "both_membrane_related": int(
                fa["membrane_related"] == 1 and fb["membrane_related"] == 1
            ),
            "both_has_coiled_coil": int(
                fa["has_coiled_coil"] == 1 and fb["has_coiled_coil"] == 1
            ),
            "both_long_protein": int(fa["long_protein"] == 1 and fb["long_protein"] == 1),
            "same_location": has_overlap(fa["location_set"], fb["location_set"]),
            "same_pathway": has_overlap(fa["pathway_set"], fb["pathway_set"]),
            "go_bp_overlap": has_overlap(fa["go_bp_set"], fb["go_bp_set"]),
            "domain_match": has_overlap(fa["domain_set"], fb["domain_set"]),
            "family_match": has_overlap(fa["family_set"], fb["family_set"]),
            "length_similar": int(length_similar),
            "physical_interaction": int(edge_row["physical_interaction"]),
            "high_conf_interaction": int(edge_row["high_conf_interaction"]),
            "sampled_non_interaction": int(edge_row["sampled_non_interaction"]),
        }
        rows.append(row)

    return rows


def main():
    # argparse is included as requested, but this script does not rely on
    # command-line parameters. Edit the file-level path constants instead.
    parser = argparse.ArgumentParser(description="Build pair-wise protein table.")
    parser.parse_args([])

    random.seed(SEED)
    np.random.seed(SEED)

    base_dir = Path(__file__).resolve().parent
    print(base_dir)
    protein_path = PROTEIN_CSV
    interaction_path = INTERACTION_CSV
    output_path = base_dir / OUTPUT_CSV

    protein_df = pd.read_csv(protein_path, low_memory=False)
    interaction_df = pd.read_csv(interaction_path, low_memory=False)

    # Build node-level feature map from protein.csv
    protein_map = build_protein_feature_map(protein_df)
    protein_node_ids = sorted(protein_map.keys())

    # Build positive interaction edges from interaction.csv
    edge_map = build_positive_edges(interaction_df)
    positive_edge_set = set(edge_map.keys())
    num_positive = len(positive_edge_set)

    # Sample the same number of negative edges by default
    num_negative = int(round(num_positive * NEG_RATIO))
    negative_edges = sample_negative_edges(
        node_ids=protein_node_ids,
        positive_edge_set=positive_edge_set,
        num_negatives=num_negative,
        seed=SEED,
    )

    # Construct the final pair-wise predicate table
    rows = build_pairwise_rows(edge_map, negative_edges, protein_map)
    out_df = pd.DataFrame(rows)

    # Force all predicate/label columns to be integer 0/1
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

    # Write the final table to CSV
    out_df = out_df.sort_values(["protein_a", "protein_b"]).reset_index(drop=True)
    out_df.to_csv(output_path, index=False)

    # Final summary printed to terminal
    print(f"Protein nodes: {len(protein_node_ids)}")
    print(f"Positive edges: {num_positive}")
    print(f"Negative edges: {len(negative_edges)}")
    print(f"Output rows: {len(out_df)}")


if __name__ == "__main__":
    main()
