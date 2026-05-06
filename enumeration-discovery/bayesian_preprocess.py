import argparse
import re
from pathlib import Path

import pandas as pd


PROTEIN_CSV = "/home/yyyy/codework/GARplus/GNN/code/DDA_test/data/去病图数据/protein.csv"
INTERACTION_CSV = "/home/yyyy/codework/GARplus/GNN/code/DDA_test/data/去病图数据/protein_protein.csv"
OUTPUT_CSV = "predicate_pairwise_table.csv"

OUTPUT_COLUMNS = [
    "protein_x",
    "protein_y",
    "x_high_evidence",
    "y_high_evidence",
    "x_long_protein",
    "y_long_protein",
    "x_has_coiled_coil",
    "y_has_coiled_coil",
    "x_membrane_related",
    "y_membrane_related",
    "x_nucleus_related",
    "y_nucleus_related",
    "x_secreted_related",
    "y_secreted_related",
    "same_location",
    "same_pathway",
    "domain_match",
    "family_match",
]


def normalize_colname(name):
    return re.sub(r"[^a-z0-9]+", "", str(name).strip().lower())


def build_column_map(df):
    return {normalize_colname(col): col for col in df.columns}


def find_first_existing(df, candidates):
    colmap = build_column_map(df)
    for cand in candidates:
        key = normalize_colname(cand)
        if key in colmap:
            return colmap[key]
    return None


def require_columns(df, candidates, label):
    col = find_first_existing(df, candidates)
    if col is None:
        raise ValueError(
            f"Required column for '{label}' is missing. "
            f"Checked aliases={candidates}. Please verify the attribute name in the CSV."
        )
    return col


def find_optional_column(df, candidates):
    return find_first_existing(df, candidates)


def require_any_column(df, candidate_groups, label):
    for candidates in candidate_groups:
        col = find_first_existing(df, candidates)
        if col is not None:
            return col
    raise ValueError(
        f"Required column group for '{label}' is missing. "
        f"Checked aliases={candidate_groups}. Please verify the attribute name in the CSV."
    )


def find_id_column_protein(df):
    return require_columns(df, ["index"], "protein index")


def find_interactor_columns(df):
    a_col = require_columns(df, ["index_A"], "interaction index_A")
    b_col = require_columns(df, ["index_B"], "interaction index_B")
    return a_col, b_col


def is_missing(value):
    if value is None:
        return True
    if pd.isna(value):
        return True
    s = str(value).strip()
    return s == "" or s.lower() in {"nan", "none", "null", "-", "--", "na", "n/a"}


def parse_float(value, default=None):
    if is_missing(value):
        return default
    try:
        return float(str(value).strip())
    except Exception:
        return default


def parse_int(value, default=None):
    parsed = parse_float(value, default=None)
    if parsed is None:
        return default
    try:
        return int(parsed)
    except Exception:
        return default


def clean_text(value):
    if is_missing(value):
        return ""
    return str(value).strip().lower()


def split_to_set(value):
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
    merged = set()
    for value in values:
        merged |= split_to_set(value)
    return merged


def safe_get(row, colname):
    if colname is None or colname not in row.index:
        return None
    return row[colname]


def any_token_contains(tokens, keywords):
    for token in tokens:
        for keyword in keywords:
            if keyword in token:
                return 1
    return 0


def has_overlap(set_a, set_b):
    return int(len(set_a & set_b) > 0)


def default_protein_features():
    return {
        "high_evidence": 0,
        "long_protein": 0,
        "has_coiled_coil": 0,
        "membrane_related": 0,
        "nucleus_related": 0,
        "secreted_related": 0,
        "location_set": set(),
        "pathway_set": set(),
        "domain_set": set(),
        "family_set": set(),
    }


def build_protein_feature_map(protein_df):
    # Step 1. Read protein.csv and build node-level predicates and token sets.
    id_col = find_id_column_protein(protein_df)
    protein_existence_col = require_columns(
        protein_df, ["Protein existence"], "Protein existence"
    )
    length_col = require_columns(protein_df, ["Length"], "Length")
    coiled_col = require_columns(protein_df, ["Coiled coil"], "Coiled coil")
    transmembrane_col = require_columns(protein_df, ["Transmembrane"], "Transmembrane")
    location_col = require_columns(
        protein_df, ["Subcellular location [CC]"], "Subcellular location [CC]"
    )
    go_cc_col = require_columns(
        protein_df,
        ["Gene Ontology (cellular component)"],
        "Gene Ontology (cellular component)",
    )
    signal_peptide_col = require_columns(protein_df, ["Signal peptide"], "Signal peptide")
    pathway_col = find_optional_column(protein_df, ["pathway"])
    pathway1_col = find_optional_column(protein_df, ["Pathway1", "Pathway 1"])
    if pathway_col is None and pathway1_col is None:
        raise ValueError(
            "At least one pathway column is required. "
            "Checked aliases=['pathway'] and ['Pathway1', 'Pathway 1']."
        )

    domain_col = find_optional_column(protein_df, ["domain"])
    domain_cc_col = find_optional_column(protein_df, ["Domain [CC]", "Domain CC"])
    domain_ft_col = find_optional_column(protein_df, ["Domain [FT]", "Domain FT"])
    if domain_col is None and domain_cc_col is None and domain_ft_col is None:
        raise ValueError(
            "At least one domain column is required. "
            "Checked aliases=['domain'], ['Domain [CC]', 'Domain CC'], "
            "['Domain [FT]', 'Domain FT']."
        )
    family_col = require_columns(protein_df, ["Protein families"], "Protein families")

    protein_map = {}
    for _, row in protein_df.iterrows():
        pid = parse_int(safe_get(row, id_col), default=None)
        if pid is None:
            continue

        protein_existence = clean_text(safe_get(row, protein_existence_col))
        length_value = parse_int(safe_get(row, length_col), default=None)
        location_set = split_to_set(safe_get(row, location_col))
        go_cc_set = split_to_set(safe_get(row, go_cc_col))
        pathway_set = merge_sets(safe_get(row, pathway_col), safe_get(row, pathway1_col))
        domain_set = merge_sets(
            safe_get(row, domain_col),
            safe_get(row, domain_cc_col),
            safe_get(row, domain_ft_col),
        )
        family_set = split_to_set(safe_get(row, family_col))

        membrane_related = int(
            (not is_missing(safe_get(row, transmembrane_col)))
            or any_token_contains(location_set, ["membrane"])
            or any_token_contains(go_cc_set, ["membrane"])
        )
        nucleus_related = int(
            any_token_contains(location_set, ["nucleus", "nuclear"])
            or any_token_contains(go_cc_set, ["nucleus", "nuclear"])
        )
        secreted_related = int(
            (not is_missing(safe_get(row, signal_peptide_col)))
            or any_token_contains(location_set, ["extracellular", "secreted"])
        )

        protein_map[pid] = {
            "high_evidence": int(protein_existence == "evidence at protein level"),
            "long_protein": int(length_value is not None and length_value >= 500),
            "has_coiled_coil": int(not is_missing(safe_get(row, coiled_col))),
            "membrane_related": membrane_related,
            "nucleus_related": nucleus_related,
            "secreted_related": secreted_related,
            "location_set": location_set,
            "pathway_set": pathway_set,
            "domain_set": domain_set,
            "family_set": family_set,
        }

    return protein_map


def build_pair_list(interaction_df):
    # Step 2. Read protein_protein.csv and build unique undirected pairs.
    a_col, b_col = find_interactor_columns(interaction_df)

    pairs = set()
    for _, row in interaction_df.iterrows():
        a = parse_int(safe_get(row, a_col), default=None)
        b = parse_int(safe_get(row, b_col), default=None)
        if a is None or b is None or a == b:
            continue
        protein_x, protein_y = sorted((a, b))
        pairs.add((protein_x, protein_y))

    return sorted(pairs)


def build_pairwise_rows(pair_list, protein_map):
    # Step 3. Expand constant predicates to x_/y_ columns and keep
    # variable predicates as pair-level columns.
    rows = []
    for protein_x, protein_y in pair_list:
        x_feat = protein_map.get(protein_x, default_protein_features())
        y_feat = protein_map.get(protein_y, default_protein_features())

        row = {
            "protein_x": int(protein_x),
            "protein_y": int(protein_y),
            "x_high_evidence": int(x_feat["high_evidence"]),
            "y_high_evidence": int(y_feat["high_evidence"]),
            "x_long_protein": int(x_feat["long_protein"]),
            "y_long_protein": int(y_feat["long_protein"]),
            "x_has_coiled_coil": int(x_feat["has_coiled_coil"]),
            "y_has_coiled_coil": int(y_feat["has_coiled_coil"]),
            "x_membrane_related": int(x_feat["membrane_related"]),
            "y_membrane_related": int(y_feat["membrane_related"]),
            "x_nucleus_related": int(x_feat["nucleus_related"]),
            "y_nucleus_related": int(y_feat["nucleus_related"]),
            "x_secreted_related": int(x_feat["secreted_related"]),
            "y_secreted_related": int(y_feat["secreted_related"]),
            "same_location": has_overlap(x_feat["location_set"], y_feat["location_set"]),
            "same_pathway": has_overlap(x_feat["pathway_set"], y_feat["pathway_set"]),
            "domain_match": has_overlap(x_feat["domain_set"], y_feat["domain_set"]),
            "family_match": has_overlap(x_feat["family_set"], y_feat["family_set"]),
        }
        rows.append(row)

    return rows


def build_pairwise_table(protein_csv, interaction_csv, output_csv, neg_ratio=1.0):
    # Reusable entrypoint kept for compatibility with downstream scripts.
    protein_df = pd.read_csv(protein_csv, low_memory=False)
    interaction_df = pd.read_csv(interaction_csv, low_memory=False)

    protein_map = build_protein_feature_map(protein_df)
    pair_list = build_pair_list(interaction_df)
    rows = build_pairwise_rows(pair_list, protein_map)

    out_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    for col in OUTPUT_COLUMNS[2:]:
        out_df[col] = pd.to_numeric(out_df[col], errors="coerce").fillna(0).astype(int)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = out_df.sort_values(["protein_x", "protein_y"]).reset_index(drop=True)
    out_df.to_csv(output_csv, index=False)

    summary = {
        "protein_nodes": len(protein_map),
        "positive_edges": len(pair_list),
        "negative_edges": 0,
        "output_rows": len(out_df),
        "output_csv": str(output_csv),
    }
    return out_df, summary


def main():
    parser = argparse.ArgumentParser(description="Build endpoint-expanded PPI pairwise predicate table.")
    parser.parse_args([])

    base_dir = Path(__file__).resolve().parent
    output_path = base_dir / OUTPUT_CSV
    _, summary = build_pairwise_table(
        protein_csv=PROTEIN_CSV,
        interaction_csv=INTERACTION_CSV,
        output_csv=output_path,
    )

    print(f"Protein nodes: {summary['protein_nodes']}")
    print(f"Positive edges: {summary['positive_edges']}")
    print(f"Negative edges: {summary['negative_edges']}")
    print(f"Output rows: {summary['output_rows']}")


if __name__ == "__main__":
    main()
