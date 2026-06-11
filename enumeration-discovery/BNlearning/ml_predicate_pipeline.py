from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import predicate_constrcution as pc


CURRENT_DIR = Path(__file__).resolve().parent
ENUM_DISCOVERY_DIR = CURRENT_DIR.parent
PPI_DATA_DIR = Path("/home/yyyy/codework/GARplus/enumeration-discovery/去病图数据")
PROCESSED_PPI_DIR = Path("/home/yyyy/codework/GARplus/enumeration-discovery/processed/ppi")

# File-level config. This script is intentionally configured in code, not via CLI.
PROTEIN_CSV = str(PPI_DATA_DIR / "protein.csv")
PPI_CSV = str(PPI_DATA_DIR / "protein_protein.csv")
OUTPUT_CSV = str(PROCESSED_PPI_DIR / "ml_predicates.csv")

BIG_GRAPH_CACHE = str(PROCESSED_PPI_DIR / "ppi_big_graph.pkl")
INTERACTION_LOOKUP_CACHE = str(PROCESSED_PPI_DIR / "protein_protein_edge_lookup_rich.pkl")

NEGATIVE_RATIO = 1.0
TARGET_PRECISION = 0.9
MAX_CANDIDATE_PAIRS = 100000
RANDOM_SEED = 42
DEBUG_TRAINING = True
DEBUG_SAMPLE_PAIRS: list[tuple[int, int]] = []
POSITIVE_SCORE_QUANTILE = 0.95
NEGATIVE_SCORE_QUANTILE = 0.05

load_or_build_big_graph = pc.load_or_build_big_graph
load_or_build_interaction_lookup = pc.load_or_build_interaction_lookup


def attach_edge_records_from_lookup(
    graph: nx.Graph,
    edge_to_rows: dict[tuple[int, int], list[dict[str, Any]]],
) -> nx.Graph:
    """Attach rich interaction rows to the topology graph."""
    if hasattr(pc, "attach_biogrid_edge_records_from_lookup"):
        return pc.attach_biogrid_edge_records_from_lookup(graph, edge_to_rows)

    for _, _, attrs in graph.edges(data=True):
        attrs["records"] = []
        attrs.pop("edge_labels", None)

    for (u, v), records in edge_to_rows.items():
        if not graph.has_node(u):
            graph.add_node(u)
        if not graph.has_node(v):
            graph.add_node(v)
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)

        for record in records:
            graph[u][v].setdefault("records", []).append(record)

    return graph


# ============================================================
# 1. Field cleaning
# ============================================================


def normalize_col_name(col: str) -> str:
    """
    Normalize raw CSV column names to a stable snake_case format.

    The protein and interaction tables may contain spaces, punctuation, or
    BioGRID-style names. Normalization keeps downstream field lookup stable.
    """
    col = str(col).strip()
    col = col.replace("#", "")
    col = col.replace("[", "")
    col = col.replace("]", "")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace("/", "_")
    col = col.replace("-", "_")
    col = col.replace(".", "_")
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.lower().strip("_")


def read_csv_auto(path: str) -> pd.DataFrame:
    """
    Read a CSV file and normalize both column names and obvious missing values.
    """
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [normalize_col_name(c) for c in df.columns]
    df = df.replace(
        {
            "-": np.nan,
            "": np.nan,
            " ": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "NULL": np.nan,
        }
    )
    return df


def require_column(df: pd.DataFrame, candidates: list[str], table_name: str) -> str:
    """
    Return the first existing normalized column and fail loudly if none exist.

    For this PPI-specific version we intentionally force `index` for protein.csv
    and `index_A/index_B` for protein_protein.csv.
    """
    normalized = {normalize_col_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_col_name(candidate)
        if key in normalized:
            return normalized[key]
    raise ValueError(
        f"Cannot find required column in {table_name}. "
        f"Candidates={candidates}, available={list(df.columns)}"
    )


# ============================================================
# 2. Node attribute processing
# ============================================================


def infer_numeric_thresholds(
    df: pd.DataFrame,
    numeric_cols: list[str],
    quantiles: tuple[float, float] = (0.33, 0.66),
) -> dict[str, dict[str, float]]:
    """
    Infer low/medium/high thresholds for numeric protein attributes.
    """
    thresholds: dict[str, dict[str, float]] = {}
    for col in numeric_cols:
        col = normalize_col_name(col)
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue

        thresholds[col] = {
            "low_upper": float(values.quantile(quantiles[0])),
            "high_lower": float(values.quantile(quantiles[1])),
        }
    return thresholds


def bin_numeric_value(value: Any, low_upper: float, high_lower: float) -> str | None:
    """
    Map one numeric value into low / medium / high.
    """
    if pd.isna(value):
        return None

    value = float(value)
    if value <= low_upper:
        return "low"
    if value <= high_lower:
        return "medium"
    return "high"


def apply_numeric_bins(
    df: pd.DataFrame,
    thresholds: dict[str, dict[str, float]],
) -> pd.DataFrame:
    """
    Add binned companion columns such as length_bin for downstream features.
    """
    df = df.copy()
    for col, th in thresholds.items():
        if col not in df.columns:
            continue

        bin_col = f"{col}_bin"
        df[bin_col] = df[col].apply(
            lambda v: bin_numeric_value(
                v,
                low_upper=th["low_upper"],
                high_lower=th["high_lower"],
            )
        )
    return df


def split_tokens(value: Any) -> set[str]:
    """
    Convert a multi-value attribute cell into a token set.
    """
    if value is None or pd.isna(value):
        return set()

    if isinstance(value, list):
        return {str(x).strip() for x in value if str(x).strip()}

    value = str(value)
    for sep in [";", "|", ","]:
        value = value.replace(sep, ";")
    return {x.strip() for x in value.split(";") if x.strip()}


# ============================================================
# 3. Build / load the PPI graph
# ============================================================


def build_ppi_graph(
    protein_csv: str,
    ppi_csv: str,
    big_graph_cache: str = BIG_GRAPH_CACHE,
    interaction_lookup_cache: str = INTERACTION_LOOKUP_CACHE,
) -> nx.Graph:
    """
    Load the project-standard cached big graph and enrich it with rich edge rows.

    This replaces the earlier standalone graph-construction logic:
    1. protein.csv is aligned by `index`
    2. protein_protein.csv is aligned by `index_A` / `index_B`
    3. the script first reuses the current project cache flow
    4. only if the big graph cache is missing does it rebuild the topology
    """
    protein_df = read_csv_auto(protein_csv)
    node_id_col = require_column(protein_df, ["index"], "protein.csv")

    global_graph = load_or_build_big_graph(
        edge_csv=ppi_csv,
        node_csv=protein_csv,
        cache_path=big_graph_cache,
    )
    edge_to_rows = load_or_build_interaction_lookup(
        interaction_csv=ppi_csv,
        cache_path=interaction_lookup_cache,
    )
    global_graph = attach_edge_records_from_lookup(global_graph, edge_to_rows)

    numeric_cols = ["length", "annotation"]
    thresholds = infer_numeric_thresholds(protein_df, numeric_cols)
    protein_df = apply_numeric_bins(protein_df, thresholds)

    protein_rows: dict[int, dict[str, Any]] = {}
    for _, row in protein_df.iterrows():
        node_id = row.get(node_id_col)
        if pd.isna(node_id):
            continue
        try:
            node_id_int = int(str(node_id).strip())
        except Exception:
            continue
        protein_rows[node_id_int] = {k: v for k, v in row.items() if not pd.isna(v)}

    for node_id in list(global_graph.nodes()):
        if node_id in protein_rows:
            attrs = protein_rows[node_id]
            attrs["node_type"] = "protein"
            attrs["node_id_source"] = node_id_col
            global_graph.nodes[node_id].update(attrs)
        else:
            global_graph.nodes[node_id].setdefault("node_type", "protein")
            global_graph.nodes[node_id].setdefault("node_id_source", node_id_col)

    for u, v in global_graph.edges():
        global_graph[u][v].setdefault("edge_type", "ppi")
        global_graph[u][v].setdefault("is_observed_ppi", True)
        records = global_graph[u][v].get("records", [])
        global_graph[u][v]["interaction_count"] = max(1, len(records))

    print(f"[Graph] nodes={global_graph.number_of_nodes()}, edges={global_graph.number_of_edges()}")
    print(f"[Graph] node_id_col={node_id_col}")
    print("[Graph] edge_a_col=index_A, edge_b_col=index_B")
    return global_graph


# ============================================================
# 4. Link prediction features
# ============================================================


def get_attr(G: nx.Graph, node: Any, attr: str) -> Any:
    return G.nodes[node].get(attr, None)


def overlap_size(G: nx.Graph, x: Any, y: Any, attr: str) -> int:
    sx = split_tokens(get_attr(G, x, attr))
    sy = split_tokens(get_attr(G, y, attr))
    return len(sx & sy)


def has_overlap(G: nx.Graph, x: Any, y: Any, attr: str) -> int:
    return int(overlap_size(G, x, y, attr) > 0)


def common_neighbors_count(G: nx.Graph, x: Any, y: Any) -> int:
    if x not in G or y not in G:
        return 0
    return len(list(nx.common_neighbors(G, x, y)))


def jaccard_score(G: nx.Graph, x: Any, y: Any) -> float:
    nx_set = set(G.neighbors(x))
    ny_set = set(G.neighbors(y))
    union = nx_set | ny_set
    if not union:
        return 0.0
    return len(nx_set & ny_set) / len(union)


def preferential_attachment_score(G: nx.Graph, x: Any, y: Any) -> int:
    return int(G.degree(x) * G.degree(y))


def adamic_adar_score(G: nx.Graph, x: Any, y: Any) -> float:
    score = 0.0
    for z in nx.common_neighbors(G, x, y):
        dz = G.degree(z)
        if dz > 1:
            score += 1.0 / np.log(dz)
    return float(score)


def get_default_overlap_attrs() -> list[str]:
    """
    Return the default overlap-attribute list used by pair feature extraction.

    This helper keeps build_pair_features(...) and get_pair_feature_names(...)
    aligned so the feature order and the feature names cannot drift apart.
    """
    return [
        "location",
        "pathway",
        "domain",
        "keywords",
        "protein_families",
        "gene_ontology_ids",
        "gene_ontology_go",
        "gene_ontology_biological_process",
        "gene_ontology_molecular_function",
        "gene_ontology_cellular_component",
        "subcellular_location_cc",
    ]


def get_pair_feature_names(
    overlap_attrs: list[str] | None = None,
) -> list[str]:
    """
    Return the feature-name list corresponding exactly to build_pair_features(...).
    """
    if overlap_attrs is None:
        overlap_attrs = get_default_overlap_attrs()

    names = [
        "common_neighbors",
        "jaccard",
        "log1p_preferential_attachment",
        "adamic_adar",
        "same_length_bin",
        "same_annotation_bin",
    ]

    for attr in overlap_attrs:
        names.append(f"has_overlap_{attr}")
        names.append(f"overlap_size_{attr}")

    return names


def build_pair_features(
    G: nx.Graph,
    x: Any,
    y: Any,
    overlap_attrs: list[str] | None = None,
) -> list[float]:
    """
    Build one feature vector for a candidate protein pair.
    """
    if overlap_attrs is None:
        overlap_attrs = get_default_overlap_attrs()

    features: list[float] = []
    features.append(common_neighbors_count(G, x, y))
    features.append(jaccard_score(G, x, y))
    # Raw preferential attachment is extremely large on this graph; log1p keeps
    # the signal while preventing it from dominating every other feature.
    features.append(np.log1p(preferential_attachment_score(G, x, y)))
    features.append(adamic_adar_score(G, x, y))

    for bin_attr in ["length_bin", "annotation_bin"]:
        vx = get_attr(G, x, bin_attr)
        vy = get_attr(G, y, bin_attr)
        features.append(float(vx is not None and vy is not None and vx == vy))

    for attr in overlap_attrs:
        if attr not in G.nodes[x] and attr not in G.nodes[y]:
            features.append(0.0)
            features.append(0.0)
            continue

        features.append(float(has_overlap(G, x, y, attr)))
        features.append(float(overlap_size(G, x, y, attr)))

    return features


# ============================================================
# 5. Positive / negative samples
# ============================================================


def sample_positive_edges(G: nx.Graph) -> list[tuple[Any, Any]]:
    return list(G.edges())


def sample_negative_edges(
    G: nx.Graph,
    num_samples: int,
    seed: int = RANDOM_SEED,
    confirmed_negative_edges: list[tuple[Any, Any]] | None = None,
) -> list[tuple[Any, Any]]:
    """
    Sample non-edges as negative examples, optionally preferring confirmed negatives.
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    negatives: list[tuple[Any, Any]] = []

    if confirmed_negative_edges:
        for x, y in confirmed_negative_edges:
            if x in G and y in G and not G.has_edge(x, y):
                edge = tuple(sorted((x, y)))
                negatives.append(edge)

    negatives = list(dict.fromkeys(negatives))
    seen = set(negatives)

    while len(negatives) < num_samples:
        x, y = rng.sample(nodes, 2)
        if G.has_edge(x, y):
            continue

        edge = tuple(sorted((x, y)))
        if edge in seen:
            continue

        seen.add(edge)
        negatives.append(edge)

    return negatives[:num_samples]


def build_link_prediction_dataset(
    G: nx.Graph,
    negative_ratio: float = 1.0,
    confirmed_negative_edges: list[tuple[Any, Any]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[Any, Any]]]:
    """
    Build one binary link-prediction dataset from the current graph.
    """
    pos_edges = sample_positive_edges(G)
    num_neg = int(len(pos_edges) * negative_ratio)
    neg_edges = sample_negative_edges(
        G,
        num_samples=num_neg,
        confirmed_negative_edges=confirmed_negative_edges,
    )

    pairs = pos_edges + neg_edges
    labels = [1] * len(pos_edges) + [0] * len(neg_edges)
    X = np.array([build_pair_features(G, x, y) for x, y in pairs], dtype=float)
    y = np.array(labels, dtype=int)
    return X, y, pairs


def inspect_feature_matrix(
    X: np.ndarray,
    y: np.ndarray,
    pairs: list[tuple[Any, Any]],
    output_dir: Path = PROCESSED_PPI_DIR,
) -> None:
    """
    Inspect the training feature matrix for sparsity, duplicates, and all-zero rows.
    """
    feature_names = get_pair_feature_names()
    X_df = pd.DataFrame(X, columns=feature_names)
    X_df["label"] = y
    X_df["x"] = [p[0] for p in pairs]
    X_df["y"] = [p[1] for p in pairs]

    print("[Feature matrix]")
    print("shape =", X.shape)
    print("positive =", int((y == 1).sum()))
    print("negative =", int((y == 0).sum()))

    print("\n[Feature non-zero ratio]")
    nonzero = (X_df[feature_names] != 0).mean().sort_values()
    print(nonzero.to_string())

    print("\n[Feature describe]")
    print(X_df[feature_names].describe().T.to_string())

    print("\n[Duplicate feature rows]")
    dup_ratio = X_df[feature_names].duplicated().mean()
    print("duplicate_feature_row_ratio =", float(dup_ratio))

    print("\n[All-zero rows]")
    all_zero_ratio = (X_df[feature_names].sum(axis=1) == 0).mean()
    print("all_zero_row_ratio =", float(all_zero_ratio))

    output_dir.mkdir(parents=True, exist_ok=True)
    X_df.head(1000).to_csv(
        output_dir / "ml_training_feature_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )
    nonzero.to_csv(
        output_dir / "ml_feature_nonzero_ratio.csv",
        encoding="utf-8-sig",
    )


# ============================================================
# 6. Simple ML model
# ============================================================


@dataclass
class MLThresholds:
    eta_pos: float
    eta_neg: float


def inspect_model_coefficients(model) -> pd.DataFrame:
    """
    Inspect learned Logistic Regression coefficients after standardization.

    Positive coefficients push predictions toward PPI, negative coefficients
    push predictions toward non-PPI, and abs_coef reflects feature reliance.
    """
    feature_names = get_pair_feature_names()
    clf = model.named_steps["clf"]
    coef = clf.coef_[0]

    return pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coef,
            "abs_coef": np.abs(coef),
        }
    ).sort_values("abs_coef", ascending=False)


def train_simple_link_predictor(
    G: nx.Graph,
    negative_ratio: float = 1.0,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
    confirmed_negative_edges: list[tuple[Any, Any]] | None = None,
):
    """
    Train a simple standardized logistic-regression link predictor.
    """
    X, y, pairs = build_link_prediction_dataset(
        G,
        negative_ratio=negative_ratio,
        confirmed_negative_edges=confirmed_negative_edges,
    )
    if DEBUG_TRAINING:
        inspect_feature_matrix(X, y, pairs)

    X_train, X_val, y_train, y_val, _, pairs_val = train_test_split(
        X,
        y,
        pairs,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    val_scores = model.predict_proba(X_val)[:, 1]
    return model, val_scores, y_val, pairs_val


def choose_eta_by_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Choose the positive threshold that maximizes validation F1.
    """
    candidates = np.linspace(0.05, 0.95, 91)
    best_eta = 0.5
    best_f1 = -1.0

    for eta in candidates:
        preds = (scores >= eta).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_eta = float(eta)

    return best_eta


def choose_eta_by_precision_target(
    scores: np.ndarray,
    labels: np.ndarray,
    target_precision: float = 0.9,
) -> float:
    """
    Choose the first threshold whose precision reaches the target value.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    best_eta = 0.9
    for p, r, eta in zip(precision[:-1], recall[:-1], thresholds):
        if p >= target_precision:
            best_eta = float(eta)
            break
    return best_eta


def choose_ml_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    positive_strategy: str = "precision",
    target_precision: float = 0.9,
    neg_quantile: float = 0.1,
) -> MLThresholds:
    """
    Convert validation scores into one positive and one negative predicate threshold.
    """
    if positive_strategy == "precision":
        eta_pos = choose_eta_by_precision_target(
            scores,
            labels,
            target_precision=target_precision,
        )
    elif positive_strategy == "f1":
        eta_pos = choose_eta_by_f1(scores, labels)
    else:
        raise ValueError(f"Unknown strategy: {positive_strategy}")

    neg_scores = scores[labels == 0]
    if len(neg_scores) == 0:
        eta_neg = 0.2
    else:
        eta_neg = float(np.quantile(neg_scores, neg_quantile))

    if eta_neg >= eta_pos:
        eta_neg = min(0.2, eta_pos * 0.5)

    return MLThresholds(eta_pos=float(eta_pos), eta_neg=float(eta_neg))


def choose_ml_thresholds_by_quantile(
    scores: np.ndarray,
    pos_quantile: float = POSITIVE_SCORE_QUANTILE,
    neg_quantile: float = NEGATIVE_SCORE_QUANTILE,
) -> MLThresholds:
    """
    Choose ML thresholds by score quantiles.

    This directly caps how many pairs become positive/negative ML predicates:
    - score >= eta_pos becomes ml_pred_ppi
    - score <= eta_neg becomes ml_pred_not_ppi
    """
    eta_pos = float(np.quantile(scores, pos_quantile))
    eta_neg = float(np.quantile(scores, neg_quantile))
    return MLThresholds(eta_pos=eta_pos, eta_neg=eta_neg)


def inspect_graph_attributes(G: nx.Graph) -> None:
    """
    Print node/edge attribute coverage after build_ppi_graph(...).

    This helps verify whether protein.csv attributes were really attached to the
    cached graph, which is critical for overlap-based pair features.
    """
    from collections import Counter

    node_keys = Counter()
    edge_keys = Counter()

    for _, attrs in G.nodes(data=True):
        node_keys.update(attrs.keys())
    for _, _, attrs in G.edges(data=True):
        edge_keys.update(attrs.keys())

    print("\n[After build_ppi_graph: node attrs]")
    for k, c in node_keys.most_common(80):
        print(k, c)

    print("\n[After build_ppi_graph: edge attrs]")
    for k, c in edge_keys.most_common(80):
        print(k, c)

    needed_node_attrs = [
        "length_bin",
        "annotation_bin",
        "location",
        "pathway",
        "domain",
        "keywords",
        "protein_families",
        "gene_ontology_ids",
        "gene_ontology_go",
        "gene_ontology_biological_process",
        "gene_ontology_molecular_function",
        "gene_ontology_cellular_component",
        "subcellular_location_cc",
    ]

    print("\n[Needed node attr coverage]")
    n = G.number_of_nodes()
    for attr in needed_node_attrs:
        cnt = sum(1 for _, a in G.nodes(data=True) if attr in a and pd.notna(a.get(attr)))
        ratio = (cnt / n) if n else 0.0
        print(f"{attr:45s} {cnt:10d} / {n} = {ratio:.6f}")


def explain_pair_features(G: nx.Graph, x: Any, y: Any) -> pd.DataFrame:
    """
    Return a readable feature breakdown for one protein pair.
    """
    names = get_pair_feature_names()
    values = build_pair_features(G, x, y)
    return pd.DataFrame({"feature": names, "value": values})


def debug_ml_training_data(
    G: nx.Graph,
    model=None,
    sample_pairs: list[tuple[Any, Any]] | None = None,
) -> None:
    """
    One-stop debug entry for graph attributes, feature matrix, and model weights.
    """
    inspect_graph_attributes(G)

    feature_names = get_pair_feature_names()
    print("\n[Feature names]")
    for i, name in enumerate(feature_names):
        print(f"{i:02d} {name}")

    X, y, pairs = build_link_prediction_dataset(G, negative_ratio=NEGATIVE_RATIO)
    inspect_feature_matrix(X, y, pairs)

    if model is not None:
        coef_df = inspect_model_coefficients(model)
        print("\n[Model coefficients]")
        print(coef_df.to_string(index=False))
        PROCESSED_PPI_DIR.mkdir(parents=True, exist_ok=True)
        coef_df.to_csv(
            PROCESSED_PPI_DIR / "ml_model_feature_coefficients.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if sample_pairs:
        for x, y_ in sample_pairs:
            print(f"\n[Pair features] ({x}, {y_})")
            print(explain_pair_features(G, x, y_).to_string(index=False))


# ============================================================
# 7. Generate ML predicates
# ============================================================


def ml_score_for_pair(model, G: nx.Graph, x: Any, y: Any) -> float:
    """
    Compute the link-prediction probability for one pair.
    """
    X = np.array([build_pair_features(G, x, y)], dtype=float)
    return float(model.predict_proba(X)[0, 1])


def ml_predicate_for_pair(
    model,
    G: nx.Graph,
    x: Any,
    y: Any,
    thresholds: MLThresholds,
) -> tuple[str | None, float]:
    """
    Convert one ML score into a predicate label or None.
    """
    score = ml_score_for_pair(model, G, x, y)
    if score >= thresholds.eta_pos:
        return "ml_pred_ppi", score
    if score <= thresholds.eta_neg:
        return "ml_pred_not_ppi", score
    return None, score


def generate_ml_predicates_for_pairs(
    model,
    G: nx.Graph,
    candidate_pairs: list[tuple[Any, Any]],
    thresholds: MLThresholds,
) -> list[dict[str, Any]]:
    """
    Generate ML-derived predicates for an explicit list of candidate pairs.
    """
    rows = []
    for x, y in candidate_pairs:
        pred, score = ml_predicate_for_pair(
            model=model,
            G=G,
            x=x,
            y=y,
            thresholds=thresholds,
        )
        if pred is None:
            continue
        rows.append(
            {
                "x": x,
                "y": y,
                "predicate": f"{pred}({x},{y})",
                "predicate_name": pred,
                "score": score,
                "source": "ml",
            }
        )
    return rows


def generate_ml_predicates_for_all_non_edges(
    model,
    G: nx.Graph,
    thresholds: MLThresholds,
    max_pairs: int = MAX_CANDIDATE_PAIRS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Sample non-edges and emit ML predicates for the confident subset.
    """
    rng = random.Random(seed)
    nodes = list(G.nodes())
    candidate_pairs = []
    seen = set()

    while len(candidate_pairs) < max_pairs:
        x, y = rng.sample(nodes, 2)
        if G.has_edge(x, y):
            continue

        edge = tuple(sorted((x, y)))
        if edge in seen:
            continue

        seen.add(edge)
        candidate_pairs.append(edge)

    rows = generate_ml_predicates_for_pairs(
        model=model,
        G=G,
        candidate_pairs=candidate_pairs,
        thresholds=thresholds,
    )
    return pd.DataFrame(rows)


# ============================================================
# 8. Pipeline entry
# ============================================================


def run_ppi_ml_predicate_pipeline(
    protein_csv: str = PROTEIN_CSV,
    ppi_csv: str = PPI_CSV,
    output_csv: str = OUTPUT_CSV,
    negative_ratio: float = NEGATIVE_RATIO,
    target_precision: float = TARGET_PRECISION,
    max_candidate_pairs: int = MAX_CANDIDATE_PAIRS,
):
    """
    End-to-end pipeline:
    1. load the PPI graph through the current project cache flow
    2. train a simple link predictor
    3. choose thresholds
    4. generate ML-derived predicates on sampled non-edges
    5. save the resulting CSV
    """
    G = build_ppi_graph(
        protein_csv=protein_csv,
        ppi_csv=ppi_csv,
        big_graph_cache=BIG_GRAPH_CACHE,
        interaction_lookup_cache=INTERACTION_LOOKUP_CACHE,
    )

    model, val_scores, y_val, pairs_val = train_simple_link_predictor(
        G,
        negative_ratio=negative_ratio,
    )
    if DEBUG_TRAINING:
        debug_ml_training_data(
            G,
            model=model,
            sample_pairs=DEBUG_SAMPLE_PAIRS,
        )

    thresholds = choose_ml_thresholds_by_quantile(
        val_scores,
        pos_quantile=POSITIVE_SCORE_QUANTILE,
        neg_quantile=NEGATIVE_SCORE_QUANTILE,
    )

    print(
        f"[Thresholds] strategy=quantile "
        f"pos_quantile={POSITIVE_SCORE_QUANTILE:.2f} "
        f"neg_quantile={NEGATIVE_SCORE_QUANTILE:.2f}"
    )
    print(f"[Thresholds] eta_pos={thresholds.eta_pos:.4f}")
    print(f"[Thresholds] eta_neg={thresholds.eta_neg:.4f}")

    feature_names = get_pair_feature_names()
    print("[Feature names]")
    for i, name in enumerate(feature_names):
        print(i, name)

    coef_df = inspect_model_coefficients(model)
    print("[Model coefficients]")
    print(coef_df.head(30).to_string(index=False))
    PROCESSED_PPI_DIR.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(
        PROCESSED_PPI_DIR / "ml_model_feature_coefficients.csv",
        index=False,
        encoding="utf-8-sig",
    )

    ml_df = generate_ml_predicates_for_all_non_edges(
        model=model,
        G=G,
        thresholds=thresholds,
        max_pairs=max_candidate_pairs,
    )

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    ml_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[Output] saved ML predicates to {output_csv}")
    print(f"[Output] num_ml_predicates={len(ml_df)}")
    return G, model, thresholds, ml_df


if __name__ == "__main__":
    run_ppi_ml_predicate_pipeline()
