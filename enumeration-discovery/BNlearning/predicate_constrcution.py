from __future__ import annotations

"""
Global PPI Predicate Repository.

This module builds ONE global predicate repository from the whole PPI graph,
instead of building a predicate repository for each pattern Q.

For each sampled pattern Q, we only evaluate whether each global predicate
exists in Q. The resulting table has:

    rows    = sampled patterns
    columns = global predicates
    values  = 0/1, whether predicate exists in the pattern

This table can be used to learn a global Predicate-BN or to guide predicate
selection during level-wise rule mining.

Compared with the earlier draft, global graph loading now follows the existing
project cache flow used by match_selected_subgraphs.py:
1. load processed/ppi/ppi_big_graph.pkl when present
2. otherwise build from raw PPI csv via pick_patterns.build_ppi_graph
3. save back to the same cache path

If a rich PPI csv is provided, its row-level attributes are attached onto the
cached topology graph so edge-label predicates can still be constructed.
"""

import json
import pickle
import re
import sys
import warnings
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from pgmpy.estimators import BayesianEstimator, HillClimbSearch

try:
    from pgmpy.estimators import BIC as BICScore
except ImportError:
    from pgmpy.estimators import BicScore as BICScore

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
except ImportError:
    from pgmpy.models import BayesianNetwork as BayesianModel


# ---------------------------------------------------------------------
# Optional project imports
# ---------------------------------------------------------------------

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = MODULE_DIR.parent

# Modified: this script lives in BNlearning/, but project utilities such as
# inspect_graph.py and pick_patterns.py live one level above in
# enumeration-discovery/. Insert that parent directory into sys.path so local
# project imports resolve correctly when the script is executed directly.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

try:
    from inspect_graph import DEFAULT_SELECTED_PATH, SelectedPPIDataset
except Exception:
    DEFAULT_SELECTED_PATH = None
    SelectedPPIDataset = None

try:
    from pick_patterns import DEFAULT_EDGE_CSV, DEFAULT_NODE_CSV, build_ppi_graph
except Exception:
    DEFAULT_EDGE_CSV = None
    DEFAULT_NODE_CSV = None
    build_ppi_graph = None


CURRENT_DIR = PROJECT_DIR
OUTPUT_DIR = CURRENT_DIR / "processed" / "ppi" / "global_predicate_repo"
DEFAULT_BIG_GRAPH_CACHE = CURRENT_DIR / "processed" / "ppi" / "ppi_big_graph.pkl"
DEFAULT_INTERACTION_LOOKUP_CACHE = CURRENT_DIR / "processed" / "ppi" / "protein_protein_edge_lookup_rich.pkl"
DEFAULT_PREDICATE_CONFIG = CURRENT_DIR / "processed" / "ppi" / "predicate_config.json"
DEFAULT_ML_PREDICATES_CSV = CURRENT_DIR / "processed" / "ppi" / "ml_predicates.csv"
# Rich interaction table used only to re-attach row-level edge attributes
# onto the cached big-graph topology.
DEFAULT_RICH_PPI_CSV = str(CURRENT_DIR / "data" / "protein_protein.csv")


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

MIN_PREDICATE_SUPPORT = 5
MAX_PREDICATES_FOR_BN = 300
MAX_VARIABLES_PER_FAMILY = 100
MIN_FAMILY_SIZE = 2
MAX_INDEGREE = 2
MAX_ITER = int(1e4)
MAX_EDGE_LABEL_VALUE_LEN = 60

SELECTED_PATH = str(DEFAULT_SELECTED_PATH) if DEFAULT_SELECTED_PATH is not None else None
PPI_CSV = DEFAULT_RICH_PPI_CSV
EDGE_CSV = str(DEFAULT_EDGE_CSV) if DEFAULT_EDGE_CSV is not None else None
NODE_CSV = str(DEFAULT_NODE_CSV) if DEFAULT_NODE_CSV is not None else None
BIG_GRAPH_CACHE = str(DEFAULT_BIG_GRAPH_CACHE)
INTERACTION_LOOKUP_CACHE = str(DEFAULT_INTERACTION_LOOKUP_CACHE)
PREDICATE_CONFIG_JSON = str(DEFAULT_PREDICATE_CONFIG)
ML_PREDICATES_CSV = str(DEFAULT_ML_PREDICATES_CSV)
OUTPUT_PATH = str(OUTPUT_DIR)
INCLUDE_NEG_EDGES = False
LEARN_GLOBAL_PREDICATE_BN = True
INCLUDE_CONFIG_PREDICATES = True
INCLUDE_ML_PREDICATES = True

# BioGRID edge fields that are useful as edge predicates.
EDGE_CATEGORICAL_FIELDS = [
    "Experimental System",
    "Experimental System Type",
    "Throughput",
    "Modification",
    "Ontology Term Categories",
    "Ontology Term Types",
]

EDGE_NUMERIC_FIELDS = [
    "Score",
]

# Node fields. These are aggregated from A/B columns when BioGRID CSV is given.
NODE_CATEGORICAL_FIELDS = [
    "organism_name",
]

# Structural node attributes computed from the global PPI graph.
NODE_NUMERIC_FIELDS = [
    "degree",
    "clustering",
    "core",
]

NODE_CATEGORICAL_DERIVED_FIELDS = [
    "role",
    "clustering_bin",
    "core_bin",
]


# ---------------------------------------------------------------------
# Predicate data structures
# ---------------------------------------------------------------------


class PredicateType(str, Enum):
    NODE_CONST = "node_const"          # exists x in Q: x.A op c
    NODE_PAIR = "node_pair"            # exists edge/pair (x,y): x.A op y.B
    EDGE_LABEL = "edge_label"          # exists edge (x,y) in Q with label l
    NEG_EDGE_LABEL = "neg_edge_label"  # exists pair (x,y) in Q without label l
    # Modified: config-driven predicate families from predicate_config.json.
    CONFIG_NODE_NUMERIC_BIN = "config_node_numeric_bin"
    CONFIG_NODE_CATEGORICAL = "config_node_categorical"
    CONFIG_NODE_TOKEN = "config_node_token"
    CONFIG_EDGE_NUMERIC_BIN = "config_edge_numeric_bin"
    CONFIG_EDGE_CATEGORICAL = "config_edge_categorical"
    CONFIG_EDGE_TOKEN = "config_edge_token"
    CONFIG_NODE_PAIR_OVERLAP = "config_node_pair_overlap"
    # Modified: ML predicates from ml_predicates.csv.
    ML_PAIR = "ml_pair"


@dataclass(frozen=True)
class Predicate:
    pid: str
    ptype: PredicateType

    # For node predicates
    attr1: Optional[str] = None
    op: Optional[str] = None
    const: Optional[Any] = None

    # For node-pair comparison predicates
    attr2: Optional[str] = None

    # For edge predicates
    label: Optional[str] = None

    # Metadata
    family: Optional[str] = None
    source: Optional[str] = None


@dataclass
class PredicateRepository:
    predicates: list[Predicate]
    by_type: dict[str, list[str]]
    by_family: dict[str, list[str]]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


def normalize_token(x: Any) -> str:
    """
    Normalize one raw token into a stable predicate-friendly string.

    This function is used whenever free-text node/edge attributes are converted
    into predicate ids or label values. The goal is to keep the representation
    deterministic across different CSV rows and reruns.
    """
    if pd.isna(x):
        return "missing"
    s = str(x).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_:\\-\\.]+", "_", s)
    s = s.strip("_")
    return s if s else "missing"


def normalize_col_name(col: str) -> str:
    """
    Modified: normalize config / CSV column names to the same convention used
    by predicate_threshold.py so config-driven predicates can align with graph
    node and edge attributes.
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


def is_missing(x: Any) -> bool:
    """
    Unified missing-value check used across CSV parsing and predicate building.

    The code mixes pandas values, Python None, and possibly malformed cells, so
    this helper centralizes the check instead of duplicating edge cases.
    """
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def split_multivalue(x: Any) -> list[str]:
    """
    BioGRID fields may contain values separated by |, ;, or comma.
    """
    if is_missing(x):
        return []
    s = str(x).strip()
    if not s or s == "-":
        return []
    parts = re.split(r"[|;,]+", s)
    return [normalize_token(p) for p in parts if normalize_token(p) != "missing"]


def load_predicate_config(path: str | Path) -> dict[str, Any]:
    """
    Modified: load predicate_config.json generated by predicate_threshold.py.
    """
    path = Path(path)
    if not path.exists():
        print(f"[Warn] predicate_config.json not found: {path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"[Config] loaded predicate config: {path}")
    return config


def clean_config_token(token: Any) -> str:
    """
    Modified: normalize token values used in config-driven contains predicates.
    """
    token = str(token).strip()
    token = token.strip("{}[]()\"'")
    token = re.sub(r"\s+", " ", token)
    return token.strip()


def split_config_tokens(value: Any) -> set[str]:
    """
    Modified: split node/edge attribute values for config-driven token matching.
    The behavior is intentionally close to predicate_threshold.py.
    """
    if is_missing(value):
        return set()

    if isinstance(value, list):
        raw_tokens = value
    else:
        s = str(value)
        for sep in [";", "|", ","]:
            s = s.replace(sep, ";")
        raw_tokens = s.split(";")

    tokens = set()
    for tok in raw_tokens:
        tok = clean_config_token(tok)
        if tok and tok != "-":
            tokens.add(tok)
    return tokens


def value_to_three_bin(value: Any, threshold: dict[str, float]) -> str | None:
    """
    Modified: discretize values according to predicate_config.json thresholds.
    """
    if is_missing(value):
        return None
    try:
        v = float(value)
    except Exception:
        return None

    low_upper = float(threshold["low_upper"])
    high_lower = float(threshold["high_lower"])
    if v <= low_upper:
        return "low"
    if v <= high_lower:
        return "medium"
    return "high"


def normalize_record_keys(record: dict[str, Any]) -> dict[str, Any]:
    """
    Modified: normalize raw edge-record keys so config field names and cached
    record field names can be matched consistently.
    """
    return {normalize_col_name(k): v for k, v in record.items()}


def load_ml_predicate_index(
    ml_csv: str | Path,
    undirected: bool = True,
) -> dict[tuple[int, int], list[str]]:
    """
    Modified: load ml_predicates.csv into an undirected pair -> predicate-name
    index so ML predicates can be counted inside sampled patterns.
    """
    ml_csv = Path(ml_csv)
    if not ml_csv.exists():
        print(f"[Warn] ML predicate csv not found: {ml_csv}")
        return {}

    df = pd.read_csv(ml_csv)
    required = {"x", "y", "predicate_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ML predicates csv missing columns: {missing}")

    index: dict[tuple[int, int], list[str]] = {}
    for _, row in df.iterrows():
        try:
            x = int(row["x"])
            y = int(row["y"])
        except Exception:
            continue

        pred_name = str(row["predicate_name"]).strip()
        key = canonical_edge(x, y) if undirected else (x, y)
        index.setdefault(key, []).append(pred_name)

    print(f"[ML] loaded ML predicate pairs={len(index)} from {ml_csv}")
    return index


def compare_values(a: Any, op: str, b: Any) -> bool:
    """
    Evaluate one atomic predicate comparison.

    This helper is shared by node-constant predicates and node-pair predicates.
    For numeric operators it attempts float conversion; for equality operators
    it compares the raw values directly.
    """
    if is_missing(a) or is_missing(b):
        return False

    if op in {"=", "=="}:
        return a == b
    if op in {"!=", "<>"}:
        return a != b

    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return False

    if op == ">=":
        return af >= bf
    if op == ">":
        return af > bf
    if op == "<=":
        return af <= bf
    if op == "<":
        return af < bf

    raise ValueError(f"Unknown operator: {op}")


def quantile_bins(values: list[float], low_name="low", mid_name="mid", high_name="high") -> tuple[float, float]:
    """
    Compute the two quantile cut points used for three-way discretization.

    The pipeline bins structural statistics such as degree/clustering/core into
    low/mid/high groups. Quantiles are used instead of fixed thresholds so the
    bins adapt to the actual global PPI graph distribution.
    """
    arr = np.asarray([v for v in values if not is_missing(v)], dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    if np.all(arr == arr[0]):
        return float(arr[0]), float(arr[0])
    q1 = float(np.quantile(arr, 1 / 3))
    q2 = float(np.quantile(arr, 2 / 3))
    return q1, q2


def assign_three_bin(value: float, q1: float, q2: float, low_name="low", mid_name="mid", high_name="high") -> str:
    """
    Map one numeric value into a three-bin categorical label.

    This is the companion of quantile_bins(...). It is used to derive symbolic
    node labels such as role / clustering_bin / core_bin from global numeric
    graph statistics.
    """
    if is_missing(value):
        return "missing"
    v = float(value)
    if q1 == q2:
        return mid_name
    if v <= q1:
        return low_name
    if v >= q2:
        return high_name
    return mid_name


def canonical_edge(u: Any, v: Any) -> tuple[int, int]:
    """
    Canonicalize one undirected edge into a sorted integer tuple.

    The repository, lookup cache, and edge record attachment all treat the PPI
    graph as undirected, so this helper guarantees one unique key per edge.
    """
    left, right = sorted((int(u), int(v)))
    return left, right


def get_interaction_endpoint_columns(df: pd.DataFrame) -> tuple[str, str]:
    """
    Detect which two columns in an interaction CSV represent the edge endpoints.

    The code prefers index_A/index_B but keeps a small fallback set so older or
    external interaction tables can still be reused without rewriting them.
    """
    for src_col, dst_col in [("index_A", "index_B"), ("src", "dst"), ("source", "target")]:
        if src_col in df.columns and dst_col in df.columns:
            return src_col, dst_col
    raise ValueError(
        "Interaction CSV must contain one of these endpoint column pairs: "
        "('index_A','index_B'), ('src','dst'), or ('source','target')."
    )


def infer_edge_label_family(label: str) -> str:
    """
    Map one concrete edge label to a broader predicate family.

    This is used when edge predicates are grouped semantically. For example,
    score_bin=high and score>=0.7 both belong to the score family.
    """
    label = str(label)
    if label.startswith("experimental_system_type="):
        return "experimental_system_type"
    if label.startswith("experimental_system="):
        return "experimental_system"
    if label.startswith("throughput="):
        return "throughput"
    if label.startswith("score_bin=") or label.startswith("score>=") or label.startswith("score<"):
        return "score"
    if label.startswith("ontology_term_categories="):
        return "ontology_term_categories"
    if label.startswith("ontology_term_types="):
        return "ontology_term_types"
    if label.startswith("modification="):
        return "modification"
    if label.startswith("qualifications="):
        return "qualifications"
    return "edge_label_other"


# ---------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------


def load_or_build_big_graph(
    edge_csv: Optional[str | Path] = None,
    node_csv: Optional[str | Path] = None,
    cache_path: str | Path = DEFAULT_BIG_GRAPH_CACHE,
) -> nx.Graph:
    """
    Reuse the same big-graph cache flow as match_selected_subgraphs.py.

    The cache stores the whole PPI topology. If the cache does not exist,
    build it from pick_patterns.build_ppi_graph and save it back.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        print(f"[Cache] Loading cached big graph from: {cache_path}")
        with open(cache_path, "rb") as f:
            graph = pickle.load(f)
        print(
            f"[Cache] Loaded big graph: "
            f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
        )
        return graph

    if build_ppi_graph is None:
        raise ImportError(
            "pick_patterns.build_ppi_graph is not available, so the big graph cache "
            "cannot be rebuilt automatically."
        )
    if edge_csv is None or node_csv is None:
        raise ValueError("edge_csv and node_csv are required when the big graph cache is missing.")

    print("[Cache] Cached big graph not found. Building from raw csv...")
    graph = build_ppi_graph(str(edge_csv), str(node_csv))

    with open(cache_path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Cache] Saved big graph cache to: {cache_path}")
    return graph


def load_or_build_interaction_lookup(
    interaction_csv: str | Path,
    cache_path: str | Path = DEFAULT_INTERACTION_LOOKUP_CACHE,
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """
    Reuse an existing edge->rows lookup cache when present.

    Expected cache format:
        {
            "interaction_csv": "...",
            "columns": [...],
            "edge_to_rows": {(u, v): [row_dict, ...], ...}
        }
    """
    cache_path = Path(cache_path)
    interaction_csv = str(interaction_csv)

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        if isinstance(cache, dict) and cache.get("interaction_csv") == interaction_csv and "edge_to_rows" in cache:
            print(f"[Cache] Loading cached interaction lookup from: {cache_path}")
            return cache["edge_to_rows"]

    print(f"[Cache] Cached interaction lookup not found or source changed. Building from: {interaction_csv}")
    df = pd.read_csv(interaction_csv, low_memory=False)
    src_col, dst_col = get_interaction_endpoint_columns(df)
    edge_to_rows: dict[tuple[int, int], list[dict[str, Any]]] = {}

    for _, row in df.iterrows():
        try:
            src = int(row[src_col])
            dst = int(row[dst_col])
        except Exception:
            continue
        if src == dst:
            continue

        edge = canonical_edge(src, dst)
        edge_to_rows.setdefault(edge, []).append(row.to_dict())

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(
            {
                "interaction_csv": interaction_csv,
                "columns": list(df.columns),
                "edge_to_rows": edge_to_rows,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"[Cache] Saved interaction lookup to: {cache_path}")
    return edge_to_rows


def attach_biogrid_edge_records_from_lookup(
    graph: nx.Graph,
    edge_to_rows: dict[tuple[int, int], list[dict[str, Any]]],
) -> nx.Graph:
    """
    Enrich the cached topology graph with row-level interaction records.

    This keeps the existing cached graph as the primary loading path, while
    restoring rich edge attributes needed by predicate construction.
    """
    # Clear old records/labels so reruns stay deterministic.
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
            org_a = record.get("Organism Name Interactor A")
            if not is_missing(org_a):
                graph.nodes[u]["organism_name"] = normalize_token(org_a)

            org_b = record.get("Organism Name Interactor B")
            if not is_missing(org_b):
                graph.nodes[v]["organism_name"] = normalize_token(org_b)

            graph[u][v].setdefault("records", []).append(record)

    return graph


def attach_biogrid_edge_records(
    graph: nx.Graph,
    csv_path: str | Path,
    lookup_cache_path: str | Path = DEFAULT_INTERACTION_LOOKUP_CACHE,
) -> nx.Graph:
    """
    Public wrapper that enriches the cached topology graph with rich edge rows.

    The wrapper first resolves the edge->rows lookup cache, then delegates the
    actual graph mutation to attach_biogrid_edge_records_from_lookup(...).
    """
    edge_to_rows = load_or_build_interaction_lookup(csv_path, lookup_cache_path)
    return attach_biogrid_edge_records_from_lookup(graph, edge_to_rows)


def data_to_nx_graph(data) -> nx.Graph:
    """
    Convert one PyG sampled-subgraph object into a simple undirected nx.Graph.

    The sampled patterns are stored in PyG format, but downstream predicate
    evaluation and BN learning only need a lightweight NetworkX graph.
    """
    graph = nx.Graph()

    if hasattr(data, "orig_node_ids"):
        node_names = [int(v) for v in data.orig_node_ids.tolist()]
    else:
        node_names = list(range(int(data.num_nodes)))

    for local_idx, node_name in enumerate(node_names):
        graph.add_node(node_name, local_idx=int(local_idx))

    edge_index = data.edge_index
    seen_edges = set()
    for eid in range(edge_index.size(1)):
        src_local = int(edge_index[0, eid])
        dst_local = int(edge_index[1, eid])
        if src_local == dst_local:
            continue

        src = node_names[src_local]
        dst = node_names[dst_local]
        key = tuple(sorted((src, dst)))
        if key in seen_edges:
            continue

        seen_edges.add(key)
        graph.add_edge(src, dst)

    return graph


def load_selected_pattern_graphs(selected_path: str | Path) -> list[tuple[int, nx.Graph]]:
    """
    Load all selected sampled patterns and convert them to NetworkX graphs.

    The returned structure is a list of (pattern_id, graph) pairs, which is the
    basic input format used by the rest of the predicate-construction pipeline.
    """
    if SelectedPPIDataset is None:
        raise ImportError("Cannot import SelectedPPIDataset from inspect_graph.")

    dataset = SelectedPPIDataset(str(selected_path))
    patterns = []
    for idx in range(len(dataset)):
        data = dataset.get(idx)
        patterns.append((idx, data_to_nx_graph(data)))
    return patterns


def build_union_graph(patterns: list[tuple[int, nx.Graph]]) -> nx.Graph:
    """
    Merge all sampled patterns into one union graph.

    This helper is kept for compatibility and future extensions. It is useful
    when only sampled patterns are available and no cached global PPI graph is
    provided.
    """
    union = nx.Graph()
    for _, g in patterns:
        union.add_nodes_from(g.nodes(data=True))
        union.add_edges_from(g.edges(data=True))
    return union


# ---------------------------------------------------------------------
# Node and edge feature augmentation
# ---------------------------------------------------------------------


def augment_node_structural_attributes(graph: nx.Graph) -> None:
    """
    Add global structural attributes to each node:
        degree
        clustering
        core
        role
        clustering_bin
        core_bin
    """
    degree_map = dict(graph.degree())
    clustering_map = nx.clustering(graph)
    if graph.number_of_edges() > 0:
        core_map = nx.core_number(graph)
    else:
        core_map = {n: 0 for n in graph.nodes()}

    degree_values = [degree_map[n] for n in graph.nodes()]
    clustering_values = [clustering_map[n] for n in graph.nodes()]
    core_values = [core_map[n] for n in graph.nodes()]

    deg_q1, deg_q2 = quantile_bins(degree_values)
    clu_q1, clu_q2 = quantile_bins(clustering_values)
    core_q1, core_q2 = quantile_bins(core_values)

    for n in graph.nodes():
        degree = float(degree_map[n])
        clustering = float(clustering_map[n])
        core = float(core_map[n])

        graph.nodes[n]["degree"] = degree
        graph.nodes[n]["clustering"] = clustering
        graph.nodes[n]["core"] = core

        graph.nodes[n]["role"] = assign_three_bin(
            degree,
            deg_q1,
            deg_q2,
            low_name="leaf",
            mid_name="mid",
            high_name="hub",
        )
        graph.nodes[n]["clustering_bin"] = assign_three_bin(clustering, clu_q1, clu_q2)
        graph.nodes[n]["core_bin"] = assign_three_bin(core, core_q1, core_q2)


def build_edge_label_vocabulary(graph: nx.Graph) -> set[str]:
    """
    Convert PPI edge attributes into edge labels.

    Each edge gets a set of labels stored in graph[u][v]["edge_labels"].
    When no attached records exist, the edge simply contributes no rich labels.
    """
    score_values = []
    for _, _, attrs in graph.edges(data=True):
        for rec in attrs.get("records", []):
            val = rec.get("Score")
            if not is_missing(val):
                try:
                    score_values.append(float(val))
                except Exception:
                    pass

    score_q1, score_q2 = quantile_bins(score_values)
    all_labels = set()

    for u, v, attrs in graph.edges(data=True):
        labels = set()
        records = attrs.get("records", [])

        for rec in records:
            for field in EDGE_CATEGORICAL_FIELDS:
                if field not in rec:
                    continue
                for value in split_multivalue(rec.get(field)):
                    if len(value) > MAX_EDGE_LABEL_VALUE_LEN:
                        continue
                    label = f"{normalize_token(field)}={value}"
                    labels.add(label)

            if "Score" in rec and not is_missing(rec.get("Score")):
                try:
                    score = float(rec.get("Score"))
                    score_bin = assign_three_bin(score, score_q1, score_q2)
                    labels.add(f"score_bin={score_bin}")
                    labels.add(f"score>={score_q1:.6g}")
                    labels.add(f"score>={score_q2:.6g}")
                except Exception:
                    labels.add("score_bin=missing")
            else:
                labels.add("score_bin=missing")

        attrs["edge_labels"] = labels
        all_labels.update(labels)

    return all_labels


# ---------------------------------------------------------------------
# Global predicate repository construction
# ---------------------------------------------------------------------


def add_pred(preds: list[Predicate], pred: Predicate, seen: set[str]) -> None:
    """
    Append one predicate only if its predicate id has not appeared before.

    Global predicate construction collects predicates from many attribute
    sources; this helper keeps repository construction deterministic and deduped.
    """
    if pred.pid not in seen:
        preds.append(pred)
        seen.add(pred.pid)


def add_config_predicates_to_repository(
    preds: list[Predicate],
    seen: set[str],
    config: dict[str, Any],
) -> None:
    """
    Modified: add candidate predicates generated by predicate_config.json into
    the global predicate repository.
    """
    for field in config.get("node_numeric_thresholds", {}).keys():
        for bin_value in ["low", "medium", "high"]:
            pid = f"CFG_NODE:{field}_bin={bin_value}"
            add_pred(
                preds,
                Predicate(
                    pid=pid,
                    ptype=PredicateType.CONFIG_NODE_NUMERIC_BIN,
                    attr1=field,
                    op="=",
                    const=bin_value,
                    family=f"node_{field}",
                    source="predicate_config",
                ),
                seen,
            )

    for field, values in config.get("node_categorical_values", {}).items():
        for value in values:
            pid = f"CFG_NODE:{field}={normalize_token(value)}"
            add_pred(
                preds,
                Predicate(
                    pid=pid,
                    ptype=PredicateType.CONFIG_NODE_CATEGORICAL,
                    attr1=field,
                    op="=",
                    const=value,
                    family=f"node_{field}",
                    source="predicate_config",
                ),
                seen,
            )

    for field, tokens in config.get("node_token_values", {}).items():
        for token in tokens:
            pid = f"CFG_NODE_TOKEN:{field}={normalize_token(token)}"
            add_pred(
                preds,
                Predicate(
                    pid=pid,
                    ptype=PredicateType.CONFIG_NODE_TOKEN,
                    attr1=field,
                    op="contains",
                    const=token,
                    family=f"node_token_{field}",
                    source="predicate_config",
                ),
                seen,
            )

    for field in config.get("edge_numeric_thresholds", {}).keys():
        for bin_value in ["low", "medium", "high"]:
            pid = f"CFG_EDGE:{field}_bin={bin_value}"
            add_pred(
                preds,
                Predicate(
                    pid=pid,
                    ptype=PredicateType.CONFIG_EDGE_NUMERIC_BIN,
                    attr1=field,
                    op="=",
                    const=bin_value,
                    family=f"edge_{field}",
                    source="predicate_config",
                ),
                seen,
            )

    for field, values in config.get("edge_categorical_values", {}).items():
        for value in values:
            pid = f"CFG_EDGE:{field}={normalize_token(value)}"
            add_pred(
                preds,
                Predicate(
                    pid=pid,
                    ptype=PredicateType.CONFIG_EDGE_CATEGORICAL,
                    attr1=field,
                    op="=",
                    const=value,
                    family=f"edge_{field}",
                    source="predicate_config",
                ),
                seen,
            )

    for field, tokens in config.get("edge_token_values", {}).items():
        for token in tokens:
            pid = f"CFG_EDGE_TOKEN:{field}={normalize_token(token)}"
            add_pred(
                preds,
                Predicate(
                    pid=pid,
                    ptype=PredicateType.CONFIG_EDGE_TOKEN,
                    attr1=field,
                    op="contains",
                    const=token,
                    family=f"edge_token_{field}",
                    source="predicate_config",
                ),
                seen,
            )

    for field in config.get("node_pair_overlap_fields", []):
        pid = f"CFG_PAIR_SHARE:{field}"
        add_pred(
            preds,
            Predicate(
                pid=pid,
                ptype=PredicateType.CONFIG_NODE_PAIR_OVERLAP,
                attr1=field,
                op="share",
                const=None,
                family=f"share_{field}",
                source="predicate_config",
            ),
            seen,
        )


def add_ml_predicates_to_repository(
    preds: list[Predicate],
    seen: set[str],
    ml_predicate_index: dict[tuple[int, int], list[str]],
) -> None:
    """
    Modified: add only predicate names such as ML:ml_pred_ppi into the global
    repository, not one variable per pair.
    """
    if not ml_predicate_index:
        return

    pred_names = sorted({name for names in ml_predicate_index.values() for name in names})
    for name in pred_names:
        pid = f"ML:{name}"
        add_pred(
            preds,
            Predicate(
                pid=pid,
                ptype=PredicateType.ML_PAIR,
                attr1="ml_predicate_name",
                op="=",
                const=name,
                family="ml",
                source="ml_predicates",
            ),
            seen,
        )


def summarize_predicate_prefix_counts(predicate_ids: list[str]) -> dict[str, int]:
    """
    Modified: summarize how many predicates belong to each major source prefix.

    This is a lightweight visibility helper so we can quickly verify whether
    config-driven and ML-driven predicates have really entered the repository or
    the exported predicate tables.
    """
    prefix_counts = {
        "NODE:": 0,
        "PAIR:": 0,
        "EDGE:": 0,
        "NEG_EDGE:": 0,
        "CFG_NODE:": 0,
        "CFG_NODE_TOKEN:": 0,
        "CFG_EDGE:": 0,
        "CFG_EDGE_TOKEN:": 0,
        "CFG_PAIR_SHARE:": 0,
        "ML:": 0,
    }
    for pid in predicate_ids:
        for prefix in prefix_counts:
            if str(pid).startswith(prefix):
                prefix_counts[prefix] += 1
                break
    return prefix_counts


def build_global_predicate_repository(
    graph: nx.Graph,
    min_support: int = MIN_PREDICATE_SUPPORT,
    include_neg_edges: bool = False,
    predicate_config: Optional[dict[str, Any]] = None,
    ml_predicate_index: Optional[dict[tuple[int, int], list[str]]] = None,
) -> PredicateRepository:
    """
    Build ONE global predicate repository from the entire PPI graph.

    The repository contains:
    1. Node attribute-to-constant predicates
    2. Node pair comparison predicates
    3. Edge label predicates
    4. Optional negated edge label predicates
    """
    preds: list[Predicate] = []
    seen = set()
    #补充节点的结构属性
    augment_node_structural_attributes(graph)
    all_edge_labels = build_edge_label_vocabulary(graph)

    for attr in NODE_CATEGORICAL_DERIVED_FIELDS + NODE_CATEGORICAL_FIELDS:
        value_counts = {}
        for n in graph.nodes():
            if attr in graph.nodes[n] and not is_missing(graph.nodes[n][attr]):
                val = graph.nodes[n][attr]
                value_counts[val] = value_counts.get(val, 0) + 1

        for val, cnt in sorted(value_counts.items(), key=lambda x: (-x[1], str(x[0]))):
            if cnt < min_support:
                continue
            pid = f"NODE:{attr}={val}"
            add_pred(
                preds,
                Predicate(
                    pid=pid,
                    ptype=PredicateType.NODE_CONST,
                    attr1=attr,
                    op="=",
                    const=val,
                    family=attr,
                    source="node_categorical",
                ),
                seen,
            )

    for attr in NODE_NUMERIC_FIELDS:
        values = [graph.nodes[n].get(attr) for n in graph.nodes() if attr in graph.nodes[n]]
        values = [float(v) for v in values if not is_missing(v)]
        if not values:
            continue

        q1, q2 = quantile_bins(values)
        min_v = float(min(values))
        max_v = float(max(values))
        for threshold in sorted(set([q1, q2])):
            if threshold <= min_v or threshold >= max_v:
                continue
            for op in [">=", "<"]:
                pid = f"NODE:{attr}{op}{threshold:.6g}"
                add_pred(
                    preds,
                    Predicate(
                        pid=pid,
                        ptype=PredicateType.NODE_CONST,
                        attr1=attr,
                        op=op,
                        const=float(threshold),
                        family=attr,
                        source="node_numeric",
                    ),
                    seen,
                )

    for attr in NODE_NUMERIC_FIELDS:
        for op in [">=", "<", "="]:
            pid = f"PAIR:{attr}{op}{attr}"
            add_pred(
                preds,
                Predicate(
                    pid=pid,
                    ptype=PredicateType.NODE_PAIR,
                    attr1=attr,
                    attr2=attr,
                    op=op,
                    family=attr,
                    source="node_pair_numeric",
                ),
                seen,
            )

    for attr in NODE_CATEGORICAL_DERIVED_FIELDS + NODE_CATEGORICAL_FIELDS:
        for op in ["=", "!="]:
            pid = f"PAIR:{attr}{op}{attr}"
            add_pred(
                preds,
                Predicate(
                    pid=pid,
                    ptype=PredicateType.NODE_PAIR,
                    attr1=attr,
                    attr2=attr,
                    op=op,
                    family=attr,
                    source="node_pair_categorical",
                ),
                seen,
            )

    label_support = {label: 0 for label in all_edge_labels}
    for _, _, attrs in graph.edges(data=True):
        for label in attrs.get("edge_labels", set()):
            label_support[label] = label_support.get(label, 0) + 1

    for label, cnt in sorted(label_support.items(), key=lambda x: (-x[1], x[0])):
        if cnt < min_support:
            continue

        pid = f"EDGE:{label}"
        add_pred(
            preds,
            Predicate(
                pid=pid,
                ptype=PredicateType.EDGE_LABEL,
                label=label,
                family=infer_edge_label_family(label),
                source="edge_attribute",
            ),
            seen,
        )

        if include_neg_edges:
            neg_pid = f"NEG_EDGE:{label}"
            add_pred(
                preds,
                Predicate(
                    pid=neg_pid,
                    ptype=PredicateType.NEG_EDGE_LABEL,
                    label=label,
                    family="neg_edge_label",
                    source="edge_attribute",
                ),
                seen,
            )

    # Modified: append config-driven predicates after the structural repository
    # is built, so both predicate sources share one global table and one BN.
    if predicate_config:
        print("[Info] Adding config-driven predicates...")
        add_config_predicates_to_repository(
            preds=preds,
            seen=seen,
            config=predicate_config,
        )

    # Modified: append ML predicate names as global variables.
    if ml_predicate_index:
        print("[Info] Adding ML predicates...")
        add_ml_predicates_to_repository(
            preds=preds,
            seen=seen,
            ml_predicate_index=ml_predicate_index,
        )

    by_type: dict[str, list[str]] = {}
    by_family: dict[str, list[str]] = {}
    for p in preds:
        by_type.setdefault(p.ptype.value, []).append(p.pid)
        by_family.setdefault(p.family or "unknown", []).append(p.pid)

    return PredicateRepository(predicates=preds, by_type=by_type, by_family=by_family)


# ---------------------------------------------------------------------
# Predicate evaluation on patterns
# ---------------------------------------------------------------------


def get_global_node_attrs(global_graph: nx.Graph, node: Any) -> dict[str, Any]:
    """
    Safely fetch the attribute dictionary of one node from the global graph.

    Missing nodes are tolerated because sampled patterns and cached graphs may
    be built from slightly different data sources during experimentation.
    """
    if node not in global_graph:
        return {}
    return global_graph.nodes[node]


def get_global_edge_attrs(global_graph: nx.Graph, u: Any, v: Any) -> dict[str, Any]:
    """
    Safely fetch one edge attribute dictionary from the global graph.

    Returning an empty dict instead of failing keeps predicate evaluation
    robust when a sampled pattern edge has no rich record attached.
    """
    if global_graph.has_edge(u, v):
        return global_graph[u][v]
    return {}


def get_config_node_tokens(global_graph: nx.Graph, node: Any, field: str) -> set[str]:
    """
    Modified: fetch and tokenize one node attribute for config-driven contains
    and overlap predicates.
    """
    attrs = get_global_node_attrs(global_graph, node)
    return split_config_tokens(attrs.get(field))


def edge_record_satisfies_config_predicate(
    record: dict[str, Any],
    predicate: Predicate,
    config: dict[str, Any],
) -> bool:
    """
    Modified: test one cached edge record against a config-driven edge
    predicate.
    """
    rec = normalize_record_keys(record)
    field = predicate.attr1

    if field not in rec:
        return False
    value = rec.get(field)

    if predicate.ptype == PredicateType.CONFIG_EDGE_CATEGORICAL:
        return str(value).strip() == str(predicate.const).strip()

    if predicate.ptype == PredicateType.CONFIG_EDGE_TOKEN:
        return str(predicate.const).strip() in split_config_tokens(value)

    if predicate.ptype == PredicateType.CONFIG_EDGE_NUMERIC_BIN:
        thresholds = config.get("edge_numeric_thresholds", {})
        if field not in thresholds:
            return False
        bin_value = value_to_three_bin(value, thresholds[field])
        return bin_value == predicate.const

    return False


def evaluate_predicate_on_pattern(
    predicate: Predicate,
    pattern_graph: nx.Graph,
    global_graph: nx.Graph,
) -> int:
    """
    Evaluate whether a global predicate exists in a sampled pattern.
    """
    nodes = list(pattern_graph.nodes())

    if predicate.ptype == PredicateType.NODE_CONST:
        for n in nodes:
            attrs = get_global_node_attrs(global_graph, n)
            if predicate.attr1 not in attrs:
                continue
            if compare_values(attrs.get(predicate.attr1), predicate.op, predicate.const):
                return 1
        return 0

    if predicate.ptype == PredicateType.NODE_PAIR:
        for u, v in pattern_graph.edges():
            u_attrs = get_global_node_attrs(global_graph, u)
            v_attrs = get_global_node_attrs(global_graph, v)
            if predicate.attr1 not in u_attrs or predicate.attr2 not in v_attrs:
                continue

            if compare_values(u_attrs.get(predicate.attr1), predicate.op, v_attrs.get(predicate.attr2)):
                return 1
            if compare_values(v_attrs.get(predicate.attr1), predicate.op, u_attrs.get(predicate.attr2)):
                return 1
        return 0

    if predicate.ptype == PredicateType.EDGE_LABEL:
        for u, v in pattern_graph.edges():
            edge_attrs = get_global_edge_attrs(global_graph, u, v)
            labels = edge_attrs.get("edge_labels", set())
            if predicate.label in labels:
                return 1
        return 0

    if predicate.ptype == PredicateType.NEG_EDGE_LABEL:
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                edge_attrs = get_global_edge_attrs(global_graph, u, v)
                labels = edge_attrs.get("edge_labels", set())
                if predicate.label not in labels:
                    return 1
        return 0

    raise ValueError(f"Unsupported predicate type: {predicate.ptype}")


def evaluate_negative_predicate_count_from_data(
    predicate: Predicate,
    pattern_graph: nx.Graph,
    global_graph: nx.Graph,
) -> int:
    """
    TODO:
    Count negative predicates using the user-provided negative-edge data/table.

    Do not enumerate all node pairs here.
    The final logic should match negative edges according to the given data
    source, e.g., confirmed negative interactions or explicitly annotated
    negative edges.

    For the current first version, this returns 0 so that enabling
    include_neg_edges does not break the pipeline.
    """
    return 0


def evaluate_predicate_count_on_pattern(
    predicate: Predicate,
    pattern_graph: nx.Graph,
    global_graph: nx.Graph,
    predicate_config: Optional[dict[str, Any]] = None,
    ml_predicate_index: Optional[dict[tuple[int, int], list[str]]] = None,
) -> int:
    """
    Count how many times one predicate holds in the current sampled pattern.

    This is the count-version companion of evaluate_predicate_on_pattern(...):
    - NODE_CONST counts satisfying nodes
    - NODE_PAIR counts satisfying pattern edges
    - EDGE_LABEL counts pattern edges carrying that label
    - NEG_EDGE_LABEL is delegated to a placeholder data-driven counter
    """
    nodes = list(pattern_graph.nodes())

    if predicate.ptype == PredicateType.NODE_CONST:
        count = 0
        for n in nodes:
            attrs = get_global_node_attrs(global_graph, n)
            if predicate.attr1 not in attrs:
                continue
            if compare_values(attrs.get(predicate.attr1), predicate.op, predicate.const):
                count += 1
        return count

    if predicate.ptype == PredicateType.NODE_PAIR:
        count = 0
        for u, v in pattern_graph.edges():
            u_attrs = get_global_node_attrs(global_graph, u)
            v_attrs = get_global_node_attrs(global_graph, v)
            if predicate.attr1 not in u_attrs or predicate.attr2 not in v_attrs:
                continue

            ok_uv = compare_values(u_attrs.get(predicate.attr1), predicate.op, v_attrs.get(predicate.attr2))
            ok_vu = compare_values(v_attrs.get(predicate.attr1), predicate.op, u_attrs.get(predicate.attr2))
            if ok_uv or ok_vu:
                count += 1
        return count

    if predicate.ptype == PredicateType.EDGE_LABEL:
        count = 0
        for u, v in pattern_graph.edges():
            edge_attrs = get_global_edge_attrs(global_graph, u, v)
            labels = edge_attrs.get("edge_labels", set())
            if predicate.label in labels:
                count += 1
        return count

    if predicate.ptype == PredicateType.NEG_EDGE_LABEL:
        return evaluate_negative_predicate_count_from_data(predicate, pattern_graph, global_graph)

    # Modified: config-driven node numeric bins such as CFG_NODE:length_bin=high.
    if predicate.ptype == PredicateType.CONFIG_NODE_NUMERIC_BIN:
        if not predicate_config:
            return 0
        thresholds = predicate_config.get("node_numeric_thresholds", {})
        field = predicate.attr1
        if field not in thresholds:
            return 0

        count = 0
        for n in nodes:
            attrs = get_global_node_attrs(global_graph, n)
            bin_value = value_to_three_bin(attrs.get(field), thresholds[field])
            if bin_value == predicate.const:
                count += 1
        return count

    # Modified: config-driven node categorical predicates.
    if predicate.ptype == PredicateType.CONFIG_NODE_CATEGORICAL:
        count = 0
        for n in nodes:
            attrs = get_global_node_attrs(global_graph, n)
            value = attrs.get(predicate.attr1)
            if is_missing(value):
                continue
            if str(value).strip() == str(predicate.const).strip():
                count += 1
        return count

    # Modified: config-driven node token contains predicates.
    if predicate.ptype == PredicateType.CONFIG_NODE_TOKEN:
        count = 0
        target = str(predicate.const).strip()
        for n in nodes:
            tokens = get_config_node_tokens(global_graph, n, predicate.attr1)
            if target in tokens:
                count += 1
        return count

    # Modified: config-driven edge predicates evaluate against cached records.
    if predicate.ptype in {
        PredicateType.CONFIG_EDGE_NUMERIC_BIN,
        PredicateType.CONFIG_EDGE_CATEGORICAL,
        PredicateType.CONFIG_EDGE_TOKEN,
    }:
        if not predicate_config:
            return 0

        count = 0
        for u, v in pattern_graph.edges():
            edge_attrs = get_global_edge_attrs(global_graph, u, v)
            records = edge_attrs.get("records", [])
            ok = False
            for rec in records:
                if edge_record_satisfies_config_predicate(rec, predicate, predicate_config):
                    ok = True
                    break
            if ok:
                count += 1
        return count

    # Modified: config-driven shared-token predicates on pattern edges.
    if predicate.ptype == PredicateType.CONFIG_NODE_PAIR_OVERLAP:
        if not predicate_config:
            return 0

        share_min = predicate_config.get("settings", {}).get("share_overlap_min", 1)
        field = predicate.attr1
        count = 0
        for u, v in pattern_graph.edges():
            u_tokens = get_config_node_tokens(global_graph, u, field)
            v_tokens = get_config_node_tokens(global_graph, v, field)
            if len(u_tokens & v_tokens) >= share_min:
                count += 1
        return count

    # Modified: ML predicates count all node pairs inside the sampled pattern.
    if predicate.ptype == PredicateType.ML_PAIR:
        if not ml_predicate_index:
            return 0

        target_name = str(predicate.const)
        count = 0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                u, v = nodes[i], nodes[j]
                try:
                    key = canonical_edge(u, v)
                except Exception:
                    continue
                names = ml_predicate_index.get(key, [])
                if target_name in names:
                    count += 1
        return count

    raise ValueError(f"Unsupported predicate type: {predicate.ptype}")


def bin_predicate_count(count: int) -> int:
    """
    Convert raw predicate count into a three-level discrete state.

    0: absent, count == 0
    1: low-frequency, count in [1, 2]
    2: high-frequency, count >= 3
    """
    if count <= 0:
        return 0
    if count <= 2:
        return 1
    return 2


def build_global_predicate_table(
    patterns: list[tuple[int, nx.Graph]],
    repository: PredicateRepository,
    global_graph: nx.Graph,
    mode: str = "binary",
    predicate_config: Optional[dict[str, Any]] = None,
    ml_predicate_index: Optional[dict[tuple[int, int], list[str]]] = None,
) -> pd.DataFrame:
    """
    Build a pattern-by-predicate table in one of three modes.

    mode="binary":
        0/1 occurrence table, indicating whether the predicate appears at least
        once in the sampled pattern.

    mode="count":
        raw count table, storing how many times the predicate holds in the
        sampled pattern.

    mode="binned":
        discretized count table using three states:
        0 = absent, 1 = low-frequency, 2 = high-frequency.
    """
    rows = []
    pattern_ids = []

    for pattern_id, pattern_graph in patterns:
        row = {}
        for pred in repository.predicates:
            # Modified: pass config-driven and ML-driven predicate sources into
            # the shared counting logic so all predicate families land in the
            # same binary / count / binned tables.
            count = evaluate_predicate_count_on_pattern(
                pred,
                pattern_graph,
                global_graph,
                predicate_config=predicate_config,
                ml_predicate_index=ml_predicate_index,
            )
            if mode == "binary":
                row[pred.pid] = int(count > 0)
            elif mode == "count":
                row[pred.pid] = int(count)
            elif mode == "binned":
                row[pred.pid] = bin_predicate_count(count)
            else:
                raise ValueError(f"Unknown predicate table mode: {mode}")
        rows.append(row)
        pattern_ids.append(pattern_id)

    df = pd.DataFrame(rows)
    df.insert(0, "pattern_id", pattern_ids)
    return df


# ---------------------------------------------------------------------
# BN learning from global predicate table
# ---------------------------------------------------------------------


def filter_predicate_table_for_bn(
    table: pd.DataFrame,
    min_support: int = MIN_PREDICATE_SUPPORT,
    max_variables: int = MAX_PREDICATES_FOR_BN,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Prepare one predicate table for BN learning.

    The filter is intentionally conservative:
    1. remove identifier column pattern_id
    2. remove constant predicates
    3. remove low-occurrence predicates using (df > 0).sum()
    4. cap the variable count by occurrence support

    This function is written to work for both binary tables and binned tables.
    """
    if "pattern_id" in table.columns:
        df = table.drop(columns=["pattern_id"])
    else:
        df = table.copy()

    info = {
        "constant_columns": [],
        "low_support_columns": [],
        "selected_columns": [],
    }

    if df.empty:
        return df, info

    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    df = df.drop(columns=constant_cols)
    info["constant_columns"] = constant_cols

    if df.empty:
        return df, info

    # For binary tables this equals ordinary support.
    # For binned tables this counts in how many sampled patterns the predicate
    # appears with a non-zero state, which is the more stable notion of support
    # for pruning.
    occurrence_support = (df > 0).sum()
    low_support_cols = occurrence_support[occurrence_support < min_support].index.tolist()
    df = df.drop(columns=low_support_cols)
    info["low_support_columns"] = low_support_cols

    if df.empty:
        return df, info

    occurrence_support = (df > 0).sum().sort_values(ascending=False)
    if max_variables is not None and df.shape[1] > max_variables:
        selected = occurrence_support.head(max_variables).index.tolist()
        df = df[selected]

    info["selected_columns"] = list(df.columns)
    return df.astype(int), info


def build_family_predicate_tables(
    full_table: pd.DataFrame,
    repository: PredicateRepository,
    min_family_size: int = MIN_FAMILY_SIZE,
) -> dict[str, pd.DataFrame]:
    """
    Split the global predicate table into family-specific subtables.

    This function is retained for ablation or future comparisons, even though
    the current default pipeline learns one global Predicate-BN.
    """
    family_to_pids: dict[str, list[str]] = {}
    for pred in repository.predicates:
        family = pred.family or "unknown"
        family_to_pids.setdefault(family, []).append(pred.pid)

    family_tables: dict[str, pd.DataFrame] = {}
    keep_pattern_id = "pattern_id" in full_table.columns

    for family, pids in sorted(family_to_pids.items()):
        existing_pids = [pid for pid in pids if pid in full_table.columns]
        if len(existing_pids) < min_family_size:
            continue

        columns = ["pattern_id"] + existing_pids if keep_pattern_id else existing_pids
        family_tables[family] = full_table[columns].copy()

    return family_tables


def learn_predicate_bn(df: pd.DataFrame) -> Optional[BayesianModel]:
    """
    Learn one Bayesian network structure and parameters from a predicate table.

    The implementation uses HillClimbSearch + BIC for structure learning and
    BayesianEstimator for CPD fitting. The table is expected to be discrete
    already, e.g. binary or 0/1/2 binned counts.
    """
    if df.empty:
        warnings.warn("Predicate-BN table is empty after filtering.", RuntimeWarning)
        return None
    if df.shape[0] < 10:
        warnings.warn(f"Too few rows for BN learning: {df.shape[0]}", RuntimeWarning)
        return None
    if df.shape[1] < 2:
        warnings.warn(f"Too few variables for BN learning: {df.shape[1]}", RuntimeWarning)
        return None

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
    model.fit(
        df,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10,
    )
    return model


# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------


def cpd_to_json(cpd) -> dict[str, Any]:
    """
    Convert one pgmpy CPD object into a JSON-serializable dictionary.

    The result is saved into result.json so the learned BN can be inspected
    without reloading Python objects.
    """
    state_names = getattr(cpd, "state_names", {}) or {}
    variable_states = state_names.get(cpd.variable, list(range(cpd.variable_card)))
    evidence = list(cpd.variables[1:])
    evidence_cards = list(cpd.cardinality[1:])
    evidence_states = [
        state_names.get(ev, list(range(card)))
        for ev, card in zip(evidence, evidence_cards)
    ]

    values = cpd.get_values()
    rows = []

    if evidence:
        import itertools

        for col_idx, evidence_assignment in enumerate(itertools.product(*evidence_states)):
            assignment = {
                str(evidence[i]): str(evidence_assignment[i])
                for i in range(len(evidence))
            }
            probs = {
                str(variable_states[row_idx]): float(values[row_idx, col_idx])
                for row_idx in range(len(variable_states))
            }
            rows.append({"evidence_states": assignment, "probabilities": probs})
    else:
        probs = {
            str(variable_states[row_idx]): float(values[row_idx, 0])
            for row_idx in range(len(variable_states))
        }
        rows.append({"evidence_states": {}, "probabilities": probs})

    return {
        "variable": str(cpd.variable),
        "evidence": [str(x) for x in evidence],
        "rows": rows,
    }


def save_single_family_bn_outputs(
    output_dir: str | Path,
    family: str,
    raw_table: pd.DataFrame,
    bn_table: pd.DataFrame,
    filter_info: dict[str, Any],
    model: Optional[BayesianModel],
) -> None:
    """
    Save all artifacts of one family-specific Predicate-BN run.

    Even though family-wise BN is no longer the default, this function is kept
    intact so the old workflow can still be re-enabled for comparison.
    """
    family_dir = Path(output_dir) / "family_bns" / f"family_{family}"
    family_dir.mkdir(parents=True, exist_ok=True)

    raw_table.to_csv(family_dir / "table_full.csv", index=False)
    bn_table.to_csv(family_dir / "table_bn.csv", index=False)

    if "pattern_id" in raw_table.columns:
        support = raw_table.drop(columns=["pattern_id"]).sum().sort_values(ascending=False)
    else:
        support = raw_table.sum().sort_values(ascending=False)

    support.rename_axis("predicate").reset_index(name="support").to_csv(
        family_dir / "support.csv",
        index=False,
    )

    status = "learned" if model is not None else "skipped"
    edges: list[list[str]] = []
    cpds: list[dict[str, Any]] = []

    if model is not None:
        edges = sorted([[str(u), str(v)] for u, v in model.edges()], key=lambda x: (x[0], x[1]))
        cpds = [cpd_to_json(cpd) for cpd in model.get_cpds()]
        pd.DataFrame(edges, columns=["source", "target"]).to_csv(
            family_dir / "edges.csv",
            index=False,
        )
    else:
        pd.DataFrame(columns=["source", "target"]).to_csv(
            family_dir / "edges.csv",
            index=False,
        )

    result = {
        "family": family,
        "status": status,
        "num_patterns": int(raw_table.shape[0]),
        "num_raw_predicates": int(raw_table.shape[1] - (1 if "pattern_id" in raw_table.columns else 0)),
        "num_bn_predicates": int(bn_table.shape[1]),
        "filter_info": filter_info,
        "edges": edges,
        "cpds": cpds,
    }
    if filter_info.get("reason"):
        result["reason"] = filter_info["reason"]

    with open(family_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    visualize_predicate_bn(
        model=model,
        output_dir=family_dir,
        filename="structure.png",
        title=f"Family Predicate-BN: {family}",
    )


def learn_family_predicate_bns(
    full_table: pd.DataFrame,
    repository: PredicateRepository,
    output_dir: str | Path,
    min_support: int = MIN_PREDICATE_SUPPORT,
    max_variables_per_family: int = MAX_VARIABLES_PER_FAMILY,
    min_family_size: int = MIN_FAMILY_SIZE,
) -> dict[str, Any]:
    """
    Learn one Predicate-BN per predicate family.

    This is legacy / ablation functionality now. The current main pipeline does
    not call it by default, but the function is preserved for backward
    compatibility and controlled experiments.
    """
    family_tables = build_family_predicate_tables(
        full_table=full_table,
        repository=repository,
        min_family_size=min_family_size,
    )

    family_results: dict[str, Any] = {}

    repository_families = sorted({pred.family or "unknown" for pred in repository.predicates})
    for family in repository_families:
        raw_predicates = [pred.pid for pred in repository.predicates if (pred.family or "unknown") == family]
        raw_pred_count = len([pid for pid in raw_predicates if pid in full_table.columns])

        if family not in family_tables:
            filter_info = {
                "constant_columns": [],
                "low_support_columns": [],
                "selected_columns": [],
                "reason": "family has fewer than min_family_size predicates in full table",
            }
            empty_table = full_table[["pattern_id"]].copy() if "pattern_id" in full_table.columns else pd.DataFrame()
            save_single_family_bn_outputs(output_dir, family, empty_table, pd.DataFrame(), filter_info, None)
            family_results[family] = {
                "family": family,
                "status": "skipped",
                "reason": filter_info["reason"],
                "num_patterns": int(full_table.shape[0]),
                "num_raw_predicates": raw_pred_count,
                "num_bn_predicates": 0,
                "filter_info": filter_info,
            }
            print(f"[Family-BN] family={family} raw={raw_pred_count} status=skipped")
            continue

        raw_table = family_tables[family]
        bn_table, filter_info = filter_predicate_table_for_bn(
            raw_table,
            min_support=min_support,
            max_variables=max_variables_per_family,
        )

        model = None
        status = "learned"
        reason = None
        if bn_table.shape[1] < 2:
            status = "skipped"
            reason = "fewer than 2 variables after filtering"
            filter_info["reason"] = reason
        elif bn_table.shape[0] < 10:
            status = "skipped"
            reason = f"too few rows for BN learning: {bn_table.shape[0]}"
            filter_info["reason"] = reason
        else:
            model = learn_predicate_bn(bn_table)
            if model is None:
                status = "skipped"
                reason = "learn_predicate_bn returned None"
                filter_info["reason"] = reason

        save_single_family_bn_outputs(output_dir, family, raw_table, bn_table, filter_info, model)
        family_results[family] = {
            "family": family,
            "status": status,
            "reason": reason,
            "num_patterns": int(raw_table.shape[0]),
            "num_raw_predicates": int(raw_table.shape[1] - (1 if "pattern_id" in raw_table.columns else 0)),
            "num_bn_predicates": int(bn_table.shape[1]),
            "filter_info": filter_info,
        }

        if status == "learned":
            print(f"[Family-BN] family={family} raw={family_results[family]['num_raw_predicates']} bn={bn_table.shape[1]} status=learned")
        else:
            print(f"[Family-BN] family={family} raw={family_results[family]['num_raw_predicates']} bn={bn_table.shape[1]} status=skipped")

    summary_path = Path(output_dir) / "family_bns" / "family_bn_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(family_results, f, indent=2, ensure_ascii=False)

    return family_results


def visualize_predicate_bn(
    model: Optional[BayesianModel],
    output_dir: str | Path,
    filename: str = "global_predicate_bn_structure.png",
    title: str = "Learned Global Predicate-BN Structure",
) -> None:
    """
    Visualize the learned global Predicate-BN.

    The figure is intentionally similar to the existing BN visualization in
    bayesian_based_prune.py so outputs across scripts stay consistent.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / filename

    if model is None:
        plt.figure(figsize=(8, 4))
        plt.text(
            0.5,
            0.5,
            "No Bayesian network was learned",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(png_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[Saved] {png_path}")
        return

    graph = nx.DiGraph()
    graph.add_nodes_from(model.nodes())
    graph.add_edges_from(model.edges())

    def node_color(node_name: str) -> str:
        text = str(node_name)
        if text.startswith("NODE:"):
            return "#4c78a8"
        if text.startswith("PAIR:"):
            return "#f58518"
        if text.startswith("EDGE:"):
            return "#54a24b"
        if text.startswith("NEG_EDGE:"):
            return "#e45756"
        return "#9d9da1"

    num_nodes = max(1, graph.number_of_nodes())
    width = max(12, min(28, num_nodes * 0.22))
    height = max(8, min(20, num_nodes * 0.16))

    plt.figure(figsize=(width, height))
    pos = nx.spring_layout(graph, seed=42, k=max(0.7, min(2.0, 10.0 / np.sqrt(num_nodes))))
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color=[node_color(node) for node in graph.nodes()],
        node_size=2400,
        edgecolors="black",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=8,
        font_color="black",
    )
    nx.draw_networkx_edges(
        graph,
        pos,
        edge_color="#7f7f7f",
        arrows=True,
        arrowsize=18,
        width=1.8,
        connectionstyle="arc3,rad=0.05",
    )
    plt.title(title, fontsize=13)
    plt.axis("off")

    legend_items = [
        ("NODE:*", "#4c78a8"),
        ("PAIR:*", "#f58518"),
        ("EDGE:*", "#54a24b"),
        ("NEG_EDGE:*", "#e45756"),
    ]
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=label,
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=10,
        )
        for label, color in legend_items
    ]
    plt.legend(handles=handles, loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {png_path}")


def save_outputs(
    output_dir: str | Path,
    repository: PredicateRepository,
    full_table: pd.DataFrame,
    count_table: Optional[pd.DataFrame] = None,
    binned_table: Optional[pd.DataFrame] = None,
    bn_table: Optional[pd.DataFrame] = None,
    filter_info: Optional[dict[str, Any]] = None,
    model: Optional[BayesianModel] = None,
) -> None:
    """
    Save repository, predicate tables, support summaries, and global BN outputs.

    The saved artifacts intentionally keep the older file names as much as
    possible so downstream scripts can continue to run with minimal changes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predicates_payload = [asdict(p) for p in repository.predicates]
    with open(output_dir / "global_predicate_repository.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_predicates": len(repository.predicates),
                "by_type": repository.by_type,
                "by_family": repository.by_family,
                "predicates": predicates_payload,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Keep backward compatibility: full == binary table in the first version.
    full_table.to_csv(output_dir / "global_predicate_table_full.csv", index=False)
    full_table.to_csv(output_dir / "global_predicate_table_binary.csv", index=False)

    if count_table is not None:
        count_table.to_csv(output_dir / "global_predicate_table_count.csv", index=False)
    if binned_table is not None:
        binned_table.to_csv(output_dir / "global_predicate_table_binned.csv", index=False)

    if "pattern_id" in full_table.columns:
        binary_support = full_table.drop(columns=["pattern_id"]).sum().sort_values(ascending=False)
    else:
        binary_support = full_table.sum().sort_values(ascending=False)

    binary_support.reset_index().rename(columns={"index": "predicate", 0: "support"}).to_csv(
        output_dir / "global_predicate_binary_support.csv",
        index=False,
    )
    # Backward-compatible old name.
    binary_support.reset_index().rename(columns={"index": "predicate", 0: "support"}).to_csv(
        output_dir / "global_predicate_support.csv",
        index=False,
    )

    if count_table is not None:
        if "pattern_id" in count_table.columns:
            count_support = count_table.drop(columns=["pattern_id"]).sum().sort_values(ascending=False)
        else:
            count_support = count_table.sum().sort_values(ascending=False)
        count_support.reset_index().rename(columns={"index": "predicate", 0: "support"}).to_csv(
            output_dir / "global_predicate_count_support.csv",
            index=False,
        )

    if bn_table is None:
        bn_table = pd.DataFrame()
    if filter_info is None:
        filter_info = {}

    bn_table.to_csv(output_dir / "global_predicate_table_bn.csv", index=False)

    result = {
        "num_full_predicates": len(repository.predicates),
        "num_bn_predicates": int(bn_table.shape[1]),
        "num_patterns": int(full_table.shape[0]),
        "filter_info": filter_info,
        "bn_edges": [],
        "cpds": [],
    }

    if model is not None:
        edges = sorted([[str(u), str(v)] for u, v in model.edges()], key=lambda x: (x[0], x[1]))
        result["bn_edges"] = edges
        result["cpds"] = [cpd_to_json(cpd) for cpd in model.get_cpds()]

        pd.DataFrame(edges, columns=["source", "target"]).to_csv(
            output_dir / "global_predicate_bn_edges.csv",
            index=False,
        )
    else:
        pd.DataFrame(columns=["source", "target"]).to_csv(
            output_dir / "global_predicate_bn_edges.csv",
            index=False,
        )

    with open(output_dir / "global_predicate_bn_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    visualize_predicate_bn(model, output_dir)
    print(f"[Save] outputs written to {output_dir}")


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


def run_global_predicate_repository_pipeline(
    selected_path: str | Path,
    output_dir: str | Path = OUTPUT_DIR,
    ppi_csv: Optional[str | Path] = None,
    include_neg_edges: bool = False,
    big_graph_cache: str | Path = DEFAULT_BIG_GRAPH_CACHE,
    interaction_lookup_cache: str | Path = DEFAULT_INTERACTION_LOOKUP_CACHE,
    edge_csv: Optional[str | Path] = DEFAULT_EDGE_CSV,
    node_csv: Optional[str | Path] = DEFAULT_NODE_CSV,
    learn_global_bn: bool = LEARN_GLOBAL_PREDICATE_BN,
    max_variables_per_family: int = MAX_VARIABLES_PER_FAMILY,
    min_family_size: int = MIN_FAMILY_SIZE,
    predicate_config_json: Optional[str | Path] = PREDICATE_CONFIG_JSON,
    ml_predicates_csv: Optional[str | Path] = ML_PREDICATES_CSV,
):
    """
    Main pipeline.

    Global graph loading now follows the existing cache flow:
    - first load processed/ppi/ppi_big_graph.pkl
    - if missing, build with pick_patterns.build_ppi_graph and save it

    If ppi_csv is provided, attach rich row-level records onto the cached graph.
    This now prefers the existing edge->rows lookup cache and only falls back
    to scanning the csv when the lookup cache is missing or stale.
    """
    print(f"[Info] Loading selected patterns: {selected_path}")
    patterns = load_selected_pattern_graphs(selected_path)
    print(f"[Info] Loaded patterns: {len(patterns)}")

    print(f"[Info] Loading global PPI graph from cache flow: {big_graph_cache}")
    global_graph = load_or_build_big_graph(
        edge_csv=edge_csv,
        node_csv=node_csv,
        cache_path=big_graph_cache,
    )

    if ppi_csv:
        print(f"[Info] Attaching rich edge records via lookup cache: {interaction_lookup_cache}")
        global_graph = attach_biogrid_edge_records(global_graph, ppi_csv, interaction_lookup_cache)
    else:
        print("[Info] No rich PPI CSV provided; edge-label predicates may be sparse.")

    # Modified: load config-generated predicate candidates once, instead of
    # recomputing thresholds inside this pipeline.
    predicate_config = {}
    if INCLUDE_CONFIG_PREDICATES and predicate_config_json:
        predicate_config = load_predicate_config(predicate_config_json)

    # Modified: load ML predicate pairs once and expose them as pattern-level
    # countable variables in the global predicate table.
    ml_predicate_index = {}
    if INCLUDE_ML_PREDICATES and ml_predicates_csv:
        ml_predicate_index = load_ml_predicate_index(ml_predicates_csv)

    print(f"[Info] Global graph: nodes={global_graph.number_of_nodes()}, edges={global_graph.number_of_edges()}")

    print("[Info] Building global predicate repository...")
    repository = build_global_predicate_repository(
        global_graph,
        min_support=MIN_PREDICATE_SUPPORT,
        include_neg_edges=include_neg_edges,
        predicate_config=predicate_config,
        ml_predicate_index=ml_predicate_index,
    )
    print(f"[Info] Global predicates: {len(repository.predicates)}")
    print(f"[Info] Predicate repository size: {len(repository.predicates)}")
    # Modified: print repository-level visibility for structural/config/ML predicate sources.
    repository_prefix_counts = summarize_predicate_prefix_counts([pred.pid for pred in repository.predicates])
    print("[Info] Repository predicate prefix counts:")
    for prefix, count in repository_prefix_counts.items():
        print(f"    {prefix:18s} {count}")

    print("[Info] Evaluating predicates on sampled patterns...")
    binary_table = build_global_predicate_table(
        patterns,
        repository,
        global_graph,
        mode="binary",
        predicate_config=predicate_config,
        ml_predicate_index=ml_predicate_index,
    )
    count_table = build_global_predicate_table(
        patterns,
        repository,
        global_graph,
        mode="count",
        predicate_config=predicate_config,
        ml_predicate_index=ml_predicate_index,
    )
    binned_table = build_global_predicate_table(
        patterns,
        repository,
        global_graph,
        mode="binned",
        predicate_config=predicate_config,
        ml_predicate_index=ml_predicate_index,
    )
    full_table = binary_table

    print(f"[Info] Binary predicate table: {binary_table.shape[0]} rows x {binary_table.shape[1] - 1} predicates")
    print(f"[Info] Count predicate table: {count_table.shape[0]} rows x {count_table.shape[1] - 1} predicates")
    print(f"[Info] Binned predicate table: {binned_table.shape[0]} rows x {binned_table.shape[1] - 1} predicates")
    # Modified: print which predicate-source prefixes really survived into each exported table.
    binary_prefix_counts = summarize_predicate_prefix_counts([c for c in binary_table.columns if c != "pattern_id"])
    count_prefix_counts = summarize_predicate_prefix_counts([c for c in count_table.columns if c != "pattern_id"])
    binned_prefix_counts = summarize_predicate_prefix_counts([c for c in binned_table.columns if c != "pattern_id"])
    print("[Info] Binary table prefix counts:")
    for prefix, count in binary_prefix_counts.items():
        print(f"    {prefix:18s} {count}")
    print("[Info] Count table prefix counts:")
    for prefix, count in count_prefix_counts.items():
        print(f"    {prefix:18s} {count}")
    print("[Info] Binned table prefix counts:")
    for prefix, count in binned_prefix_counts.items():
        print(f"    {prefix:18s} {count}")

    model = None
    bn_table = pd.DataFrame()
    filter_info: dict[str, Any] = {}
    if learn_global_bn:
        print("[Info] Learning one global Predicate-BN from binned-count predicate table...")
        bn_table, filter_info = filter_predicate_table_for_bn(
            binned_table,
            min_support=MIN_PREDICATE_SUPPORT,
            max_variables=MAX_PREDICATES_FOR_BN,
        )
        print(f"[Info] Global Predicate-BN table: {bn_table.shape[0]} rows x {bn_table.shape[1]} predicates")
        print(f"[Info] Global Predicate-BN predicate count: {bn_table.shape[1]}")
        # Modified: print which prefixes survived BN filtering/top-k truncation.
        bn_prefix_counts = summarize_predicate_prefix_counts(list(bn_table.columns))
        print("[Info] BN table prefix counts:")
        for prefix, count in bn_prefix_counts.items():
            print(f"    {prefix:18s} {count}")
        model = learn_predicate_bn(bn_table)
        if model is not None:
            print(
                f"[Info] Learned global Predicate-BN: "
                f"nodes={model.number_of_nodes()}, edges={model.number_of_edges()}"
            )
        else:
            print("[Warn] Global Predicate-BN was skipped or not learned.")
    else:
        print("[Warn] Global Predicate-BN learning is disabled.")

    save_outputs(
        output_dir=output_dir,
        repository=repository,
        full_table=full_table,
        count_table=count_table,
        binned_table=binned_table,
        bn_table=bn_table,
        filter_info=filter_info,
        model=model,
    )

    print("[Done] Global predicate repository pipeline finished.")
    return repository, binary_table, count_table, binned_table, model


def main():
    """
    File-level entry point.

    This script is configured through module constants near the top of the
    file, not through command-line arguments. main() simply validates the key
    path and runs the end-to-end pipeline once.
    """
    if SELECTED_PATH is None:
        raise ValueError("SELECTED_PATH is not set. Please update the file-level config.")

    run_global_predicate_repository_pipeline(
        selected_path=SELECTED_PATH,
        output_dir=OUTPUT_PATH,
        ppi_csv=PPI_CSV,
        include_neg_edges=INCLUDE_NEG_EDGES,
        big_graph_cache=BIG_GRAPH_CACHE,
        interaction_lookup_cache=INTERACTION_LOOKUP_CACHE,
        edge_csv=EDGE_CSV,
        node_csv=NODE_CSV,
        learn_global_bn=LEARN_GLOBAL_PREDICATE_BN,
        max_variables_per_family=MAX_VARIABLES_PER_FAMILY,
        min_family_size=MIN_FAMILY_SIZE,
        predicate_config_json=PREDICATE_CONFIG_JSON,
        ml_predicates_csv=ML_PREDICATES_CSV,
    )


if __name__ == "__main__":
    main()
