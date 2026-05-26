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


CURRENT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = CURRENT_DIR / "processed" / "ppi" / "global_predicate_repo"
DEFAULT_BIG_GRAPH_CACHE = CURRENT_DIR / "processed" / "ppi" / "ppi_big_graph.pkl"
# Rich interaction table used only to re-attach row-level edge attributes
# onto the cached big-graph topology.
DEFAULT_RICH_PPI_CSV = "/home/yyyy/codework/GARplus/GNN/code/DDA_test/data/去病图数据/protein_protein.csv"


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
OUTPUT_PATH = str(OUTPUT_DIR)
INCLUDE_NEG_EDGES = False
LEARN_GLOBAL_PREDICATE_BN = False

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
    if pd.isna(x):
        return "missing"
    s = str(x).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9_:\\-\\.]+", "_", s)
    s = s.strip("_")
    return s if s else "missing"


def is_missing(x: Any) -> bool:
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


def compare_values(a: Any, op: str, b: Any) -> bool:
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
    arr = np.asarray([v for v in values if not is_missing(v)], dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    if np.all(arr == arr[0]):
        return float(arr[0]), float(arr[0])
    q1 = float(np.quantile(arr, 1 / 3))
    q2 = float(np.quantile(arr, 2 / 3))
    return q1, q2


def assign_three_bin(value: float, q1: float, q2: float, low_name="low", mid_name="mid", high_name="high") -> str:
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
    left, right = sorted((int(u), int(v)))
    return left, right


def infer_edge_label_family(label: str) -> str:
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


def attach_biogrid_edge_records(graph: nx.Graph, csv_path: str | Path) -> nx.Graph:
    """
    Enrich the cached topology graph with row-level records from a BioGRID/PPI csv.

    This keeps the existing cached graph as the primary loading path, while
    restoring rich edge attributes needed by predicate construction.
    """
    df = pd.read_csv(csv_path)
    if "index_A" not in df.columns or "index_B" not in df.columns:
        raise ValueError("CSV must contain index_A and index_B columns.")

    # Clear old records/labels so reruns stay deterministic.
    for _, _, attrs in graph.edges(data=True):
        attrs["records"] = []
        attrs.pop("edge_labels", None)

    for _, row in df.iterrows():
        u = int(row["index_A"])
        v = int(row["index_B"])
        if u == v:
            continue

        if not graph.has_node(u):
            graph.add_node(u)
        if not graph.has_node(v):
            graph.add_node(v)
        if not graph.has_edge(u, v):
            graph.add_edge(u, v)

        if "Organism Name Interactor A" in df.columns:
            org_a = row.get("Organism Name Interactor A")
            if not is_missing(org_a):
                graph.nodes[u]["organism_name"] = normalize_token(org_a)

        if "Organism Name Interactor B" in df.columns:
            org_b = row.get("Organism Name Interactor B")
            if not is_missing(org_b):
                graph.nodes[v]["organism_name"] = normalize_token(org_b)

        record = row.to_dict()
        graph[u][v].setdefault("records", []).append(record)

    return graph


def data_to_nx_graph(data) -> nx.Graph:
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
    if SelectedPPIDataset is None:
        raise ImportError("Cannot import SelectedPPIDataset from inspect_graph.")

    dataset = SelectedPPIDataset(str(selected_path))
    patterns = []
    for idx in range(len(dataset)):
        data = dataset.get(idx)
        patterns.append((idx, data_to_nx_graph(data)))
    return patterns


def build_union_graph(patterns: list[tuple[int, nx.Graph]]) -> nx.Graph:
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
    if pred.pid not in seen:
        preds.append(pred)
        seen.add(pred.pid)


def build_global_predicate_repository(
    graph: nx.Graph,
    min_support: int = MIN_PREDICATE_SUPPORT,
    include_neg_edges: bool = False,
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
    if node not in global_graph:
        return {}
    return global_graph.nodes[node]


def get_global_edge_attrs(global_graph: nx.Graph, u: Any, v: Any) -> dict[str, Any]:
    if global_graph.has_edge(u, v):
        return global_graph[u][v]
    return {}


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


def build_global_predicate_table(
    patterns: list[tuple[int, nx.Graph]],
    repository: PredicateRepository,
    global_graph: nx.Graph,
) -> pd.DataFrame:
    rows = []
    pattern_ids = []

    for pattern_id, pattern_graph in patterns:
        row = {}
        for pred in repository.predicates:
            row[pred.pid] = evaluate_predicate_on_pattern(pred, pattern_graph, global_graph)
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

    support = df.sum()
    low_support_cols = support[support < min_support].index.tolist()
    df = df.drop(columns=low_support_cols)
    info["low_support_columns"] = low_support_cols

    if df.empty:
        return df, info

    support = df.sum().sort_values(ascending=False)
    if max_variables is not None and df.shape[1] > max_variables:
        selected = support.head(max_variables).index.tolist()
        df = df[selected]

    info["selected_columns"] = list(df.columns)
    return df.astype(int), info


def build_family_predicate_tables(
    full_table: pd.DataFrame,
    repository: PredicateRepository,
    min_family_size: int = MIN_FAMILY_SIZE,
) -> dict[str, pd.DataFrame]:
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
    bn_table: Optional[pd.DataFrame] = None,
    filter_info: Optional[dict[str, Any]] = None,
    model: Optional[BayesianModel] = None,
) -> None:
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

    full_table.to_csv(output_dir / "global_predicate_table_full.csv", index=False)

    if "pattern_id" in full_table.columns:
        support = full_table.drop(columns=["pattern_id"]).sum().sort_values(ascending=False)
    else:
        support = full_table.sum().sort_values(ascending=False)

    support.reset_index().rename(columns={"index": "predicate", 0: "support"}).to_csv(
        output_dir / "global_predicate_support.csv",
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
    edge_csv: Optional[str | Path] = DEFAULT_EDGE_CSV,
    node_csv: Optional[str | Path] = DEFAULT_NODE_CSV,
    learn_global_bn: bool = LEARN_GLOBAL_PREDICATE_BN,
    max_variables_per_family: int = MAX_VARIABLES_PER_FAMILY,
    min_family_size: int = MIN_FAMILY_SIZE,
):
    """
    Main pipeline.

    Global graph loading now follows the existing cache flow:
    - first load processed/ppi/ppi_big_graph.pkl
    - if missing, build with pick_patterns.build_ppi_graph and save it

    If ppi_csv is provided, attach rich row-level records onto the cached graph
    so edge-label predicates can be constructed from the original interaction
    table.
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
        print(f"[Info] Attaching rich edge records from CSV: {ppi_csv}")
        global_graph = attach_biogrid_edge_records(global_graph, ppi_csv)
    else:
        print("[Info] No rich PPI CSV provided; edge-label predicates may be sparse.")

    print(f"[Info] Global graph: nodes={global_graph.number_of_nodes()}, edges={global_graph.number_of_edges()}")

    print("[Info] Building global predicate repository...")
    repository = build_global_predicate_repository(
        global_graph,
        min_support=MIN_PREDICATE_SUPPORT,
        include_neg_edges=include_neg_edges,
    )
    print(f"[Info] Global predicates: {len(repository.predicates)}")

    print("[Info] Evaluating predicates on sampled patterns...")
    full_table = build_global_predicate_table(patterns, repository, global_graph)
    print(f"[Info] Full predicate table: {full_table.shape[0]} rows x {full_table.shape[1] - 1} predicates")

    save_outputs(
        output_dir=output_dir,
        repository=repository,
        full_table=full_table,
    )

    print("[Info] Learning family-wise Predicate-BNs...")
    family_results = learn_family_predicate_bns(
        full_table=full_table,
        repository=repository,
        output_dir=output_dir,
        min_support=MIN_PREDICATE_SUPPORT,
        max_variables_per_family=max_variables_per_family,
        min_family_size=min_family_size,
    )

    if learn_global_bn:
        print("[Info] Filtering predicate table for global BN learning...")
        bn_table, filter_info = filter_predicate_table_for_bn(full_table)
        print(f"[Info] Global BN predicate table: {bn_table.shape[0]} rows x {bn_table.shape[1]} predicates")

        print("[Info] Learning global Predicate-BN...")
        model = learn_predicate_bn(bn_table)
        save_outputs(
            output_dir=output_dir,
            repository=repository,
            full_table=full_table,
            bn_table=bn_table,
            filter_info=filter_info,
            model=model,
        )

    print("[Done] Global predicate repository pipeline finished.")
    return repository, full_table, family_results


def main():
    if SELECTED_PATH is None:
        raise ValueError("SELECTED_PATH is not set. Please update the file-level config.")

    run_global_predicate_repository_pipeline(
        selected_path=SELECTED_PATH,
        output_dir=OUTPUT_PATH,
        ppi_csv=PPI_CSV,
        include_neg_edges=INCLUDE_NEG_EDGES,
        big_graph_cache=BIG_GRAPH_CACHE,
        edge_csv=EDGE_CSV,
        node_csv=NODE_CSV,
        learn_global_bn=LEARN_GLOBAL_PREDICATE_BN,
        max_variables_per_family=MAX_VARIABLES_PER_FAMILY,
        min_family_size=MIN_FAMILY_SIZE,
    )


if __name__ == "__main__":
    main()
