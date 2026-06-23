import os
import argparse
import random
import sys
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data, InMemoryDataset
from tqdm import tqdm
from collections import Counter
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DIGRESS_ROOT = os.path.join(PROJECT_ROOT, "DiGress", "DiGress-main")
if DIGRESS_ROOT not in sys.path:
    sys.path.insert(0, DIGRESS_ROOT)

from sampling_utils import (
    GraphOrderEncoder,
    compute_graph_embeddings,
    order_energy,
    select_graphs,
    train_order_encoder,
)

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

RETRAIN_ENCODER = True
RECOMPUTE_EMBEDDINGS = True
RUN_SANITY_CHECK = True

RAW_NUM_SUBGRAPHS = 5000
K_HOP = 2
RAW_MIN_NODES = 5
RAW_MAX_NODES = 10

PICK_K = 1000
PICK_SIGMA = 500
PICK_CHI = 0.7

HIDDEN_DIM = 128
EMB_DIM = 64
NUM_LAYERS = 3
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 5
MARGIN = 1.0

SELECTOR = "fps"
# SELECTOR = "pickpatterns"
TARGET_NUM = 2000

# Raw subgraph construction mode.
# - "vertex": original behavior; randomly sample k-hop neighborhoods around vertices.
# - "signed_negative_aware": sample a mixed raw pool where part of subgraphs are
#   centered on negative signed edges, then still use order embeddings + SELECTOR
#   to pick representative subgraphs. This is the recommended mode for GAR+ ML
#   refinement / negative-rule mining.
SAMPLING_MODE = "signed_negative_aware"
NEGATIVE_CENTER_RATIO = 0.5
# The raw-pool ratio alone is not enough: the embedding selector is label-blind
# and can discard most negative-centred graphs.  Reserve this share *after*
# selection as well, so ppi_selected.pt / dda_selected.pt / ti_selected.pt
# contain negative edges without the later isolated-edge augmentation.
SELECTED_NEGATIVE_CENTER_RATIO = 0.5
# PyG reuses the existing *_raw.pt by default.  Set this to True only when you
# want the script to delete that cache and regenerate the raw pool.
FORCE_RESAMPLE_RAW_POOL = False
INCLUDE_POSITIVE_EDGE_CENTERS = True
INTERACTION_LABEL_COL = "interaction_label"
NEGATIVE_LABEL = "negative"
POSITIVE_LABEL = "positive"

SIGNED_DATA_DIR = r"/home/yyyy/codework/GARplus/enumeration-discovery/去病图数据"
DEFAULT_RELATION = os.environ.get("GARPLUS_SAMPLING_RELATION", "ppi").strip().lower()

RELATION_CONFIGS = {
    "ppi": {
        "name": "PPI",
        "directed": False,
        "edge_csv": os.path.join(SIGNED_DATA_DIR, "protein_protein_signed.csv"),
        "node_csv": os.path.join(SIGNED_DATA_DIR, "protein.csv"),
        "node_csvs": [
            {"path": os.path.join(SIGNED_DATA_DIR, "protein.csv"), "node_type": "Protein", "id_offset": 0},
        ],
        "processed_dir": os.path.join(BASE_DIR, "processed", "ppi"),
        "src_dst_candidates": [
            ("index_a", "index_b"),
            ("entrez gene interactor a", "entrez gene interactor b"),
            ("x_index", "y_index"),
            ("src", "dst"),
        ],
    },
    "dda": {
        "name": "DDA",
        "directed": True,
        "edge_csv": os.path.join(SIGNED_DATA_DIR, "drug_disease_signed.csv"),
        "node_csv": None,
        "node_csvs": [
            {"path": os.path.join(SIGNED_DATA_DIR, "drug.csv"), "node_type": "Drug", "id_offset": 0},
            {"path": os.path.join(SIGNED_DATA_DIR, "disease.csv"), "node_type": "Disease", "id_offset": 1000000000},
        ],
        "processed_dir": os.path.join(BASE_DIR, "processed", "dda"),
        "src_dst_candidates": [
            ("chemical_index", "disease_index"),
            ("src", "dst"),
        ],
        "dst_node_offset": 1000000000,
    },
    "ti": {
        "name": "TI",
        "directed": True,
        "edge_csv": os.path.join(SIGNED_DATA_DIR, "gene_disease_signed.csv"),
        "node_csv": None,
        "node_csvs": [
            {"path": os.path.join(SIGNED_DATA_DIR, "gene.csv"), "node_type": "Gene", "id_offset": 0},
            {"path": os.path.join(SIGNED_DATA_DIR, "disease.csv"), "node_type": "Disease", "id_offset": 1000000000},
        ],
        "processed_dir": os.path.join(BASE_DIR, "processed", "ti"),
        "src_dst_candidates": [
            ("gene_index", "disease_index"),
            ("src", "dst"),
        ],
        "dst_node_offset": 1000000000,
    },
}

if DEFAULT_RELATION not in RELATION_CONFIGS:
    raise ValueError(f"Unsupported GARPLUS_SAMPLING_RELATION={DEFAULT_RELATION!r}")

DEFAULT_EDGE_CSV = RELATION_CONFIGS[DEFAULT_RELATION]["edge_csv"]
DEFAULT_NODE_CSV = RELATION_CONFIGS[DEFAULT_RELATION]["node_csv"]
PROCESSED_DIR = RELATION_CONFIGS[DEFAULT_RELATION]["processed_dir"]

ENCODER_FILENAME = f"{DEFAULT_RELATION}_order_encoder.pt"
EMB_FILENAME = f"{DEFAULT_RELATION}_train_embeddings.pt"
RAW_FILENAME = f"{DEFAULT_RELATION}_raw.pt"
SELECTED_FILENAME = f"{DEFAULT_RELATION}_selected_negative_centered.pt"


class PickPatternPPIDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        edge_csv,
        node_csv=None,
        num_subgraphs=2000,
        min_nodes=5,
        max_nodes=10,
        k_hop=2,
        relation_config=None,
        raw_filename=RAW_FILENAME,
    ):
        self.edge_csv = edge_csv
        self.node_csv = node_csv
        self.num_subgraphs = num_subgraphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.k_hop = k_hop
        self.relation_config = relation_config or RELATION_CONFIGS[DEFAULT_RELATION]
        self.raw_filename = raw_filename
        super().__init__(root)

        load_path = self.processed_paths[0]
        print(f"[Load] path={load_path}")
        self.data, self.slices = torch.load(load_path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.raw_filename]

    def download(self):
        return

    def process(self):
        graph = build_signed_graph(self.edge_csv, self.node_csv, self.relation_config)
        if SAMPLING_MODE == "signed_negative_aware":
            data_list = sample_signed_k_hop_subgraphs(
                graph=graph,
                num_subgraphs=self.num_subgraphs,
                min_nodes=self.min_nodes,
                max_nodes=self.max_nodes,
                k_hop=self.k_hop,
                negative_center_ratio=NEGATIVE_CENTER_RATIO,
                include_positive_edge_centers=INCLUDE_POSITIVE_EDGE_CENTERS,
            )
        elif SAMPLING_MODE == "vertex":
            data_list = sample_k_hop_subgraphs(
                graph=graph,
                num_subgraphs=self.num_subgraphs,
                min_nodes=self.min_nodes,
                max_nodes=self.max_nodes,
                k_hop=self.k_hop,
            )
        else:
            raise ValueError(f"Unsupported SAMPLING_MODE: {SAMPLING_MODE}")
        if not data_list:
            raise RuntimeError("No sampled subgraphs were generated from the signed graph.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"[Done] Saved RAW sampled pool: {len(data_list)} -> {self.processed_paths[0]}")


def _find_src_dst_columns(df_edges: pd.DataFrame, candidates) -> tuple:
    for src_col, dst_col in candidates:
        if src_col in df_edges.columns and dst_col in df_edges.columns:
            return src_col, dst_col
    raise ValueError(
        f"Expected one of src/dst column pairs {candidates}, got {list(df_edges.columns)}"
    )


def _merge_signed_label(old_label: str, new_label: str) -> str:
    priority = {NEGATIVE_LABEL: 3, POSITIVE_LABEL: 2, "neutral": 1, "unknown": 0}
    old_label = str(old_label or "unknown").strip().lower()
    new_label = str(new_label or "unknown").strip().lower()
    return new_label if priority.get(new_label, 0) > priority.get(old_label, 0) else old_label


def build_signed_graph(edge_csv: str, node_csv: str = None, relation_config=None) -> nx.Graph:
    if not os.path.exists(edge_csv):
        raise FileNotFoundError(f"Edge CSV not found: {edge_csv}")

    relation_config = relation_config or RELATION_CONFIGS[DEFAULT_RELATION]
    relation_name = relation_config.get("name", "signed")
    dst_node_offset = int(relation_config.get("dst_node_offset", 0) or 0)
    df_edges = pd.read_csv(edge_csv)
    df_edges.columns = [str(c).strip().lower() for c in df_edges.columns]

    src_col, dst_col = _find_src_dst_columns(
        df_edges,
        relation_config.get("src_dst_candidates", [("src", "dst")]),
    )
    rel_col = "rel" if "rel" in df_edges.columns else None
    label_col = INTERACTION_LABEL_COL if INTERACTION_LABEL_COL in df_edges.columns else None
    edge_label_col = "edgelabel" if "edgelabel" in df_edges.columns else None
    experimental_system_col = "experimental system" if "experimental system" in df_edges.columns else None

    if rel_col is not None and relation_name.upper() == "PPI":
        rel_series = df_edges[rel_col].astype(str).str.strip().str.lower()
        df_edges = df_edges[rel_series.isin(["protein_protein", "protein-protein"])].copy()

    graph = nx.DiGraph() if relation_config.get("directed", False) else nx.Graph()
    valid_nodes = None
    node_sources = []
    if node_csv:
        node_sources = [{"path": node_csv, "node_type": relation_name, "id_offset": 0}]
    else:
        node_sources = list(relation_config.get("node_csvs", []))

    if node_sources:
        valid_nodes = set()
        for source in node_sources:
            source_path = source.get("path")
            if not source_path or not os.path.exists(source_path):
                continue
            df_nodes = pd.read_csv(source_path)
            df_nodes.columns = [str(c).strip().lower() for c in df_nodes.columns]
            if "node_type" in df_nodes.columns and relation_name.upper() == "PPI":
                df_nodes = df_nodes[df_nodes["node_type"].astype(str).str.lower() == "protein"].copy()
            if "node_index" in df_nodes.columns:
                node_id_col = "node_index"
            elif "node_id" in df_nodes.columns:
                node_id_col = "node_id"
            elif "index" in df_nodes.columns:
                node_id_col = "index"
            else:
                node_id_col = df_nodes.columns[0]
            id_offset = int(source.get("id_offset", 0) or 0)
            node_type = source.get("node_type", relation_name)
            count = 0
            for raw_node_id in df_nodes[node_id_col].dropna().tolist():
                node_id = int(raw_node_id) + id_offset
                graph.add_node(node_id, node_type=node_type, source_node_id=int(raw_node_id))
                valid_nodes.add(node_id)
                count += 1
            print(f"[Graph] Node file loaded: type={node_type} path={source_path} nodes={count} offset={id_offset}")

    for _, row in df_edges.iterrows():
        if pd.isna(row[src_col]) or pd.isna(row[dst_col]):
            continue
        src = int(row[src_col])
        dst = int(row[dst_col]) + dst_node_offset
        if src == dst:
            continue
        if valid_nodes is not None:
            if src not in valid_nodes or dst not in valid_nodes:
                continue
        interaction_label = str(row[label_col]).strip().lower() if label_col is not None and pd.notna(row[label_col]) else "unknown"
        edge_type = str(row[edge_label_col]).strip() if edge_label_col is not None and pd.notna(row[edge_label_col]) else relation_name
        experimental_system = str(row[experimental_system_col]).strip() if experimental_system_col is not None and pd.notna(row[experimental_system_col]) else edge_type
        if graph.has_edge(src, dst):
            old_label = graph[src][dst].get("interaction_label", "unknown")
            graph[src][dst]["interaction_label"] = _merge_signed_label(old_label, interaction_label)
            graph[src][dst]["edge_multiplicity"] = int(graph[src][dst].get("edge_multiplicity", 1)) + 1
        else:
            graph.add_edge(
                src,
                dst,
                interaction_label=interaction_label,
                experimental_system=experimental_system,
                edge_type=edge_type,
                edge_multiplicity=1,
            )

    label_counts = Counter(
        str(attrs.get("interaction_label", "unknown")).strip().lower()
        for _, _, attrs in graph.edges(data=True)
    )
    print(
        f"[Graph] Built {relation_name} graph from signed edges: "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges, "
        f"labels={dict(label_counts)}"
    )
    return graph

def graph_to_pyg(
    subgraph: nx.Graph,
    center_node: int = None,
    center_nodes=None,
    sampling_center_label: str = "unknown",
    directed: bool = False,
) -> Data:
    node_list = list(subgraph.nodes())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}
    if center_nodes is None:
        center_nodes = [center_node] if center_node is not None else []
    center_node_set = set(center_nodes)

    degrees = np.array([subgraph.degree(node_id) for node_id in node_list], dtype=np.float32)
    clustering = np.array(
        [nx.clustering(subgraph, node_id) for node_id in node_list], dtype=np.float32
    )
    center_flag = np.array(
        [1.0 if node_id in center_node_set else 0.0 for node_id in node_list], dtype=np.float32
    )

    deg_max = float(degrees.max()) if len(degrees) else 1.0
    if deg_max <= 0:
        deg_max = 1.0

    x = torch.tensor(
        np.stack([degrees / deg_max, clustering, center_flag], axis=1),
        dtype=torch.float32,
    )

    directed_edges = []
    edge_labels = []
    for src, dst, attrs in subgraph.edges(data=True):
        label = str(attrs.get("interaction_label", "unknown")).strip().lower()
        directed_edges.append((node_to_idx[src], node_to_idx[dst]))
        edge_labels.append(label)
        if not directed:
            directed_edges.append((node_to_idx[dst], node_to_idx[src]))
            edge_labels.append(label)

    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    num_nodes = len(node_list)
    data = Data(
        x=x,
        edge_index=edge_index,
        num_nodes=num_nodes,
        n_nodes=torch.tensor([num_nodes], dtype=torch.long),
        y=torch.zeros(1, 0).float(),
    )
    if center_nodes:
        center_indices = [node_to_idx[node_id] for node_id in center_nodes if node_id in node_to_idx]
    else:
        center_indices = []
    data.center_id = torch.tensor(center_indices[:2] or [0], dtype=torch.long)
    data.orig_node_ids = torch.tensor(node_list, dtype=torch.long)
    data.center_orig_id = torch.tensor(center_nodes[:2] if center_nodes else [node_list[0]], dtype=torch.long)
    label_to_id = {NEGATIVE_LABEL: 1, POSITIVE_LABEL: 2, "neutral": 3, "unknown": 0}
    data.edge_label = torch.tensor([label_to_id.get(label, 0) for label in edge_labels], dtype=torch.long)
    # Persist the label of the edge used to construct this subgraph.  This is
    # deliberately separate from edge_label: it survives batching and lets the
    # final diversity selector enforce a class quota.
    data.sampling_center_label = torch.tensor(
        [label_to_id.get(sampling_center_label, 0)], dtype=torch.long
    )
    return data


def sample_k_hop_subgraphs(
    graph: nx.Graph,
    num_subgraphs: int,
    min_nodes: int,
    max_nodes: int,
    k_hop: int,
) -> List[Data]:
    nodes = list(graph.nodes())
    if not nodes:
        return []

    subgraphs = []
    seen_signatures = set()
    attempts = 0
    max_attempts = num_subgraphs * 20

    print(
        f"[Sample] Extracting up to {num_subgraphs} subgraphs with "
        f"{k_hop}-hop neighborhoods and size in [{min_nodes}, {max_nodes}]"
    )
    pbar = tqdm(total=num_subgraphs, desc="Sampling subgraphs", unit="subgraph")

    while len(subgraphs) < num_subgraphs and attempts < max_attempts:
        attempts += 1
        if attempts % 1000 == 0:
            pbar.set_postfix(
                attempts=attempts,
                unique=len(subgraphs),
                coverage=f"{len(subgraphs)}/{num_subgraphs}",
            )
        center = random.choice(nodes)
        ego = nx.ego_graph(graph, center, radius=k_hop, undirected=True)

        if ego.number_of_nodes() < min_nodes:
            continue

        if ego.number_of_nodes() > max_nodes:
            bfs_nodes = []
            traversal_graph = ego.to_undirected() if ego.is_directed() else ego
            for node in nx.bfs_tree(traversal_graph, source=center).nodes():
                bfs_nodes.append(node)
                if len(bfs_nodes) >= max_nodes:
                    break
            ego = ego.subgraph(bfs_nodes).copy()

        if ego.number_of_nodes() < min_nodes or ego.number_of_edges() == 0:
            continue

        signature = tuple(sorted(ego.nodes()))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        subgraphs.append(graph_to_pyg(ego, center_node=center, directed=graph.is_directed()))
        pbar.update(1)

    pbar.close()
    print(f"[Sample] Collected {len(subgraphs)} unique subgraphs after {attempts} attempts")
    return subgraphs




def _limited_edge_centered_subgraph(graph: nx.Graph, src: int, dst: int, k_hop: int, max_nodes: int) -> nx.Graph:
    """Construct a bounded k-hop context around a signed center edge."""

    src_ego = nx.ego_graph(graph, src, radius=k_hop, undirected=True)
    dst_ego = nx.ego_graph(graph, dst, radius=k_hop, undirected=True)
    node_candidates = []
    seen = set()
    traversal_graph = graph.to_undirected() if graph.is_directed() else graph
    for root in (src, dst):
        for node_id in nx.bfs_tree(traversal_graph, source=root).nodes():
            if node_id in src_ego or node_id in dst_ego:
                if node_id not in seen:
                    node_candidates.append(node_id)
                    seen.add(node_id)
            if len(node_candidates) >= max_nodes:
                break
        if len(node_candidates) >= max_nodes:
            break
    for endpoint in (src, dst):
        if endpoint not in seen:
            node_candidates.insert(0, endpoint)
            seen.add(endpoint)
    node_candidates = node_candidates[:max_nodes]
    if src not in node_candidates or dst not in node_candidates:
        node_candidates = [src, dst] + [node for node in node_candidates if node not in (src, dst)]
        node_candidates = node_candidates[:max_nodes]
    return graph.subgraph(node_candidates).copy()


def _signed_edges(graph: nx.Graph, label: str):
    """Return graph edges whose `interaction_label` matches label."""

    return [
        (src, dst)
        for src, dst, attrs in graph.edges(data=True)
        if str(attrs.get("interaction_label", "unknown")).strip().lower() == label
    ]


def sample_signed_k_hop_subgraphs(
    graph: nx.Graph,
    num_subgraphs: int,
    min_nodes: int,
    max_nodes: int,
    k_hop: int,
    negative_center_ratio: float = 0.5,
    include_positive_edge_centers: bool = True,
) -> List[Data]:
    """Sample a raw pool with explicit negative-edge-centered subgraphs.

    This keeps the later order-embedding selection unchanged, but ensures that
    negative links enter the candidate pool with their k-hop context instead of
    being appended later as isolated edges.
    """

    negative_edges = _signed_edges(graph, NEGATIVE_LABEL)
    positive_edges = _signed_edges(graph, POSITIVE_LABEL)
    nodes = list(graph.nodes())
    if not nodes:
        return []

    negative_target = min(len(negative_edges), int(num_subgraphs * negative_center_ratio))
    remaining_target = num_subgraphs - negative_target
    print(
        f"[SampleSigned] negative_edges={len(negative_edges)} positive_edges={len(positive_edges)} "
        f"negative_target={negative_target} remaining_target={remaining_target}"
    )

    subgraphs: List[Data] = []
    seen_signatures = set()

    random.shuffle(negative_edges)
    for src, dst in tqdm(negative_edges[:negative_target], desc="Sampling negative-centered", unit="subgraph"):
        subgraph = _limited_edge_centered_subgraph(graph, src, dst, k_hop=k_hop, max_nodes=max_nodes)
        if subgraph.number_of_nodes() < min_nodes or subgraph.number_of_edges() == 0:
            continue
        signature = ("neg", tuple(sorted(subgraph.nodes())), tuple(sorted((min(a, b), max(a, b)) for a, b in subgraph.edges())))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        subgraphs.append(
            graph_to_pyg(
                subgraph,
                center_nodes=[src, dst],
                sampling_center_label=NEGATIVE_LABEL,
                directed=graph.is_directed(),
            )
        )

    attempts = 0
    max_attempts = max(num_subgraphs * 20, 1000)
    pbar = tqdm(total=max(0, num_subgraphs - len(subgraphs)), desc="Sampling background", unit="subgraph")
    while len(subgraphs) < num_subgraphs and attempts < max_attempts:
        attempts += 1
        if include_positive_edge_centers and positive_edges and random.random() < 0.5:
            src, dst = random.choice(positive_edges)
            ego = _limited_edge_centered_subgraph(graph, src, dst, k_hop=k_hop, max_nodes=max_nodes)
            center_nodes = [src, dst]
        else:
            center = random.choice(nodes)
            ego = nx.ego_graph(graph, center, radius=k_hop, undirected=True)
            if ego.number_of_nodes() > max_nodes:
                bfs_nodes = []
                traversal_graph = ego.to_undirected() if ego.is_directed() else ego
                for node in nx.bfs_tree(traversal_graph, source=center).nodes():
                    bfs_nodes.append(node)
                    if len(bfs_nodes) >= max_nodes:
                        break
                ego = ego.subgraph(bfs_nodes).copy()
            center_nodes = [center]
        if ego.number_of_nodes() < min_nodes or ego.number_of_edges() == 0:
            continue
        signature = ("bg", tuple(sorted(ego.nodes())), tuple(sorted((min(a, b), max(a, b)) for a, b in ego.edges())))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        center_label = POSITIVE_LABEL if len(center_nodes) == 2 else "unknown"
        subgraphs.append(
            graph_to_pyg(
                ego,
                center_nodes=center_nodes,
                sampling_center_label=center_label,
                directed=graph.is_directed(),
            )
        )
        pbar.update(1)
    pbar.close()
    print(f"[SampleSigned] Collected {len(subgraphs)} signed-aware subgraphs")
    return subgraphs


def load_data_list_from_inmemory(dataset):
    return [dataset.get(i) for i in range(len(dataset))]


def _center_label_id(graph: Data) -> int:
    """Read a persisted centre-edge label; old raw caches default to unknown."""

    value = getattr(graph, "sampling_center_label", None)
    if value is None:
        return 0
    return int(value.reshape(-1)[0].item())


def select_graphs_with_negative_quota(
    embs: np.ndarray,
    graph_list: List[Data],
    target_num: int,
    negative_ratio: float,
) -> List[int]:
    """Run the diversity selector within label strata, then merge its results.

    A negative-centred graph is a graph whose construction edge was labelled
    negative.  Selecting it does not trim its context, therefore both endpoints
    retain their real k-hop neighbours instead of becoming degree-1 nodes.
    """

    target_num = min(target_num, len(graph_list))
    if target_num <= 0:
        return []

    negative_idx = [i for i, graph in enumerate(graph_list) if _center_label_id(graph) == 1]
    other_idx = [i for i, graph in enumerate(graph_list) if _center_label_id(graph) != 1]
    negative_target = min(len(negative_idx), round(target_num * negative_ratio))

    def select_from(indices: List[int], count: int) -> List[int]:
        if count <= 0 or not indices:
            return []
        count = min(count, len(indices))
        local_idx = select_graphs(
            embs=embs[indices],
            method=SELECTOR,
            k=count,
            seed=seed,
            sigma=PICK_SIGMA,
            chi=PICK_CHI,
        )
        return [indices[int(i)] for i in local_idx]

    selected = select_from(negative_idx, negative_target)
    selected.extend(select_from(other_idx, target_num - len(selected)))

    # If a stratum is smaller than its quota, use any still-unselected graphs.
    if len(selected) < target_num:
        selected_set = set(selected)
        remainder = [i for i in range(len(graph_list)) if i not in selected_set]
        selected.extend(select_from(remainder, target_num - len(selected)))

    negative_set = set(negative_idx)
    print(
        f"[SelectSigned] raw_negative_centered={len(negative_idx)} "
        f"target_negative_centered={negative_target} "
        f"selected_negative_centered={sum(i in negative_set for i in selected)}"
    )
    return selected


@torch.no_grad()
def sanity_check_order(model, graph_list, num_trials=10):
    if len(graph_list) < 2:
        print("[Sanity] Skip: not enough graphs.")
        return

    model.eval()
    device = next(model.parameters()).device
    total = 0
    ok = 0
    sample_count = min(num_trials, len(graph_list))

    for graph in random.sample(graph_list, sample_count):
        if graph.num_nodes <= max(3, RAW_MIN_NODES):
            continue

        keep_num = max(3, int(graph.num_nodes * 0.7))
        keep_nodes = sorted(random.sample(range(graph.num_nodes), keep_num))
        keep_set = set(keep_nodes)
        remap = {old: new for new, old in enumerate(keep_nodes)}

        edges = []
        for eid in range(graph.edge_index.size(1)):
            src = int(graph.edge_index[0, eid])
            dst = int(graph.edge_index[1, eid])
            if src in keep_set and dst in keep_set:
                edges.append([remap[src], remap[dst]])

        if not edges:
            continue

        small = Data(
            x=graph.x[keep_nodes],
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
            num_nodes=len(keep_nodes),
        )

        z_small = model(Batch.from_data_list([small]).to(device))
        z_large = model(Batch.from_data_list([graph]).to(device))
        e_small_large = float(order_energy(z_small, z_large).item())
        e_large_small = float(order_energy(z_large, z_small).item())
        print(
            f"[Sanity] E(small,large)={e_small_large:.4f}, "
            f"E(large,small)={e_large_small:.4f}"
        )
        total += 1
        if e_small_large <= e_large_small:
            ok += 1

    print(f"[Sanity] pass rate: {ok}/{max(total, 1)}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Signed-aware GAR+ subgraph sampling for PPI, DDA, and TI.")
    parser.add_argument("--relation", choices=sorted(RELATION_CONFIGS), default=DEFAULT_RELATION)
    parser.add_argument("--edge-csv", default=None)
    parser.add_argument("--node-csv", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--target-num", type=int, default=TARGET_NUM)
    parser.add_argument("--raw-num-subgraphs", type=int, default=RAW_NUM_SUBGRAPHS)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    relation = args.relation.strip().lower()
    relation_config = dict(RELATION_CONFIGS[relation])
    edge_csv = args.edge_csv or relation_config["edge_csv"]
    node_csv = args.node_csv if args.node_csv is not None else relation_config.get("node_csv")
    processed_dir = args.processed_dir or relation_config["processed_dir"]
    encoder_filename = f"{relation}_order_encoder.pt"
    emb_filename = f"{relation}_train_embeddings.pt"
    raw_filename = f"{relation}_raw.pt"
    selected_filename = f"{relation}_selected.pt"

    os.makedirs(processed_dir, exist_ok=True)

    print(f"[Config] relation={relation} edge_csv={edge_csv} node_csv={node_csv}")
    encoder_path = os.path.join(processed_dir, encoder_filename)
    emb_path = os.path.join(processed_dir, emb_filename)
    selected_path = os.path.join(processed_dir, selected_filename)
    raw_path = os.path.join(processed_dir, raw_filename)

    if not 0.0 <= SELECTED_NEGATIVE_CENTER_RATIO <= 1.0:
        raise ValueError("SELECTED_NEGATIVE_CENTER_RATIO must be within [0, 1]")
    if FORCE_RESAMPLE_RAW_POOL and os.path.exists(raw_path):
        os.remove(raw_path)
        print(f"[Cache] Removed raw pool so it will be rebuilt: {raw_path}")

    raw_dataset = PickPatternPPIDataset(
        root=processed_dir,
        edge_csv=edge_csv,
        node_csv=node_csv,
        num_subgraphs=args.raw_num_subgraphs,
        min_nodes=RAW_MIN_NODES,
        max_nodes=RAW_MAX_NODES,
        k_hop=K_HOP,
        relation_config=relation_config,
        raw_filename=raw_filename,
    )

    data_list = load_data_list_from_inmemory(raw_dataset)
    print(f"[Info] Loaded raw sampled subgraphs: {len(data_list)}")
    print(f"[Info] Raw pool file: {raw_dataset.processed_paths[0]}")

    if len(data_list) == 0:
        raise RuntimeError("No sampled graphs found in raw dataset.")

    in_dim = data_list[0].x.size(-1)

    if (not RETRAIN_ENCODER) and os.path.exists(encoder_path):
        print(f"[Info] Loading existing order encoder from: {encoder_path}")
        model = GraphOrderEncoder(
            in_dim=in_dim,
            hidden_dim=HIDDEN_DIM,
            emb_dim=EMB_DIM,
            num_layers=NUM_LAYERS,
        )
        state_dict = torch.load(encoder_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
    else:
        print("[Info] Training a new order encoder...")
        model = train_order_encoder(
            graph_list=data_list,
            in_dim=in_dim,
            hidden_dim=HIDDEN_DIM,
            emb_dim=EMB_DIM,
            num_layers=NUM_LAYERS,
            batch_size=BATCH_SIZE,
            lr=LR,
            epochs=EPOCHS,
            margin=MARGIN,
        )
        torch.save(model.state_dict(), encoder_path)
        print(f"[Info] Saved order encoder to: {encoder_path}")

    if RUN_SANITY_CHECK:
        sanity_check_order(model, data_list, num_trials=10)

    if (not RECOMPUTE_EMBEDDINGS) and os.path.exists(emb_path):
        print(f"[Info] Loading existing embeddings from: {emb_path}")
        emb_ckpt = torch.load(emb_path, map_location="cpu")
        embs = emb_ckpt["embeddings"]
        if isinstance(embs, np.ndarray):
            embs_np = embs
            embs = torch.from_numpy(embs_np)
        else:
            embs_np = embs.numpy()
    else:
        print("[Info] Computing graph embeddings...")
        embs = compute_graph_embeddings(model, data_list)
        embs_np = embs.numpy()

    selected_idx = select_graphs_with_negative_quota(
        embs=embs_np,
        graph_list=data_list,
        target_num=args.target_num,
        negative_ratio=SELECTED_NEGATIVE_CENTER_RATIO,
    )

    if len(selected_idx) == 0:
        print("[WARN] selector selected 0 graphs, fallback to all raw graphs.")
        selected_graphs = data_list
    else:
        selected_graphs = [data_list[i] for i in selected_idx]

    print(f"[Info] Selected {len(selected_graphs)} / {len(data_list)} graphs")

    torch.save(
        {
            "embeddings": embs,
            "selected_idx": selected_idx,
            "pick_k": PICK_K,
            "pick_sigma": PICK_SIGMA,
            "pick_chi": PICK_CHI,
            "raw_num_graphs": len(data_list),
            "selected_num_graphs": len(selected_graphs),
            "edge_csv": edge_csv,
            "relation": relation,
            "k_hop": K_HOP,
            "sampling_mode": SAMPLING_MODE,
            "negative_center_ratio": NEGATIVE_CENTER_RATIO,
            "selected_negative_center_ratio": SELECTED_NEGATIVE_CENTER_RATIO,
        },
        emb_path,
    )
    print(f"[Info] Saved embeddings to: {emb_path}")

    selected_data, selected_slices = raw_dataset.collate(selected_graphs)
    torch.save((selected_data, selected_slices), selected_path)
    print(f"[Done] Saved selected dataset -> {selected_path}")


if __name__ == "__main__":
    main()




