import os
import random
import sys
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data, InMemoryDataset
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
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

RAW_NUM_SUBGRAPHS = 500
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
TARGET_NUM = 200

DEFAULT_EDGE_CSV = os.path.join(CURRENT_DIR, "data", "protein_protein.csv")
DEFAULT_NODE_CSV = os.path.join(CURRENT_DIR, "data", "node.csv")
PROCESSED_DIR = os.path.join(CURRENT_DIR, "processed", "ppi")

ENCODER_FILENAME = "ppi_order_encoder.pt"
EMB_FILENAME = "ppi_train_embeddings.pt"
RAW_FILENAME = "ppi_raw.pt"
SELECTED_FILENAME = "ppi_selected.pt"


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
    ):
        self.edge_csv = edge_csv
        self.node_csv = node_csv
        self.num_subgraphs = num_subgraphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.k_hop = k_hop
        super().__init__(root)

        load_path = self.processed_paths[0]
        print(f"[Load] path={load_path}")
        self.data, self.slices = torch.load(load_path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [RAW_FILENAME]

    def download(self):
        return

    def process(self):
        graph = build_ppi_graph(self.edge_csv, self.node_csv)
        data_list = sample_k_hop_subgraphs(
            graph=graph,
            num_subgraphs=self.num_subgraphs,
            min_nodes=self.min_nodes,
            max_nodes=self.max_nodes,
            k_hop=self.k_hop,
        )
        if not data_list:
            raise RuntimeError("No sampled subgraphs were generated from the PPI graph.")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"[Done] Saved RAW sampled pool: {len(data_list)} -> {self.processed_paths[0]}")


def build_ppi_graph(edge_csv: str, node_csv: str = None) -> nx.Graph:
    if not os.path.exists(edge_csv):
        raise FileNotFoundError(f"Edge CSV not found: {edge_csv}")

    df_edges = pd.read_csv(edge_csv)
    df_edges.columns = [str(c).strip().lower() for c in df_edges.columns]

    if "index_a" in df_edges.columns and "index_b" in df_edges.columns:
        src_col, dst_col = "index_a", "index_b"
    elif "x_index" in df_edges.columns and "y_index" in df_edges.columns:
        src_col, dst_col = "x_index", "y_index"
    else:
        src_col, dst_col = "src", "dst"
    rel_col = "rel" if "rel" in df_edges.columns else None

    if src_col not in df_edges.columns or dst_col not in df_edges.columns:
        raise ValueError(
            f"Expected src/dst, x_index/y_index, or index_A/index_B columns in {edge_csv}, got {list(df_edges.columns)}"
        )

    if rel_col is not None:
        rel_series = df_edges[rel_col].astype(str).str.strip().str.lower()
        df_edges = df_edges[rel_series.isin(["protein_protein", "protein-protein"])].copy()

    graph = nx.Graph()
    valid_protein_nodes = None
    if node_csv and os.path.exists(node_csv):
        df_nodes = pd.read_csv(node_csv)
        df_nodes.columns = [str(c).strip().lower() for c in df_nodes.columns]
        if "node_type" in df_nodes.columns:
            df_nodes = df_nodes[df_nodes["node_type"].astype(str).str.lower() == "protein"].copy()
        if "node_index" in df_nodes.columns:
            node_id_col = "node_index"
        elif "node_id" in df_nodes.columns:
            node_id_col = "node_id"
        else:
            node_id_col = df_nodes.columns[0]
        valid_protein_nodes = set(int(node_id) for node_id in df_nodes[node_id_col].tolist())
        for node_id in valid_protein_nodes:
            graph.add_node(node_id)
        print(f"[Graph] Protein-only node filter enabled: {len(valid_protein_nodes)} protein nodes")

    for _, row in df_edges.iterrows():
        src = int(row[src_col])
        dst = int(row[dst_col])
        if src == dst:
            continue
        if valid_protein_nodes is not None:
            if src not in valid_protein_nodes or dst not in valid_protein_nodes:
                continue
        graph.add_edge(src, dst)

    print(
        f"[Graph] Built PPI graph from protein-protein edges: "
        f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges"
    )
    return graph


def graph_to_pyg(subgraph: nx.Graph, center_node: int) -> Data:
    node_list = list(subgraph.nodes())
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_list)}

    degrees = np.array([subgraph.degree(node_id) for node_id in node_list], dtype=np.float32)
    clustering = np.array(
        [nx.clustering(subgraph, node_id) for node_id in node_list], dtype=np.float32
    )
    center_flag = np.array(
        [1.0 if node_id == center_node else 0.0 for node_id in node_list], dtype=np.float32
    )

    deg_max = float(degrees.max()) if len(degrees) else 1.0
    if deg_max <= 0:
        deg_max = 1.0

    x = torch.tensor(
        np.stack([degrees / deg_max, clustering, center_flag], axis=1),
        dtype=torch.float32,
    )

    directed_edges = []
    for src, dst in subgraph.edges():
        directed_edges.append((node_to_idx[src], node_to_idx[dst]))
        directed_edges.append((node_to_idx[dst], node_to_idx[src]))

    edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    num_nodes = len(node_list)
    data = Data(
        x=x,
        edge_index=edge_index,
        num_nodes=num_nodes,
        n_nodes=torch.tensor([num_nodes], dtype=torch.long),
        y=torch.zeros(1, 0).float(),
    )
    data.center_id = torch.tensor([node_to_idx[center_node]], dtype=torch.long)
    data.orig_node_ids = torch.tensor(node_list, dtype=torch.long)
    data.center_orig_id = torch.tensor([center_node], dtype=torch.long)
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
            for node in nx.bfs_tree(ego, source=center).nodes():
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
        subgraphs.append(graph_to_pyg(ego, center_node=center))
        pbar.update(1)

    pbar.close()
    print(f"[Sample] Collected {len(subgraphs)} unique subgraphs after {attempts} attempts")
    return subgraphs


def load_data_list_from_inmemory(dataset):
    return [dataset.get(i) for i in range(len(dataset))]


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


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    encoder_path = os.path.join(PROCESSED_DIR, ENCODER_FILENAME)
    emb_path = os.path.join(PROCESSED_DIR, EMB_FILENAME)
    selected_path = os.path.join(PROCESSED_DIR, SELECTED_FILENAME)

    raw_dataset = PickPatternPPIDataset(
        root=PROCESSED_DIR,
        edge_csv=DEFAULT_EDGE_CSV,
        node_csv=DEFAULT_NODE_CSV,
        num_subgraphs=RAW_NUM_SUBGRAPHS,
        min_nodes=RAW_MIN_NODES,
        max_nodes=RAW_MAX_NODES,
        k_hop=K_HOP,
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

    selected_idx = select_graphs(
        embs=embs_np,
        method=SELECTOR,
        k=TARGET_NUM,
        seed=seed,
        sigma=PICK_SIGMA,
        chi=PICK_CHI,
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
            "edge_csv": DEFAULT_EDGE_CSV,
            "k_hop": K_HOP,
        },
        emb_path,
    )
    print(f"[Info] Saved embeddings to: {emb_path}")

    selected_data, selected_slices = raw_dataset.collate(selected_graphs)
    torch.save((selected_data, selected_slices), selected_path)
    print(f"[Done] Saved selected dataset -> {selected_path}")


if __name__ == "__main__":
    main()

