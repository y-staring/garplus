import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import Counter

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# =========================
# 开关配置
# =========================
RETRAIN_ENCODER = False
RECOMPUTE_EMBEDDINGS = False
RUN_SANITY_CHECK = True
SAVE_VISUALS = True

RAW_NUM_SUBGRAPHS = 4000
RAW_MIN_NODES = 4
RAW_MAX_NODES = 8

PICK_K = 1000
PICK_SIGMA = 10
PICK_CHI = 0.7

HIDDEN_DIM = 128
EMB_DIM = 64
NUM_LAYERS = 3
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 15
MARGIN = 1.0

ENCODER_FILENAME = "ppi_order_encoder.pt"
EMB_FILENAME = "ppi_train_embeddings.pt"
FINAL_FILENAME = "ppi_train.pt"

CURRENT_FILE = os.path.realpath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.datasets.ppi_dataset_order_embedding import (
    PPIGraphDataset,
    train_order_encoder,
    compute_graph_embeddings,
    pick_patterns,
    sanity_check_order,
    GraphOrderEncoder,
)


def load_data_list_from_inmemory(dataset):
    return [dataset.get(i) for i in range(len(dataset))]


def pca_2d(x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return x @ vt[:2].T


def get_graph_basic_stats(graph_list):
    node_counts = []
    edge_counts = []
    node_label_counter = Counter()

    for g in graph_list:
        n = int(g.num_nodes)
        node_counts.append(n)
        edge_counts.append(g.edge_index.size(1) // 2 if g.edge_index is not None else 0)

        if g.x is not None and g.x.numel() > 0:
            labels = g.x.argmax(dim=-1).cpu().numpy().tolist()
            node_label_counter.update(labels)

    return node_counts, edge_counts, node_label_counter


def save_visualizations(graph_list, embs_np, selected_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    selected_set = set(selected_idx)
    selected_mask = np.array([i in selected_set for i in range(len(graph_list))], dtype=bool)

    # 1) PCA of embeddings
    if len(embs_np) >= 2:
        z2 = pca_2d(embs_np)

        plt.figure(figsize=(7, 6))
        plt.scatter(
            z2[~selected_mask, 0],
            z2[~selected_mask, 1],
            s=12,
            alpha=0.35,
            label="Not selected"
        )
        if selected_mask.any():
            plt.scatter(
                z2[selected_mask, 0],
                z2[selected_mask, 1],
                s=22,
                alpha=0.9,
                label="Selected"
            )
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("PCA of Graph Embeddings")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "embeddings_pca.png"), dpi=180)
        plt.close()

    node_counts, edge_counts, node_label_counter = get_graph_basic_stats(graph_list)
    selected_graphs = [graph_list[i] for i in selected_idx]
    sel_node_counts, sel_edge_counts, sel_node_label_counter = get_graph_basic_stats(selected_graphs)

    # 2) Node count histogram
    plt.figure(figsize=(7, 5))
    bins = np.arange(min(node_counts), max(node_counts) + 2) - 0.5
    plt.hist(node_counts, bins=bins, alpha=0.5, label="All raw graphs")
    if len(sel_node_counts) > 0:
        plt.hist(sel_node_counts, bins=bins, alpha=0.7, label="Selected graphs")
    plt.xlabel("Number of nodes")
    plt.ylabel("Count")
    plt.title("Node Count Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "node_count_hist.png"), dpi=180)
    plt.close()

    # 3) Edge count histogram
    plt.figure(figsize=(7, 5))
    bins = np.arange(min(edge_counts), max(edge_counts) + 2) - 0.5
    plt.hist(edge_counts, bins=bins, alpha=0.5, label="All raw graphs")
    if len(sel_edge_counts) > 0:
        plt.hist(sel_edge_counts, bins=bins, alpha=0.7, label="Selected graphs")
    plt.xlabel("Number of undirected edges")
    plt.ylabel("Count")
    plt.title("Edge Count Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "edge_count_hist.png"), dpi=180)
    plt.close()

    # 4) Node label bar chart
    all_labels = sorted(set(node_label_counter.keys()) | set(sel_node_label_counter.keys()))
    all_counts = [node_label_counter.get(k, 0) for k in all_labels]
    sel_counts = [sel_node_label_counter.get(k, 0) for k in all_labels]

    x = np.arange(len(all_labels))
    width = 0.38

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, all_counts, width=width, label="All raw graphs")
    plt.bar(x + width / 2, sel_counts, width=width, label="Selected graphs")
    plt.xticks(x, [str(k) for k in all_labels])
    plt.xlabel("Node label")
    plt.ylabel("Count")
    plt.title("Node Label Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "node_label_bar.png"), dpi=180)
    plt.close()

    # 5) Summary txt
    summary_path = os.path.join(out_dir, "selection_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"num_raw_graphs: {len(graph_list)}\n")
        f.write(f"num_selected_graphs: {len(selected_idx)}\n")
        f.write(f"pick_k: {PICK_K}\n")
        f.write(f"pick_sigma: {PICK_SIGMA}\n")
        f.write(f"pick_chi: {PICK_CHI}\n")
        f.write(f"raw_node_count_mean: {np.mean(node_counts):.4f}\n")
        f.write(f"selected_node_count_mean: {np.mean(sel_node_counts) if len(sel_node_counts) else 0:.4f}\n")
        f.write(f"raw_edge_count_mean: {np.mean(edge_counts):.4f}\n")
        f.write(f"selected_edge_count_mean: {np.mean(sel_edge_counts) if len(sel_edge_counts) else 0:.4f}\n")


def main():
    datadir = os.path.join(PROJECT_ROOT, "data", "PPI")
    processed_dir = os.path.join(datadir, "processed")
    visuals_dir = os.path.join(processed_dir, "visuals")
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(visuals_dir, exist_ok=True)

    encoder_path = os.path.join(processed_dir, ENCODER_FILENAME)
    emb_path = os.path.join(processed_dir, EMB_FILENAME)
    final_path = os.path.join(processed_dir, FINAL_FILENAME)

    # 1) raw sampled subgraphs
    raw_dataset = PPIGraphDataset(
        root=datadir,
        split="train",
        stage="raw",
        num_subgraphs=RAW_NUM_SUBGRAPHS,
        min_nodes=RAW_MIN_NODES,
        max_nodes=RAW_MAX_NODES,
    )

    data_list = load_data_list_from_inmemory(raw_dataset)
    print(f"[Info] Loaded raw sampled subgraphs: {len(data_list)}")

    if len(data_list) == 0:
        raise RuntimeError("No sampled graphs found in raw dataset.")

    in_dim = data_list[0].x.size(-1)

    # 2) train or load encoder
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

    # 3) sanity check
    if RUN_SANITY_CHECK:
        sanity_check_order(model, data_list, num_trials=10)

    # 4) embeddings
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

    # 5) PickPatterns
    selected_idx = pick_patterns(
        embs=embs_np,
        k=PICK_K,
        sigma=PICK_SIGMA,
        chi=PICK_CHI,
    )

    if len(selected_idx) == 0:
        print("[WARN] PickPatterns selected 0 graphs, fallback to all raw graphs.")
        selected_graphs = data_list
    else:
        selected_graphs = [data_list[i] for i in selected_idx]

    print(f"[Info] PickPatterns selected {len(selected_graphs)} / {len(data_list)} graphs")

    # 6) save embeddings
    torch.save(
        {
            "embeddings": embs,
            "selected_idx": selected_idx,
            "pick_k": PICK_K,
            "pick_sigma": PICK_SIGMA,
            "pick_chi": PICK_CHI,
            "raw_num_graphs": len(data_list),
        },
        emb_path,
    )
    print(f"[Info] Saved embeddings to: {emb_path}")

    # 7) save final dataset
    data, slices = raw_dataset.collate(selected_graphs)
    torch.save((data, slices), final_path)
    print(f"[Done] Saved final diffusion dataset to: {final_path}")

    # 8) save visuals
    if SAVE_VISUALS:
        save_visualizations(
            graph_list=data_list,
            embs_np=embs_np,
            selected_idx=selected_idx,
            out_dir=visuals_dir,
        )
        print(f"[Info] Saved visualizations to: {visuals_dir}")


if __name__ == "__main__":
    main()