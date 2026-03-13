import os
import sys
import random
import numpy as np
import torch

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

RAW_NUM_SUBGRAPHS = 4000
RAW_MIN_NODES = 4
RAW_MAX_NODES = 8

PICK_K = 1000
PICK_SIGMA = 10
PICK_CHI = 0.5

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

# 保证能从项目根目录导入
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
    GraphOrderEncoder,   # 需要能直接构造模型
)


def load_data_list_from_inmemory(dataset):
    return [dataset.get(i) for i in range(len(dataset))]


def main():
    datadir = os.path.join(PROJECT_ROOT, "data", "PPI")
    processed_dir = os.path.join(datadir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    encoder_path = os.path.join(processed_dir, ENCODER_FILENAME)
    emb_path = os.path.join(processed_dir, EMB_FILENAME)
    final_path = os.path.join(processed_dir, FINAL_FILENAME)

    # ==========================================================
    # 1) 读取 raw sampled subgraphs
    # ==========================================================
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

    # ==========================================================
    # 2) 训练 / 加载 order encoder
    # ==========================================================
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

    # ==========================================================
    # 3) Sanity check
    # ==========================================================
    if RUN_SANITY_CHECK:
        sanity_check_order(model, data_list, num_trials=10)

    # ==========================================================
    # 4) 计算 / 加载 embeddings
    # ==========================================================
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

    # ==========================================================
    # 5) PickPatterns
    # ==========================================================
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

    # ==========================================================
    # 6) 保存 embeddings
    # ==========================================================
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

    # ==========================================================
    # 7) 保存最终 diffusion 训练集
    # ==========================================================
    data, slices = raw_dataset.collate(selected_graphs)
    torch.save((data, slices), final_path)
    print(f"[Done] Saved final diffusion dataset to: {final_path}")


if __name__ == "__main__":
    main()