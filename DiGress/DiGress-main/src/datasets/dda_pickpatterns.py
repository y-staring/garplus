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
RETRAIN_ENCODER = True
RECOMPUTE_EMBEDDINGS = True
RUN_SANITY_CHECK = True

RAW_NUM_SUBGRAPHS = 4000
RAW_MIN_NODES = 5
RAW_MAX_NODES = 10

PICK_K = 1000
PICK_SIGMA = 500
PICK_CHI = 0.1

HIDDEN_DIM = 128
EMB_DIM = 64
NUM_LAYERS = 3
BATCH_SIZE = 32
LR = 1e-3
EPOCHS = 15
MARGIN = 1.0

SELECTOR = "fps"
TARGET_NUM = 2000


ENCODER_FILENAME = "dda_order_encoder.pt"
EMB_FILENAME = "dda_train_embeddings.pt"

RAW_FILENAME = "dda_train_raw.pt"
TRAIN_FILENAME = "dda_train.pt"
VAL_FILENAME = "dda_val.pt"
TEST_FILENAME = "dda_test.pt"

# 保证能从项目根目录导入
CURRENT_FILE = os.path.realpath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.datasets.dda_dataset_order_embedding import (
    DDAGraphDataset,
    train_order_encoder,
    compute_graph_embeddings,
    pick_patterns,
    select_graphs,
    sanity_check_order,
    GraphOrderEncoder,   # 需要能直接构造模型
)


def load_data_list_from_inmemory(dataset):
    return [dataset.get(i) for i in range(len(dataset))]

def split_graphs(graph_list, seed=42):
    num_graphs = len(graph_list)

    test_len = int(round(num_graphs * 0.2))
    train_len = int(round((num_graphs - test_len) * 0.8))
    val_len = num_graphs - train_len - test_len

    rng = np.random.default_rng(seed)
    indices = np.arange(num_graphs)
    rng.shuffle(indices)

    train_idx = indices[:train_len]
    val_idx = indices[train_len:train_len + val_len]
    test_idx = indices[train_len + val_len:]

    train_graphs = [graph_list[i] for i in train_idx]
    val_graphs = [graph_list[i] for i in val_idx]
    test_graphs = [graph_list[i] for i in test_idx]

    print(f"[Split] Total: {num_graphs}")
    print(f"[Split] Train: {len(train_graphs)}")
    print(f"[Split] Val:   {len(val_graphs)}")
    print(f"[Split] Test:  {len(test_graphs)}")

    return train_graphs, val_graphs, test_graphs


def main():
    datadir = os.path.join(PROJECT_ROOT, "data", "DDA")
    processed_dir = os.path.join(datadir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    encoder_path = os.path.join(processed_dir, ENCODER_FILENAME)
    emb_path = os.path.join(processed_dir, EMB_FILENAME)

    raw_path = os.path.join(processed_dir, RAW_FILENAME)
    train_path = os.path.join(processed_dir, TRAIN_FILENAME)
    val_path = os.path.join(processed_dir, VAL_FILENAME)
    test_path = os.path.join(processed_dir, TEST_FILENAME)

    # ==========================================================
    # 1) 读取 raw sampled subgraphs
    # ==========================================================
    raw_dataset = DDAGraphDataset(
        root=datadir,
        split="train",
        stage="raw",
        num_subgraphs=RAW_NUM_SUBGRAPHS,
        min_nodes=RAW_MIN_NODES,
        max_nodes=RAW_MAX_NODES,
    )

    data_list = load_data_list_from_inmemory(raw_dataset)
    print(f"[Info] Loaded raw sampled subgraphs: {len(data_list)}")
    print(f"[Info] Raw pool file: {raw_path}")

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
    # selected_idx = pick_patterns(
    #     embs=embs_np,
    #     k=PICK_K,
    #     sigma=PICK_SIGMA,
    #     chi=PICK_CHI,
    # )

    # selected_idx = pick_patterns(
    #     embs=embs_np,
    #     k=PICK_K,
    #     sigma=PICK_SIGMA,
    #     chi=PICK_CHI,
    #     remove_dominated= False,
    #     # dominated_keep_ratio=0.3
    # )

    selected_idx = select_graphs(
        embs=embs_np,
        method=SELECTOR,
        k=TARGET_NUM,
        seed=seed,
        sigma=PICK_SIGMA,
        chi=0.7,
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
        },
        emb_path,
    )
    print(f"[Info] Saved embeddings to: {emb_path}")

    train_graphs, val_graphs, test_graphs = split_graphs(selected_graphs, seed=seed)

    train_data, train_slices = raw_dataset.collate(train_graphs)
    val_data, val_slices = raw_dataset.collate(val_graphs)
    test_data, test_slices = raw_dataset.collate(test_graphs)

    torch.save((train_data, train_slices), train_path)
    torch.save((val_data, val_slices), val_path)
    torch.save((test_data, test_slices), test_path)

    print(f"[Done] Saved train dataset -> {train_path}")
    print(f"[Done] Saved val dataset   -> {val_path}")
    print(f"[Done] Saved test dataset  -> {test_path}")


if __name__ == "__main__":
    main()