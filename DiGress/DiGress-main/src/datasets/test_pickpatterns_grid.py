import os
import torch
import numpy as np

from ppi_dataset_order_embedding import (
    remove_dominated_embeddings,
    pick_patterns,
    embedding_volume,
)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

def main():
    emb_path = os.path.join(PROJECT_ROOT, "data", "PPI", "processed", "ppi_train_embeddings.pt")
    ckpt = torch.load(emb_path, map_location="cpu")
    embs = ckpt["embeddings"].numpy()

    print("=== Embedding Stats ===")
    print("shape:", embs.shape)
    print("min:", embs.min(), "max:", embs.max())
    print("mean:", embs.mean(), "std:", embs.std())

    # 1) dominated removal
    remain_idx = remove_dominated_embeddings(embs)
    remain_embs = embs[remain_idx]

    print("\n=== Dominated Removal ===")
    print("before:", len(embs))
    print("after :", len(remain_idx))
    print("removed:", len(embs) - len(remain_idx))
    print("non-dominated ratio:", len(remain_idx) / len(embs))

    # 2) volumes of non-dominated embeddings
    vols = np.array([embedding_volume(e) for e in remain_embs], dtype=np.float64)
    vols = np.sort(vols)[::-1]

    print("\n=== Non-dominated Volume Stats ===")
    print("count:", len(vols))
    if len(vols) > 0:
        print("max:", vols.max())
        print("min:", vols.min())
        print("median:", np.median(vols))
        print("top 10 volumes:", vols[:10])

    # 3) try a few configs
    print("\n=== Representative configs ===")
    configs = [
        (500, 10, 1.0),
        (1000, 5, 0.8),
        (1000, 3, 0.6),
        (2000, 2, 0.4),
        (3000, 1, 0.2),
    ]
    for k, sigma, chi in configs:
        selected_idx = pick_patterns(embs, k=k, sigma=sigma, chi=chi)
        print(f"k={k}, sigma={sigma}, chi={chi} -> selected={len(selected_idx)}")

if __name__ == "__main__":
    main()