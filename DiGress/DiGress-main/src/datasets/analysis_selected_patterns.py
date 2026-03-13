import os
import sys
import torch
from collections import Counter

CURRENT_FILE = os.path.realpath(__file__)
CURRENT_DIR = os.path.dirname(CURRENT_FILE)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from ppi_dataset_order_embedding import PPIGraphDataset, EDGE_BIT_MAP


def load_data_list_from_inmemory(dataset):
    return [dataset.get(i) for i in range(len(dataset))]


def has_negative_edge(data):
    if not hasattr(data, "edge_label_mask"):
        return False
    masks = data.edge_label_mask.cpu().tolist()
    bit = EDGE_BIT_MAP["is_negative"]
    return any((m & (1 << bit)) != 0 for m in masks)


def main():
    datadir = os.path.join(PROJECT_ROOT, "data", "PPI")

    final_dataset = PPIGraphDataset(
        root=datadir,
        split="train",
        stage="final",
        num_subgraphs=2000,
        min_nodes=4,
        max_nodes=8,
    )

    data_list = load_data_list_from_inmemory(final_dataset)
    print(f"Loaded final selected graphs: {len(data_list)}")

    node_counter = Counter()
    edge_counter = Counter()
    bitmask_counter = Counter()
    neg_graphs = 0

    for data in data_list:
        n = int(data.num_nodes)
        # 你的 edge_index 是双向存边，所以 /2 看成无向边数量
        e_undirected = int(data.edge_index.size(1) // 2)

        node_counter[n] += 1
        edge_counter[e_undirected] += 1

        if hasattr(data, "edge_label_mask"):
            masks = data.edge_label_mask.cpu().tolist()
            bitmask_counter.update(masks)

        if has_negative_edge(data):
            neg_graphs += 1

    print("\n=== Node Count Distribution ===")
    for k, v in sorted(node_counter.items()):
        print(f"{k}: {v}")

    print("\n=== Edge Count Distribution ===")
    for k, v in sorted(edge_counter.items()):
        print(f"{k}: {v}")

    print("\n=== Negative-edge Graphs ===")
    print(f"{neg_graphs}/{len(data_list)}")

    print("\n=== Top 20 Edge Bitmasks ===")
    for item in bitmask_counter.most_common(20):
        print(item)


if __name__ == "__main__":
    main()