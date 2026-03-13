import os
import sys
from collections import Counter

import torch
import numpy as np
from tqdm import tqdm

# ===== 路径配置 =====
PROJECT_ROOT = "/home/yyyy/codework/GARplus/DiGress/DiGress-main"
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "PPI")
OUTPUT_TXT = "/home/yyyy/codework/GARplus/DiGress/outputs/2026-03-11/12-13-57-ppi_gar/ppi_train_ground_truth.txt"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.datasets.ppi_dataset_order_embedding import PPIGraphDataset, EDGE_BIT_MAP

# from src.datasets.ppi_dataset import PPIGraphDataset

def get_int_labels(tensor):
    """
    兼容:
      - one-hot float [N, C]
      - long labels [N]
      - [N, 1]
    """
    if tensor is None or tensor.numel() == 0:
        return None

    t = tensor.cpu()

    if t.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
        return t.numpy().reshape(-1)

    if t.dim() > 1 and t.shape[-1] > 1:
        return t.argmax(dim=-1).cpu().numpy().reshape(-1)

    return t.int().cpu().numpy().reshape(-1)


def tensor_to_dense_edge_matrix(edge_index, edge_attr, num_nodes):
    """
    输出 E 矩阵，元素是 edge_type:
      0 = no-edge
      1..128 = bitmask + 1
    """
    E = np.zeros((num_nodes, num_nodes), dtype=int)

    if edge_index is None or edge_index.numel() == 0:
        return E

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    if edge_attr is not None and edge_attr.numel() > 0:
        edge_types = get_int_labels(edge_attr)
    else:
        edge_types = np.ones(len(src), dtype=int)

    for i in range(len(src)):
        u = int(src[i])
        v = int(dst[i])
        et = int(edge_types[i])
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            E[u, v] = et

    return E


def edge_type_to_bitmask(edge_type):
    if edge_type == 0:
        return None
    return int(edge_type) - 1


def is_negative_edge_type(edge_type):
    if edge_type == 0:
        return False
    bitmask = edge_type_to_bitmask(edge_type)
    return bool(bitmask & (1 << EDGE_BIT_MAP["is_negative"]))


def export_dataset(split="train", stage="final", output_txt=OUTPUT_TXT):
    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    print(f"[Info] Loading dataset: root={DATA_ROOT}, split={split}, stage={stage}")
    dataset = PPIGraphDataset(root=DATA_ROOT, split=split, stage=stage)
    print(f"[Info] Loaded graphs: {len(dataset)}")

    stats = {
        "num_graphs": 0,
        "node_labels": Counter(),
        "edge_types": Counter(),
        "graph_num_nodes": Counter(),
        "graph_num_edges": Counter(),
        "negative_edges": 0,
        "total_edges": 0,
    }

    with open(output_txt, "w", encoding="utf-8") as f:
        for data in tqdm(dataset, total=len(dataset)):
            n = int(data.num_nodes)
            stats["num_graphs"] += 1
            stats["graph_num_nodes"][n] += 1

            # 节点标签: x 是 10 维 one-hot
            x_arr = get_int_labels(data.x)
            if x_arr is None or len(x_arr) != n:
                x_arr = np.zeros(n, dtype=int)

            for x in x_arr:
                stats["node_labels"][int(x)] += 1

            # 边矩阵: E 存 edge_type = bitmask + 1
            E = tensor_to_dense_edge_matrix(data.edge_index, data.edge_attr, n)

            undirected_edge_count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    et = int(E[i, j])
                    if et != 0:
                        undirected_edge_count += 1
                        stats["edge_types"][et] += 1
                        stats["total_edges"] += 1
                        if is_negative_edge_type(et):
                            stats["negative_edges"] += 1

            stats["graph_num_edges"][undirected_edge_count] += 1

            # 写 txt
            f.write(f"N={n}\n")
            f.write("X:\n")
            f.write(" ".join(map(str, x_arr.tolist())) + "\n")
            f.write("E:\n")
            for row in E:
                f.write(" ".join(map(str, row.tolist())) + "\n")
            f.write("\n")

    print("\n" + "=" * 70)
    print("Export Diagnosis Report")
    print("=" * 70)
    print(f"[Graphs] total = {stats['num_graphs']}")

    print("\n[Graph node count distribution]")
    for k, v in sorted(stats["graph_num_nodes"].items()):
        print(f"  N={k:<3} -> {v}")

    print("\n[Graph edge count distribution]")
    for k, v in sorted(stats["graph_num_edges"].items()):
        print(f"  E={k:<3} -> {v}")

    print("\n[Node label distribution]")
    total_nodes = sum(stats["node_labels"].values())
    for k, v in sorted(stats["node_labels"].items()):
        print(f"  label {k:<2} -> {v:<8} ({v / max(total_nodes,1):.2%})")

    print("\n[Top edge_type distribution]")
    for et, cnt in stats["edge_types"].most_common(20):
        print(f"  edge_type={et:<3} bitmask={et-1:<3} count={cnt}")

    neg_ratio = stats["negative_edges"] / max(stats["total_edges"], 1)
    print(f"\n[Negative edges] {stats['negative_edges']} / {stats['total_edges']} = {neg_ratio:.4%}")

    print("\n[Done]")
    print(f"Saved txt to: {output_txt}")


if __name__ == "__main__":
    export_dataset(split="train", stage="final", output_txt=OUTPUT_TXT)