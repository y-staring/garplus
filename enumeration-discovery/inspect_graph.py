import argparse
import os

import torch
from torch_geometric.data import InMemoryDataset


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SELECTED_PATH = os.path.join(CURRENT_DIR, "processed", "ppi", "ppi_selected.pt")


class SelectedPPIDataset(InMemoryDataset):
    def __init__(self, data_path):
        self.data_path = data_path
        super().__init__(root=os.path.dirname(data_path))
        self.data, self.slices = torch.load(data_path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [os.path.basename(self.data_path)]

    def download(self):
        return

    def process(self):
        return


def to_original_edge_list(data):
    orig_ids = data.orig_node_ids.tolist()
    edge_index = data.edge_index

    undirected_edges = []
    seen = set()
    for eid in range(edge_index.size(1)):
        src = int(edge_index[0, eid])
        dst = int(edge_index[1, eid])
        orig_src = int(orig_ids[src])
        orig_dst = int(orig_ids[dst])
        key = tuple(sorted((orig_src, orig_dst)))
        if orig_src == orig_dst or key in seen:
            continue
        seen.add(key)
        undirected_edges.append((orig_src, orig_dst))

    undirected_edges.sort()
    return undirected_edges


def summarize_graph(data):
    num_nodes = int(data.num_nodes)
    num_edges_directed = int(data.edge_index.size(1))
    num_edges_undirected = num_edges_directed // 2
    avg_degree = (2.0 * num_edges_undirected / num_nodes) if num_nodes > 0 else 0.0
    density = (
        (2.0 * num_edges_undirected) / (num_nodes * (num_nodes - 1))
        if num_nodes > 1
        else 0.0
    )
    return {
        "num_nodes": num_nodes,
        "num_edges_directed": num_edges_directed,
        "num_edges_undirected": num_edges_undirected,
        "avg_degree": avg_degree,
        "density": density,
    }


def summarize_dataset(dataset):
    node_counts = []
    edge_counts_directed = []
    edge_counts_undirected = []
    densities = []

    for idx in range(len(dataset)):
        data = dataset.get(idx)
        stats = summarize_graph(data)
        node_counts.append(stats["num_nodes"])
        edge_counts_directed.append(stats["num_edges_directed"])
        edge_counts_undirected.append(stats["num_edges_undirected"])
        densities.append(stats["density"])

    def _summary(values):
        values = list(values)
        return {
            "min": min(values) if values else 0,
            "max": max(values) if values else 0,
            "mean": (sum(values) / len(values)) if values else 0.0,
        }

    return {
        "dataset_size": len(dataset),
        "num_nodes": _summary(node_counts),
        "num_edges_directed": _summary(edge_counts_directed),
        "num_edges_undirected": _summary(edge_counts_undirected),
        "density": _summary(densities),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect one selected PPI subgraph, including graph size and original index_A/index_B mapping."
    )
    parser.add_argument("--data", default=DEFAULT_SELECTED_PATH, help="Path to ppi_selected.pt")
    parser.add_argument("--graph-index", type=int, default=0, help="Selected subgraph index")
    parser.add_argument(
        "--show-edges",
        action="store_true",
        help="Print original edge list as index_A,index_B pairs",
    )
    parser.add_argument(
        "--dataset-summary",
        action="store_true",
        help="Print summary statistics for the whole selected dataset",
    )
    args = parser.parse_args()

    dataset = SelectedPPIDataset(args.data)
    if len(dataset) == 0:
        raise RuntimeError(f"No graphs found in {args.data}")

    if args.dataset_summary:
        summary = summarize_dataset(dataset)
        print(f"dataset_size: {summary['dataset_size']}")
        print(
            "num_nodes: "
            f"min={summary['num_nodes']['min']}, "
            f"max={summary['num_nodes']['max']}, "
            f"mean={summary['num_nodes']['mean']:.4f}"
        )
        print(
            "num_edges_directed: "
            f"min={summary['num_edges_directed']['min']}, "
            f"max={summary['num_edges_directed']['max']}, "
            f"mean={summary['num_edges_directed']['mean']:.4f}"
        )
        print(
            "num_edges_undirected: "
            f"min={summary['num_edges_undirected']['min']}, "
            f"max={summary['num_edges_undirected']['max']}, "
            f"mean={summary['num_edges_undirected']['mean']:.4f}"
        )
        print(
            "density: "
            f"min={summary['density']['min']:.6f}, "
            f"max={summary['density']['max']:.6f}, "
            f"mean={summary['density']['mean']:.6f}"
        )
        return

    if args.graph_index < 0 or args.graph_index >= len(dataset):
        raise IndexError(f"graph-index out of range: {args.graph_index}, dataset size={len(dataset)}")

    data = dataset.get(args.graph_index)
    stats = summarize_graph(data)
    center_local = int(data.center_id.item()) if hasattr(data, "center_id") else None
    center_orig = int(data.center_orig_id.item()) if hasattr(data, "center_orig_id") else None
    orig_ids = data.orig_node_ids.tolist() if hasattr(data, "orig_node_ids") else []

    print(f"graph_index: {args.graph_index}")
    print(f"dataset_size: {len(dataset)}")
    print(f"num_nodes: {stats['num_nodes']}")
    print(f"num_edges_directed: {stats['num_edges_directed']}")
    print(f"num_edges_undirected: {stats['num_edges_undirected']}")
    print(f"avg_degree: {stats['avg_degree']:.4f}")
    print(f"density: {stats['density']:.6f}")
    print(f"center_local_id: {center_local}")
    print(f"center_orig_id: {center_orig}")
    print(f"orig_node_ids: {orig_ids}")

    if args.show_edges:
        edges = to_original_edge_list(data)
        print("original_edges(index_A,index_B):")
        for src, dst in edges:
            print(f"{src},{dst}")


if __name__ == "__main__":
    main()
