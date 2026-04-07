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


def main():
    parser = argparse.ArgumentParser(
        description="Inspect one selected PPI subgraph and map local nodes back to original index_A/index_B ids."
    )
    parser.add_argument("--data", default=DEFAULT_SELECTED_PATH, help="Path to ppi_selected.pt")
    parser.add_argument("--graph-index", type=int, default=0, help="Selected subgraph index")
    args = parser.parse_args()

    dataset = SelectedPPIDataset(args.data)
    if len(dataset) == 0:
        raise RuntimeError(f"No graphs found in {args.data}")
    if args.graph_index < 0 or args.graph_index >= len(dataset):
        raise IndexError(f"graph-index out of range: {args.graph_index}, dataset size={len(dataset)}")

    data = dataset.get(args.graph_index)
    center_local = int(data.center_id.item()) if hasattr(data, "center_id") else None
    center_orig = int(data.center_orig_id.item()) if hasattr(data, "center_orig_id") else None
    orig_ids = data.orig_node_ids.tolist() if hasattr(data, "orig_node_ids") else []
    edges = to_original_edge_list(data)

    print(f"graph_index: {args.graph_index}")
    print(f"num_nodes: {data.num_nodes}")
    print(f"num_edges_undirected: {len(edges)}")
    print(f"center_local_id: {center_local}")
    print(f"center_orig_id: {center_orig}")
    print(f"orig_node_ids: {orig_ids}")
    print("original_edges(index_A,index_B):")
    for src, dst in edges:
        print(f"{src},{dst}")


if __name__ == "__main__":
    main()
