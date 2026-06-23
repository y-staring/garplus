import torch

PT_PATH = "/home/yyyy/codework/GARplus/enumeration-discovery/processed/ti/ti_selected.pt"

data, slices = torch.load(PT_PATH, map_location="cpu")

# 1=negative，2=positive，3=neutral，0=unknown
labels = data.edge_label.reshape(-1)

negative_directed = int((labels == 1).sum())
positive_directed = int((labels == 2).sum())
neutral_directed = int((labels == 3).sum())
unknown_directed = int((labels == 0).sum())

print("图数量:", len(slices["x"]) - 1)
print("负边出现次数（有向）:", negative_directed)
print("负边出现次数（无向，约）:", negative_directed // 2)
print("正边出现次数（无向，约）:", positive_directed // 2)
print("中性边出现次数（无向，约）:", neutral_directed // 2)
print("未知边出现次数（无向，约）:", unknown_directed // 2)

total = len(labels)
print("负边比例:", f"{negative_directed / total:.2%}")

negative_edges = set()

num_graphs = len(slices["orig_node_ids"]) - 1
for graph_id in range(num_graphs):
    node_start = int(slices["orig_node_ids"][graph_id])
    node_end = int(slices["orig_node_ids"][graph_id + 1])
    edge_start = int(slices["edge_index"][graph_id])
    edge_end = int(slices["edge_index"][graph_id + 1])

    orig_ids = data.orig_node_ids[node_start:node_end].tolist()
    edge_index = data.edge_index[:, edge_start:edge_end]  # 不要减 node_start
    edge_labels = data.edge_label[edge_start:edge_end]

    for (src, dst), label in zip(edge_index.t().tolist(), edge_labels.tolist()):
        if label == 1:
            negative_edges.add(
                tuple(sorted((orig_ids[src], orig_ids[dst])))
            )

print("去重后的原始负边数:", len(negative_edges))

center_labels = data.sampling_center_label.reshape(-1)
print("负边中心子图数:", int((center_labels == 1).sum()))