import pandas as pd
import numpy as np
import os

# 1. 设置路径与载入数据
file_name = "edges_ratio10x.csv"
input_csv = f"./benchmarks/data_signed/ratio_datasets/{file_name}"
output_dir = "./benchmarks/data_signed/ratio_datasets/ratio10x/" # 为改比例单独建一个文件夹

os.makedirs(output_dir, exist_ok=True)
edges = pd.read_csv(input_csv)

# 假设你的CSV包含 src, dst, rel 等列，如果没有rel列，则需要你指定一个虚拟的关系
if 'rel' not in edges.columns:
    edges['rel'] = 'default_relation'

# 2. 提取并映射实体 (Entity)
# 将 src 和 dst 的所有独特节点合并
unique_entities = pd.concat([edges['src'], edges['dst']]).unique()
entity2id = {ent: idx for idx, ent in enumerate(unique_entities)}

with open(os.path.join(output_dir, "entity2id.txt"), "w") as f:
    f.write(f"{len(entity2id)}\n")
    for ent, idx in entity2id.items():
        f.write(f"{ent}\t{idx}\n")

# 3. 提取并映射关系 (Relation)
unique_relations = edges['rel'].unique()
relation2id = {rel: idx for idx, rel in enumerate(unique_relations)}

with open(os.path.join(output_dir, "relation2id.txt"), "w") as f:
    f.write(f"{len(relation2id)}\n")
    for rel, idx in relation2id.items():
        f.write(f"{rel}\t{idx}\n")

# 4. 切分数据集并保存 (80% Train, 10% Valid, 10% Test)
edges = edges.sample(frac=1, random_state=42).reset_index(drop=True)
n_total = len(edges)
n_train = int(n_total * 0.8)
n_valid = int(n_total * 0.1)

train_edges = edges.iloc[:n_train]
valid_edges = edges.iloc[n_train:n_train+n_valid]
test_edges = edges.iloc[n_train+n_valid:]

def save_triples(dataset, filename):
    with open(os.path.join(output_dir, filename), "w") as f:
        f.write(f"{len(dataset)}\n")
        for _, row in dataset.iterrows():
            h_id = entity2id[row['src']]
            t_id = entity2id[row['dst']]
            r_id = relation2id[row['rel']]
            f.write(f"{h_id} {t_id} {r_id}\n")

save_triples(train_edges, "train2id.txt")
save_triples(valid_edges, "valid2id.txt")
save_triples(test_edges, "test2id.txt")

print(f"{file_name} 转换完成！保存在 {output_dir}")