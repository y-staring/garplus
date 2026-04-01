
import pandas as pd
import numpy as np
import os

def sample_non_edges_fixed(num_nodes, forbidden_src, forbidden_dst, num_samples):
    """
    采样无边 (Label 0)。
    forbidden_src/dst: 必须包含所有真实存在的边 (Label 1 和 Label 2)，防止生成的 0 与 2 冲突。
    """
    exist = set()
    for s, d in zip(forbidden_src, forbidden_dst):
        if s > d: s, d = d, s # 统一存为 (min, max) 处理无向冲突
        exist.add((s, d))
    
    samples = set()
    while len(samples) < num_samples:
        s = np.random.randint(0, num_nodes)
        d = np.random.randint(0, num_nodes)
        
        if s == d: continue
        
        k = (s, d) if s < d else (d, s)
        if k not in exist:
            samples.add((s, d)) 
            
    return list(samples)

def generate_train_valid_sets():
    node_csv = "/mnt/e/OpenKE/benchmarks/data_updated/node.csv"
    edge_csv = "/mnt/e/OpenKE/benchmarks/data_updated/edge_old.csv"
    out_dir = "benchmarks/PPI/random_old/"
    
    os.makedirs(out_dir, exist_ok=True)
    print("正在准备数据...")
    
    # 1. 读取基础数据
    nodes = pd.read_csv(node_csv)
    num_nodes = len(nodes)
    edges = pd.read_csv(edge_csv)
    
    # 如果数据集里有 rel，可以打印一下方便排查
    # 分离不同 Label 的边（假设 CSV 中有 label 列，包含 1 和 2）
    if 'label' in edges.columns:
        df_pos = edges[edges['label'] == 1]
        df_neg = edges[edges['label'] == 2]
    else:
        # Fallback，如果只有单一类别
        df_pos = edges
        df_neg = []
        
    print(f"原始数据统计: Label 1 (Pos) = {len(df_pos)}, Label 2 (Neg) = {len(df_neg)}")

    # 取 Label 1 (Pos) 的数量作为生成 Label 0 的数量
    num_samples_L0 = len(df_pos) if len(df_pos) > 0 else len(edges)
    
    all_real_src = edges["src"].values
    all_real_dst = edges["dst"].values
    
    # 采样 Label 0
    generated_L0_pairs = sample_non_edges_fixed(
        num_nodes, 
        forbidden_src=all_real_src, 
        forbidden_dst=all_real_dst, 
        num_samples=num_samples_L0
    )
    
    src_L0 = np.array([p[0] for p in generated_L0_pairs])
    dst_L0 = np.array([p[1] for p in generated_L0_pairs])
    lbl_L0 = np.zeros(len(generated_L0_pairs), dtype=int)
    
    # 合并
    all_src = np.concatenate([edges["src"].values, src_L0])
    all_dst = np.concatenate([edges["dst"].values, dst_L0])
    if 'label' in edges.columns:
        all_lbl = np.concatenate([edges["label"].values, lbl_L0])
    else:
        # 没有label则用1填充原有边
        all_lbl = np.concatenate([np.ones(len(edges), dtype=int), lbl_L0])
        
    print(f"采样后总数据量: {len(all_lbl)} (其中 Label 0: {len(lbl_L0)})")
    
    # 为了避免依赖 sklearn，使用 numpy 进行随机乱序并划分
    shuffled_idx = np.random.permutation(len(all_lbl))
    split_point = int(len(all_lbl) * 0.8) # 80% 作为训练集
    train_idx = shuffled_idx[:split_point]
    val_idx = shuffled_idx[split_point:]

    # 生成 train2id.txt
    train_file = os.path.join(out_dir, "train2id.txt")
    with open(train_file, 'w') as f:
        f.write(f"{len(train_idx)}\n")
        for i in train_idx:
            f.write(f"{all_src[i]} {all_dst[i]} {all_lbl[i]}\n")
            
    # 生成 valid2id.txt
    valid_file = os.path.join(out_dir, "valid2id.txt")
    with open(valid_file, 'w') as f:
        f.write(f"{len(val_idx)}\n")
        for i in val_idx:
            f.write(f"{all_src[i]} {all_dst[i]} {all_lbl[i]}\n")
    
    # 生成 test2id.txt（这里我们直接用 valid2id.txt 的内容作为测试集，或者你也可以重新划分）
    test_file = os.path.join(out_dir, "test2id.txt")
    with open(test_file, 'w') as f:
        f.write(f"{len(val_idx)}\n")
        for i in val_idx:
            f.write(f"{all_src[i]} {all_dst[i]} {all_lbl[i]}\n")

            
    # 额外：顺便在当前文件夹生成配套的 entity2id.txt 和 relation2id.txt
    with open(os.path.join(out_dir, "entity2id.txt"), 'w') as f:
        f.write(f"{num_nodes}\n")
        for _, row in nodes.iterrows():
            f.write(f"{row['node_name']}\t{int(row['node_id'])}\n")
            
    unique_rels = np.unique(all_lbl)
    with open(os.path.join(out_dir, "relation2id.txt"), 'w') as f:
        f.write(f"{len(unique_rels)}\n")
        for rel in unique_rels:
            f.write(f"Label_{rel}\t{rel}\n")

    print(f"=== 已成功在 {out_dir} 生成训练所需的所有 txt 文件 ===")


def convert_nodes_to_entity2id():
    input_file = "benchmarks/data_signed/node_labeled.csv"
    output_file = "benchmarks/data_signed/entity2id.txt"

    print(f"Reading {input_file}...")
    # 读取原始CSV文件
    df = pd.read_csv(input_file)

    # 获取节点总数
    num_entities = len(df)

    print(f"Total entities found: {num_entities}")
    print(f"Writing to {output_file}...")
    
    # 写入 entity2id.txt
    with open(output_file, 'w') as f:
        # 第一行写入实体总数
        f.write(f"{num_entities}\n")
        
        # 遍历每一行，按 OpenKE 要求的格式写入: 实体名 \t 实体ID
        # 将 old_index 作为实体名，node_id 作为对应的ID
        for _, row in df.iterrows():
            old_index = row['old_index']
            node_id = int(row['node_id'])
            f.write(f"{old_index}\t{node_id}\n")
            
    print("Entity conversion completed successfully!")

def convert_edges_to_relation2id():
    input_file = "benchmarks/data_signed/edges_labeled_with_reason.csv"
    output_file = "benchmarks/data_signed/relation2id.txt"

    print(f"Reading {input_file}...")
    # 读取包含边和关系的CSV文件
    df = pd.read_csv(input_file)

    # 提取所有独特的关系（这里默认取 'rel' 列，如果想按标签分类可改为 'label' 或 'edge_semantic'）
    unique_relations = df['edge_semantic'].unique()
    num_relations = len(unique_relations)

    print(f"Total unique relations found: {num_relations}")
    print(f"Writing to {output_file}...")
    
    # 写入 relation2id.txt
    with open(output_file, 'w') as f:
        # 第一行写入关系总数
        f.write(f"{num_relations}\n")
        
        # 遍历每种关系，写入: 关系名 \t 关系ID
        for idx, rel in enumerate(unique_relations):
            f.write(f"{rel}\t{idx}\n")
            
    print("Relation conversion completed successfully!")

if __name__ == "__main__":
    generate_train_valid_sets()
    # convert_nodes_to_entity2id()
    # convert_edges_to_relation2id()
