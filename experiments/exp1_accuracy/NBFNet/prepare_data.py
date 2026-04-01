import os
import random
import pandas as pd
import numpy as np

def sample_non_edges_fixed(num_nodes, forbidden_src, forbidden_dst, num_samples):
    """
    随机采样负边，确保不属于在 forbidden_src 和 forbidden_dst 中出现过的实际边
    """
    forbidden_edges = set(zip(forbidden_src, forbidden_dst))
    generated_edges = set()
    
    while len(generated_edges) < num_samples:
        u = random.randint(0, num_nodes - 1)
        v = random.randint(0, num_nodes - 1)
        
        if u != v and (u, v) not in forbidden_edges and (u, v) not in generated_edges:
            generated_edges.add((u, v))
            
    return list(generated_edges)

def generate_nbfnet_datasets():
    node_csv = "data_updated/node.csv"
    edge_csv = "data_updated/edge_update.csv"
    out_dir = "data_updated/my_dataset/"
    
    os.makedirs(out_dir, exist_ok=True)
    print("正在准备数据...")
    
    # 1. 读取基础数据
    nodes = pd.read_csv(node_csv)
    num_nodes = len(nodes)
    edges = pd.read_csv(edge_csv)
    
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
    
    # 为了构建 NBFNet 所需的三元组格式 (head, relation, tail)
    # 此处我们将分类类型 (`all_lbl`) 作为 NBFNet 中的关系 (relation) 处理
    data_triplets = pd.DataFrame({
        'head': all_src,
        'relation': all_lbl,  # Label 0, 1, 2 作为不同的关系
        'tail': all_dst
    })
    
    # 使用 numpy 进行随机乱序并划分: 80% 训练, 10% 验证, 10% 测试
    shuffled_idx = np.random.permutation(len(all_lbl))
    train_split = int(len(all_lbl) * 0.8)
    valid_split = int(len(all_lbl) * 0.9)
    
    train_idx = shuffled_idx[:train_split]
    val_idx = shuffled_idx[train_split:valid_split]
    test_idx = shuffled_idx[valid_split:]
    
    train_df = data_triplets.iloc[train_idx]
    val_df = data_triplets.iloc[val_idx]
    test_df = data_triplets.iloc[test_idx]
    
    # NBFNet 要求输入的文件是以制表符（\t）分隔的 .txt 格式，并且没有表头
    train_df.to_csv(os.path.join(out_dir, "train.txt"), sep='\t', index=False, header=False)
    val_df.to_csv(os.path.join(out_dir, "valid.txt"), sep='\t', index=False, header=False)
    test_df.to_csv(os.path.join(out_dir, "test.txt"), sep='\t', index=False, header=False)
    
    print(f"数据处理完成，已保存至 {out_dir}")
    print(f"划分明细 - 训练集：{len(train_df)}，验证集：{len(val_df)}，测试集：{len(test_df)}")

if __name__ == "__main__":
    generate_nbfnet_datasets()