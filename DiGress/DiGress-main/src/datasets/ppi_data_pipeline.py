import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np
import os

# ==============================================================================
# 1. 配置与定义
# ==============================================================================
# 节点功能分类 (x.A)
GO_CATEGORIES = [
    "Enzyme", "Transcription", "Transporter", "Receptor", 
    "Structural", "Ubiquitin", "Immune", "Metabolic", "Signaling", "Other"
]
NUM_NODE_CLASSES = len(GO_CATEGORIES)

# 边类别定义 (GAR+ Predicates)
# 0: No Edge
# 1: l(x,y) + ¬M(x,y)  (Positive Edge, Low Conf)
# 2: l(x,y) + M(x,y)   (Positive Edge, High Conf)
# 3: ¬l(x,y) + ¬M(x,y) (Negative Edge, Low Conf)
# 4: ¬l(x,y) + M(x,y)  (Negative Edge, High Conf)
NUM_EDGE_CLASSES = 5 

ML_THRESHOLD = 0.5  # ML Score 阈值

# ==============================================================================
# 2. 辅助函数：属性映射逻辑
# ==============================================================================
def map_ontology_to_category(ontology_str):
    """ 将 BioGRID 的 Ontology 字符串映射为 0-9 的整数 """
    s = str(ontology_str).lower()
    if 'kinase' in s or 'phosphatase' in s or 'enzyme' in s: return 0
    if 'transcription' in s: return 1
    if 'transporter' in s or 'channel' in s: return 2
    if 'receptor' in s: return 3
    if 'structural' in s or 'cytoskeleton' in s: return 4
    if 'ubiquitin' in s: return 5
    if 'immune' in s: return 6
    if 'metabolic' in s: return 7
    if 'signaling' in s: return 8
    return 9 # Other

def get_gar_edge_index(sys_type, score):
    """ 
    根据边类型和分数计算 DiGress 离散索引 
    Returns: 1, 2, 3, or 4
    """
    # 1. 判断 l(x,y) 还是 ¬l(x,y)
    # physical -> Positive (l), genetic -> Negative (¬l)
    is_negative = 0  # Default Positive
    if 'genetic' in str(sys_type).lower() or 'negative' in str(sys_type).lower():
        is_negative = 1
        
    # 2. 判断 M(x,y)
    is_high_conf = 1 if float(score) >= ML_THRESHOLD else 0
    
    # 3. 组合公式: 1 + (IsNeg * 2) + IsHigh
    # Pos/Low  = 1 + 0 + 0 = 1
    # Pos/High = 1 + 0 + 1 = 2
    # Neg/Low  = 1 + 2 + 0 = 3
    # Neg/High = 1 + 2 + 1 = 4
    return 1 + (is_negative * 2) + is_high_conf

# ==============================================================================
# 3. 载入与预处理 (Pandas -> NetworkX)
# ==============================================================================
def load_biogrid_graph(path):
    print(f"[Info] Loading Raw Data from {path}...")
    df = pd.read_csv(path, sep='\t')
    
    G = nx.Graph()
    
    for _, row in df.iterrows():
        u = str(row['Official Symbol Interactor A'])
        v = str(row['Official Symbol Interactor B'])
        
        # 提取属性
        sys_type = row['Experimental System Type']
        score = row['Score']
        
        # 节点属性 (A 和 B 分别处理)
        ont_A = row['Ontology Term Names Interactor A']
        ont_B = row['Ontology Term Names Interactor B']
        
        # 添加节点 (如果已存在会更新属性，这里假设属性一致)
        G.add_node(u, ontology=ont_A)
        G.add_node(v, ontology=ont_B)
        
        # 添加边
        G.add_edge(u, v, system_type=sys_type, score=score)
        
    print(f"[Info] Graph Constructed. Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    return G

# ==============================================================================
# 4. 核心转换 (NetworkX -> PyG DiGress Data)
# ==============================================================================
def to_digress_data(subG: nx.Graph):
    """ 将一个 NetworkX 子图转换为包含 GAR+ 逻辑的 PyG Data 对象 """
    # 重新映射 ID 到 0 ~ N-1
    mapping = {n: i for i, n in enumerate(subG.nodes())}
    
    # --- A. 构建节点特征 X (One-Hot) ---
    xs_indices = []
    for n in subG.nodes():
        ont_str = subG.nodes[n].get('ontology', '')
        cat_idx = map_ontology_to_category(ont_str)
        xs_indices.append(cat_idx)
    
    x_idx = torch.tensor(xs_indices, dtype=torch.long)
    # [N, 10]
    x = F.one_hot(x_idx, num_classes=NUM_NODE_CLASSES).float()
    
    # --- B. 构建边特征 E (One-Hot) ---
    src, dst = [], []
    edge_attr_indices = []
    
    for u, v, d in subG.edges(data=True):
        if u not in mapping or v not in mapping: continue
        u_idx, v_idx = mapping[u], mapping[v]
        
        # 获取 GAR+ 编码索引 (1-4)
        attr_idx = get_gar_edge_index(d.get('system_type'), d.get('score', 0))
        
        # 双向添加 (无向图)
        src.extend([u_idx, v_idx])
        dst.extend([v_idx, u_idx])
        edge_attr_indices.extend([attr_idx, attr_idx]) # 对称

    if not src: return None

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # [M, 5] (0=NoEdge, 1..4=Types)
    ea_idx = torch.tensor(edge_attr_indices, dtype=torch.long)
    edge_attr = F.one_hot(ea_idx, num_classes=NUM_EDGE_CLASSES).float()
    
    # 全局变量 (DiGress 需要)
    y = torch.zeros(1, 0).float()
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, n_nodes=subG.number_of_nodes(), y=y)
    return data

# ==============================================================================
# 5. 解码器 (Tensor -> GAR+ Literals)
# ==============================================================================
def decode_gar_literals(data: Data):
    """ 从生成的 Tensor 中反向解析出 GAR+ 规则谓词 """
    literals = []
    
    # 1. 还原节点属性
    X_ids = data.x.argmax(dim=-1).tolist()
    node_props = {}
    
    for i, cat_idx in enumerate(X_ids):
        cat_name = GO_CATEGORIES[cat_idx]
        node_props[i] = cat_name
        literals.append(f"n{i}.Func == {cat_name}")
        
    # 2. 还原边属性
    # 将稀疏 edge_index 转为稠密矩阵以便遍历 (模拟生成时的全图)
    num_nodes = data.n_nodes
    E_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
    
    src, dst = data.edge_index
    attr_ids = data.edge_attr.argmax(dim=-1)
    
    for u, v, a in zip(src, dst, attr_ids):
        E_matrix[u, v] = a

    # 遍历上三角
    rows, cols = torch.triu_indices(num_nodes, num_nodes, offset=1)
    for u, v in zip(rows, cols):
        val = E_matrix[u, v].item()
        if val == 0: continue
        
        # 反向逻辑: val = 1 + (is_neg * 2) + is_high
        val_shifted = val - 1
        is_neg = val_shifted // 2
        is_high = val_shifted % 2
        
        # 拓扑谓词
        rel = "¬l" if is_neg else "l"
        literals.append(f"{rel}({u},{v})")
        
        # ML 谓词
        ml = "M" if is_high else "¬M"
        literals.append(f"{ml}({u},{v})")
        
        # 自动推导的二元谓词 (GAR+ 特性)
        if node_props[u] == node_props[v]:
            literals.append(f"n{u}.Func == n{v}.Func")

    return literals

# ==============================================================================
# 6. 主程序流程
# ==============================================================================
if __name__ == "__main__":
    # --- Step 0: 生成模拟数据 (Dummy BioGRID File) ---
    dummy_file = "dummy_biogrid.txt"
    with open(dummy_file, "w") as f:
        # 写入 Header
        f.write("Official Symbol Interactor A\tOfficial Symbol Interactor B\t"
                "Experimental System Type\tScore\t"
                "Ontology Term Names Interactor A\tOntology Term Names Interactor B\n")
        # 写入一些模拟样本
        # 样本1: Kinase 和 Substrate 的物理结合，高分
        f.write("ProtA\tProtB\tphysical\t0.9\tProtein Kinase Activity\tMetabolic Process\n")
        # 样本2: 两个 Transcription Factors 的遗传负向关联，低分
        f.write("ProtC\tProtD\tgenetic\t0.2\tTranscription Factor\tTranscription Factor\n")
        # 样本3: 免疫蛋白和受体的物理结合，低分
        f.write("ProtE\tProtF\tphysical\t0.4\tImmune Response\tReceptor Activity\n")

    # --- Step 1: 载入数据 ---
    full_graph = load_biogrid_graph(dummy_file)
    
    # --- Step 2: 采样子图 (模拟 Dataset 行为) ---
    # 这里直接拿全图当一个 Batch
    subgraph = full_graph 
    
    # --- Step 3: 构建 DiGress Data ---
    print("\n[Info] Converting to DiGress Data format...")
    pyg_data = to_digress_data(subgraph)
    
    print("\n--- Generated Tensors (Input for Diffusion Model) ---")
    print(f"X (Node Features) shape: {pyg_data.x.shape}")
    print(f"E (Edge Features) shape: {pyg_data.edge_attr.shape}")
    # 打印前几个节点的 One-Hot
    print("Sample Node X:", pyg_data.x[0]) 
    
    # --- Step 4: 解码验证 (模拟 Rule Mining) ---
    print("\n--- Decoding GAR+ Literals from Tensors ---")
    gar_rules = decode_gar_literals(pyg_data)
    
    print("Recovered Literals:")
    for rule in gar_rules:
        print(f"  {rule}")

    # 清理模拟文件
    if os.path.exists(dummy_file):
        os.remove(dummy_file)