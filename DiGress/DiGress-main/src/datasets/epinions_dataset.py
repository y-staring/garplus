# src/datasets/epinions_dataset.py
# -*- coding: utf-8 -*-
"""
Epinions（有向带符号社交信任图）→ 随机游走子图数据集（适配 DiGress）

设计要点：
- 先把 SNAP 的 soc-sign-epinions 原始大图下载/解压
- 在“无向视图”上做二阶随机游走（node2vec 风格），收集节点集合作为子图（保证连通、覆盖局部）
- 但最终从“原始有向图”里取边，并保留 sign ∈ {+1,-1}
- 边类型 one-hot: [none, neg, pos] —— 与 DiGress 的 edge_counts() 口径一致（非边计入 slot 0）
- 节点特征：度分箱 one-hot（入度+出度的总度进行分箱），与 AbstractDataModule.node_types() 的 one-hot 假设一致
- 将采样得到的子图节点集合划分为 train/val/test 三个 raw 文件；每个 split 在 process() 中各自转换为 PyG Data 并缓存为 processed 文件
- 提供 DataModule 与 DatasetInfos，按 DiGress 新数据集指南集成

依赖：
- torch, torch_geometric, networkx
- 你项目中的：
  from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
"""

import os
import gzip
import random
import pathlib
from typing import List, Tuple, Dict
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data, download_url
import torch_geometric.utils
import networkx as nx
import torch.nn.functional as F
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

# 在文件开头补充：
from collections import deque
try:
    from tqdm import tqdm
except Exception:
    tqdm = lambda x, **kw: x  # 没有 tqdm 也能跑

def _check_rw_sample(UG, ns, min_nodes: int, max_nodes: int):
    n = len(ns)
    if n < min_nodes: return False, f"too_small:{n}"
    if n > max_nodes: return False, f"too_large:{n}"
    if len(ns) != len(set(ns)): return False, "dup_nodes"

    import networkx as nx
    H = UG.subgraph(ns)
    if not nx.is_connected(H):
        return False, "not_connected"
    return True, "ok"



# -----------------------------
# 常量 & 随机数生成器（保证可复现）
# -----------------------------
EPINIONS_URL = "https://snap.stanford.edu/data/soc-sign-epinions.txt.gz"
RNG = random.Random(42)  # 只用于 Python 端随机；与 torch 的随机分开

# -----------------------------
# 工具：下载/读取 SNAP 文件
# -----------------------------
def _download_if_needed(root: str) -> str:
    """
    若本地无原始文件则下载并解压；返回解压后的 .txt 路径。
    这么做让数据集“开箱即用”，且一旦缓存，下次不再重复下载。
    """
    os.makedirs(root, exist_ok=True)
    gz_path = os.path.join(root, "soc-sign-epinions.txt.gz")
    txt_path = os.path.join(root, "soc-sign-epinions.txt")

    if (not os.path.exists(gz_path)) and (not os.path.exists(txt_path)):
        print("[Info] Downloading Epinions ...")
        download_url(EPINIONS_URL, root)  # 会把文件保存为 raw_dir 下的同名 .gz

    # 若仅存在 .gz 则解压
    if (not os.path.exists(txt_path)) and os.path.exists(gz_path):
        print("[Info] Decompressing Epinions ...")
        with gzip.open(gz_path, "rb") as f_in, open(txt_path, "wb") as f_out:
            f_out.write(f_in.read())

    if not os.path.exists(txt_path):
        # 某些环境 download_url 会把文件放在 root/ 下，用上面的路径拼接可能不同
        # 再兜底找一下
        alt_gz = os.path.join(root, os.path.basename(EPINIONS_URL))
        if os.path.exists(alt_gz):
            with gzip.open(alt_gz, "rb") as f_in, open(txt_path, "wb") as f_out:
                f_out.write(f_in.read())

    assert os.path.exists(txt_path), "Epinions txt not found after download/decompress."
    return txt_path



def _read_signed_digraph(txt_path: str) -> nx.DiGraph:
    """
    读取 SNAP soc-sign-epinions: 每行 u v sign [time?]。
    - 仅使用 sign（>0 视为 +1，<=0 视为 -1）
    - 读完后把节点 relabel 为 0..N-1，便于张量索引
    """
    G = nx.DiGraph()
    with open(txt_path, "r") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            u, v = int(parts[0]), int(parts[1])
            s = 1 if int(parts[2]) > 0 else -1
            G.add_edge(u, v, sign=s)
            # 显式加点以保证无边节点也被计入
            G.add_node(u)
            G.add_node(v)

    mapping = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    return G


# -----------------------------
# 节点/边特征构造
# -----------------------------
def _degree_bins(G: nx.DiGraph, num_bins: int = 10) -> Dict[int, torch.Tensor]:
    """
    节点特征 = 度分箱 one-hot（使用 total-degree = in + out）
    - 结构理由：非属性图上，度是最稳健的结构线索之一。分箱的 one-hot 与 DiGress 的“节点类型 one-hot”假设匹配。
    - 最后一箱为“>= num_bins-1”
    """
    deg_dict = {n: G.in_degree(n) + G.out_degree(n) for n in G.nodes()}

    def encode_onehot(d: int) -> torch.Tensor:
        b = min(d, num_bins - 1)
        x = torch.zeros(num_bins, dtype=torch.float)
        x[b] = 1.0
        return x

    return {n: encode_onehot(deg) for n, deg in deg_dict.items()}


def _onehot_sign(sign: int) -> torch.Tensor:
    """
    边类型 one-hot: [none, neg, pos]
    - 真实边只用 neg/pos；none 预留给“非边”计数（与 AbstractDataModule.edge_counts 的口径一致）
    """
    if sign >= 0:
        return torch.tensor([0.0, 1.0, 0.0], dtype=torch.float)
    else:
        return torch.tensor([0.0, 0.0, 1.0], dtype=torch.float)


#以BFD形式扩展，防止全生成长条状的
def _bfs_expansion_from_seed(
    G: nx.DiGraph,
    seed_nodes: List[int], 
    max_nodes: int = 50
) -> List[int]:
    """ 从种子节点集合开始 BFS """
    UG = G.to_undirected(as_view=True)
    q = deque(seed_nodes)
    visited = set(seed_nodes)
    nodes_list = list(seed_nodes)
    
    while q and len(nodes_list) < max_nodes:
        curr = q.popleft()
        neighbors = list(UG.neighbors(curr))
        RNG.shuffle(neighbors)
        for nxt in neighbors:
            if nxt not in visited:
                visited.add(nxt)
                nodes_list.append(nxt)
                q.append(nxt)
                if len(nodes_list) >= max_nodes:
                    break
    return nodes_list


def _find_negative_edges(G: nx.DiGraph) -> List[Tuple[int, int]]:
    """
    扫描图中所有的负边 (u, v)，sign < 0。
    """
    print("[Info] Scanning for Negative Edges to use as seeds...")
    neg_edges = []
    for u, v, data in tqdm(G.edges(data=True), desc="Finding Neg Edges"):
        # 原始 txt 里 sign 是整数，可能是 -1
        if data.get('sign', 0) < 0:
            neg_edges.append((u, v))
            
    print(f"[Info] Found {len(neg_edges)} negative edges.")
    return neg_edges

def _inject_gar_semantics(subG: nx.DiGraph):
    """
    为子图注入 GAR+ 所需的节点属性和 ML 评分。
    直接修改 subG 的节点和边属性。
    """
    # 1. 计算 Influence (基于 PageRank) - 代表用户权威度
    # 使用无向图计算以保证稳定性，或者用有向图体现权威
    try:
        pr = nx.pagerank(subG, alpha=0.85)
    except:
        # 极小图或不连通可能导致收敛问题，兜底
        pr = {n: 1.0/subG.number_of_nodes() for n in subG.nodes()}
    
    # 2. 计算 Community (基于 Louvain 或 Greedy Modularity) - 代表兴趣圈层
    # 需要先转无向
    UG = subG.to_undirected()
    try:
        # 使用 greedy_modularity_communities 比较快
        communities = nx.community.greedy_modularity_communities(UG)
        comm_map = {}
        for c_id, nodes in enumerate(communities):
            for node in nodes:
                comm_map[node] = c_id
    except:
        comm_map = {n: 0 for n in subG.nodes()}

    # 3. 注入属性到节点
    for n in subG.nodes():
        # 属性 A: Influence (连续值)
        subG.nodes[n]['influence'] = float(pr[n])
        # 属性 B: Community (离散值/常量)
        subG.nodes[n]['group'] = int(comm_map[n])
        # 属性 C: Activity (基于度的等级)
        degree = subG.degree(n)
        subG.nodes[n]['activity'] = int(np.log2(degree + 1))

    # 4. (可选) 计算 ML Predicate 值: 假设我们需要预测任意两点间的 link score
    # 这里我们只计算图中已存在的边的 "ML Score" 作为演示
    # 真实场景下，ML Predicate 通常用于预测 "Missing Edge"
    # 这里用 Jaccard Coefficient 作为简单的 ML 模型代理
    # 注意：Jaccard 定义在无向图上
    jaccard_preds = nx.jaccard_coefficient(UG, list(UG.edges()))
    
    # 将 ML 分数存入边属性 (模拟 M(x,y))
    edge_ml_scores = {}
    for u, v, score in jaccard_preds:
        # 有向图可能双向都有边，需要处理
        if subG.has_edge(u, v):
            subG.edges[u, v]['ml_score'] = score
        if subG.has_edge(v, u):
            subG.edges[v, u]['ml_score'] = score
            
    return subG

# -----------------------------
# 随机游走采样（node2vec 二阶随机游走）
# -----------------------------
def _random_walk_nodes(
    G: nx.DiGraph,
    center: int,
    walk_length: int = 40,
    num_walks: int = 4,
    max_nodes: int = 200,
    p: float = 1.0,
    q: float = 1.0,
) -> List[int]:
    """
    在“无向视图”上进行若干条二阶随机游走，汇总访问到的节点作为子图。
    这么做的原因：
    - 行走阶段只为“选节点集合”，无向视图更有助于连通性与覆盖局部结构；
    - 最终仍然从“原始有向图”取边并保留 sign，用于学习方向性与正/负关系；
    - node2vec 的二阶偏置：
        p > 1 抑制回溯；q > 1 偏向 BFS（团簇/社区）；q < 1 偏向 DFS（长链/路径）。
    """
    UG = G.to_undirected(as_view=True)
    nodes_collected = set([center])

    neigh = {u: list(UG.neighbors(u)) for u in UG.nodes()}
    if not neigh.get(center, []):
        return [center]

    rng = RNG
    for _ in range(num_walks):
        path = [center]
        # 第一步：随机选邻居，增加样本多样性
        first_neighs = neigh.get(path[-1], [])
        if not first_neighs:
            continue
        nxt = rng.choice(first_neighs)
        path.append(nxt)
        nodes_collected.add(nxt)

        # 后续：二阶转移
        for _t in range(2, walk_length + 1):
            prev, cur = path[-2], path[-1]
            cands = neigh.get(cur, [])
            if not cands:
                break

            # node2vec 权重
            prev_set = set(neigh.get(prev, []))
            weights = []
            for x in cands:
                if x == prev:
                    w = 1.0 / p
                elif x in prev_set:
                    w = 1.0
                else:
                    w = 1.0 / q
                weights.append(w)

            # 归一化按权采样（纯 Python，不依赖 numpy）
            s = sum(weights)
            r = rng.random() * s
            acc, choice = 0.0, cands[0]
            for x, w in zip(cands, weights):
                acc += w
                if acc >= r:
                    choice = x
                    break

            path.append(choice)
            nodes_collected.add(choice)

            if len(nodes_collected) >= max_nodes:
                break

        if len(nodes_collected) >= max_nodes:
            break

    nodes_list = list(nodes_collected)
    if len(nodes_list) > max_nodes:
        rng.shuffle(nodes_list)
        nodes_list = nodes_list[:max_nodes]
    return nodes_list




def _to_gar_pyg_data(H: nx.DiGraph) -> Data:
    # 1. 注入语义 (保持你的逻辑)
    # 这会给节点加上 'influence', 'group', 'activity' 属性
    H = _inject_gar_semantics(H) 

    # =========================================================
    # ★★★ 核心修改 1: 强制转无向图 (适配 DiGress) ★★★
    # =========================================================
    # DiGress 只能处理对称邻接矩阵。
    # 这里我们把有向图转为无向，如果两点间有冲突边，networkx 会保留其中一条。
    G_undir = H.to_undirected()

    mapping = {n: i for i, n in enumerate(G_undir.nodes())}
    inv_mapping = {i: n for n, i in mapping.items()}
    
    # =========================================================
    # ★★★ 核心修改 2: 多属性特征拼接 (Concatenation) ★★★
    # 目标 X 维度: Group(5) + Influence(3) + Activity(3) = 11
    # =========================================================
    xs_indices = []
    # 定义维度基数
    DIM_G = 5  # Group
    DIM_I = 3  # Influence
    DIM_A = 3  # Activity
    # 总类别数 = 5 * 3 * 3 = 45
    # 这里的 total_bins 必须写入 yaml
    
    for i in range(len(G_undir.nodes())):
        original_node = inv_mapping[i]
        node_attrs = G_undir.nodes[original_node]
        
        # 1. 获取三个分量的索引
        g_idx = int(node_attrs.get('group', 0)) % DIM_G
        
        inf_val = float(node_attrs.get('influence', 0.0))
        if inf_val < 0.01: i_idx = 0
        elif inf_val < 0.05: i_idx = 1
        else: i_idx = 2
        
        act_val = int(node_attrs.get('activity', 0))
        if act_val <= 1: a_idx = 0
        elif act_val <= 3: a_idx = 1
        else: a_idx = 2
        
        # 2. ★★★ 计算唯一扁平化索引 ★★★
        # 公式: Index = G * (3*3) + I * (3) + A
        flat_idx = g_idx * (DIM_I * DIM_A) + i_idx * DIM_A + a_idx
        
        xs_indices.append(flat_idx)

    # 堆叠成矩阵 (N, 11)，必须是 Float 类型
    # 转为 One-Hot (总维度 45)
    # 此时每个节点真的只有一个 1，完全符合 DiGress 要求
    TOTAL_BINS = DIM_G * DIM_I * DIM_A  # 45
    x_idx = torch.tensor(xs_indices, dtype=torch.long)
    x = F.one_hot(x_idx, num_classes=TOTAL_BINS).float()
    # x = torch.stack(xs_list).float()

    # =========================================================
    # ★★★ 核心修改 3: 边特征双向添加 (Symmetry) ★★★
    # =========================================================
    src, dst = [], []
    edge_attr_indices = []
    num_edge_classes = 3  # 0:None, 1:Trust, 2:Distrust
    
    for u, v, d in G_undir.edges(data=True):
        if u not in mapping or v not in mapping: continue
        
        u_idx, v_idx = mapping[u], mapping[v]
        
        # 符号映射
        sign = d.get('sign', 1)
        attr_idx = 1 if sign > 0 else 2
        
        # 添加双向边，确保邻接矩阵对称
        # u -> v
        src.append(u_idx)
        dst.append(v_idx)
        edge_attr_indices.append(attr_idx)
        
        # v -> u
        src.append(v_idx)
        dst.append(u_idx)
        edge_attr_indices.append(attr_idx)

    if not src: return None

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    
    # 边特征转 One-Hot Float
    ea_idx = torch.tensor(edge_attr_indices, dtype=torch.long)
    edge_attr = F.one_hot(ea_idx, num_classes=num_edge_classes).float()

    # =========================================================
    # 4. 构造 Data
    # =========================================================
    y = torch.zeros(1, 0).float()
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.n_nodes = G_undir.number_of_nodes()
    data.y = y 

    return data



# -----------------------------
# 子图转换为 PyG Data
# -----------------------------
def _to_pyg_data(H: nx.DiGraph, node_x_dict: Dict[int, torch.Tensor]) -> Data:
    """
    将 networkx DiGraph 子图转换为 PyG Data：
    - x: [n, Cx] 节点 one-hot（度分箱）
    - edge_index: [2, m]（保留有向）
    - edge_attr: [m, 3] one-hot（[none, neg, pos]；真实边只用 neg/pos）
    - y: [1] 图级条件（本任务为空，DiGress 会在输入维度中额外 +1 时间条件）
    - 若子图无边，返回 None（对扩散式建模帮助有限）
    """
    mapping = {n: i for i, n in enumerate(H.nodes())}
    inv = [n for n, _ in sorted(mapping.items(), key=lambda kv: kv[1])]

    # 节点特征
    x = torch.stack([node_x_dict[n] for n in inv], dim=0)  # [n, Cx]

    # 边特征
    src, dst, eattr = [], [], []
    for (u, v, d) in H.edges(data=True):
        src.append(mapping[u])
        dst.append(mapping[v])
        eattr.append(_onehot_sign(d.get("sign", 1)))
    if len(src) == 0:
        return None

     # ======================================================
    # ★★★ 强制对称化（DiGress discrete 必须无向 E=E^T）★★★
    # ======================================================
    src2 = []
    dst2 = []
    attr2 = []
    for s, t, a in zip(src, dst, eattr):
        src2.append(s); dst2.append(t); attr2.append(a)
        if t != s:      # 添加反向边
            src2.append(t)
            dst2.append(s)
            attr2.append(a.clone())

    edge_index = torch.tensor([src2, dst2], dtype=torch.long)
    edge_attr = torch.stack(attr2, dim=0)

    y = torch.zeros(1, 0)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


    
    # edge_index = torch.tensor([src, dst], dtype=torch.long)
    # edge_attr = torch.stack(eattr, dim=0)  # [m, 3]
    # y = torch.zeros(1, 0,dtype=torch.float)  # 无图级条件时占位

    # return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
# -----------------------------
# ★★★ 新增：导出为 TXT 供分析脚本使用 ★★★
# -----------------------------
def _export_samples_to_txt(samples: List[List[int]], G: nx.DiGraph, filepath: str):
    """
    将采样到的子图保存为文本格式，格式兼容 analysis 脚本。
    格式：
    N=xx
    X node_label (全是0，因为我们没有真实的node label，只有度信息，这里填占位符)
    E
    Adjacency Matrix (0, 1, 2) where 2=neg, 1=pos
    """
    print(f"[Info] Exporting {len(samples)} graphs to {filepath} for structural analysis...")
    with open(filepath, "w") as f:
        for nodes in samples:
            subG = G.subgraph(nodes)
            n = subG.number_of_nodes()
            # 建立局部映射 0..n-1
            mapping = {node: i for i, node in enumerate(subG.nodes())}
            
            f.write(f"N={n}\n")
            # 写入虚拟的 Node Labels (全0)
            f.write("X " + " ".join(["0"] * n) + "\n")
            f.write("E\n")
            
            # 构建 n x n 矩阵
            mat = [[0] * n for _ in range(n)]
            for u, v, d in subG.edges(data=True):
                r, c = mapping[u], mapping[v]
                s = d.get('sign', 1)
                # 2=Neg, 1=Pos (对应分析脚本的 POS_LBL=2, NEG_LBL=1)
                val = 2 if s < 0 else 1
                mat[r][c] = val
                # 分析脚本如果是无向处理，这里只存一边即可；如果是有向，存有向矩阵
                # 假设分析脚本处理的是 Adjacency Matrix
            
            for row in mat:
                f.write(" ".join(map(str, row)) + "\n")

# -----------------------------
# Dataset（仅随机游走）
# -----------------------------
class EpinionsSubgraphDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        max_nodes: int = 40,
        min_nodes: int = 10,
        num_bins: int = 10,
        num_subgraphs: int = 2000,
        train_val_test: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        export_txt: bool = False,
        # 新增参数：负边混合比例
        # 1.0 表示只采以负边为中心的图（极端关注冲突）
        # 0.5 表示一半负边中心，一半随机（平衡）
        neg_sample_ratio: float = 1.0 
    ):
        self.split = split
        self.max_nodes = max_nodes
        self.min_nodes = min_nodes
        self.num_bins = num_bins
        self.num_subgraphs = num_subgraphs
        self.train_val_test = train_val_test
        self.export_txt = export_txt
        self.neg_sample_ratio = neg_sample_ratio
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return ["train_nodes.pt", "val_nodes.pt", "test_nodes.pt"]

    @property
    def processed_file_names(self) -> List[str]:
        # 改个名字防止和之前的缓存冲突
        return [f"epinions_neg_seed_{self.split}.pt"]

    def download(self):
        all_exist = all(os.path.exists(p) for p in self.raw_paths)
        if all_exist: return

        txt_path = _download_if_needed(self.raw_dir)
        G = _read_signed_digraph(txt_path)
        all_nodes = list(G.nodes())

        # ==========================================
        # ★★★ 策略：获取负边池 ★★★
        # ==========================================
        neg_edges = _find_negative_edges(G)
        if not neg_edges:
            print("[Warning] No negative edges found! Fallback to random.")
            neg_edges = []

        samples: List[List[int]] = []
        target = self.num_subgraphs
        pbar = tqdm(total=target, desc="Sampling(Neg-Focused)")

        tries = 0
        while len(samples) < target:
            tries += 1
            if tries > target * 100: break # 防止死循环

            # 决定这次采样是用“负边种子”还是“随机节点”
            is_neg_seed = (RNG.random() < self.neg_sample_ratio) and (len(neg_edges) > 0)
            
            if is_neg_seed:
                # 随机挑一条负边 (u, v)
                u, v = RNG.choice(neg_edges)
                seed = [u, v]
            else:
                # 随机挑一个点
                seed = [RNG.choice(all_nodes)]

            # 执行 BFS 扩张
            nodes = _bfs_expansion_from_seed(G, seed, self.max_nodes)

            # 过滤太小的图
            if len(nodes) < self.min_nodes:
                continue
            
            # (可选) 过滤掉纯散点，虽然 BFS 保证连通，但可以根据需要加其他过滤
            
            samples.append(nodes)
            pbar.update(1)

        pbar.close()

        # 划分 & 保存
        RNG.shuffle(samples)
        n = len(samples)
        n_train = int(self.train_val_test[0] * n)
        n_val = int(self.train_val_test[1] * n)

        train_nodes = samples[:n_train]
        val_nodes = samples[n_train : n_train + n_val]
        test_nodes = samples[n_train + n_val :]

        torch.save(train_nodes, self.raw_paths[0])
        torch.save(val_nodes, self.raw_paths[1])
        torch.save(test_nodes, self.raw_paths[2])

        if self.export_txt:
            export_path = os.path.join(self.raw_dir, "epinions_neg_focused.txt")
            _export_samples_to_txt(samples, G, export_path)
            print(f"[Info] Exported negative-focused samples to '{export_path}'")


    # ----- 转换为 PyG Data 并缓存 -----
    def process(self):
        """
        将当前 split 的“节点集合列表”加载 → 读取原始图 → 构造节点特征映射 → 逐个生成 Data → 缓存
        """
        txt_path = os.path.join(self.raw_dir, "soc-sign-epinions.txt")
        if not os.path.exists(txt_path):
            # 兜底，确保 txt 存在（某些环境 download() 可能尚未运行）
            _download_if_needed(self.raw_dir)
        G = _read_signed_digraph(txt_path)
        node_x = _degree_bins(G, self.num_bins)

        file_idx = {"train": 0, "val": 1, "test": 2}
        nodes_lists: List[List[int]] = torch.load(self.raw_paths[file_idx[self.split]])

        data_list: List[Data] = []
        for nodes in nodes_lists:
            H = G.subgraph(set(nodes)).copy()  # 从有向原图取子图，保留方向与 sign
            data = _to_gar_pyg_data(H)
            if data is None:
                continue  # 过滤掉无边子图
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
        print(f"[Info] Processed split '{self.split}': {len(data_list)} graphs")


# -----------------------------
# DataModule（适配 DiGress）
# -----------------------------
class EpinionsDataModule(AbstractDataModule):
    def __init__(self, cfg):
        # 1. 接收 cfg
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        
        # 2. 定位路径
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        # 3. 从 cfg 中提取参数（这里会自动获取 debug 模式的覆盖值）
        # 使用 getattr 设置默认值，防止 config 里没写报错
        target_num_subgraphs = getattr(cfg.dataset, 'num_subgraphs', 2000)
        
        # 如果是 debug 模式，通常 num_subgraphs 会很小，这里无需额外处理，
        # 只要 hydra 正确加载了 debug.yaml，cfg.dataset.num_subgraphs 已经是小数值了。

        common_args = dict(
            root=root_path,
            max_nodes=getattr(cfg.dataset, 'max_nodes', 50),
            min_nodes=getattr(cfg.dataset, 'min_nodes', 10),
            num_bins=getattr(cfg.dataset, 'num_bins', 10),
            num_subgraphs=target_num_subgraphs,
            # 其他你需要的特定参数，比如随机游走参数
            # p=getattr(cfg.dataset, 'p', 2.0),
            # q=getattr(cfg.dataset, 'q', 4.0),
            neg_sample_ratio=getattr(cfg.dataset, 'neg_sample_ratio', 0.8),
            export_txt=True 
        )

        self.ds_train = EpinionsSubgraphDataset(split="train", **common_args)
        self.ds_val = EpinionsSubgraphDataset(split="val", **common_args)
        self.ds_test = EpinionsSubgraphDataset(split="test", **common_args)
        
        super().__init__(cfg, {"train": self.ds_train, "val": self.ds_val, "test": self.ds_test})
        self.inner = self.ds_train

    def __getitem__(self, idx):
        return self.inner[idx]


# -----------------------------
# DatasetInfos（适配 DiGress）
# -----------------------------
class EpinionsDatasetInfos(AbstractDatasetInfos):
    """
    注册数据分布与容量信息，供 DiGress 噪声模型与采样器使用
    - 节点数分布：来自 DataModule.node_counts()
    - 节点类型分布：这里为“度分箱 one-hot”的维度数（num_bins）；用训练集统计其出现比例即可
    - 边类型分布：DataModule.edge_counts()（slot 0 = non-edge，1.. = 真实边类型）
    """

    def __init__(self, datamodule, dataset_config=None):
        self.datamodule = datamodule
        self.name = "epinions_rw"

        # 统计分布
        self.n_nodes = self.datamodule.node_counts()       # P(N=n)
        self.node_types = self.datamodule.node_types()     # P(node class) —— 按度分箱 one-hot 的维度统计
        self.edge_types = self.datamodule.edge_counts()    # P(edge-type including non-edge)

        # 完成注册（含 max_n_nodes, num_classes, nodes_dist）
        super().complete_infos(self.n_nodes, self.node_types)

        # 注意：input_dims / output_dims 需要在构建模型前调用
        # infos.compute_input_output_dims(datamodule, extra_features, domain_features)
        # 其中 extra_features/domain_features 由你在训练脚本中提供（可为空特征）
