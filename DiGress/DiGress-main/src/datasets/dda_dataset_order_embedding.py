import os
import math
import random
import signal

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx

from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset

from torch_geometric.data import Data, InMemoryDataset, Batch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from networkx.algorithms.isomorphism import GraphMatcher

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

# ==============================================================================
# 0. 全局配置 & GAR+ 谓词编码逻辑
# ==============================================================================

# --- 超时控制 ---
class TimeoutException(Exception): 
    pass

class TimeLimit:
    def __init__(self, seconds):
        self.seconds = seconds
        self.old_handler = signal.SIG_DFL

    def __enter__(self):
        if hasattr(signal, 'SIGALRM'):
            self.old_handler = signal.getsignal(signal.SIGALRM)
            def handler(signum, frame):
                raise TimeoutException()
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, type, value, traceback):
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.old_handler)

# --- 验证相关常量 ---
MATCH_LIMIT = 400  
TIME_LIMIT = 5
ML_THRESHOLD = 0.3



# --- [NEW] 节点特征编码映射 ---
import pandas as pd

# --- [NEW] 节点特征编码映射 ---
NODE_BIT_MAP = {
    'is_kinase': 0,          # +1
    'is_disease_related': 1  # +2
}

# --- [FIXED] 连续位的边特征编码映射（只保留你要的 key） ---
EDGE_BIT_MAP = {
    "is_negative": 0,           # +1  
    "location_match_y": 1,      # +2
    "M_sim": 2,                 # +4
    "physical_interaction": 3,  # +8
    "x_hub_degree": 4,          # +16
    "x_low_betweenness": 5,     # +32
    "Affinity_Capture_MS": 6,   # +64
}


# --- [MODIFIED] 特征维度定义 ---
#======给每个节点赋予一个类别的值============
NUM_NODE_CLASSES = 10
# NUM_EDGE_CLASSES = 1 + (2 ** len(EDGE_BIT_MAP))  # 1 + 2^7 = 129
NUM_EDGE_CLASSES = 0

EDGE_LABEL_MAPPING = None



def _norm_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


#compress bitmask
def is_negative_raw_bitmask(raw_bitmask: int) -> bool:
    raw_bitmask = int(raw_bitmask)
    return bool(raw_bitmask & (1 << EDGE_BIT_MAP["is_negative"]))


def collect_used_edge_bitmasks(graphs) -> list:
    """
    从一组 NetworkX 图中收集实际出现过的 raw bitmask（非零）。
    """
    used = set()
    for G in graphs:
        for _, _, d in G.edges(data=True):
            raw_val = int(d.get("raw_label", d.get("label", 0)))
            if raw_val != 0:
                used.add(raw_val)
    return sorted(used)


def build_edge_label_mapping(graphs):
    """
    构建压缩映射：
      raw bitmask -> compressed class id

    约定：
      0 = no-edge
      1..K = 实际出现过的 raw bitmask
    """
    used_masks = collect_used_edge_bitmasks(graphs)
    bitmask_to_class = {mask: i + 1 for i, mask in enumerate(used_masks)}
    class_to_bitmask = {i + 1: mask for i, mask in enumerate(used_masks)}

    return {
        "bitmask_to_class": bitmask_to_class,
        "class_to_bitmask": class_to_bitmask,
        "num_edge_classes": len(used_masks) + 1,
        "used_masks": used_masks,
    }


def compress_edge_label(raw_bitmask: int, bitmask_to_class: dict) -> int:
    """
    raw bitmask -> compressed class id
    """
    raw_bitmask = int(raw_bitmask)
    if raw_bitmask == 0:
        return 0
    if raw_bitmask not in bitmask_to_class:
        raise ValueError(f"Unknown raw edge bitmask: {raw_bitmask}")
    return int(bitmask_to_class[raw_bitmask])


def decompress_edge_label(edge_class: int, class_to_bitmask: dict) -> int:
    """
    compressed class id -> raw bitmask
    """
    edge_class = int(edge_class)
    if edge_class == 0:
        return 0
    if edge_class not in class_to_bitmask:
        raise ValueError(f"Unknown compressed edge class: {edge_class}")
    return int(class_to_bitmask[edge_class])


def print_edge_label_stats(graphs, mapping):
    raw_counter = Counter()
    comp_counter = Counter()

    for G in graphs:
        for _, _, d in G.edges(data=True):
            raw_counter[int(d.get("raw_label", 0))] += 1
            comp_counter[int(d.get("label", 0))] += 1

    print("==== Bitmask stats ====")
    print(f"不同 bitmask（谓词组合）数量 = {len([k for k in raw_counter if k != 0])}")
    print(f"总边数 = {sum(raw_counter.values())}")
    print(f"最常见的 10 种 bitmask = {raw_counter.most_common(10)}")

    print("==== Compressed edge label stats ====")
    print(f"used bitmasks = {mapping['used_masks']}")
    print(f"num compressed edge classes = {mapping['num_edge_classes']}")
    print(f"最常见的 10 种 compressed labels = {comp_counter.most_common(10)}")




# --- 辅助函数 ---
def map_loc_to_category(loc_str):
    """ 
    [Legacy] 用于保留 cat_idx 以便可视化或调试，不参与训练特征生成。
    """
    s = str(loc_str).lower()
    if s == 'nan' or s == '' or s == '-': return 9
    if 'nucleus' in s or 'nuclear' in s or 'nucleoplasm' in s: return 0
    if 'membrane' in s: return 2
    if 'mitochondri' in s: return 4
    if 'reticulum' in s: return 5
    if 'golgi' in s: return 6
    if 'lysosome' in s or 'peroxisome' in s or 'endosome' in s: return 7
    if 'secreted' in s or 'extracellular' in s: return 3
    if 'cytoplasm' in s or 'cytosol' in s: return 1
    return 8 

def encode_node_feature(node_attr: dict) -> int:
    """ 
    [NEW] 使用 location category 作为节点特征 (0-9)
    """
    # 优先取 canon 字段，其次是原始字段
    loc_raw = node_attr.get(
        "canon_Subcellular_location_CC",
        node_attr.get("Subcellular location [CC]", node_attr.get("location", ""))
    )
    return map_loc_to_category(loc_raw)

def encode_edge_feature(
    id_x,
    id_y,
    node_x_attr: dict,
    node_y_attr: dict,
    edge_row,
    global_stats: dict,
    sim_threshold: float = ML_THRESHOLD
) -> int:
    """
    连续位 bitmask 编码（只使用 EDGE_BIT_MAP 里的 key）：
      - location_match_y
      - M_sim            (edge_row["ml_score"] > sim_threshold)
      - physical_interaction
      - is_negative
      - x_hub_degree
      - x_low_betweenness
      - Affinity_Capture_MS
    """
    e_feature = 0

    # A) location_match_y：canon 位置严格相等
    loc_x = _norm_str(node_x_attr.get(
        "canon_Subcellular_location_CC",
        node_x_attr.get("Subcellular location [CC]", node_x_attr.get("location", ""))
    ))
    loc_y = _norm_str(node_y_attr.get(
        "canon_Subcellular_location_CC",
        node_y_attr.get("Subcellular location [CC]", node_y_attr.get("location", ""))
    ))
    if loc_x and (loc_x == loc_y):
        e_feature |= (1 << EDGE_BIT_MAP["location_match_y"])

    # B) x_hub_degree：x.degree >= global q75(degree)
    try:
        deg_x = float(node_x_attr.get("degree", 0.0))
    except Exception:
        deg_x = 0.0
    q75_deg = float(global_stats.get("degree", {}).get("q75", float("inf")))
    if deg_x >= q75_deg:
        e_feature |= (1 << EDGE_BIT_MAP["x_hub_degree"])

    # C) x_low_betweenness：x.betweenness <= global q25(betweenness)
    try:
        bet_x = float(node_x_attr.get("betweenness_centrality", 0.0))
    except Exception:
        bet_x = 0.0
    q25_bet = float(global_stats.get("betweenness_centrality", {}).get("q25", float("-inf")))
    if bet_x <= q25_bet:
        e_feature |= (1 << EDGE_BIT_MAP["x_low_betweenness"])

    # D/E) edge_row 相关邮票：M_sim / Affinity Capture-MS / physical / negative
    # if edge_row:
    if edge_row is not None:
        # D) M_sim：看 ml_score 是否过阈值
        try:
            ml_score = float(edge_row.get("ml_score", 0.0))
        except Exception:
            ml_score = 0.0
        if ml_score > sim_threshold:
            e_feature |= (1 << EDGE_BIT_MAP["M_sim"])

        # E1) Affinity Capture-MS
        exp_sys = _norm_str(edge_row.get("Experimental System", ""))
        if exp_sys == "Affinity Capture-MS":
            e_feature |= (1 << EDGE_BIT_MAP["Affinity_Capture_MS"])

        # E2) physical_interaction
        sys_type = str(edge_row.get("Experimental System Type", "")).lower()
        if "physical" in sys_type:
            e_feature |= (1 << EDGE_BIT_MAP["physical_interaction"])

        # E3) is_negative
        raw_type = str(edge_row.get("type", "positive")).lower()
        if ("negative" in raw_type) or (edge_row.get("is_negative") is True):
            e_feature |= (1 << EDGE_BIT_MAP["is_negative"])

    return int(e_feature)


# [REMOVED] get_gar_edge_index 函数已删除，因为它与新编码逻辑冲突

# ==============================================================================
# 1. 验证函数 (Validation)
# ==============================================================================
def edge_match(d1, d2):
    # GraphMatcher 用压缩后的训练标签做匹配
    return d1.get('label') == d2.get('label')


def node_match_fn(n1, n2):
    return n1.get('feature_val') == n2.get('feature_val')


def calculate_subgraph_metrics(subG, bigG):
    """
    计算子图在生成它的母图(BigG)中的 Support 和 Confidence。
    负边判断必须基于 raw_label，而不是压缩后的 label。
    """
    edges_to_test = list(subG.edges(data=True))

    candidates = []
    for u, v, d in edges_to_test:
        raw_val = d.get('raw_label', 0)
        if is_negative_raw_bitmask(raw_val):
            candidates.append((u, v))

    if not candidates:
        return None

    target_u, target_v = candidates[0]

    premiseG = subG.copy()
    premiseG.remove_edge(target_u, target_v)

    if premiseG.number_of_edges() == 0:
        return None

    GM = GraphMatcher(bigG, premiseG, node_match=node_match_fn, edge_match=edge_match)

    unique_matches = set()
    supp_neg = 0

    try:
        with TimeLimit(TIME_LIMIT):
            for mapping in GM.subgraph_isomorphisms_iter():
                real_u = mapping.get(target_u)
                real_v = mapping.get(target_v)

                if real_u is None or real_v is None:
                    continue

                edge_key = frozenset([real_u, real_v])
                if edge_key in unique_matches:
                    continue
                unique_matches.add(edge_key)

                if bigG.has_edge(real_u, real_v):
                    real_raw_label = bigG[real_u][real_v].get('raw_label', 0)
                    if is_negative_raw_bitmask(real_raw_label):
                        supp_neg += 1

                if len(unique_matches) >= MATCH_LIMIT:
                    break

    except TimeoutException:
        pass
    except Exception:
        return None

    supp_shape = len(unique_matches)
    conf = supp_neg / supp_shape if supp_shape > 0 else 0.0

    return {
        'conf': conf,
        'supp_neg': supp_neg,
        'supp_shape': supp_shape,
        'label': subG[target_u][target_v].get('label', 0),          # compressed
        'raw_label': subG[target_u][target_v].get('raw_label', 0),  # original bitmask
    }

# ==============================================================================
# Order-Embedding 模型与算法
# ==============================================================================

class GraphOrderEncoder(nn.Module):
    """
    子图 -> graph embedding
    GIN + global pooling
    输出非负 embedding，适合 order relation
    """
    def __init__(self, in_dim, hidden_dim=128, emb_dim=64, num_layers=3):
        super().__init__()
        self.node_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim),
        )

        self.out_act = nn.Softplus()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.node_proj(x)
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        g = global_mean_pool(x, batch)
        z = self.readout(g)
        z = self.out_act(z)
        return z


def order_energy(z_small, z_large):
    """
    E(A,B) = || max(0, phi(A)-phi(B)) ||^2
    """
    return torch.sum(F.relu(z_small - z_large) ** 2, dim=-1)


def order_embedding_loss(z1, z2, y, margin=1.0):
    """
    y=1: z1 对应图是 z2 的子图
    y=0: 不满足子图关系
    """
    e = order_energy(z1, z2)
    pos_loss = y * e
    neg_loss = (1.0 - y) * F.relu(margin - e)
    return (pos_loss + neg_loss).mean()


def induced_subgraph_by_node_drop(data, keep_ratio=0.7, min_nodes=3):
    """
    从一个 PyG Data 中随机删点，生成真正的正样本 A ⊆ B
    """
    num_nodes = data.num_nodes
    keep_num = max(min_nodes, int(num_nodes * keep_ratio))

    if keep_num >= num_nodes or keep_num < min_nodes:
        return None

    keep_nodes = sorted(random.sample(range(num_nodes), keep_num))
    keep_set = set(keep_nodes)
    old_to_new = {old: new for new, old in enumerate(keep_nodes)}

    edge_index = data.edge_index
    new_edges = []
    kept_eids = []

    for eid in range(edge_index.size(1)):
        u = int(edge_index[0, eid])
        v = int(edge_index[1, eid])
        if u in keep_set and v in keep_set:
            new_edges.append([old_to_new[u], old_to_new[v]])
            kept_eids.append(eid)

    if len(new_edges) == 0:
        return None

    new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
    new_x = data.x[keep_nodes]
    new_n = len(keep_nodes)
    new_data = Data(
        x=new_x,
        edge_index=new_edge_index,
        # n_nodes=len(keep_nodes),
        num_nodes = new_n,
        n_nodes=torch.tensor([new_n], dtype=torch.long),
        y=torch.zeros(1, 0).float()
    )

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        new_data.edge_attr = data.edge_attr[kept_eids]

    if hasattr(data, "edge_label_mask") and data.edge_label_mask is not None:
        new_data.edge_label_mask = data.edge_label_mask[kept_eids]

    return new_data


class OrderPairDataset(Dataset):
    """
    动态构造训练对:
      正样本: (small_subgraph, original_graph, 1)
      负样本: (random_graph_i, random_graph_j, 0)
    """
    def __init__(self, graph_list, pos_ratio=0.5, min_keep=0.5, max_keep=0.9):
        self.graph_list = graph_list
        self.pos_ratio = pos_ratio
        self.min_keep = min_keep
        self.max_keep = max_keep

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        if random.random() < self.pos_ratio:
            large = self.graph_list[idx]
            small = None
            for _ in range(10):
                keep_ratio = random.uniform(self.min_keep, self.max_keep)
                small = induced_subgraph_by_node_drop(
                    large, keep_ratio=keep_ratio, min_nodes=3
                )
                if small is not None and small.num_nodes < large.num_nodes:
                    break

            if small is None:
                j = random.randrange(len(self.graph_list))
                while j == idx:
                    j = random.randrange(len(self.graph_list))
                return self.graph_list[j], self.graph_list[idx], torch.tensor(0.0)

            return small, large, torch.tensor(1.0)

        j = random.randrange(len(self.graph_list))
        while j == idx:
            j = random.randrange(len(self.graph_list))
        return self.graph_list[idx], self.graph_list[j], torch.tensor(0.0)


def order_collate_fn(batch):
    g1_list, g2_list, y_list = zip(*batch)
    b1 = Batch.from_data_list(list(g1_list))
    b2 = Batch.from_data_list(list(g2_list))
    y = torch.stack(list(y_list), dim=0).float()
    return b1, b2, y


def train_order_encoder(
    graph_list,
    in_dim,
    hidden_dim=128,
    emb_dim=64,
    num_layers=3,
    batch_size=32,
    lr=1e-3,
    epochs=10,
    margin=1.0,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GraphOrderEncoder(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        emb_dim=emb_dim,
        num_layers=num_layers,
    ).to(device)

    dataset = OrderPairDataset(graph_list, pos_ratio=0.5)

    loader = TorchDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=order_collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_num = 0

        for b1, b2, y in loader:
            b1 = b1.to(device)
            b2 = b2.to(device)
            y = y.to(device)

            z1 = model(b1)
            z2 = model(b2)

            loss = order_embedding_loss(z1, z2, y, margin=margin)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_num += bs

        print(f"[OrderTrain] Epoch {epoch:03d} | loss={total_loss / max(total_num,1):.6f}")

    return model


@torch.no_grad()
def compute_graph_embeddings(model, graph_list, batch_size=64, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    loader = PyGDataLoader(
        graph_list,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: Batch.from_data_list(batch),
    )

    all_embs = []
    for batch in loader:
        batch = batch.to(device)
        z = model(batch)
        all_embs.append(z.cpu())

    return torch.cat(all_embs, dim=0)


def dominates(e1, e2, eps=1e-9):
    return np.all(e1 <= e2 + eps)


def remove_dominated_embeddings(embs):
    n = len(embs)
    keep = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            # per coordinate less than
            if dominates(embs[i], embs[j]):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


# def embedding_volume(e):
#     return float(np.prod(e))
def embedding_volume(e, eps=1e-8):
    e = np.asarray(e, dtype=np.float64)
    return float(np.sum(np.log(e + eps)))

# def compute_thresholds(embs, sigma, chi=1.0):
#     """
#     Threshold[i] = 第 i 维 top-(chi*sigma) 大值中的最小值
#     论文要求 chi in [0,1]
#     """
#     embs = np.asarray(embs, dtype=np.float32)
#     n, d = embs.shape

#     if not (0 < chi <= 1.0):
#         raise ValueError(f"chi should be in (0,1], got {chi}")

#     m = max(1, min(n, int(math.ceil(chi * sigma))))

#     thresholds = np.zeros(d, dtype=np.float32)
#     for i in range(d):
#         vals = np.sort(embs[:, i])[::-1]
#         thresholds[i] = vals[m - 1]

#     return thresholds


import heapq
def compute_thresholds(embs, sigma, chi=1.0):
    """
    按论文思路计算 Threshold[i]:
    Threshold[i] = 第 i 维 top-(chi*sigma) 大值中的最小值

    参数
    ----
    embs : array-like, shape (n, d)
        所有 embedding
    sigma : int
        论文中的 sigma
    chi : float, default=1.0
        论文中的 chi, 通常 in [0, 1]

    返回
    ----
    thresholds : np.ndarray, shape (d,)
    """
    embs = np.asarray(embs, dtype=np.float32)
    n, d = embs.shape

    if sigma <= 0:
        raise ValueError(f"sigma should be positive, got {sigma}")
    if not (0 < chi <= 1.0):
        raise ValueError(f"chi should be in (0,1], got {chi}")

    # 这里沿用你之前的实现约定：ceil(chi * sigma)
    m = max(1, min(n, int(math.ceil(chi * sigma))))

    thresholds = np.zeros(d, dtype=np.float32)

    # 对每一维维护一个大小最多为 m 的最小堆
    # 堆里始终保存“当前 top-m 大值”
    for i in range(d):
        heap = []

        for e in embs:
            ai = float(e[i])

            # (i) 若 |S_th^i| < m，则直接加入
            if len(heap) < m:
                heapq.heappush(heap, ai)
            # 否则若 ai > Threshold[i]（也就是当前堆顶最小值），替换之
            elif ai > heap[0]:
                heapq.heapreplace(heap, ai)

        # (ii) Threshold[i] 定义为 S_th^i 中最小值
        thresholds[i] = heap[0]

    return thresholds

def select_graphs(
    embs,
    method="topk",
    k=1000,
    seed=42,
    sigma=10,
    chi=0.7,
):
    """
    Unified graph selector.

    Args:
        embs: [N, d] embedding matrix
        method: "pickpatterns" | "topk" | "fps" | "random"
        k: number of graphs to select
        seed: random seed
        sigma, chi: only used by pickpatterns
    """

    if method == "pickpatterns":
        print("[Selector] Using PickPatterns")
        return pick_patterns(
            embs=embs,
            k=k,
            sigma=sigma,
            chi=chi,
        )

    elif method == "topk":
        print("[Selector] Using Top-K volume")
        return select_topk_embeddings(embs, k)

    elif method == "fps":
        print("[Selector] Using Farthest Point Sampling")
        return select_by_fps(embs, k, seed)

    elif method == "random":
        print("[Selector] Using Random sampling")
        return select_random(embs, k, seed)

    else:
        raise ValueError(f"Unknown selector: {method}")
    
def select_topk_embeddings(embs, k):
    embs = np.asarray(embs, dtype=np.float32)
    n = len(embs)

    volumes = np.array([embedding_volume(e) for e in embs], dtype=np.float32)
    order = np.argsort(-volumes)

    k = min(k, n)

    print("Top volume:", volumes[order[:5]])
    print("Total graphs:", n)
    print("Selected:", k)

    return order[:k].tolist()

def select_by_fps(embs, k, seed=42):

    embs = np.asarray(embs, dtype=np.float32)
    n = len(embs)

    if k >= n:
        return list(range(n))

    rng = np.random.default_rng(seed)

    selected = []
    first = int(rng.integers(0, n))
    selected.append(first)

    dist = np.full(n, np.inf)

    for _ in range(1, k):

        last = selected[-1]

        d = np.sum((embs - embs[last]) ** 2, axis=1)
        dist = np.minimum(dist, d)

        dist[selected] = -1

        nxt = int(np.argmax(dist))

        if dist[nxt] < 0:
            break

        selected.append(nxt)

    return selected

import random

def select_random(embs, k, seed=42):
    n = len(embs)
    print("len(embs):", n)
    if k >= n:
        return list(range(n))

    random.seed(seed)
    return random.sample(range(n), k)


def pick_patterns(
    embs,
    k,
    sigma,
    chi=1.0,
    remove_dominated=True,
    dominated_keep_ratio=0.0,
    verbose=True,
    debug_topk=10,
):
    """
    PickPatterns with optional relaxed dominated filtering + debug prints.
    """
    embs = np.asarray(embs, dtype=np.float32)
    n = len(embs)

    if n == 0:
        return []

    print("=" * 80)
    print(f"[PickPatterns] total embeddings = {n}")
    print(f"[PickPatterns] remove_dominated = {remove_dominated}")
    print(f"[PickPatterns] dominated_keep_ratio = {dominated_keep_ratio}")
    print(f"[PickPatterns] k = {k}, sigma = {sigma}, chi = {chi}")

    # -----------------------------
    # Step 1: dominated filtering
    # -----------------------------
    if remove_dominated:
        nd_idx = remove_dominated_embeddings(embs)
        print(f"[Step1] left embedding count = {len(nd_idx)}")

        if dominated_keep_ratio <= 0.0:
            remain_idx = nd_idx
            print(f"[Step1] strict paper mode, remain = {len(remain_idx)}")
        else:
            nd_set = set(nd_idx)
            dom_idx = [i for i in range(n) if i not in nd_set]
            extra_num = int(len(dom_idx) * dominated_keep_ratio)

            print(f"[Step1] dominated count = {len(dom_idx)}")
            print(f"[Step1] extra dominated kept = {extra_num}")

            if extra_num > 0:
                dom_vols = np.array([embedding_volume(embs[i]) for i in dom_idx], dtype=np.float32)
                order_dom = np.argsort(-dom_vols)
                extra_idx = [dom_idx[i] for i in order_dom[:extra_num]]
            else:
                extra_idx = []

            remain_idx = nd_idx + extra_idx
            print(f"[Step1] remain after relaxed filtering = {len(remain_idx)}")
    else:
        remain_idx = list(range(n))
        print(f"[Step1] skipped dominated filtering, remain = {len(remain_idx)}")

    remain_embs = embs[remain_idx]

    if len(remain_embs) == 0:
        print("[PickPatterns] remain_embs is empty")
        return []

    # -----------------------------
    # Step 2: sort by volume desc
    # -----------------------------
    volumes = np.array([embedding_volume(e) for e in remain_embs], dtype=np.float32)
    order = np.argsort(-volumes)

    remain_embs = remain_embs[order]
    remain_idx = [remain_idx[i] for i in order]
    volumes = volumes[order]

    print(f"[Step2] remain_embs sorted by volume, count = {len(remain_embs)}")
    print(f"[Step2] volume stats: min={volumes.min():.4f}, max={volumes.max():.4f}, "
          f"mean={volumes.mean():.4f}, median={np.median(volumes):.4f}")
    print(f"[Step2] top-{min(debug_topk, len(volumes))} volumes = {volumes[:debug_topk]}")

    # -----------------------------
    # Step 3: thresholds
    # -----------------------------
    thresholds = compute_thresholds(remain_embs, sigma=sigma, chi=chi)

    print(f"[Step3] threshold shape = {thresholds.shape}")
    print(f"[Step3] threshold stats: min={thresholds.min():.6f}, "
          f"max={thresholds.max():.6f}, mean={thresholds.mean():.6f}, "
          f"median={np.median(thresholds):.6f}")
    print(f"[Step3] first-{min(debug_topk, len(thresholds))} thresholds = {thresholds[:debug_topk]}")

    # 看 embedding 相对 threshold 的关系
    num_all_above = 0
    num_all_below = 0
    num_partial = 0

    for e in remain_embs:
        le_mask = (e <= thresholds)
        if np.all(le_mask):
            num_all_below += 1
        elif np.any(le_mask):
            num_partial += 1
        else:
            num_all_above += 1

    print(f"[Step3] embeddings all <= threshold: {num_all_below}")
    print(f"[Step3] embeddings partially <= threshold: {num_partial}")
    print(f"[Step3] embeddings all > threshold: {num_all_above}")

    # -----------------------------
    # Step 4: greedy cover
    # -----------------------------
    d = remain_embs.shape[1]
    cover = np.zeros(d, dtype=np.float32)
    selected = []

    reject_cover_only = 0
    reject_threshold_only = 0
    reject_both = 0
    accept_count = 0

    print(f"[Step4] before select: {len(remain_idx)} candidates")

    for t, (idx, e) in enumerate(zip(remain_idx, remain_embs)):
        gt_cover = (e > cover)
        le_thresh = (e <= thresholds)

        cond = gt_cover & le_thresh
        should_add = np.any(cond)

        if should_add:
            selected.append(idx)
            # cover = np.maximum(cover, e)
            cover = np.maximum(cover, cover + 0.3 * (e - cover))
            accept_count += 1

            if verbose and (accept_count <= debug_topk):
                print(f"[ACCEPT {accept_count}] idx={idx}, step={t}, "
                      f"num_dims_gt_cover={gt_cover.sum()}, "
                      f"num_dims_le_thresh={le_thresh.sum()}, "
                      f"num_dims_valid={(cond).sum()}, "
                      f"cover_mean={cover.mean():.6f}, cover_max={cover.max():.6f}")
        else:
            has_gt_cover = np.any(gt_cover)
            has_le_thresh = np.any(le_thresh)

            if has_gt_cover and not has_le_thresh:
                reject_threshold_only += 1
            elif (not has_gt_cover) and has_le_thresh:
                reject_cover_only += 1
            else:
                reject_both += 1

            if verbose and (t < debug_topk):
                print(f"[REJECT] idx={idx}, step={t}, "
                      f"num_dims_gt_cover={gt_cover.sum()}, "
                      f"num_dims_le_thresh={le_thresh.sum()}, "
                      f"num_dims_valid={(cond).sum()}")

        if len(selected) >= k:
            print(f"[Step4] reached k={k}, terminate early")
            break

    print(f"[Step4] after select: {len(selected)}")
    print(f"[Step4] accepted = {accept_count}")
    print(f"[Step4] rejected by threshold-only = {reject_threshold_only}")
    print(f"[Step4] rejected by cover-only = {reject_cover_only}")
    print(f"[Step4] rejected by both/other = {reject_both}")
    print(f"[Step4] final cover stats: min={cover.min():.6f}, max={cover.max():.6f}, "
          f"mean={cover.mean():.6f}, median={np.median(cover):.6f}")
    print("=" * 80)

    return selected


@torch.no_grad()
def sanity_check_order(model, graph_list, num_trials=10, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval().to(device)
    ok = 0
    total = 0

    for _ in range(num_trials):
        g_large = random.choice(graph_list)
        g_small = induced_subgraph_by_node_drop(g_large, keep_ratio=0.7, min_nodes=3)
        if g_small is None:
            continue

        b_small = Batch.from_data_list([g_small]).to(device)
        b_large = Batch.from_data_list([g_large]).to(device)

        z_small = model(b_small)
        z_large = model(b_large)

        e1 = order_energy(z_small, z_large).item()
        e2 = order_energy(z_large, z_small).item()

        print(f"[Sanity] E(small,large)={e1:.4f}, E(large,small)={e2:.4f}")
        total += 1
        if e1 < e2:
            ok += 1

    print(f"[Sanity] pass rate: {ok}/{max(total,1)}")

# ==============================================================================
# 2. Dataset 类定义
# ==============================================================================

class DDAGraphDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        split='train',
        stage='final',   # 'raw' or 'final'
        transform=None,
        pre_transform=None,
        pre_filter=None,
        num_subgraphs=2000,
        min_nodes=3,
        max_nodes=5
    ):
        self.num_subgraphs = num_subgraphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.split = split
        self.stage = stage

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['drug_disease_with_semantic.csv', 'disease.csv','drug.csv']

    # @property
    # def processed_file_names(self):
    #     return [f'ppi_{self.split}.pt']
    # 区分两个阶段的训练数据
    @property
    def processed_file_names(self):
        if self.stage == 'raw':
            return [f'dda_{self.split}_raw.pt']
        return [f'dda_{self.split}.pt']

    def download(self):
        pass 

    def process(self):
        dda_path = os.path.join(self.root, 'raw', 'drug_disease_with_semantic.csv')
        drug_path = os.path.join(self.root, 'raw', 'drug.csv')
        disease_path = os.path.join(self.root, 'raw', 'disease.csv')
        
        # --- A. 加载 Metadata ---
        print(f"Loading metadata from {meta_path}...")
        id_to_attrs = {} 
        try:
            df_meta = pd.read_csv(meta_path, low_memory=False)
            df_meta.columns = df_meta.columns.str.strip()
            id_col = 'biogrid_id'
            
            df_meta[id_col] = pd.to_numeric(df_meta[id_col], errors='coerce')
            df_meta = df_meta.dropna(subset=[id_col])
            
            for _, row in df_meta.iterrows():
                bid = str(int(row[id_col]))
                attr_dict = row.to_dict()
                loc_raw = row.get('location', '')
                attr_dict['cat_idx'] = map_loc_to_category(loc_raw) # 仅用于Legacy显示
                id_to_attrs[bid] = attr_dict
            print(f"Metadata map built. Covered proteins: {len(id_to_attrs)}")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return

                # --- B. 构建大图 ---
        print(f"Loading PPI structure from {dda_path}...")
        bigG = nx.Graph()
        try:
            df_ppi = pd.read_csv(dda_path, sep=',' if ',' in open(dda_path).readline() else '\t')
            df_ppi.columns = df_ppi.columns.str.strip()

            # 第一遍：只加点、加边，边 label 先占位 0
            for _, row in df_ppi.iterrows():
                u_bid = str(row.get('chemical_index', '')).split('.')[0]
                v_bid = str(row.get('disease_index', '')).split('.')[0]
                if not u_bid or not v_bid:
                    continue

                u_attrs = id_to_attrs.get(u_bid, {})
                v_attrs = id_to_attrs.get(v_bid, {})

                loc_u = u_attrs.get('location', '')
                loc_v = v_attrs.get('location', '')

                u_enc = map_loc_to_category(loc_u)   # 0..9
                v_enc = map_loc_to_category(loc_v)


                # 添加节点
                bigG.add_node(u_bid, **u_attrs, feature_val=u_enc)
                bigG.add_node(v_bid, **v_attrs, feature_val=v_enc)

                edge_data = row.to_dict()

                # 先占位，第二遍再用全局统计 + GNN + Affinity Capture-MS 规则重算
                # bigG.add_edge(u_bid, v_bid, label=0, **edge_data)
                bigG.add_edge(u_bid, v_bid, raw_label=0, label=0, **edge_data)

            # --- B2. 计算节点统计量与全局分位数（给 x.degree / x.betweenness 的谓词用）---
            deg_dict = dict(bigG.degree())
            nx.set_node_attributes(bigG, deg_dict, "degree")

            bet_dict = nx.betweenness_centrality(bigG,k=256,seed=42)
            nx.set_node_attributes(bigG, bet_dict, "betweenness_centrality")

            deg_vals = np.array(list(deg_dict.values()), dtype=float)
            bet_vals = np.array(list(bet_dict.values()), dtype=float)
            global_stats = {
                "degree": {"q75": float(np.quantile(deg_vals, 0.75))},
                "betweenness_centrality": {"q25": float(np.quantile(bet_vals, 0.25))}
            }

            # --- B3. 准备 pred_scores（没有就先空着，不会影响流程）---
            pred_scores = {"GNN": {}}
            GNN_THRESHOLD = 0.5

            # --- B4. 第二遍：重算边 label（严格按你的谓词策略）---
            for u, v, d in bigG.edges(data=True):
                ux = bigG.nodes[u]
                vy = bigG.nodes[v]

                raw_label = int(
                    encode_edge_feature(
                        id_x=u,
                        id_y=v,
                        node_x_attr=ux,
                        node_y_attr=vy,
                        edge_row=d,
                        global_stats=global_stats
                    )
                )

                d["raw_label"] = raw_label
                d["label"] = 0   # 先占位，后面统一压缩

        except Exception as e:
            print(f"Error building graph: {e}")
            import traceback
            traceback.print_exc()
            return

        print(f"Big Graph loaded. Nodes: {bigG.number_of_nodes()}, Edges: {bigG.number_of_edges()}")

    
        # --- C. 扫描负边 ---
        neg_edge_nodes = set()
        print("Scanning for negative edges to prioritize sampling...")
        for u, v, d in bigG.edges(data=True):
            raw_val = d.get('raw_label', 0)
            if is_negative_raw_bitmask(raw_val):
                neg_edge_nodes.add(u)
        neg_edge_nodes.add(v)
                
        neg_edge_nodes = list(neg_edge_nodes)
        print(f"Found {len(neg_edge_nodes)} nodes involved in negative edges.")
        
        if len(neg_edge_nodes) == 0:
            print("Warning: No negative edges found! Sampling will be purely random.")

        # --- D. 采样与转换 ---
        nx_subgraphs = []
        data_list = []
        all_nodes = list(bigG.nodes())

        print(f"Sampling {self.num_subgraphs} subgraphs (Strategy: Negative-Centric)...")

        pbar = tqdm(total=self.num_subgraphs)
        high_support_cnt = 0
        neg_sub_cnt = 0

        while len(nx_subgraphs) < self.num_subgraphs:
            use_biased_sampling = (len(neg_edge_nodes) > 0) and (np.random.rand() < 0.8)
            seed = np.random.choice(neg_edge_nodes) if use_biased_sampling else np.random.choice(all_nodes)

            target_size = np.random.randint(self.min_nodes, self.max_nodes + 1)
            sub_nodes = [seed]

            try:
                bfs_successors = dict(nx.bfs_successors(bigG, seed))
                queue = [seed]
                steps = 0
                while len(sub_nodes) < target_size and queue and steps < 1000:
                    curr = queue.pop(0)
                    steps += 1
                    if curr in bfs_successors:
                        neighbors = bfs_successors[curr]
                        np.random.shuffle(neighbors)
                        for n in neighbors:
                            if n not in sub_nodes:
                                sub_nodes.append(n)
                                queue.append(n)
                                if len(sub_nodes) >= target_size:
                                    break
            except Exception:
                pass

            if len(sub_nodes) < self.min_nodes:
                continue

            G_sub = bigG.subgraph(sub_nodes).copy()

            has_neg = False
            for _, _, d in G_sub.edges(data=True):
                raw_val = d.get('raw_label', 0)
                if is_negative_raw_bitmask(raw_val):
                    has_neg = True
                    break

            # if has_neg:
            #     neg_sub_cnt += 1
            #     metrics = calculate_subgraph_metrics(G_sub, bigG)
            #     if metrics and metrics['supp_shape'] >= MATCH_LIMIT:
            #         high_support_cnt += 1

            G_sub = nx.convert_node_labels_to_integers(G_sub, label_attribute='orig_symbol')
            nx_subgraphs.append(G_sub)
            pbar.update(1)

        print(f"*Overall neg sub pattern: {neg_sub_cnt}")
        print(f"*Overall high support pattern: {high_support_cnt}")
        pbar.close()

        if len(nx_subgraphs) == 0:
            print("[FATAL] No sampled subgraphs collected.")
            return

        # ===== 建立压缩映射 =====
        global EDGE_LABEL_MAPPING, NUM_EDGE_CLASSES
        EDGE_LABEL_MAPPING = build_edge_label_mapping(nx_subgraphs)
        NUM_EDGE_CLASSES = EDGE_LABEL_MAPPING["num_edge_classes"]

        print_edge_label_stats(nx_subgraphs, EDGE_LABEL_MAPPING)

        # ===== 把每个子图里的 raw_label -> compressed label =====
        for G_sub in nx_subgraphs:
            for u, v, d in G_sub.edges(data=True):
                raw_label = int(d.get("raw_label", 0))
                d["label"] = compress_edge_label(
                    raw_label,
                    EDGE_LABEL_MAPPING["bitmask_to_class"]
                )

        # ===== 再转 PyG =====
        for G_sub in nx_subgraphs:
            pyg_data = self._to_pyg_data(G_sub)
            if pyg_data is not None:
                data_list.append(pyg_data)

        if len(data_list) == 0:
            print("[FATAL] No PyG subgraphs generated after compression.")
            return

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"[Done] Saved sampled subgraphs: {len(data_list)} -> {self.processed_paths[0]}")

    def _to_pyg_data(self, G):
        global NUM_EDGE_CLASSES

        if NUM_EDGE_CLASSES <= 0:
            raise ValueError("NUM_EDGE_CLASSES is not initialized before _to_pyg_data.")

        # 1) 节点特征：feature_val (0..9) -> 10维 one-hot
        xs = [G.nodes[n].get('feature_val', 9) for n in G.nodes()]
        x_idx = torch.tensor(xs, dtype=torch.long)
        x = F.one_hot(x_idx, num_classes=NUM_NODE_CLASSES).float()

        # 2) 构造有向边 + 特征
        src, dst = [], []
        edge_type_ids = []
        edge_bitmasks = []

        cand = []  # (s, t, edge_class, raw_bitmask)
        for u, v, d in G.edges(data=True):
            raw_bitmask = int(d.get('raw_label', 0))
            edge_class = int(d.get('label', 0))   # 压缩后的类别 id

            # 无向图 -> 双向边
            cand.append((u, v, edge_class, raw_bitmask))
            cand.append((v, u, edge_class, raw_bitmask))

        if not cand:
            return None

        seen = set()
        dup = 0
        self_loops = 0

        for s, t, edge_class, raw_bitmask in cand:
            s = int(s)
            t = int(t)

            if s == t:
                self_loops += 1
                continue

            key = (s, t)
            if key in seen:
                dup += 1
                continue
            seen.add(key)

            src.append(s)
            dst.append(t)
            edge_type_ids.append(edge_class)
            edge_bitmasks.append(raw_bitmask)

        if len(src) == 0:
            return None

        if self_loops > 0:
            print(f"[WARN] removed self-loops: {self_loops}")
        if dup > 0:
            print(f"[WARN] duplicate directed edges removed: {dup}")

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_type_ids = torch.tensor(edge_type_ids, dtype=torch.long)

        # 压缩后的 one-hot，不再是固定 129
        edge_attr = F.one_hot(edge_type_ids, num_classes=NUM_EDGE_CLASSES).float()

        # 保留原始 raw bitmask 供 debug / rule analysis
        edge_label_mask = torch.tensor(edge_bitmasks, dtype=torch.long)

        n = G.number_of_nodes()
        if edge_index.size(1) > n * (n - 1):
            print(f"[FATAL] edges > n*(n-1): n={n}, E={edge_index.size(1)}")
            return None

        y = torch.zeros(1, 0).float()
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_label_mask=edge_label_mask,  # 这里存原始 bitmask
            n_nodes=torch.tensor([n], dtype=torch.long),
            num_nodes=n,
            y=y
        )




# ==============================================================================
# 3. DataModule & Infos (保持基本不变)
# ==============================================================================

class DDADataModule(AbstractDataModule):
    def __init__(self, cfg):
        current_file_path = os.path.realpath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        abs_datadir = os.path.join(project_root, 'data', 'DDA')
        self.datadir = abs_datadir
        print(f"[Info] Absolute Data Directory: {self.datadir}")

        final_path = os.path.join(self.datadir, "processed", "dda_train.pt")
        if not os.path.exists(final_path):
            raise FileNotFoundError(
                f"Final DDA training set not found: {final_path}\n"
                f"Please run: python src/preprocess/train_dda_order_and_pick.py"
            )

        base_dataset = DDAGraphDataset(
            root=self.datadir,
            split='train',
            stage='final',
            num_subgraphs=cfg.dataset.num_subgraphs,
            min_nodes=cfg.dataset.min_nodes,
            max_nodes=cfg.dataset.max_nodes,
        )

        datasets = {'train': base_dataset, 'val': base_dataset, 'test': base_dataset}
        super().__init__(cfg, datasets)

class DDADatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'ppi'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        print(">>> 真实数据节点分布:", self.node_types)
        self.edge_types = self.datamodule.edge_counts()
        print(">>> Edge type distribution:", self.edge_types)
        print(">>> Edge type dim:", len(self.edge_types))

        super().complete_infos(self.n_nodes, self.node_types)