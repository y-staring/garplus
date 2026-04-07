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
