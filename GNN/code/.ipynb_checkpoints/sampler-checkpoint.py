import dgl

from dgl.dataloading import EdgeCollator

import torch
from torch.utils.data import DataLoader as TorchDataLoader

import random

from collections import defaultdict
import math

class HomoGraphDataSampler:
    def __init__(self, g: dgl.DGLGraph, pos_sampler, neg_sampler):
        self.pos_sampler = pos_sampler
        self.neg_sampler = neg_sampler
        self.g = g


    def construct_batch_data_sampler(self, eids, reverse_eids, batch_size=2048) -> TorchDataLoader:
        edge_collator = EdgeCollator(
            self.g, eids, self.pos_sampler,
            negative_sampler=self.neg_sampler,
            exclude='reverse_id',
            reverse_eids=reverse_eids
        )
        train_loader = TorchDataLoader(
            dataset=edge_collator.dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=edge_collator.collate,
            num_workers=0,
        )
        return train_loader
    
     # ================== 新增：按标签 1:1 采样 ==================
    def construct_balanced_batch_data_sampler(
        self,
        eids,
        reverse_eids,
        batch_size=2048,
        label_key="type_id",   # 或者 "label"，看你想按哪个字段来平衡
        seed=42,
    ) -> TorchDataLoader:
        """
        在给定的 eids 子集里，按 label_key (0/1) 做 1:1 采样，然后再交给 EdgeCollator。

        用法示例：
            sampler = HomoGraphDataSampler(g, pos_sampler, neg_sampler)
            train_loader = sampler.construct_balanced_batch_data_sampler(
                train_eids_full, reverse_eids,
                batch_size=train_batch_size,
                label_key="type_id"   # 或 "label"
            )
        """

        g = self.g

        # 1) 取出这些边的标签（0/1）
        labels_all = g.edata[label_key]          # shape [num_edges]
        labels_sub = labels_all[eids]           # 只看子集里的 label
        # 保证是 1D tensor
        if labels_sub.dim() > 1:
            labels_sub = labels_sub.view(-1)

        # 2) 找到子集中 0 / 1 的索引
        pos_mask = (labels_sub == 1)
        neg_mask = (labels_sub == 0)

        pos_eids = eids[pos_mask]
        neg_eids = eids[neg_mask]

        num_pos = pos_eids.shape[0]
        num_neg = neg_eids.shape[0]
        if num_pos == 0 or num_neg == 0:
            raise ValueError(
                f"在给定 eids 中，无法按 {label_key} 做 1:1 采样："
                f"正边数={num_pos}，负边数={num_neg}"
            )

        # 3) 取两者中的最小值，做 1:1 采样
        min_num = min(num_pos, num_neg)

        torch.manual_seed(seed)
        perm_pos = torch.randperm(num_pos)[:min_num]
        perm_neg = torch.randperm(num_neg)[:min_num]

        pos_sample = pos_eids[perm_pos]
        neg_sample = neg_eids[perm_neg]

        # 4) 拼在一起并打乱
        balanced_eids = torch.cat([pos_sample, neg_sample], dim=0)

        perm_all = torch.randperm(balanced_eids.shape[0])
        balanced_eids = balanced_eids[perm_all]

        print(
            f"[Balanced Sampler] 使用 {label_key} 做 1:1 采样："
            f"正边 {min_num}，负边 {min_num}，总共 {balanced_eids.shape[0]}"
        )

        # 5) 和原来一样，用 EdgeCollator + TorchDataLoader
        edge_collator = EdgeCollator(
            g,
            balanced_eids,
            self.pos_sampler,
            negative_sampler=self.neg_sampler,
            exclude='reverse_id',
            reverse_eids=reverse_eids,
        )

        train_loader = TorchDataLoader(
            dataset=edge_collator.dataset,
            batch_size=batch_size,
            shuffle=True,        # 这里建议 shuffle=True
            drop_last=False,
            collate_fn=edge_collator.collate,
            num_workers=0,
        )

        return train_loader


class HardNegEdgeSampler:
    def __init__(self, base_sampler, hard_neg_edge_pool: dict, k=1, replace_ratio=0.2, save_sample_edges=None):
        self.base_sampler = base_sampler  # 基础的采样
        self.k = k  # 每条边找几条负边
        self.hard_neg_edge_pool = hard_neg_edge_pool  # 在基础采样的基础上进行替换
        self.replace_ratio = replace_ratio  # 替换比例
        self.save_sample_edges = save_sample_edges # 保存采样的负边

    def __call__(self, g: dgl.DGLGraph, eids):
        # 返回(srcs, dsts)
        neg_srcs, neg_dsts = self.base_sampler(g, eids)
        # 要根据base_sampler采样完的结果来看，k不一定为1
        num_to_replace = int(len(neg_srcs) * self.replace_ratio)
        replace_idx = torch.randperm(len(neg_srcs))[:num_to_replace]
        k_eids = eids.repeat_interleave(self.k)  # 重复k次
        for i in replace_idx.tolist():
            pos_eid = k_eids[i].item()
            if pos_eid in self.hard_neg_edge_pool.keys() and len(self.hard_neg_edge_pool[pos_eid]) > 0:
                src, new_neg = random.choice(self.hard_neg_edge_pool[pos_eid])
                neg_srcs[i] = src
                neg_dsts[i] = new_neg
                if self.save_sample_edges is not None:
                    self.save_sample_edges.add((src, new_neg))
        return neg_srcs, neg_dsts



@torch.no_grad()
def generate_hard_negative_edge_pool_fast(
    G_dgl, model, node_feats, train_eids,
    band_low=0.80, band_high=0.95,   # 难度带：如 80%~95%
    semi_hard_margin=0.02,           # 半难约束：s_neg ≤ s_pos - margin
    alpha=3.0,                       # 软自对抗强度：权重 ∝ exp(alpha * s_hat)
    top_k=3,                         # 每条正边要的neg个数
    src_batch_size=1024,             # 源点批大小：按显存调
    device='cuda'
):
    device = torch.device(device)
    model.eval()
    G_dgl = G_dgl.to(device)

    # 1) 预计算节点表示并归一化
    h_all = model.encoder([G_dgl for _ in range(model.num_layers)], node_feats.to(device))
    h_all = torch.nn.functional.normalize(h_all, dim=-1)  # [N, d]
    N, d = h_all.shape

    # 2) CSR 邻接放 GPU，快速屏蔽邻居
    indptr, indices, _ = G_dgl.adj_sparse('csr')
    indptr  = indptr.to(device)
    indices = indices.to(device)

    # 3) 取训练边的 src/dst，并按 src 分组，避免重复算
    src_pos, dst_pos = G_dgl.find_edges(train_eids)
    src_pos   = src_pos.to(device, non_blocking=True)
    dst_pos   = dst_pos.to(device, non_blocking=True)
    train_eids = train_eids.to(device, non_blocking=True)

    by_src = defaultdict(list)
    for i in range(train_eids.numel()):
        by_src[int(src_pos[i])].append((int(train_eids[i]), int(dst_pos[i])))
    unique_src = torch.tensor(list(by_src.keys()), device=device, dtype=torch.long)

    # 4) 排名带的下标范围函数（基于 valid 数量，避免 quantile）
    def k_counts(n_valid):
        # 取“上 1 - band_low”的窗口，如 20%最高分
        k_max = max(1, int(math.ceil((1.0 - band_low)  * n_valid)))
        # 从“上 1 - band_high”开始切片，如 5%位置
        k_start = max(0, int(math.floor((1.0 - band_high) * n_valid)))
        return k_start, k_max

    # 5) 提升 GEMM 速度（可选）
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    pool = {}
    # 6) 按源点批量计算：sim = H @ H[src_batch].T  -> [N, B]
    for b in range(0, unique_src.numel(), src_batch_size):
        src_batch = unique_src[b:b+src_batch_size]             # [B]
        B = src_batch.numel()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            sim = h_all @ h_all[src_batch].T                   # [N, B]

        # 屏蔽自连
        sim[src_batch, torch.arange(B, device=device)] = -1e9

        # 屏蔽邻居（CSR 快速索引；小循环在 GPU 上成本可接受）
        for j in range(B):
            u = int(src_batch[j])
            beg, end = int(indptr[u]), int(indptr[u+1])
            if end > beg:
                nbrs = indices[beg:end]                        # [deg(u)]
                sim[nbrs, j] = -1e9

        # 对每个源点 u，先取“高分窗口”，再给这个 u 下的所有 (eid, dst) 复用
        for j in range(B):
            u = int(src_batch[j])
            col = sim[:, j]                                    # [N]
            valid_mask = col > -1e9
            n_valid = int(valid_mask.sum().item())
            if n_valid == 0:
                for (eid, _) in by_src[u]:
                    pool[eid] = []
                continue

            k_start, k_max = k_counts(n_valid)
            k_max = min(k_max, n_valid)

            # 取 top k_max（最大在前），再切 5%~20% 的带
            top_vals, top_idx = torch.topk(col, k=k_max, largest=True, sorted=True)
            band_vals = top_vals[k_start:]                     # [k_band]
            band_idx  = top_idx[k_start:]                      # [k_band]
            if band_idx.numel() == 0:
                for (eid, _) in by_src[u]:
                    pool[eid] = []
                continue

            # 软自对抗：对 band 内做 0-1 归一化，再做 exp(alpha * s_hat)
            vmin, vmax = band_vals.min(), band_vals.max()
            if float(vmax - vmin) > 1e-6:
                s_hat = (band_vals - vmin) / (vmax - vmin)
            else:
                s_hat = torch.zeros_like(band_vals)
            base_w = torch.exp(alpha * s_hat).clamp_min_(1e-12)
            base_w = base_w / (base_w.sum() + 1e-12)

            h_u = h_all[u]
            # 复用同一个窗口，针对 (eid, dst) 做“半难约束”
            for (eid, v_pos) in by_src[u]:
                s_pos = torch.dot(h_all[v_pos], h_u)
                keep = band_vals <= (s_pos - semi_hard_margin)
                cand_idx = band_idx[keep]
                if cand_idx.numel() == 0:
                    cand_idx = band_idx
                    w = base_w
                else:
                    w = base_w[keep]
                    w = w / (w.sum() + 1e-12)

                k_take = min(top_k, int(cand_idx.numel()))
                if k_take == 0:
                    pool[eid] = []
                    continue

                picked_local = torch.multinomial(w, num_samples=k_take, replacement=False)
                chosen = cand_idx[picked_local]
                pool[eid] = [(u, int(x)) for x in chosen.tolist()]

    return pool