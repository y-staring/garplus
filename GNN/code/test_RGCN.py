import torch
import numpy as np
import random
import dgl
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def reset_seed(seed):
    np.random.seed(seed)  # 设置NumPy随机种子
    random.seed(seed)  # 设置Python随机种子

    torch.manual_seed(seed)
    dgl.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况

    # torch.backends.cudnn.deterministic = True

seed = 42
reset_seed(seed)


import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv


class RGCNEncoder(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, num_rels, num_layers=1, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        self.layers.append(RelGraphConv(in_feats, h_feats, num_rels, activation=F.relu))
        if num_layers > 2:
            for _ in range(num_layers - 2):
                self.layers.append(RelGraphConv(h_feats, h_feats, num_rels, activation=F.relu))
        self.layers.append(RelGraphConv(h_feats, out_feats, num_rels))  # 最后一层不激活

    def forward(self, blocks, feat):
        h = feat
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            etypes = block.edata['type_id']
            h = layer(block, h, etypes)
            if l != len(self.layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h

#decoder：拼接 + MLP
class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, emb, src, dst):
        h_src = emb[src]
        h_dst = emb[dst]
        h_cat = torch.cat([h_src, h_dst], dim=1)
        return self.mlp(h_cat).squeeze()  # [B]，logits（不加 sigmoid）

class RGCNModel(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, num_rels, num_layers=1, dropout=0.2):
        super().__init__()
        self.num_layers = num_layers
        self.encoder = RGCNEncoder(in_feats, h_feats, out_feats, num_rels, num_layers, dropout)
        self.decoder = MLPDecoder(out_feats)

    def forward(self, blocks, feats, src, dst):
        emb = self.encoder(blocks, feats)  # shape [N, out_feats]
        return self.decoder(emb, src, dst)  # shape [B]

    def print_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param}")

from dgl.nn.pytorch import GATConv


class GATEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, num_layers=1, dropout=0.2):
        super(GATEncoder, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop=0, attn_drop=0))
        if num_layers > 2:
            for _ in range(num_layers - 2):
                self.layers.append(GATConv(hidden_dim, hidden_dim, 1, feat_drop=0, attn_drop=0))
        self.layers.append(GATConv(hidden_dim, out_dim, 1))

    def forward(self, blocks, feat):
        h = feat
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            h = torch.mean(h, dim=1)  # 对各个head的结果求平均
            if l != len(self.layers) - 1:
                h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, num_layers=1, dropout=0.2):
        super(GATModel, self).__init__()
        self.num_layers = num_layers
        self.encoder = GATEncoder(in_dim, hidden_dim, out_dim, num_heads, num_layers, dropout)
        self.decoder = MLPDecoder(out_dim)

    def forward(self, blocks, feats, src, dst):
        emb = self.encoder(blocks, feats)  # shape [N, out_feats]
        return self.decoder(emb, src, dst)  # shape [B]

    def print_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param}")



from dgl.nn.pytorch import HGTConv

class HGTEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, head_size, num_ntypes, num_etypes, num_layers=1, dropout=0.2):
        super(HGTEncoder, self).__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(HGTConv(in_dim, head_size, num_heads, num_ntypes, num_etypes, dropout=dropout))
        if num_layers > 2:
            for _ in range(num_layers - 2):
                self.layers.append(HGTConv(head_size*num_heads, head_size, num_heads, num_ntypes, num_etypes, dropout=dropout))
        self.layers.append(HGTConv(head_size*num_heads, out_dim, 1, num_ntypes, num_etypes, use_norm=True))

    def forward(self, blocks, feat):
        h = feat
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # print(block.ndata['type_id'])
            # print(block.edata['type_id'])
            # print(block.ndata['type_id'])
            if type(block.ndata['type_id']) == dict:
                # fixme:为什么会有两种类型呢
                h = layer(block, h, block.ndata['type_id']['_N'], block.edata['type_id'])
            else:
                 h = layer(block, h, block.ndata['type_id'], block.edata['type_id'])
            # h = torch.mean(h, dim=1)  # 对各个head的结果求平均
            # if l != len(self.layers) - 1:
            #     h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class HGTModel(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, head_size, num_ntypes, num_etypes, num_layers=1, dropout=0.2):
        super(HGTModel, self).__init__()
        self.num_layers = num_layers
        self.encoder = HGTEncoder(in_dim, out_dim, num_heads, head_size, num_ntypes, num_etypes, num_layers, dropout)
        self.decoder = MLPDecoder(out_dim)

    def forward(self, blocks, feats, src, dst):
        emb = self.encoder(blocks, feats)  # shape [N, out_feats]
        return self.decoder(emb, src, dst)  # shape [B]

    def print_params(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param}")

# import pandas as pd
# import torch
# import dgl

# def load_graph_with_label(node_csv, edge_csv, rel_types=None, device="cpu"):
#     # 1) 读节点
#     nodes = pd.read_csv(node_csv)
#     num_nodes = len(nodes)

#     # 2) 读边
#     edges = pd.read_csv(edge_csv)  # 必须有: src, dst, rel, label
#     if rel_types is not None:
#         edges = edges[edges["rel"].isin(rel_types)]

#     src = torch.tensor(edges["src"].values, dtype=torch.int64)
#     dst = torch.tensor(edges["dst"].values, dtype=torch.int64)
#     labels = torch.tensor(edges["label"].values, dtype=torch.float32)  # 1 or 0

#     # 3) 构图（这里只加原始方向，不强制加反向；要加的话要考虑 label 对应谁）
#     g = dgl.graph((src, dst), num_nodes=num_nodes)

#     # 如果你还是想加反向边，也可以复制 label 一份：
#     # g = dgl.add_edges(g, dst, src)
#     # labels = torch.cat([labels, labels], dim=0)

#     # 4) 把标签挂到边上
#     g.edata["label"] = labels

#     return g.to(device)
import dgl
import torch
import pandas as pd

def load_graph_with_label(node_csv, edge_csv, rel_types=None, device="cpu"):
    # 1. 节点
    nodes = pd.read_csv(node_csv)
    num_nodes = len(nodes)

    # 2. 边
    edges = pd.read_csv(edge_csv)
    if rel_types is not None:
        edges = edges[edges["rel"].isin(rel_types)]

    src = torch.tensor(edges["src"].values, dtype=torch.long)
    dst = torch.tensor(edges["dst"].values, dtype=torch.long)

    # 语义标签：0/1
    edge_label_float = torch.tensor(edges["label"].values, dtype=torch.float32)
    edge_label_long  = torch.tensor(edges["label"].values, dtype=torch.long)

    # 3. 先建“正向边”图
    g = dgl.graph((src, dst), num_nodes=num_nodes)

    # 4. 再手动加一份“反向边”（为了和之前 reverse_eids 的代码兼容）
    # g = dgl.add_edges(g, dst, src)

    # 正向一份，反向一份 → 拼起来
    # labels_all  = torch.cat([edge_label_float, edge_label_float], dim=0)  # 给损失函数用
    # types_all   = torch.cat([edge_label_long,  edge_label_long],  dim=0)  # 给 RGCN 用

    # 5. 写到图上
    g.edata["label"]   = edge_label_float          # 0/1，float32
    g.edata["type_id"] = edge_label_long           # 0/1，long，关系 id

    return g.to(device)


import os
from dataloader import save_dgl_graph, reload_dgl_graph
from train_test_split import homo_graph_train_test_split
# 记得把 load_graph_with_sign 的函数定义放在前面，或者从你自己的模块里 import 进来

node_csv = r"data_signed\node_labeled.csv"
edge_csv = r"data_signed\edges_labeled.csv"
graph_save = "ppi_graph_signed_labeled.pt"

if os.path.exists(graph_save):
    print("Reload graph from cache:", graph_save)
    g = reload_dgl_graph(graph_save)
else:
    print("Build graph from csv...")
    g = load_graph_with_label(
        node_csv,
        edge_csv,
        rel_types={"protein_protein"},  # 只保留这一类关系
        device="cpu"                    # 先放在 CPU，后面再 .to(device)
    )
    save_dgl_graph(g, graph_save)

print(g)
device = "cpu"
g = g.to(device)
print("g.edata keys:", g.edata.keys())   # 这里应该看到 'label' 和 'type_id'

# print("splitting dataset...")
# g, train_eids, test_eids, reverse_eids = homo_graph_train_test_split(
#     g, split_ratio=(0.8, 0.2), seed=seed
# )
# g = dgl.add_self_loop(g)
# print(train_eids, test_eids)


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["DGLBACKEND"] = "pytorch"
num_rels = int(g.edata['type_id'].max().item()) + 1
print("num_rels:",num_rels)
# num_ntypes = int(g.ndata['type_id'].max().item()) + 1
model_layers = 2
in_feats = 100
h_feats = 128
out_feats = 64
num_heads = 4  # 对于GAT, HGT
head_size = 32 # 对于HGT

num_epochs = 45
train_batch_size = 8192
initial_lr=1e-3
scheduler_step_size=15
scheduler_gamma=0.7
scheduler_threshold=1e-4

# GAT 
"""
num_epochs = 45
train_batch_size = 8192
initial_lr=1e-3
scheduler_step_size=15
scheduler_gamma=0.7
scheduler_threshold=1e-4
"""

# RGCN
"""
num_epochs = 40
train_batch_size = 8192*4
initial_lr=1e-3
scheduler_step_size=10
scheduler_gamma=0.7
scheduler_threshold=1e-4
"""

# HGT
"""
num_epochs = 45
train_batch_size = 8192
initial_lr=1e-3
scheduler_step_size=8
scheduler_gamma=0.6
scheduler_threshold=1e-4
"""


reset_seed(seed)

initial_emb = nn.Parameter(torch.Tensor(g.num_nodes(), in_feats), requires_grad=False)
nn.init.xavier_uniform_(initial_emb)
g.ndata['feat'] = initial_emb.to(device)
keep_feat = g.ndata['feat']

import torch
import dgl
from dgl.dataloading import MultiLayerFullNeighborSampler, DataLoader  # ✅ 用 DataLoader
from dgl.dataloading.negative_sampler import GlobalUniform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
g = g.to(device)

model_layers = 2
train_ratio = 0.8
batch_size = 8192
seed = 42

labels = g.edata["label"]   # 0/1
all_eids = torch.arange(g.num_edges())[ (labels == 0) | (labels == 1) ]
print("总有标签边数:", len(all_eids))

torch.manual_seed(seed)
perm = torch.randperm(len(all_eids))
num_train = int(train_ratio * len(all_eids))

train_eids_full = all_eids[perm[:num_train]]   # 训练集（目前仍然是不平衡的）
test_eids       = all_eids[perm[num_train:]]   # 测试集（保持真实分布）

print("train edges:", len(train_eids_full))
print("test  edges:", len(test_eids))

# reverse_eids 就先用“指向自己”这版
reverse_eids = torch.arange(g.num_edges()).to(device)
print("reverseids_edges:", len(reverse_eids))
g = dgl.add_self_loop(g)
print("after self-loop, g.num_edges:", g.num_edges())
# sampler 公用
from dgl.dataloading import MultiLayerFullNeighborSampler
from dgl.dataloading.negative_sampler import GlobalUniform
from sampler import HomoGraphDataSampler

pos_sampler = MultiLayerFullNeighborSampler(model_layers)
neg_sampler = GlobalUniform(k=1)
sampler = HomoGraphDataSampler(g, pos_sampler, neg_sampler)



import torch.nn as nn
from tqdm import tqdm
from train import Trainer
# from model import GATModel, RGCNModel, HGTModel  # 看你实际 import 名字

def run_one_experiment(
    g,
    train_eids,
    test_eids,
    reverse_eids,
    sampler,
    balanced: bool,
    num_epochs: int = 30,
    batch_size: int = 8192,
    seed: int = 42,
    model_type: str = "GAT",   # or "RGCN" / "HGT"
):
    """
    balanced = False → 用原始不平衡训练集
    balanced = True  → 从 train_eids 中做 1:1 采样训练
    """

    device = g.device
    torch.manual_seed(seed)

    # ===== 1) 构造 train_loader =====
    if balanced:
        print("\n[Experiment] 使用【类别平衡】训练集 (1:1)")
        train_loader = sampler.construct_balanced_batch_data_sampler(
            train_eids,
            reverse_eids,
            batch_size=batch_size,
            label_key="type_id",   # 或 "label"，你现在 type_id=0/1 就是正负语义
            seed=seed,
        )
    else:
        print("\n[Experiment] 使用【类别不平衡】原始训练集")
        train_loader = sampler.construct_batch_data_sampler(
            train_eids,
            reverse_eids,
            batch_size=batch_size,
        )

    # test_loader 一般保持真实分布，不做平衡
    test_loader = sampler.construct_batch_data_sampler(
        test_eids,
        reverse_eids,
        batch_size=batch_size,
    )

    # ===== 2) 初始化模型 =====
    in_feats = g.ndata["feat"].shape[1]
    h_feats = 128
    out_feats = 64
    num_heads = 4
    head_size = 32
    num_rels = int(g.edata["type_id"].max().item()) + 1
    num_ntypes = 1  # 你现在只有一种节点

    if model_type == "GAT":
        model = GATModel(in_feats, h_feats, out_feats, num_heads).to(device)
    elif model_type == "RGCN":
        model = RGCNModel(in_feats, h_feats, out_feats, num_rels).to(device)
    elif model_type == "HGT":
        model = HGTModel(in_feats, out_feats, num_heads, head_size, num_ntypes, num_rels).to(device)
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    # ===== 3) Trainer、优化器等 =====
    initial_lr = 1e-3
    scheduler_step_size = 15
    scheduler_gamma = 0.7
    scheduler_threshold = 1e-4

    trainer = Trainer(
        model,
        nn.BCELoss(),
        lr=initial_lr,
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        threshold=scheduler_threshold,
    )

    # ===== 4) 训练循环，记录曲线 =====
    train_loss_list = []
    val_loss_list = []
    prc_list = []
    auc_list = []

    for epoch in tqdm(range(num_epochs), desc=f"{'Balanced' if balanced else 'Imbalanced'}"):
        train_loss, p1, p2, lr = trainer.train_one_epoch(train_loader, device)
        val_loss, prc, auc = trainer.evaluate(test_loader, device)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        prc_list.append(prc)
        auc_list.append(auc)

        print(
            f"[{'BAL' if balanced else 'IMB'}] "
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Lr: {lr:.5f}, Train: {train_loss:.4f}, "
            f"Val: {val_loss:.4f}, PRC: {prc:.4f}, AUC: {auc:.4f}"
        )

    results = {
        "train_loss": train_loss_list,
        "val_loss": val_loss_list,
        "prc": prc_list,
        "auc": auc_list,
        "final_train_loss": train_loss_list[-1],
        "final_val_loss": val_loss_list[-1],
        "final_prc": prc_list[-1],
        "final_auc": auc_list[-1],
    }
    return results


num_epochs = 30
import pickle

# 1) 不平衡训练
results_imbalanced = run_one_experiment(
    g,
    train_eids_full,
    test_eids,
    reverse_eids,
    sampler,
    balanced=False,
    num_epochs=num_epochs,
    batch_size=batch_size,
    seed=42,
    model_type="GAT",       # 或 "RGCN"
)
with open("label_imbalanced.pkl", "wb") as f:
    pickle.dump(results_imbalanced, f)
# 2) 平衡训练
results_balanced = run_one_experiment(
    g,
    train_eids_full,        # 注意这里还是传“全训练边”，由 sampler 内部做 1:1 抽样
    test_eids,
    reverse_eids,
    sampler,
    balanced=True,
    num_epochs=num_epochs,
    batch_size=batch_size,
    seed=42,
    model_type="GAT",       # 与上面保持一致
)


with open("label_balanced.pkl", "wb") as f:
    pickle.dump(results_balanced, f)

