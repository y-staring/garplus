import dgl


import torch

def homo_graph_train_test_split(g: dgl.DGLGraph, split_ratio=(0.8, 0.2), seed=42):
    # 按给定比例划分训练集、测试集，两点之间的正向、反向边在一个集合中
    device = g.device
    # 设置随机种子
    torch.manual_seed(seed)
    G_dgl = g.to('cpu')
    u, v = G_dgl.edges()
    reverse_eids = G_dgl.edge_ids(v, u)
    # G_dgl.edata['reverse_id'] = reverse_eids

    # === 2) 以“无向对”为单位做 split（去重每对，只保留 u < v 的一侧作为代表）
    mask_rep = u < v  # 代表这一对
    pair_rep_eids = torch.nonzero(mask_rep).squeeze()  # 每对的代表 eid
    perm = torch.randperm(len(pair_rep_eids))
    pair_rep_eids = pair_rep_eids[perm]
    # todo:看一下如果有不同类型的边对不同类型的边分别采样

    train_ratio = split_ratio[0]
    num_train_pairs = int(train_ratio * pair_rep_eids.numel())
    train_rep = pair_rep_eids[:num_train_pairs]
    val_rep = pair_rep_eids[num_train_pairs:]

    # 展开成成对的 eids（代表 + 其反向）
    train_eids = torch.cat([train_rep, reverse_eids[train_rep]]).to(device)
    val_eids = torch.cat([val_rep, reverse_eids[val_rep]]).to(device)

    # === 3) 构建训练图：从整图里移除「验证对」的两条边
    # 注意很多图编辑 API 需要 CPU tensor
    # G_train = dgl.remove_edges(G_dgl, val_eids.cpu())
    # G_dgl = G_dgl.to(device)
    # G_train = G_train.to(device)

    # # 随机打乱eid，切分训练集 验证集
    # G_dgl = G_dgl.to(device)
    # all_eids = torch.arange(G_dgl.number_of_edges())
    # perm = torch.randperm(len(all_eids))
    # shuffled_eids = all_eids[perm]
    # train_size = int(0.8 * len(shuffled_eids))
    # train_eids = shuffled_eids[:train_size].to(device)
    # val_eids = shuffled_eids[train_size:].to(device)
    # # 初始化sampler
    # sampler = MultiLayerFullNeighborSampler(2)
    # neg_sampler = GlobalUniform(k=1)

    return G_dgl.to(device), train_eids, val_eids, reverse_eids.to(device)


def old_homo_graph_train_test_split(g: dgl.DGLGraph, split_ratio=(0.8, 0.2), seed=42):
    # 按给定比例划分训练集、测试集，两点之间的正向、反向边在一个集合中
    device = g.device
    # 设置随机种子
    torch.manual_seed(seed)
    G_dgl = g.to('cpu')
    u, v = G_dgl.edges()
    # todo:不考虑自连边

    num_e = G_dgl.num_edges()
    eid_map = {}  # (u,v) -> eid
    for eid in range(num_e):
        eid_map[(int(u[eid]), int(v[eid]))] = eid

    reverse_eids = torch.empty(num_e, dtype=torch.long)
    for eid in range(num_e):
        ru = int(v[eid])
        rv = int(u[eid])
        reverse_eids[eid] = eid_map[(ru, rv)]
    G_dgl.edata['reverse_id'] = reverse_eids

    # === 2) 以“无向对”为单位做 split（去重每对，只保留 u < v 的一侧作为代表）
    mask_rep = (u < v)  # 代表这一对
    pair_rep_eids = torch.nonzero(mask_rep, as_tuple=False).squeeze(1)  # 每对的代表 eid
    perm = torch.randperm(pair_rep_eids.numel())
    pair_rep_eids = pair_rep_eids[perm]
    # todo:看一下如果有不同类型的边对不同类型的边分别采样

    train_ratio = split_ratio[0]
    num_train_pairs = int(train_ratio * pair_rep_eids.numel())
    train_rep = pair_rep_eids[:num_train_pairs]
    val_rep = pair_rep_eids[num_train_pairs:]

    # 展开成成对的 eids（代表 + 其反向）
    train_eids = torch.cat([train_rep, reverse_eids[train_rep]]).to(device)
    val_eids = torch.cat([val_rep, reverse_eids[val_rep]]).to(device)

    # === 3) 构建训练图：从整图里移除「验证对」的两条边
    # 注意很多图编辑 API 需要 CPU tensor
    # G_train = dgl.remove_edges(G_dgl, val_eids.cpu())
    # G_dgl = G_dgl.to(device)
    # G_train = G_train.to(device)

    # # 随机打乱eid，切分训练集 验证集
    # G_dgl = G_dgl.to(device)
    # all_eids = torch.arange(G_dgl.number_of_edges())
    # perm = torch.randperm(len(all_eids))
    # shuffled_eids = all_eids[perm]
    # train_size = int(0.8 * len(shuffled_eids))
    # train_eids = shuffled_eids[:train_size].to(device)
    # val_eids = shuffled_eids[train_size:].to(device)
    # # 初始化sampler
    # sampler = MultiLayerFullNeighborSampler(2)
    # neg_sampler = GlobalUniform(k=1)

    return G_dgl.to(device), train_eids, val_eids, reverse_eids.to(device)


