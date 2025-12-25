import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import RelGraphConv
from dgl.nn.pytorch import GATConv, HGTConv


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


# ============================================RGCN===========================================
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


# ========================================GAT=============================================
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


# ========================================HGT=============================================
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