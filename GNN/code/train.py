import dgl
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score


class Trainer:
    def __init__(self,
                 model,
                 # g: dgl.DGLGraph,
                 # train_loader: DataLoader,
                 # val_loader: DataLoader,
                 loss_fn,
                 lr=1e-3,
                 scheduler_step_size=5,
                 scheduler_gamma=0.5,
                 threshold=1e-4):
        self.model = model
        # self.g = g
        # self.train_data_loader = train_loader
        # self.val_data_loader = val_loader
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)
        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', scheduler_gamma, patience=scheduler_step_size, threshold=threshold)
        self.loss_fn = loss_fn


    # 在外面初始化吧，要指定维度
    # def initialize(self):
    #     self.g.ndata['feat'] =

    def train_one_epoch(self, train_data_loader: DataLoader, device):
        # device = self.model.device
        self.model.train()
        total_loss = 0
        preds, trues = [], []
        # 需要在外面指定初始特征
        # node_feats = self.g.ndata['feat']
        for input_nodes, pair_graph, neg_graph, blocks in train_data_loader:
            pos_src, pos_dst = pair_graph.edges()
            neg_src, neg_dst = neg_graph.edges()
            

            # neg_src, neg_dst = neg_graph.edges()
            # 兼容异构图和同构图两种情况
            feat_raw = blocks[0].ndata['feat']

            if isinstance(feat_raw, dict):
                # 异构图：我们不关心具体类型名，直接取第一个类型的特征
                # 例如 {'protein': tensor(...)}，或者 {'_N': tensor(...)}
                feats = next(iter(feat_raw.values()))
            else:
                # 同构图：直接就是一个 tensor
                feats = feat_raw


            all_src = torch.cat([pos_src, neg_src])
            all_dst = torch.cat([pos_dst, neg_dst])
            # todo:这里是只考虑一种点类型
            # # print(blocks[0].canonical_etypes)
            # feats = blocks[0].ndata['feat']["_N"]

            # all_src = torch.cat([pos_src, neg_src])
            # all_dst = torch.cat([pos_dst, neg_dst])

            labels = torch.cat([
                torch.ones(pos_src.shape[0]),
                torch.zeros(neg_src.shape[0])
            ]).to(device)

            # 前向
            pred = torch.sigmoid(self.model(blocks, feats, all_src, all_dst))
            preds.append(pred.detach().cpu())
            trues.append(labels.detach().cpu())

            # 反向传播
            loss = self.loss_fn(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            # self.model.print_params()
        # 一轮结束之后
        self.scheduler.step()
        # self.scheduler.step(total_loss / len(train_data_loader)) # 如果是ReduceLROnPlateau的话
        preds = torch.cat(preds)
        trues = torch.cat(trues)
        prc = average_precision_score(trues.numpy(), preds.numpy())
        auc = roc_auc_score(trues.numpy(), preds.numpy())
        return total_loss / len(train_data_loader), prc, auc, self.optimizer.param_groups[0]['lr']



    @torch.no_grad()
    def evaluate(self, val_data_loader: DataLoader, device):
        # device = self.model.device
        self.model.eval()
        total_loss = 0
        preds, trues = [], []
        for input_nodes, pair_graph, neg_graph, blocks in val_data_loader:
            pos_src, pos_dst = pair_graph.edges()
            neg_src, neg_dst = neg_graph.edges()

            # input_ids = blocks[0].srcdata[dgl.NID]
            # feats = node_feats[input_ids]
            # todo:这里是只考虑一种点类型
            # print(blocks[0].canonical_etypes)
            feats = blocks[0].ndata['feat']["_N"]

            all_src = torch.cat([pos_src, neg_src])
            all_dst = torch.cat([pos_dst, neg_dst])
            labels = torch.cat([
                torch.ones(pos_src.shape[0]),
                torch.zeros(neg_src.shape[0])
            ]).to(device)

            pred = torch.sigmoid(self.model(blocks, feats, all_src, all_dst))

            preds.append(pred.cpu())
            trues.append(labels.cpu())
            loss = self.loss_fn(pred, labels)
            total_loss += loss.item()


        preds = torch.cat(preds)
        trues = torch.cat(trues)

        prc = average_precision_score(trues.numpy(), preds.numpy())
        auc = roc_auc_score(trues.numpy(), preds.numpy())
        return total_loss / len(val_data_loader), prc, auc