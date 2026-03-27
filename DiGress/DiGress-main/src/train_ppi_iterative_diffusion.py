import graph_tool as gt
import os
import json
import copy

import torch
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch_geometric.data import Data, InMemoryDataset

from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from diffusion_model import LiftedDenoisingDiffusion
from diffusion_model_discrete import DiscreteDenoisingDiffusion
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures

from datasets.ppi_dataset_order_embedding import (
    PPIDatasetInfos,
    NUM_NODE_CLASSES,
    map_loc_to_category,
    encode_edge_feature,
    compress_edge_label,
    # calculate_subgraph_metrics,
    is_negative_raw_bitmask,
)
from analysis.visualization import NonMolecularVisualization
from analysis.spectre_utils import SpectreSamplingMetrics
import pandas as pd
import numpy as np
import networkx as nx

from analysis.ppi_rule_utils import (
    build_updated_edge_file_with_random_negatives,
    build_reference_big_graph,
    compute_rule_metrics_for_graph,
    save_negative_edges_csv,
)

# ============================================================
# 可调参数（第一版先写死，后面你可以再 Hydra 化）
# ============================================================
# OUTER_ROUNDS = 3                  # 外层迭代轮数
# EPOCHS_PER_ROUND = 50             # 每轮新增训练 epoch 数
# GENERATE_PER_ROUND = 200          # 每轮生成多少候选图
# MAX_NEW_PER_ROUND = 100           # 每轮最多加入多少新图
# KEEP_ONLY_NEGATIVE_RULE = True    # 只保留含负边规则
# DEDUP_AGAINST_HISTORY = True      # 对历史训练集去重
# SAMPLE_BATCH_SIZE = 32            # 生成时每次 sample_batch 的 batch_size


# ============================================================
# 1) 动态训练集 Dataset
# ============================================================
class DynamicPPIDataset(InMemoryDataset):
    def __init__(self, pt_path, transform=None, pre_transform=None, pre_filter=None):
        self.pt_path = pt_path
        super().__init__(".", transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.pt_path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self):
        pass

    def process(self):
        pass


class DynamicPPIDataModule(utils.LightningDataset if hasattr(utils, "LightningDataset") else object):
    """
    为了尽量少改你现有代码，这里不用继承你的 PPIDataModule，
    直接复用 AbstractDataModule 的逻辑。
    """
    def __init__(self, cfg, pt_path):
        from src.datasets.abstract_dataset import AbstractDataModule
        dataset = DynamicPPIDataset(pt_path)
        datasets = {"train": dataset, "val": dataset, "test": dataset}
        self._inner = AbstractDataModule(cfg, datasets)

    def __getattr__(self, item):
        return getattr(self._inner, item)


# ============================================================
# 2) 读取 edge label mapping
# ============================================================
def load_edge_mapping(datadir):
    mapping_path = os.path.join(datadir, "processed", "edge_label_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    bitmask_to_class = {int(k): int(v) for k, v in payload["bitmask_to_class"].items()}
    class_to_bitmask = {int(k): int(v) for k, v in payload["class_to_bitmask"].items()}
    num_edge_classes = int(payload["num_edge_classes"])

    return {
        "bitmask_to_class": bitmask_to_class,
        "class_to_bitmask": class_to_bitmask,
        "num_edge_classes": num_edge_classes,
        "used_masks": [int(x) for x in payload["used_masks"]],
    }


# ============================================================
# 3) 从原始 CSV 重建 bigG（用于规则验证）
# ============================================================
def build_big_graph_from_raw(datadir, edge_mapping):
    ppi_path = os.path.join(datadir, "raw", "protein_protein_with_type.csv")
    meta_path = os.path.join(datadir, "raw", "protein.csv")

    print(f"[Build bigG] metadata: {meta_path}")
    id_to_attrs = {}
    df_meta = pd.read_csv(meta_path, low_memory=False)
    df_meta.columns = df_meta.columns.str.strip()
    id_col = "biogrid_id"
    df_meta[id_col] = pd.to_numeric(df_meta[id_col], errors="coerce")
    df_meta = df_meta.dropna(subset=[id_col])

    for _, row in df_meta.iterrows():
        bid = str(int(row[id_col]))
        attr_dict = row.to_dict()
        loc_raw = row.get("location", "")
        attr_dict["cat_idx"] = map_loc_to_category(loc_raw)
        id_to_attrs[bid] = attr_dict

    print(f"[Build bigG] PPI: {ppi_path}")
    bigG = nx.Graph()
    df_ppi = pd.read_csv(ppi_path, sep="," if "," in open(ppi_path).readline() else "\t")
    df_ppi.columns = df_ppi.columns.str.strip()

    # 先加点和边
    for _, row in df_ppi.iterrows():
        u_bid = str(row.get("BioGRID ID Interactor A", "")).split(".")[0]
        v_bid = str(row.get("BioGRID ID Interactor B", "")).split(".")[0]
        if not u_bid or not v_bid:
            continue

        u_attrs = id_to_attrs.get(u_bid, {})
        v_attrs = id_to_attrs.get(v_bid, {})

        loc_u = u_attrs.get("location", "")
        loc_v = v_attrs.get("location", "")

        u_enc = map_loc_to_category(loc_u)
        v_enc = map_loc_to_category(loc_v)

        bigG.add_node(u_bid, **u_attrs, feature_val=u_enc)
        bigG.add_node(v_bid, **v_attrs, feature_val=v_enc)

        edge_data = row.to_dict()
        bigG.add_edge(u_bid, v_bid, raw_label=0, label=0, **edge_data)

    # 节点统计量
    deg_dict = dict(bigG.degree())
    nx.set_node_attributes(bigG, deg_dict, "degree")

    bet_dict = nx.betweenness_centrality(bigG, k=256, seed=42)
    nx.set_node_attributes(bigG, bet_dict, "betweenness_centrality")

    deg_vals = np.array(list(deg_dict.values()), dtype=float)
    bet_vals = np.array(list(bet_dict.values()), dtype=float)
    global_stats = {
        "degree": {"q75": float(np.quantile(deg_vals, 0.75))},
        "betweenness_centrality": {"q25": float(np.quantile(bet_vals, 0.25))}
    }

    # 重算 raw_label + compressed label
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
        d["label"] = compress_edge_label(raw_label, edge_mapping["bitmask_to_class"])

    print(f"[Build bigG] done. nodes={bigG.number_of_nodes()}, edges={bigG.number_of_edges()}")
    return bigG


# ============================================================
# 4) sample_batch 输出 -> PyG Data
# ============================================================
def sampled_graph_to_pyg(atom_types, edge_types, edge_mapping):
    """
    atom_types: [n]，节点离散类别
    edge_types: [n, n]，边离散类别（压缩后的 edge class）
    """
    atom_types = atom_types.long()
    edge_types = edge_types.long()

    n = atom_types.shape[0]
    if n <= 1:
        return None

    x = F.one_hot(atom_types, num_classes=NUM_NODE_CLASSES).float()

    src, dst = [], []
    edge_type_ids = []
    edge_bitmasks = []

    class_to_bitmask = edge_mapping["class_to_bitmask"]
    num_edge_classes = edge_mapping["num_edge_classes"]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            edge_class = int(edge_types[i, j].item())
            if edge_class == 0:
                continue
            src.append(i)
            dst.append(j)
            edge_type_ids.append(edge_class)
            edge_bitmasks.append(int(class_to_bitmask[edge_class]))

    if len(src) == 0:
        return None

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_type_ids = torch.tensor(edge_type_ids, dtype=torch.long)
    edge_attr = F.one_hot(edge_type_ids, num_classes=num_edge_classes).float()
    edge_label_mask = torch.tensor(edge_bitmasks, dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label_mask=edge_label_mask,
        n_nodes=torch.tensor([n], dtype=torch.long),
        num_nodes=n,
        y=torch.zeros(1, 0).float()
    )


def pyg_to_nx_for_rule_check(data):
    """
    用于 calculate_subgraph_metrics:
    - 节点需要 feature_val
    - 边需要 label / raw_label
    """
    G = nx.Graph()

    x_idx = data.x.argmax(dim=-1).tolist()
    for i, feat in enumerate(x_idx):
        G.add_node(i, feature_val=int(feat))

    edge_index = data.edge_index
    edge_attr = data.edge_attr.argmax(dim=-1).tolist()
    raw_masks = data.edge_label_mask.tolist()

    # directed -> undirected，去重
    seen = set()
    for eid in range(edge_index.size(1)):
        u = int(edge_index[0, eid])
        v = int(edge_index[1, eid])
        if u == v:
            continue
        a, b = min(u, v), max(u, v)
        if (a, b) in seen:
            continue
        seen.add((a, b))

        G.add_edge(
            a, b,
            label=int(edge_attr[eid]),
            raw_label=int(raw_masks[eid])
        )

    return G


# ============================================================
# 5) 简单图签名去重
# ============================================================
def graph_signature(data):
    node_labels = tuple(sorted(data.x.argmax(dim=-1).tolist()))

    undirected_edges = []
    ei = data.edge_index
    raws = data.edge_label_mask.tolist()
    for k in range(ei.size(1)):
        u = int(ei[0, k])
        v = int(ei[1, k])
        if u == v:
            continue
        a, b = min(u, v), max(u, v)
        undirected_edges.append((a, b, int(raws[k])))

    undirected_edges = tuple(sorted(set(undirected_edges)))
    return (data.num_nodes, node_labels, undirected_edges)


def load_data_list_from_pt(pt_path):
    dataset = DynamicPPIDataset(pt_path)
    return [dataset.get(i) for i in range(len(dataset))]


def save_data_list_to_pt(data_list, out_path):
    tmp_ds = DynamicPPIDataset.__new__(DynamicPPIDataset)
    data, slices = InMemoryDataset.collate(tmp_ds, data_list)
    torch.save((data, slices), out_path)


# ============================================================
# 6) 生成候选图
# ============================================================
@torch.no_grad()
def generate_candidate_graphs(model, edge_mapping, total_to_generate, batch_size):
    model.eval()
    model.visualization_tools = None  # 关闭可视化，避免生成时写图
    generated = []

    left = total_to_generate
    batch_id = 0

    while left > 0:
        cur_bs = min(batch_size, left)
        samples = model.sample_batch(
            batch_id=batch_id,
            batch_size=cur_bs,
            keep_chain=0,
            number_chain_steps=1,
            save_final=0,
            num_nodes=None,
        )
        for atom_types, edge_types in samples:
            pyg = sampled_graph_to_pyg(atom_types, edge_types, edge_mapping)
            if pyg is not None:
                generated.append(pyg)

        left -= cur_bs
        batch_id += cur_bs

    print(f"[Generate] got {len(generated)} candidate graphs")
    return generated


# ============================================================
# 7) 规则筛选
# ============================================================
# def filter_generated_graphs(generated_graphs, bigG, keep_only_negative_rule=True):
def filter_generated_graphs(
    generated_graphs,
    bigG,
    keep_only_negative_rule=True,
    confidence_threshold=0.0,
    support_threshold=1,
    match_limit=400,
    time_limit=5,
    enable_node_match=True,
):
    kept = []
    metrics_log = []
    matched_negative_edges = set()



    for data in generated_graphs:
        nx_g = pyg_to_nx_for_rule_check(data)

        # has_neg = False
        # for _, _, d in nx_g.edges(data=True):
        #     if is_negative_raw_bitmask(d.get("raw_label", 0)):
        #         has_neg = True
        #         break
        has_neg = any(is_negative_raw_bitmask(d.get("raw_label", 0)) for _, _, d in nx_g.edges(data=True))
        if keep_only_negative_rule and not has_neg:
            continue


        m = compute_rule_metrics_for_graph(
                    subG=nx_g,
                    bigG=bigG,
                    match_limit=match_limit,
                    time_limit=time_limit,
                    confidence_threshold=confidence_threshold,
                    support_threshold=support_threshold,
                    # denominator_mode=denominator_mode,
                    enable_node_match=enable_node_match,
                    keep_only_negative_rule=keep_only_negative_rule,
                )
        if m is None:
            continue

        kept.append(data)
        # metrics_log.append(m)
        metrics_log.append({
            "conf": m.confidence,
            "supp_neg": m.support_negative,
            "supp_base": m.support_base,
            "supp_shape": m.support_shape,
            "status": m.status,
            "denominator_mode": m.denominator_mode,
            "matched_negative_edges": m.matched_negative_edges,
        })
        matched_negative_edges.update(tuple(edge) for edge in m.matched_negative_edges)

    print(f"[Filter] kept {len(kept)} / {len(generated_graphs)}")
    # return kept, metrics_log
    return kept, metrics_log, sorted(matched_negative_edges)



# ============================================================
# 8) 合并训练集
# ============================================================
def merge_training_set(base_pt, new_graphs, out_pt, max_new_per_round, dedup_against_history=True):
    old_list = load_data_list_from_pt(base_pt)
    print(f"[Merge] old dataset size = {len(old_list)}")

    if dedup_against_history:
        sigs = set(graph_signature(g) for g in old_list)
    else:
        sigs = set()

    added = 0
    for g in new_graphs:
        sig = graph_signature(g)
        if sig in sigs:
            continue
        old_list.append(g)
        sigs.add(sig)
        added += 1
        if added >= max_new_per_round:
            break

    save_data_list_to_pt(old_list, out_pt)
    print(f"[Merge] added = {added}, new dataset size = {len(old_list)}")
    print(f"[Merge] saved -> {out_pt}")
    return out_pt, added

# ============================================================
# 9) 构造 PPI 训练组件
# ============================================================
def build_ppi_components(cfg, datamodule):
    dataset_config = cfg["dataset"]
    dataset_infos = PPIDatasetInfos(datamodule, dataset_config)

    train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
    visualization_tools = NonMolecularVisualization()

    if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()

    domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(
        datamodule=datamodule,
        extra_features=extra_features,
        domain_features=domain_features
    )

    sampling_metrics = SpectreSamplingMetrics(
        datamodule,
        compute_emd=True,
        metrics_list=['degree', 'clustering', 'orbit']
    )

    model_kwargs = {
        'dataset_infos': dataset_infos,
        'train_metrics': train_metrics,
        'sampling_metrics': sampling_metrics,
        'visualization_tools': visualization_tools,
        'extra_features': extra_features,
        'domain_features': domain_features
    }
    return model_kwargs


# ============================================================
# 10) 单轮训练
# ============================================================
def train_one_round(cfg, train_pt_path, round_id, run_checkpoints_dir, resume_ckpt=None):
    datamodule = DynamicPPIDataModule(cfg, train_pt_path)
    model_kwargs = build_ppi_components(cfg, datamodule)

    if cfg.model.type == 'discrete':
        model = DiscreteDenoisingDiffusion(cfg=cfg, **model_kwargs)
    else:
        model = LiftedDenoisingDiffusion(cfg=cfg, **model_kwargs)

    callbacks = []
    ckpt_dir = run_checkpoints_dir

    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='{epoch}',
            monitor='val/epoch_NLL',
            save_top_k=5,
            mode='min',
            every_n_epochs=1
        )
        last_ckpt_save = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename='last',
            every_n_epochs=1
        )
        callbacks.extend([last_ckpt_save, checkpoint_callback])

    if cfg.train.ema_decay > 0:
        ema_callback = utils.EMA(decay=cfg.train.ema_decay)
        callbacks.append(ema_callback)

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true",
        accelerator='gpu' if use_gpu else 'cpu',
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=cfg.general.name == 'debug',
        enable_progress_bar=False,
        callbacks=callbacks,
        log_every_n_steps=50 if cfg.general.name != 'debug' else 1,
        logger=[]
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)

    last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
    if not os.path.exists(last_ckpt_path):
        raise FileNotFoundError(f"last.ckpt not found after round {round_id}: {last_ckpt_path}")

    return model, datamodule, last_ckpt_path


# ============================================================
# 11) 外层迭代主流程
# ============================================================
@hydra.main(version_base='1.3', config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    print("cfg.dataset =", cfg.dataset)
    # print("cfg.dataset.name =", getattr(cfg.dataset, "name", None))
    # print("full cfg =")
    # from omegaconf import OmegaConf
    # print(OmegaConf.to_yaml(cfg))

    if cfg.dataset.name != "ppi":
        raise ValueError("This script is only for dataset.name == 'ppi'")

    source_datadir = cfg.paths.source_datadir
    source_train_pt = cfg.paths.source_train_pt
    run_root = cfg.paths.run_root

    outer_rounds = cfg.iterative.outer_rounds
    epochs_per_round = cfg.iterative.epochs_per_round
    generate_per_round = cfg.iterative.generate_per_round
    max_new_per_round = cfg.iterative.max_new_per_round
    sample_batch_size = cfg.iterative.sample_batch_size
    keep_only_negative_rule = cfg.iterative.keep_only_negative_rule
    dedup_against_history = cfg.iterative.dedup_against_history
    reference_source = cfg.iterative.reference_source
    reference_split = cfg.iterative.reference_split
    raw_edge_file = cfg.iterative.raw_edge_file
    raw_node_file = cfg.iterative.raw_node_file
    ml_threshold = cfg.iterative.ml_threshold
    confidence_threshold = cfg.iterative.confidence_threshold
    support_threshold = cfg.iterative.support_threshold
    rule_match_limit = cfg.iterative.rule_match_limit
    rule_time_limit = cfg.iterative.rule_time_limit
    # rule_denominator_mode = cfg.iterative.rule_denominator_mode
    enable_node_match = cfg.iterative.enable_node_match
    export_discovered_negative_edges = cfg.iterative.export_discovered_negative_edges
    discovered_negative_edges_file = cfg.iterative.discovered_negative_edges_file
    edge_old_file = cfg.iterative.edge_old_file
    edge_updated_output_file = cfg.iterative.edge_updated_output_file
    edge_src_col = cfg.iterative.edge_src_col
    edge_dst_col = cfg.iterative.edge_dst_col
    edge_rel_col = cfg.iterative.edge_rel_col
    edge_rel_name = cfg.iterative.edge_rel_name
    edge_label_col = cfg.iterative.edge_label_col
    edge_negative_label = cfg.iterative.edge_negative_label
    num_additional_negatives = cfg.iterative.num_additional_negatives
    random_seed = cfg.iterative.random_seed
    datasets_dir = os.path.join(run_root, "datasets")
    checkpoints_dir = os.path.join(run_root, "checkpoints")
    metrics_dir = os.path.join(run_root, "metrics")
    generated_dir = os.path.join(run_root, "generated")
    logs_dir = os.path.join(run_root, "logs")

    for d in [run_root, datasets_dir, checkpoints_dir, metrics_dir, generated_dir, logs_dir]:
        os.makedirs(d, exist_ok=True)

    edge_mapping = load_edge_mapping(source_datadir)
    bigG = build_big_graph_from_raw(source_datadir, edge_mapping)

    if not os.path.exists(source_train_pt):
        raise FileNotFoundError(source_train_pt)

    init_copy_path = os.path.join(datasets_dir, "ppi_train_iter_0.pt")
    if not os.path.exists(init_copy_path):
        old_list = load_data_list_from_pt(source_train_pt)
        save_data_list_to_pt(old_list, init_copy_path)

    current_train_pt = init_copy_path
    resume_ckpt = None
    all_discovered_negative_edges = set()
    for round_id in range(outer_rounds):
        print("\n" + "=" * 100)
        print(f"[Round {round_id}] current_train_pt = {current_train_pt}")
        print(f"[Round {round_id}] resume_ckpt = {resume_ckpt}")
        print("=" * 100)

        round_cfg = copy.deepcopy(cfg)
        round_cfg.train.n_epochs = (round_id + 1) * epochs_per_round
        round_cfg.general.name = f"{cfg.general.name}_iter"
        round_cfg.general.resume = resume_ckpt

        model, datamodule, last_ckpt = train_one_round(
            round_cfg,
            train_pt_path=current_train_pt,
            round_id=round_id,
            run_checkpoints_dir=checkpoints_dir,
            resume_ckpt=resume_ckpt
        )

        generated = generate_candidate_graphs(
            model=model,
            edge_mapping=edge_mapping,
            total_to_generate=generate_per_round,
            batch_size=sample_batch_size
        )

        kept, metrics_log, matched_negative_edges = filter_generated_graphs(
            generated_graphs=generated,
            bigG=bigG,
            keep_only_negative_rule=keep_only_negative_rule,
            confidence_threshold=confidence_threshold,
            support_threshold=support_threshold,
            match_limit=rule_match_limit,
            time_limit=rule_time_limit,
            # denominator_mode=rule_denominator_mode,
            enable_node_match=enable_node_match,
        )
        all_discovered_negative_edges.update(tuple(edge) for edge in matched_negative_edges)
        print(len(all_discovered_negative_edges))
        metrics_json = os.path.join(metrics_dir, f"iter_{round_id}_metrics.json")
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(metrics_log, f, indent=2)

        if len(kept) == 0:
            print(f"[Round {round_id}] no valid generated graphs, stop.")
            break

        next_train_pt = os.path.join(datasets_dir, f"ppi_train_iter_{round_id + 1}.pt")
        next_train_pt, added = merge_training_set(
            base_pt=current_train_pt,
            new_graphs=kept,
            out_pt=next_train_pt,
            max_new_per_round=max_new_per_round,
            dedup_against_history=dedup_against_history
        )

        if added == 0:
            print(f"[Round {round_id}] no new unique graphs added, stop.")
            break

        current_train_pt = next_train_pt
        resume_ckpt = last_ckpt

        if export_discovered_negative_edges:
            save_negative_edges_csv(
                negative_edges=sorted(all_discovered_negative_edges),
                output_path=discovered_negative_edges_file,
                src_col=edge_src_col,
                dst_col=edge_dst_col,
                rel_col=edge_rel_col,
                rel_name=edge_rel_name,
                label_col=edge_label_col,
                negative_label=edge_negative_label,
            )
            print(f"[Done] discovered negative edges saved -> {discovered_negative_edges_file}")

            if edge_old_file:
                build_updated_edge_file_with_random_negatives(
                    edge_old_file=edge_old_file,
                    discovered_negative_edges=sorted(all_discovered_negative_edges),
                    output_path=edge_updated_output_file,
                    src_col=edge_src_col,
                    dst_col=edge_dst_col,
                    rel_col=edge_rel_col,
                    rel_name=edge_rel_name,
                    label_col=edge_label_col,
                    negative_label=edge_negative_label,
                    num_additional_negatives=num_additional_negatives,
                    random_seed=random_seed,
                )
                print(f"[Done] updated edge file saved -> {edge_updated_output_file}")


    print("[Done] iterative diffusion training finished.")

if __name__ == "__main__":
    main()