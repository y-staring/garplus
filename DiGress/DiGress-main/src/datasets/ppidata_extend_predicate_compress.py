import json
import os
import signal
import sys
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.isomorphism import GraphMatcher

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.datasets.ppi_dataset import PPIGraphDataset
from src.datasets.ppi_dataset_order_embedding import EDGE_BIT_MAP, encode_edge_feature, map_loc_to_category


class TimeoutException(Exception):
    pass


class TimeLimit:
    def __init__(self, seconds: int):
        self.seconds = seconds
        self.old_handler = signal.SIG_DFL

    def __enter__(self):
        if hasattr(signal, "SIGALRM"):
            self.old_handler = signal.getsignal(signal.SIGALRM)

            def handler(signum, frame):
                raise TimeoutException()

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.old_handler)


@dataclass
class RuleMetric:
    graph_id: int
    target_edge: Tuple[int, int]
    confidence: float
    support_negative: int
    support_shape: int
    status: str
    label_dist: Dict[int, int]
    updates: List[Tuple[str, str]]


@dataclass
class PipelineConfig:
    # 你的生成子图文件（始终作为规则候选输入）
    generated_sample_file: str = ""
    # support/confidence 的来源大图：train_data 或 raw_csv_graph
    reference_source: str = "raw_csv_graph"  # train_data | raw_csv_graph

    # 仅当 reference_source=train_data 时使用
    data_root: str = "DiGress/DiGress-main/data/PPI"
    split: str = "train"

    # 仅当 reference_source=raw_csv_graph 时使用
    raw_edge_file: str = ""
    raw_node_file: str = ""

    # 编码映射（用于 edge class <-> raw bitmask）
    edge_label_mapping: str = ""
    update_edge_file: str = ""
    update_ppi: bool = False

    ml_threshold: float = 0.3
    confidence_threshold: float = 0.5
    support_threshold: int = 5
    match_limit: int = 500
    time_limit: int = 10
    enable_node_match: bool = False
    use_negative_centered_triads: bool = True
    triad_min_edges: int = 2
    triad_topk_per_negative_edge: int = 5


def load_edge_label_mapping(mapping_json: str) -> Dict:
    with open(mapping_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return {
        "bitmask_to_class": {int(k): int(v) for k, v in payload["bitmask_to_class"].items()},
        "class_to_bitmask": {int(k): int(v) for k, v in payload["class_to_bitmask"].items()},
        "num_edge_classes": int(payload["num_edge_classes"]),
        "used_masks": [int(x) for x in payload.get("used_masks", [])],
    }


def edge_class_to_raw_bitmask(edge_class: int, edge_label_mapping: Dict) -> int:
    edge_class = int(edge_class)
    if edge_class == 0:
        return 0
    class_to_bitmask = edge_label_mapping["class_to_bitmask"]
    if edge_class in class_to_bitmask:
        return int(class_to_bitmask[edge_class])
    # 兼容旧可视化导出：edge_type = raw_bitmask + 1
    return max(edge_class - 1, 0)


def raw_bitmask_to_edge_class(raw_bitmask: int, edge_label_mapping: Dict) -> int:
    raw_bitmask = int(raw_bitmask)
    if raw_bitmask == 0:
        return 0
    bitmask_to_class = edge_label_mapping["bitmask_to_class"]
    if raw_bitmask in bitmask_to_class:
        return int(bitmask_to_class[raw_bitmask])
    # 兼容旧导出格式：class = raw + 1
    return raw_bitmask + 1


def parse_graph_txt(filepath: str) -> List[Tuple[List[int], List[List[int]]]]:
    graphs = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("N="):
            i += 1
            continue

        n = int(line.split("=")[1])

        i += 1
        while i < len(lines) and not lines[i].strip().startswith("X:"):
            i += 1
        i += 1
        x_list = list(map(int, lines[i].strip().split()))

        i += 1
        while i < len(lines) and not lines[i].strip().startswith("E:"):
            i += 1
        i += 1

        edge_matrix = []
        for _ in range(n):
            if i >= len(lines):
                break
            edge_matrix.append(list(map(int, lines[i].strip().replace(",", " ").split())))
            i += 1
        graphs.append((x_list, edge_matrix))

    return graphs


def export_train_groundtruth(data_root: str, split: str, output_file: str) -> str:
    dataset = PPIGraphDataset(root=data_root, split=split)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for data in dataset:
            num_nodes = int(data.num_nodes)
            x_idx = data.x.argmax(dim=-1).cpu().numpy().tolist()

            E = np.zeros((num_nodes, num_nodes), dtype=int)
            if data.edge_attr is not None and data.edge_attr.numel() > 0:
                e_idx = data.edge_attr.argmax(dim=-1).cpu().numpy()
                src = data.edge_index[0].cpu().numpy()
                dst = data.edge_index[1].cpu().numpy()
                for k in range(len(src)):
                    E[int(src[k]), int(dst[k])] = int(e_idx[k])

            f.write(f"N={num_nodes}\n")
            f.write("X:\n")
            f.write(" ".join(map(str, x_idx)) + "\n")
            f.write("E:\n")
            for r in E:
                f.write(" ".join(map(str, r.tolist())) + "\n")
            f.write("\n")

    return output_file


def build_big_graph_from_train_dataset(data_root: str, split: str, edge_label_mapping: Dict) -> nx.Graph:
    dataset = PPIGraphDataset(root=data_root, split=split)
    bigG = nx.Graph()

    for gid, data in enumerate(dataset):
        num_nodes = int(data.num_nodes)
        x_idx = data.x.argmax(dim=-1).cpu().numpy().tolist()

        local_to_global = {}
        for i in range(num_nodes):
            gnid = f"g{gid}_n{i}"
            local_to_global[i] = gnid
            bigG.add_node(gnid, feature_val=int(x_idx[i]), deg=0)

        if data.edge_attr is None or data.edge_attr.numel() == 0:
            continue

        edge_classes = data.edge_attr.argmax(dim=-1).cpu().numpy()
        src = data.edge_index[0].cpu().numpy()
        dst = data.edge_index[1].cpu().numpy()

        for k in range(len(src)):
            u_local = int(src[k])
            v_local = int(dst[k])
            if u_local == v_local:
                continue
            u = local_to_global[u_local]
            v = local_to_global[v_local]

            raw_bitmask = edge_class_to_raw_bitmask(int(edge_classes[k]), edge_label_mapping)
            if raw_bitmask == 0:
                continue
            bigG.add_edge(u, v, label=raw_bitmask, raw_label=raw_bitmask)

    for n in bigG.nodes():
        bigG.nodes[n]["deg"] = bigG.degree(n)
    return bigG


def export_raw_ppi_groundtruth(
    raw_edge_file: str,
    raw_node_file: str,
    edge_label_mapping: Dict,
    output_file: str,
    ml_threshold: float,
) -> str:
    bigG = build_big_graph_from_raw(raw_edge_file, raw_node_file, ml_threshold=ml_threshold)
    nodes = list(bigG.nodes())
    node_to_idx = {nid: i for i, nid in enumerate(nodes)}
    n = len(nodes)

    x_list = [int(bigG.nodes[nid].get("feature_val", 9)) for nid in nodes]
    E = np.zeros((n, n), dtype=int)

    for u, v, d in bigG.edges(data=True):
        i, j = node_to_idx[u], node_to_idx[v]
        raw_label = int(d.get("label", 0))
        edge_class = raw_bitmask_to_edge_class(raw_label, edge_label_mapping)
        E[i, j] = edge_class
        E[j, i] = edge_class

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"N={n}\n")
        f.write("X:\n")
        f.write(" ".join(map(str, x_list)) + "\n")
        f.write("E:\n")
        for r in E:
            f.write(" ".join(map(str, r.tolist())) + "\n")
        f.write("\n")
    return output_file


def to_nx_graph(x_list: List[int], edge_matrix: List[List[int]], edge_label_mapping: Dict) -> nx.Graph:
    n = len(edge_matrix)
    G = nx.Graph()

    for i in range(n):
        G.add_node(i, feature_val=int(x_list[i]))

    for i in range(n):
        for j in range(i + 1, n):
            val = int(edge_matrix[i][j])
            raw_bitmask = edge_class_to_raw_bitmask(val, edge_label_mapping)
            if raw_bitmask != 0:
                G.add_edge(i, j, label=raw_bitmask)

    for node in G.nodes():
        G.nodes[node]["deg"] = G.degree(node)
    return G


def _norm_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def build_big_graph_from_raw(raw_edge_file: str, raw_node_file: str, ml_threshold: float) -> nx.Graph:
    id_to_attrs = {}
    df_meta = pd.read_csv(raw_node_file, low_memory=False)
    df_meta.columns = df_meta.columns.str.strip()
    id_col = "biogrid_id" if "biogrid_id" in df_meta.columns else "BioGRID ID"
    df_meta[id_col] = pd.to_numeric(df_meta[id_col], errors="coerce")
    df_meta = df_meta.dropna(subset=[id_col])

    for _, row in df_meta.iterrows():
        bid = str(int(row[id_col]))
        attrs = row.to_dict()
        attrs["cat_idx"] = map_loc_to_category(attrs.get("location", ""))
        id_to_attrs[bid] = attrs

    bigG = nx.Graph()
    df_ppi = pd.read_csv(raw_edge_file, sep="," if "," in open(raw_edge_file).readline() else "\t")
    df_ppi.columns = df_ppi.columns.str.strip()

    for _, row in df_ppi.iterrows():
        u_bid = str(row.get("BioGRID ID Interactor A", "")).split(".")[0]
        v_bid = str(row.get("BioGRID ID Interactor B", "")).split(".")[0]
        if not u_bid or not v_bid:
            continue

        u_attrs = id_to_attrs.get(u_bid, {})
        v_attrs = id_to_attrs.get(v_bid, {})
        bigG.add_node(u_bid, **u_attrs, feature_val=map_loc_to_category(u_attrs.get("location", "")))
        bigG.add_node(v_bid, **v_attrs, feature_val=map_loc_to_category(v_attrs.get("location", "")))
        bigG.add_edge(u_bid, v_bid, **row.to_dict(), raw_label=0, label=0)

    deg_dict = dict(bigG.degree())
    nx.set_node_attributes(bigG, deg_dict, "degree")
    bet_dict = nx.betweenness_centrality(bigG, k=256, seed=42)
    nx.set_node_attributes(bigG, bet_dict, "betweenness_centrality")

    deg_vals = np.array(list(deg_dict.values()), dtype=float)
    bet_vals = np.array(list(bet_dict.values()), dtype=float)
    global_stats = {
        "degree": {"q75": float(np.quantile(deg_vals, 0.75))},
        "betweenness_centrality": {"q25": float(np.quantile(bet_vals, 0.25))},
    }

    for u, v, d in bigG.edges(data=True):
        raw_label = int(
            encode_edge_feature(
                id_x=u,
                id_y=v,
                node_x_attr=bigG.nodes[u],
                node_y_attr=bigG.nodes[v],
                edge_row=d,
                global_stats=global_stats,
                sim_threshold=ml_threshold,
            )
        )
        d["raw_label"] = raw_label
        d["label"] = raw_label

    for node in bigG.nodes():
        bigG.nodes[node]["deg"] = bigG.degree(node)

    return bigG


def node_match_fn(n1, n2) -> bool:
    return (n1.get("feature_val") == n2.get("feature_val")) and (n1.get("deg", 0) <= n2.get("deg", 0))


def edge_match_fn(d1, d2) -> bool:
    return int(d1.get("label", 0)) == int(d2.get("label", 0))


def compute_rule_metrics_for_graph(
    graph_id: int,
    subG: nx.Graph,
    bigG: nx.Graph,
    match_limit: int,
    time_limit: int,
    confidence_threshold: float,
    support_threshold: int,
    enable_node_match: bool,
) -> List[RuleMetric]:
    neg_targets = [
        (u, v)
        for u, v, d in subG.edges(data=True)
        if bool(int(d.get("label", 0)) & (1 << EDGE_BIT_MAP["is_negative"]))
    ]

    metrics: List[RuleMetric] = []
    if not neg_targets:
        return metrics

    nm_func = node_match_fn if enable_node_match else None

    for target_u, target_v in neg_targets:
        premiseG = subG.copy()
        premiseG.remove_edge(target_u, target_v)
        if premiseG.number_of_edges() <= 0:
            continue

        GM = GraphMatcher(bigG, premiseG, node_match=nm_func, edge_match=edge_match_fn)

        try:
            with TimeLimit(time_limit):
                has_iso = GM.subgraph_is_isomorphic()
        except TimeoutException:
            continue

        if not has_iso:
            continue

        supp_premise = 0
        supp_negative = 0
        label_dist = Counter()
        updates = []
        status = "Finished"

        try:
            with TimeLimit(time_limit):
                for mapping in GM.subgraph_isomorphisms_iter():
                    real_u = mapping.get(target_u)
                    real_v = mapping.get(target_v)
                    if real_u is None or real_v is None:
                        continue

                    supp_premise += 1
                    real_label = int(bigG[real_u][real_v].get("label", 0)) if bigG.has_edge(real_u, real_v) else 0
                    label_dist[real_label] += 1

                    if bool(real_label & (1 << EDGE_BIT_MAP["is_negative"])):
                        supp_negative += 1
                    elif real_label == 0:
                        updates.append((str(real_u), str(real_v)))

                    if supp_premise >= match_limit:
                        status = "Limit"
                        break
        except TimeoutException:
            status = "TimeOut"

        if supp_premise == 0:
            continue

        confidence = supp_negative / supp_premise
        if confidence >= confidence_threshold and supp_premise >= support_threshold:
            metrics.append(
                RuleMetric(
                    graph_id=graph_id,
                    target_edge=(target_u, target_v),
                    confidence=confidence,
                    support_negative=supp_negative,
                    support_shape=supp_premise,
                    status=status,
                    label_dist=dict(label_dist),
                    updates=updates,
                )
            )

    return metrics


def extract_negative_centered_triads(
    G: nx.Graph,
    min_edges: int = 2,
    topk_per_neg: int = 5,
) -> List[Tuple[Tuple[int, int, int], nx.Graph]]:
    triads: List[Tuple[Tuple[int, int, int], nx.Graph]] = []
    neg_edges = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if bool(int(d.get("label", 0)) & (1 << EDGE_BIT_MAP["is_negative"]))
    ]

    for u, v in neg_edges:
        nbr_u = set(G.neighbors(u))
        nbr_v = set(G.neighbors(v))
        candidates = (nbr_u | nbr_v) - {u, v}
        common = (nbr_u & nbr_v) - {u, v}

        ordered = list(common) + [w for w in candidates if w not in common]
        picked = 0
        for w in ordered:
            sub = G.subgraph([u, v, w]).copy()
            for n in sub.nodes():
                sub.nodes[n]["deg"] = sub.degree(n)

            if sub.number_of_edges() < min_edges:
                continue
            triads.append(((u, v, w), sub))
            picked += 1
            if picked >= topk_per_neg:
                break

    return triads


def apply_updates_to_ppi(raw_edge_file: str, output_file: str, update_pairs: List[Tuple[str, str]]) -> Tuple[int, int]:
    df = pd.read_csv(raw_edge_file, sep="," if "," in open(raw_edge_file).readline() else "\t")
    df.columns = df.columns.str.strip()

    col_u = "BioGRID ID Interactor A"
    col_v = "BioGRID ID Interactor B"
    col_type = "type"
    if col_type not in df.columns:
        df[col_type] = "positive"

    modified = 0
    appended = 0

    update_pairs = list(set((str(a), str(b)) for a, b in update_pairs))
    for u, v in update_pairs:
        mask = ((df[col_u].astype(str) == u) & (df[col_v].astype(str) == v)) | (
            (df[col_u].astype(str) == v) & (df[col_v].astype(str) == u)
        )
        idxs = df.index[mask].tolist()

        if idxs:
            for idx in idxs:
                if str(df.at[idx, col_type]).lower() != "added_negative":
                    df.at[idx, col_type] = "added_negative"
                    modified += 1
        else:
            new_row = {c: np.nan for c in df.columns}
            new_row[col_u] = u
            new_row[col_v] = v
            new_row[col_type] = "added_negative"
            df.loc[len(df)] = new_row
            appended += 1

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    return modified, appended


def run_pipeline(cfg: PipelineConfig):
    edge_mapping = load_edge_label_mapping(cfg.edge_label_mapping)
    if cfg.reference_source == "train_data":
        bigG = build_big_graph_from_train_dataset(cfg.data_root, cfg.split, edge_mapping)
    elif cfg.reference_source == "raw_csv_graph":
        bigG = build_big_graph_from_raw(cfg.raw_edge_file, cfg.raw_node_file, cfg.ml_threshold)
    else:
        raise ValueError("reference_source must be 'train_data' or 'raw_csv_graph'")

    parsed = parse_graph_txt(cfg.generated_sample_file)
    all_rule_metrics: List[RuleMetric] = []
    all_updates: List[Tuple[str, str]] = []

    for gid, (x_list, edge_matrix) in enumerate(parsed):
        g = to_nx_graph(x_list, edge_matrix, edge_mapping)
        if g.number_of_edges() == 0:
            continue

        comps = list(nx.connected_components(g))
        if not comps:
            continue
        lcc = g.subgraph(max(comps, key=len)).copy()

        candidate_subgraphs: List[nx.Graph]
        if cfg.use_negative_centered_triads:
            triads = extract_negative_centered_triads(
                lcc,
                min_edges=cfg.triad_min_edges,
                topk_per_neg=cfg.triad_topk_per_negative_edge,
            )
            candidate_subgraphs = [sub for _, sub in triads]
        else:
            candidate_subgraphs = [lcc]

        for sub in candidate_subgraphs:
            metrics = compute_rule_metrics_for_graph(
                graph_id=gid,
                subG=sub,
                bigG=bigG,
                match_limit=cfg.match_limit,
                time_limit=cfg.time_limit,
                confidence_threshold=cfg.confidence_threshold,
                support_threshold=cfg.support_threshold,
                enable_node_match=cfg.enable_node_match,
            )
            if not metrics:
                continue

            all_rule_metrics.extend(metrics)
            for m in metrics:
                all_updates.extend(m.updates)

    all_rule_metrics.sort(key=lambda x: (x.confidence, x.support_shape), reverse=True)

    print("\n=== Rules (hit threshold) ===")
    for idx, m in enumerate(all_rule_metrics, 1):
        print(
            f"[{idx}] graph={m.graph_id} target={m.target_edge} "
            f"conf={m.confidence:.4f} support={m.support_shape} neg={m.support_negative} "
            f"status={m.status} updates={len(m.updates)}"
        )

    print(f"\nRules count: {len(all_rule_metrics)}")
    print(f"Candidate update pairs: {len(set(all_updates))}")

    if cfg.update_ppi and all_updates:
        modified, appended = apply_updates_to_ppi(cfg.raw_edge_file, cfg.update_edge_file, all_updates)
        print(f"Updated PPI file saved to: {cfg.update_edge_file}")
        print(f"Modified existing rows: {modified}, appended rows: {appended}")


if __name__ == "__main__":
    # 不使用命令行参数；直接在这里改配置即可运行
    config = PipelineConfig(
        generated_sample_file="",
        reference_source="raw_csv_graph",
        raw_edge_file="",
        raw_node_file="",
        edge_label_mapping="",
        update_edge_file="",
        update_ppi=False,
    )
    if not config.generated_sample_file:
        raise ValueError("Please set PipelineConfig.generated_sample_file before running.")
    run_pipeline(config)
