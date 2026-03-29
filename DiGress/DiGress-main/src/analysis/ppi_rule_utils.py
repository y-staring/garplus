import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from networkx.algorithms.isomorphism import GraphMatcher

from src.datasets.ppi_dataset import PPIGraphDataset
from src.datasets.ppi_dataset_order_embedding import (
    EDGE_BIT_MAP,
    TimeLimit,
    TimeoutException,
    encode_edge_feature,
    is_negative_raw_bitmask,
    map_loc_to_category,
)


@dataclass
class RuleMetric:
    confidence: float
    support_negative: int
    # support_base: int
    support_shape: int
    status: str
    has_negative_target: bool
    matched_negative_edges: List[Tuple[str, str]]


def _norm_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def node_match_fn(n1, n2) -> bool:
    return (n1.get("feature_val") == n2.get("feature_val")) #and (n1.get("deg", 0) <= n2.get("deg", 0))


def edge_match_fn(d1, d2) -> bool:
    return int(d1.get("label", 0)) == int(d2.get("label", 0))


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

def edge_class_to_raw_bitmask(edge_class: int, edge_label_mapping: Dict) -> int:
    edge_class = int(edge_class)
    if edge_class == 0:
        return 0
    class_to_bitmask = edge_label_mapping["class_to_bitmask"]
    if edge_class in class_to_bitmask:
        return int(class_to_bitmask[edge_class])
    # 兼容旧可视化导出：edge_type = raw_bitmask + 1
    return max(edge_class - 1, 0)


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


def build_reference_big_graph(
    source_mode: str,
    data_root: str,
    split: str,
    edge_label_mapping: Dict,
    raw_edge_file: str,
    raw_node_file: str,
    ml_threshold: float,
) -> nx.Graph:
    if source_mode == "train_data":
        return build_big_graph_from_train_dataset(data_root=data_root, split=split, edge_label_mapping=edge_label_mapping)
    if source_mode == "raw_csv_graph":
        return build_big_graph_from_raw(raw_edge_file=raw_edge_file, raw_node_file=raw_node_file, ml_threshold=ml_threshold)
    raise ValueError("source_mode must be one of: train_data, raw_csv_graph")


def compute_rule_metrics_for_graph(
    subG: nx.Graph,
    bigG: nx.Graph,
    match_limit: int,
    time_limit: int,
    confidence_threshold: float,
    support_threshold: int,
    # denominator_mode: str,
    enable_node_match: bool,
    keep_only_negative_rule: bool,
) -> Optional[RuleMetric]:
    neg_targets = [
        (u, v)
        for u, v, d in subG.edges(data=True)
        if bool(int(d.get("label", 0)) & (1 << EDGE_BIT_MAP["is_negative"]))
    ]

    if keep_only_negative_rule and not neg_targets:
        return None

    if not neg_targets:
        return None

    #为什么是第一个
    target_u, target_v = neg_targets[0]
    premiseG = subG.copy()
    premiseG.remove_edge(target_u, target_v)
    if premiseG.number_of_edges() <= 0:
        return None

    nm_func = node_match_fn if enable_node_match else None
    GM = GraphMatcher(bigG, premiseG, node_match=nm_func, edge_match=edge_match_fn)

    supp_premise = 0
    supp_pattern = 0
    supp_negative = 0
    unique_target_pairs = set()
    matched_negative_edges: Set[Tuple[str, str]] = set()
    status = "Finished"

    try:
        with TimeLimit(time_limit):
            for mapping in GM.subgraph_isomorphisms_iter():
                inv_mapping = {pnode: gnode for gnode, pnode in mapping.items()}
                real_u = inv_mapping.get(target_u)
                real_v = inv_mapping.get(target_v)
                if real_u is None or real_v is None:
                    continue

                supp_premise += 1
                pair_key = tuple(sorted((str(real_u), str(real_v))))
                unique_target_pairs.add(pair_key)

                real_label = int(bigG[real_u][real_v].get("label", 0)) if bigG.has_edge(real_u, real_v) else 0
                if bool(real_label & (1 << EDGE_BIT_MAP["is_negative"])):
                    supp_negative += 1
                elif real_label == 0:
                    supp_pattern += 1
                    matched_negative_edges.add(tuple(sorted((str(real_u), str(real_v)))))

                if supp_premise >= match_limit:
                    status = "Limit"
                    break
    except TimeoutException:
        status = "TimeOut"
    except Exception:
        return None

    # supp_shape = len(unique_target_pairs)
    
    if supp_premise == 0:
        return None

    confidence = supp_pattern / supp_premise
    if confidence >= confidence_threshold and supp_pattern >= support_threshold:
        return RuleMetric(
            confidence=confidence,
            support_negative=supp_negative,
            support_shape=supp_premise,
            status=status,
            # denominator_mode=denominator_mode,
            has_negative_target=True,
            matched_negative_edges=sorted(matched_negative_edges),
        )


def save_negative_edges_csv(
    negative_edges: List[Tuple[str, str]],
    output_path: str,
    src_col: str = "src",
    dst_col: str = "dst",
    rel_col: str = "rel",
    rel_name: str = "ppi",
    label_col: str = "label",
    negative_label: int = 2,
) -> str:
    rows = []
    for u, v in sorted(set(tuple(sorted((str(u), str(v)))) for u, v in negative_edges)):
        rows.append({src_col: u, dst_col: v, rel_col: rel_name, label_col: negative_label})

    out_df = pd.DataFrame(rows, columns=[src_col, dst_col, rel_col, label_col])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)
    return output_path


def build_updated_edge_file_with_random_negatives(
    edge_old_file: str,
    discovered_negative_edges: List[Tuple[str, str]],
    output_path: str,
    src_col: str = "src",
    dst_col: str = "dst",
    rel_col: str = "rel",
    rel_name: str = "ppi",
    label_col: str = "label",
    negative_label: int = 2,
    num_additional_negatives: int = 0,
    random_seed: int = 42,
) -> str:
    df_base = pd.read_csv(edge_old_file)
    required = {src_col, dst_col}
    if not required.issubset(df_base.columns):
        raise ValueError(f"edge_old_file must contain columns: {required}")

    existing_edges: Set[Tuple[str, str]] = set()
    for _, row in df_base.iterrows():
        u, v = str(row[src_col]), str(row[dst_col])
        existing_edges.add((u, v))
        existing_edges.add((v, u))

    generated_negatives = []
    for u, v in discovered_negative_edges:
        uu, vv = str(u), str(v)
        if uu == vv:
            continue
        if (uu, vv) in existing_edges:
            continue
        generated_negatives.append({
            src_col: uu,
            dst_col: vv,
            rel_col: rel_name,
            label_col: negative_label,
        })
        existing_edges.add((uu, vv))
        existing_edges.add((vv, uu))

    rng = random.Random(random_seed)
    all_ids = sorted(set(str(x) for x in df_base[src_col].tolist() + df_base[dst_col].tolist()))
    if len(all_ids) >= 2 and num_additional_negatives > 0:
        count = 0
        while count < num_additional_negatives:
            u = rng.choice(all_ids)
            v = rng.choice(all_ids)
            if u == v:
                continue
            if (u, v) in existing_edges:
                continue
            generated_negatives.append({
                src_col: u,
                dst_col: v,
                rel_col: rel_name,
                label_col: negative_label,
            })
            existing_edges.add((u, v))
            existing_edges.add((v, u))
            count += 1

    df_new_neg = pd.DataFrame(generated_negatives)
    df_random_out = pd.concat([df_base, df_new_neg], ignore_index=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_random_out.to_csv(output_path, index=False)
    return output_path