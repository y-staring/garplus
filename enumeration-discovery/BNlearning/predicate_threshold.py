from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def normalize_col_name(col: str) -> str:
    col = str(col).strip()
    col = col.replace("#", "")
    col = col.replace("[", "")
    col = col.replace("]", "")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace("/", "_")
    col = col.replace("-", "_")
    col = col.replace(".", "_")
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"_+", "_", col)
    return col.lower().strip("_")


def read_csv_auto(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    df.columns = [normalize_col_name(c) for c in df.columns]
    df = df.replace(
        {
            "-": np.nan,
            "": np.nan,
            " ": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "NULL": np.nan,
        }
    )
    return df


def split_tokens(value: Any) -> list[str]:
    """
    将文本/list字段拆成 token。

    对 GO、pathway、keyword、domain 等字段都可以用。
    """
    if value is None or pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    value = str(value)

    for sep in [";", "|", ","]:
        value = value.replace(sep, ";")

    tokens = [x.strip() for x in value.split(";") if x.strip()]

    return tokens


def infer_numeric_thresholds(
    df: pd.DataFrame,
    numeric_cols: list[str],
    quantiles: tuple[float, float] = (0.33, 0.66),
) -> dict[str, dict[str, float]]:
    """
    为数值字段自动确定 low / medium / high 分桶阈值。
    """
    thresholds: dict[str, dict[str, float]] = {}

    for col in numeric_cols:
        col = normalize_col_name(col)

        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce").dropna()

        if values.empty:
            continue

        thresholds[col] = {
            "low_upper": float(values.quantile(quantiles[0])),
            "high_lower": float(values.quantile(quantiles[1])),
        }

    return thresholds


def infer_frequent_values(
    df: pd.DataFrame,
    cols: list[str],
    min_count: int = 5,
    top_k: int = 100,
) -> dict[str, list[str]]:
    """
    为类别字段保留高频值。

    例如：
        experimental_system
        experimental_system_type
        throughput
        organism
    """
    result: dict[str, list[str]] = {}

    for col in cols:
        col = normalize_col_name(col)

        if col not in df.columns:
            continue

        counter = Counter()

        for value in df[col].dropna():
            value = str(value).strip()
            if value:
                counter[value] += 1

        items = [(v, c) for v, c in counter.items() if c >= min_count]
        items.sort(key=lambda x: x[1], reverse=True)

        result[col] = [v for v, _ in items[:top_k]]

    return result


def infer_frequent_tokens(
    df: pd.DataFrame,
    cols: list[str],
    min_count: int = 5,
    top_k: int = 100,
) -> dict[str, list[str]]:
    """
    为文本/list字段保留高频 token。

    例如：
        pathway
        keywords
        gene_ontology_ids
        protein_families
    """
    result: dict[str, list[str]] = {}

    for col in cols:
        col = normalize_col_name(col)

        if col not in df.columns:
            continue

        counter = Counter()

        for value in df[col].dropna():
            tokens = split_tokens(value)
            counter.update(tokens)

        items = [(t, c) for t, c in counter.items() if c >= min_count]
        items.sort(key=lambda x: x[1], reverse=True)

        result[col] = [t for t, _ in items[:top_k]]

    return result


def infer_overlap_thresholds(
    df: pd.DataFrame,
    cols: list[str],
    quantiles: tuple[float, float] = (0.33, 0.66),
) -> dict[str, dict[str, float]]:
    """
    可选：为 overlap_size 分桶准备阈值。

    注意：
        这个函数只根据单个节点字段的 token 数量估计阈值。
        更严格的做法是从真实边或采样节点对上计算 overlap_size。
        第一版可以不用这个。
    """
    result: dict[str, dict[str, float]] = {}

    for col in cols:
        col = normalize_col_name(col)

        if col not in df.columns:
            continue

        lengths = []

        for value in df[col].dropna():
            lengths.append(len(split_tokens(value)))

        if not lengths:
            continue

        arr = pd.Series(lengths)

        result[col] = {
            "low_upper": float(arr.quantile(quantiles[0])),
            "high_lower": float(arr.quantile(quantiles[1])),
        }

    return result


def build_predicate_config(
    protein_csv: str,
    ppi_csv: str,
    output_json: str = "predicate_config.json",
    min_count: int = 5,
    top_k: int = 100,
) -> dict[str, Any]:
    """
    生成 predicate_config.json。

    这个文件是 predicate_construction.py 的输入。
    """
    protein_df = read_csv_auto(protein_csv)
    ppi_df = read_csv_auto(ppi_csv)

    # =========================
    # 节点数值字段
    # =========================
    node_numeric_cols = [
        "Length",
        "Annotation",
        # "index",  # 内部编号，没有语义，先不用
    ]

    # =========================
    # 边数值字段
    # =========================
    edge_numeric_cols = [
        "Score",
    ]

    # =========================
    # 节点类别字段
    # =========================
    node_categorical_cols = [
        "Reviewed",
        "Organism",
        "Protein existence",
        "DNA binding",
    ]

    # =========================
    # 边类别字段
    # =========================
    edge_categorical_cols = [
        "Experimental System",
        "Experimental System Type",
        "Throughput",
        "Modification",
        # "Qualifications",  # 长文本，太稀疏，先不用
        "Ontology Term Categories",
        "Ontology Term Types",
        # "Organism Name Interactor A",  # 如果基本都是 Homo sapiens，也可以不用
        # "Organism Name Interactor B",
    ]

    # =========================
    # 节点文本/list字段
    # =========================
    node_token_cols = [
        "location",
        "pathway",
        "domain",
        "Keywords",
        "Gene Ontology biological process",
        "Gene Ontology cellular component",
        "Gene Ontology molecular function",
        "Gene Ontology GO",
        "Gene Ontology IDs",
        "Subcellular location CC",
        "Transmembrane",
        "Cross-link",
        "Disulfide bond",
        "Glycosylation",
        "Modified residue",
        "Post-translational modification",
        "Domain CC",
        "Domain FT",
        "Motif",
        "Region",
        "Repeat",
        "Sequence similarities",
        "Zinc finger",
        "Protein families",
        # 下面这些先不用，太长、太脏、太稀疏
        # "Function CC",
        # "Activity regulation",
        # "Topological domain",
        # "Chain",
    ]

    # =========================
    # 边文本/list字段
    # =========================
    edge_token_cols = [
        "Ontology Term IDs",
        "Ontology Term Names",
        "Ontology Term Qualifier IDs",
        "Ontology Term Qualifier Names",
    ]

    config: dict[str, Any] = {
        "node_numeric_thresholds": infer_numeric_thresholds(
            protein_df,
            node_numeric_cols,
        ),
        "edge_numeric_thresholds": infer_numeric_thresholds(
            ppi_df,
            edge_numeric_cols,
        ),
        "node_categorical_values": infer_frequent_values(
            protein_df,
            node_categorical_cols,
            min_count=min_count,
            top_k=top_k,
        ),
        "edge_categorical_values": infer_frequent_values(
            ppi_df,
            edge_categorical_cols,
            min_count=min_count,
            top_k=top_k,
        ),
        "node_token_values": infer_frequent_tokens(
            protein_df,
            node_token_cols,
            min_count=min_count,
            top_k=top_k,
        ),
        "edge_token_values": infer_frequent_tokens(
            ppi_df,
            edge_token_cols,
            min_count=min_count,
            top_k=top_k,
        ),
        "node_pair_overlap_fields": [
            normalize_col_name(c)
            for c in [
                "location",
                "pathway",
                "domain",
                "Keywords",
                "Gene Ontology IDs",
                "Gene Ontology biological process",
                "Gene Ontology cellular component",
                "Gene Ontology molecular function",
                "Protein families",
                "Subcellular location CC",
                "Domain CC",
                "Domain FT",
                "Motif",
                "Region",
                "Repeat",
                "Sequence similarities",
            ]
        ],
        "settings": {
            "min_count": min_count,
            "top_k": top_k,
            "numeric_bins": ["low", "medium", "high"],
            "share_overlap_min": 1,
        },
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"[Predicate Config] saved to {output_json}")
    print(f"[Node numeric] {list(config['node_numeric_thresholds'].keys())}")
    print(f"[Edge numeric] {list(config['edge_numeric_thresholds'].keys())}")
    print(f"[Node token fields] {list(config['node_token_values'].keys())}")
    print(f"[Edge categorical fields] {list(config['edge_categorical_values'].keys())}")

    return config


if __name__ == "__main__":
    build_predicate_config(
        protein_csv="/home/yyyy/codework/GARplus/enumeration-discovery/去病图数据/protein.csv",
        ppi_csv="/home/yyyy/codework/GARplus/enumeration-discovery/去病图数据/protein_protein.csv",
        output_json="/home/yyyy/codework/GARplus/enumeration-discovery/processed/ppi/predicate_config.json",
        min_count=5,
        top_k=100,
    )
