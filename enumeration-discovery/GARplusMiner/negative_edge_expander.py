from __future__ import annotations

import ast
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


MISSING_LABELS = {"", "unknown", "candidate", "unlabeled", "none", "nan", "na", "n/a"}


# =========================
# 在这里直接改运行参数
# =========================
# ACTIVE_DATASET 控制当前处理哪个数据集，可选 "PPI"、"DDA"、"TI"。
# 主流程会把每个数据集挖到的 deduped_rule 写入对应的 processed/*/deduped_rules.txt。
# 本脚本读取这些规则，扫描 interaction CSV，只输出“会被规则扩展为 negative 的边”的端点索引。
# 输出不再是整张 interaction 表：
# - PPI 输出 index_A,index_B
# - DDA 输出 chemical_index,disease_index
# - TI 输出 gene_index,disease_index
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "\u53bb\u75c5\u56fe\u6570\u636e"
PROCESSED_DIR = BASE_DIR / "processed"

ACTIVE_DATASET = "PPI"
LABEL_COLUMN = "interaction_label"
NEGATIVE_VALUE = "negative"
SIMILARITY_THRESHOLD = 0.85
ONLY_LABELS = MISSING_LABELS
OVERWRITE_EXISTING = False


@dataclass(frozen=True)
class NegativeExpansionRule:
    antecedent: tuple[str, ...]
    consequent: str
    raw_text: str = ""

    @property
    def negative_label(self) -> str:
        if "=" not in self.consequent:
            return "negative"
        return self.consequent.split("=", 1)[1].strip()


@dataclass(frozen=True)
class ExpansionConfig:
    dataset_name: str
    input_csv: Path
    output_csv: Path
    rules_file: Path
    src_column: str
    dst_column: str
    source_node_csv: Optional[Path] = None
    target_node_csv: Optional[Path] = None
    source_node_index_column: str = "index"
    target_node_index_column: str = "index"
    label_column: str = LABEL_COLUMN
    negative_value: str = NEGATIVE_VALUE
    similarity_threshold: float = SIMILARITY_THRESHOLD
    only_labels: Optional[set[str]] = None
    overwrite_existing: bool = OVERWRITE_EXISTING


DATASET_CONFIGS = {
    "PPI": ExpansionConfig(
        dataset_name="PPI",
        input_csv=DATA_DIR / "protein_protein_signed.csv",
        output_csv=PROCESSED_DIR / "ppi" / "rule_negative_pairs.csv",
        rules_file=PROCESSED_DIR / "ppi" / "deduped_rules.txt",
        source_node_csv=DATA_DIR / "protein.csv",
        target_node_csv=DATA_DIR / "protein.csv",
        src_column="index_A",
        dst_column="index_B",
        only_labels=set(ONLY_LABELS),
    ),
    "DDA": ExpansionConfig(
        dataset_name="DDA",
        input_csv=DATA_DIR / "drug_disease_signed.csv",
        output_csv=PROCESSED_DIR / "dda" / "rule_negative_pairs.csv",
        rules_file=PROCESSED_DIR / "dda" / "deduped_rules.txt",
        source_node_csv=DATA_DIR / "drug.csv",
        target_node_csv=DATA_DIR / "disease.csv",
        src_column="chemical_index",
        dst_column="disease_index",
        only_labels=set(ONLY_LABELS),
    ),
    "TI": ExpansionConfig(
        dataset_name="TI",
        input_csv=DATA_DIR / "gene_disease_signed.csv",
        output_csv=PROCESSED_DIR / "ti" / "rule_negative_pairs.csv",
        rules_file=PROCESSED_DIR / "ti" / "deduped_rules.txt",
        source_node_csv=DATA_DIR / "gene.csv",
        target_node_csv=DATA_DIR / "disease.csv",
        src_column="gene_index",
        dst_column="disease_index",
        only_labels=set(ONLY_LABELS),
    ),
}

CONFIG = DATASET_CONFIGS[ACTIVE_DATASET]


def normalize_key(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^0-9a-zA-Z]+", "_", text)
    return text.strip("_")


def normalize_value(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def split_literal(literal: str) -> tuple[str, str]:
    if "=" not in literal:
        raise ValueError(f"rule literal must contain '=': {literal}")
    key, value = literal.split("=", 1)
    return key.strip(), normalize_value(value)


def extract_python_tuple_after(text: str, marker: str) -> Optional[tuple[str, ...]]:
    start = text.find(marker)
    if start < 0:
        return None
    start += len(marker)
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "(":
        return None

    depth = 0
    quote: Optional[str] = None
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                parsed = ast.literal_eval(text[start : index + 1])
                if isinstance(parsed, str):
                    return (parsed,)
                return tuple(str(item) for item in parsed)
    raise ValueError(f"could not parse tuple after {marker}")


def extract_value_after(text: str, marker: str) -> Optional[str]:
    match = re.search(rf"{re.escape(marker)}([^\s]+)", text)
    return match.group(1) if match else None


def parse_rule_line(text: str) -> NegativeExpansionRule:
    """解析主流程输出的一行 `deduped_rule`。

    优先读取 raw_antecedent/raw_consequent，因为它保留了具体端点，例如
    `v0.degree_bin=low`。如果规则文件只有去重后的 antecedent，则 `v*.xxx`
    会在匹配时解释成 v0 或 v1 任意一端满足即可。
    """

    antecedent = extract_python_tuple_after(text, "raw_antecedent=")
    if antecedent is None:
        antecedent = extract_python_tuple_after(text, "antecedent=")
    if antecedent is None:
        raise ValueError(f"missing antecedent/raw_antecedent in rule line: {text}")
    consequent = extract_value_after(text, "raw_consequent=") or extract_value_after(text, "consequent=")
    if not consequent:
        raise ValueError(f"missing consequent/raw_consequent in rule line: {text}")
    return NegativeExpansionRule(antecedent=antecedent, consequent=consequent, raw_text=text.strip())


def load_rules(rules_file: Path) -> list[NegativeExpansionRule]:
    """从 `deduped_rules_output_path` 写出的规则文件读取所有 deduped_rule。"""

    rules: list[NegativeExpansionRule] = []
    with Path(rules_file).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line or "deduped_rule" not in line:
                continue
            rules.append(parse_rule_line(line))
    if not rules:
        raise ValueError(f"no deduped_rule lines found in {rules_file}")
    return rules


def read_rows(path: str) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def load_node_attrs(path: Optional[str], index_column: str) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    result: dict[str, dict[str, str]] = {}
    rows, _fields = read_rows(path)
    for row in rows:
        node_id = normalize_value(row.get(index_column))
        if not node_id:
            continue
        result[node_id] = {normalize_key(key): normalize_value(value) for key, value in row.items()}
    return result


def row_value(row: dict[str, str], normalized_key: str) -> str:
    if normalized_key in row:
        return normalize_value(row.get(normalized_key))
    for key, value in row.items():
        if normalize_key(key) == normalized_key:
            return normalize_value(value)
    return ""


def endpoint_ids(row: dict[str, str], src_column: str, dst_column: str) -> tuple[str, str]:
    src = row_value(row, normalize_key(src_column))
    dst = row_value(row, normalize_key(dst_column))
    return src, dst


def degree_bins(rows: list[dict[str, str]], src_column: str, dst_column: str) -> dict[str, str]:
    counts: Counter[str] = Counter()
    for row in rows:
        src, dst = endpoint_ids(row, src_column, dst_column)
        if src:
            counts[src] += 1
        if dst:
            counts[dst] += 1
    values = sorted(counts.values())
    low_cut = values[max(0, len(values) // 3 - 1)] if values else 0
    high_cut = values[max(0, (2 * len(values)) // 3 - 1)] if values else 0
    bins: dict[str, str] = {}
    for node_id, degree in counts.items():
        if degree <= low_cut:
            bins[node_id] = "low"
        elif degree <= high_cut:
            bins[node_id] = "medium"
        else:
            bins[node_id] = "high"
    return bins


def edge_context(
    row: dict[str, str],
    source_node_attrs: dict[str, dict[str, str]],
    target_node_attrs: dict[str, dict[str, str]],
    node_degree_bins: dict[str, str],
    src_column: str,
    dst_column: str,
    similarity_threshold: float,
) -> dict[str, dict[str, str]]:
    """构造规则匹配时使用的 e0/v0/v1 命名空间。

    `e0.xxx` 来自 interaction CSV 的边属性。
    `v0.xxx` 来自源端点属性：PPI 是 index_A，DDA 是 Drug，TI 是 Gene。
    `v1.xxx` 来自目标端点属性：PPI 是 index_B，DDA/TI 是 Disease。
    `v*.xxx` 在 literal_matches 中解释成 v0 或 v1 任意一端满足。
    """

    src, dst = endpoint_ids(row, src_column, dst_column)
    edge_attrs = {normalize_key(key): normalize_value(value) for key, value in row.items()}

    if "ml_similarity_pred" not in edge_attrs:
        raw_pred = edge_attrs.get("similarity_pred")
        if raw_pred in {"0", "1"}:
            edge_attrs["ml_similarity_pred"] = "yes" if raw_pred == "1" else "no"
        elif edge_attrs.get("similarity_score"):
            try:
                edge_attrs["ml_similarity_pred"] = (
                    "yes" if float(edge_attrs["similarity_score"]) >= similarity_threshold else "no"
                )
            except ValueError:
                pass

    left = dict(source_node_attrs.get(src, {}))
    right = dict(target_node_attrs.get(dst, {}))
    for node_id, attrs in ((src, left), (dst, right)):
        degree_bin = node_degree_bins.get(node_id, "")
        attrs.setdefault("degree_bin", degree_bin)
        attrs.setdefault("degree_bucket", degree_bin)

    return {"e0": edge_attrs, "v0": left, "v1": right, "v*": {"_src": src, "_dst": dst}}


def literal_matches(literal: str, context: dict[str, dict[str, str]]) -> bool:
    """检查一个前件 literal，例如 `e0.ml_similarity_pred=yes`。"""

    key, expected = split_literal(literal)
    if "." not in key:
        return False
    entity, attr = key.split(".", 1)
    attr = normalize_key(attr)
    if entity == "v*":
        return any(context.get(vertex, {}).get(attr) == expected for vertex in ("v0", "v1"))
    return context.get(entity, {}).get(attr) == expected


def rule_matches(rule: NegativeExpansionRule, context: dict[str, dict[str, str]]) -> bool:
    """一条规则只有在所有前件 literal 都成立时才命中。"""

    return all(literal_matches(literal, context) for literal in rule.antecedent)


def can_export(row: dict[str, str], label_column: str, allowed_existing: set[str], overwrite_existing: bool) -> bool:
    if overwrite_existing:
        return True
    return normalize_value(row.get(label_column)) in allowed_existing


def expand_negative_edges(config: ExpansionConfig) -> dict[str, int]:
    """导出会被负规则扩展为 negative 的 interaction 端点索引。

    扩展流程：
    1. 读取 interaction CSV。
    2. 读取主流程写出的 deduped_rules.txt。
    3. 对每条 interaction 构造 e0/v0/v1 属性上下文。
    4. 如果满足某条 consequent=negative 的规则，就输出该 interaction 的端点索引。
    """

    rows, _fields = read_rows(str(config.input_csv))
    rules = load_rules(config.rules_file)
    source_node_attrs = load_node_attrs(
        str(config.source_node_csv) if config.source_node_csv else None,
        config.source_node_index_column,
    )
    target_node_attrs = load_node_attrs(
        str(config.target_node_csv) if config.target_node_csv else None,
        config.target_node_index_column,
    )
    bins = degree_bins(rows, config.src_column, config.dst_column)
    allowed_existing = set(config.only_labels or MISSING_LABELS)

    matched = 0
    exported = 0
    skipped_existing = 0
    output_rows: list[dict[str, str]] = []
    for row in rows:
        context = edge_context(
            row,
            source_node_attrs,
            target_node_attrs,
            bins,
            config.src_column,
            config.dst_column,
            config.similarity_threshold,
        )
        hit_index = None
        hit_rule = None
        for index, rule in enumerate(rules):
            if rule.negative_label != config.negative_value:
                continue
            if rule_matches(rule, context):
                hit_index = index
                hit_rule = rule
                break
        if hit_rule is None:
            continue
        matched += 1
        if not can_export(row, config.label_column, allowed_existing, config.overwrite_existing):
            skipped_existing += 1
            continue
        output_rows.append(
            {
                config.src_column: row_value(row, normalize_key(config.src_column)),
                config.dst_column: row_value(row, normalize_key(config.dst_column)),
                "negative_rule_index": str(hit_index),
                "negative_rule_antecedent": " & ".join(hit_rule.antecedent),
            }
        )
        exported += 1

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with Path(config.output_csv).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[config.src_column, config.dst_column, "negative_rule_index", "negative_rule_antecedent"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    return {
        "rows": len(rows),
        "rules": len(rules),
        "matched": matched,
        "exported_pairs": exported,
        "skipped_existing_label": skipped_existing,
    }


def main() -> None:
    summary = expand_negative_edges(CONFIG)
    print(
        "[NegativeEdgeExpansion] "
        + f"dataset={CONFIG.dataset_name} "
        + " ".join(f"{key}={value}" for key, value in summary.items())
        + f" output={CONFIG.output_csv}"
    )


if __name__ == "__main__":
    main()
