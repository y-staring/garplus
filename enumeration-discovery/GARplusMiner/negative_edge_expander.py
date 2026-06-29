from __future__ import annotations

import ast
import csv
import json
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys
import csv
import time

max_int = sys.maxsize
while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)
MISSING_LABELS = {"", "unknown", "candidate", "unlabeled", "none", "nan", "na", "n/a"}
NEUTRAL_LABELS = {"neutral", "netural"}
MISSING_VALUES = MISSING_LABELS | {"-", "null", "inf", "-inf"}


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

ACTIVE_DATASET = "DDA"
LABEL_COLUMN = "interaction_label"
NEGATIVE_VALUE = "negative"
SIMILARITY_THRESHOLD = 0.85
ONLY_LABELS = MISSING_LABELS | NEUTRAL_LABELS
OVERWRITE_EXISTING = False
ALLOW_POSITIVE_RELABEL = False
ALLOW_EXISTING_NEGATIVE_RELABEL = False
EXPANSION_MODE = "anchored_existing_edge_labeling"  # "anchored_existing_edge_labeling", "existing_edge_labeling", "matched_existing", "candidate_non_edges", or "body_rematch_non_edges"
MAX_CANDIDATES_PER_ANCHOR = 50
MAX_NEW_NEG_PER_NODE = 100
MAX_NEW_NEG_TOTAL = None
MAX_BODY_MATCHES_PER_RULE = 200000
MAX_ANCHORED_PARTIAL_MATCHES = 1000
DEBUG_PROGRESS = True
DEBUG_USABLE_RULES = True
DEBUG_MAX_PRINT_RULES = 50
DEBUG_EVERY_ROWS = 50000
DEBUG_EVERY_CHECKED_ROWS = 10000
EARLY_STOP_ON_FIRST_MATCH = True
USE_FIRST_ROW_PER_PAIR = False
INCREMENTAL_WRITE_OUTPUT = True
FLUSH_EVERY_EXPORTS = 100
MIN_SRC_DEGREE = 1
MIN_DST_DEGREE = 1
REQUIRE_RULE_HAS_PAIR_OR_CONTEXT = True

COMPUTABLE_VIRTUAL_E0_ATTRS = {
    "ml_similarity_pred",
    "similarity_pred",
    "similarity_score",
    "common_neighbor_bin",
    "common_neighbor_count",
}


def configured_only_labels() -> Optional[set[str]]:
    return None if ONLY_LABELS is None else set(ONLY_LABELS)


def debug_log(config: "ExpansionConfig", message: str) -> None:
    if config.debug_progress:
        print(f"[NegativeEdgeExpansionDebug] {message}", flush=True)


@dataclass(frozen=True)
class NegativeExpansionRule:
    pattern_id: int
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
    pattern_instances_file: Path
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
    allow_positive_relabel: bool = ALLOW_POSITIVE_RELABEL
    allow_existing_negative_relabel: bool = ALLOW_EXISTING_NEGATIVE_RELABEL
    expansion_mode: str = EXPANSION_MODE
    max_candidates_per_anchor: int = MAX_CANDIDATES_PER_ANCHOR
    max_new_neg_per_node: int = MAX_NEW_NEG_PER_NODE
    max_new_neg_total: Optional[int] = MAX_NEW_NEG_TOTAL
    max_body_matches_per_rule: int = MAX_BODY_MATCHES_PER_RULE
    max_anchored_partial_matches: int = MAX_ANCHORED_PARTIAL_MATCHES
    debug_progress: bool = DEBUG_PROGRESS
    debug_usable_rules: bool = DEBUG_USABLE_RULES
    debug_max_print_rules: int = DEBUG_MAX_PRINT_RULES
    debug_every_rows: int = DEBUG_EVERY_ROWS
    debug_every_checked_rows: int = DEBUG_EVERY_CHECKED_ROWS
    early_stop_on_first_match: bool = EARLY_STOP_ON_FIRST_MATCH
    use_first_row_per_pair: bool = USE_FIRST_ROW_PER_PAIR
    incremental_write_output: bool = INCREMENTAL_WRITE_OUTPUT
    flush_every_exports: int = FLUSH_EVERY_EXPORTS
    min_src_degree: int = MIN_SRC_DEGREE
    min_dst_degree: int = MIN_DST_DEGREE
    require_rule_has_pair_or_context: bool = REQUIRE_RULE_HAS_PAIR_OR_CONTEXT


DATASET_CONFIGS = {
    "PPI": ExpansionConfig(
        dataset_name="PPI",
        input_csv=DATA_DIR / "protein_protein_signed.csv",
        output_csv=PROCESSED_DIR / "ppi" / "rule_negative_pairs_existing_edge_labeling.csv" 
        if (EXPANSION_MODE == "anchored_existing_edge_labeling") 
        else PROCESSED_DIR / "ppi" / "rule_negative_pairs.csv",
        rules_file=PROCESSED_DIR / "ppi" / "deduped_rules.txt",
        pattern_instances_file=PROCESSED_DIR / "ppi" / "pattern_instances.jsonl",
        source_node_csv=DATA_DIR / "protein.csv",
        target_node_csv=DATA_DIR / "protein.csv",
        src_column="index_A",
        dst_column="index_B",
        only_labels=configured_only_labels(),
    ),
    "DDA": ExpansionConfig(
        dataset_name="DDA",
        input_csv=DATA_DIR / "drug_disease_signed.csv",
        output_csv=PROCESSED_DIR / "dda" / "rule_negative_pairs_existing_edge_labeling.csv" 
        if (EXPANSION_MODE == "anchored_existing_edge_labeling") 
        else PROCESSED_DIR / "dda" / "rule_negative_pairs_0626.csv",
        rules_file=PROCESSED_DIR / "dda" / "deduped_rules.txt",
        pattern_instances_file=PROCESSED_DIR / "dda" / "pattern_instances.jsonl",
        source_node_csv=DATA_DIR / "drug.csv",
        target_node_csv=DATA_DIR / "disease.csv",
        src_column="chemical_index",
        dst_column="disease_index",
        only_labels=configured_only_labels(),
    ),
    "TI": ExpansionConfig(
        dataset_name="TI",
        input_csv=DATA_DIR / "gene_disease_signed.csv",
        output_csv=PROCESSED_DIR / "ti" / "rule_negative_pairs_existing_edge_labeling.csv" 
        if (EXPANSION_MODE == "anchored_existing_edge_labeling") 
        else PROCESSED_DIR / "ti" / "rule_negative_pairs_0626.csv",

        rules_file=PROCESSED_DIR / "ti" / "deduped_rules.txt",
        pattern_instances_file=PROCESSED_DIR / "ti" / "pattern_instances.jsonl",
        source_node_csv=DATA_DIR / "gene.csv",
        target_node_csv=DATA_DIR / "disease.csv",
        src_column="gene_index",
        dst_column="disease_index",
        only_labels=configured_only_labels(),
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


def split_literal(literal: str) -> tuple[str, str, str]:
    if "!=" in literal:
        key, value = literal.split("!=", 1)
        return key.strip(), "!=", normalize_value(value)
    if "=" in literal:
        key, value = literal.split("=", 1)
        return key.strip(), "=", normalize_value(value)
    raise ValueError(f"rule literal must contain '=' or '!=': {literal}")


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


def extract_pattern_id(text: str) -> int:
    match = re.search(r"pattern_id=(\d+)", text)
    if not match:
        raise ValueError(f"missing pattern_id in rule line: {text}")
    return int(match.group(1))


def parse_rule_line(text: str) -> NegativeExpansionRule:
    """解析主流程输出的一行 `deduped_rule`。

    优先读取 raw_antecedent/raw_consequent，因为它保留了具体端点，例如
    `v0.degree_bin=low`。如果规则文件只有去重后的 antecedent，则 `v*.xxx`
    会在匹配时解释成 v0 或 v1 任意一端满足即可。
    """

    pattern_id = extract_pattern_id(text)
    antecedent = extract_python_tuple_after(text, "raw_antecedent=")
    if antecedent is None:
        antecedent = extract_python_tuple_after(text, "antecedent=")
    if antecedent is None:
        raise ValueError(f"missing antecedent/raw_antecedent in rule line: {text}")
    consequent = extract_value_after(text, "raw_consequent=") or extract_value_after(text, "consequent=")
    if not consequent:
        raise ValueError(f"missing consequent/raw_consequent in rule line: {text}")
    return NegativeExpansionRule(
        pattern_id=pattern_id,
        antecedent=antecedent,
        consequent=consequent,
        raw_text=text.strip(),
    )


def load_rules(rules_file: Path) -> list[NegativeExpansionRule]:
    """从 `deduped_rules_output_path` 写出的规则文件读取所有 deduped_rule。"""

    rules: list[NegativeExpansionRule] = []
    with Path(rules_file).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line or ("deduped_rule" not in line and "cover_rule" not in line):
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


def compute_score_bins(
    rows: list[dict[str, str]],
    src_column: str,
    dst_column: str,
) -> tuple[Optional[float], Optional[float]]:
    score_values = []
    for row in rows:
        src, dst = endpoint_ids(row, src_column, dst_column)
        if not src or not dst:
            continue
        try:
            score_values.append(float(row_value(row, "inferencescore")))
        except (TypeError, ValueError):
            continue
    score_values.sort()
    if not score_values:
        return None, None
    low_index = min(len(score_values) - 1, max(0, int(round((len(score_values) - 1) * 0.33))))
    high_index = min(len(score_values) - 1, max(0, int(round((len(score_values) - 1) * 0.66))))
    return score_values[low_index], score_values[high_index]


def build_enriched_edge_attrs(
    row: dict[str, str],
    dataset_name: str,
    score_low: Optional[float] = None,
    score_high: Optional[float] = None,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> dict[str, str]:
    attrs = {normalize_key(key): normalize_value(value) for key, value in row.items()}

    direct_evidence = attrs.get("directevidence", attrs.get("direct_evidence", ""))
    attrs["direct_evidence_category"] = (
        "inference_evidence"
        if direct_evidence in MISSING_VALUES
        else "marker_mechanism" if direct_evidence == "marker/mechanism" else "other"
    )

    if dataset_name == "TI":
        presence_value = attrs.get("inferencegenesymbol", "")
        attrs["inference_gene_present"] = "no" if presence_value in MISSING_VALUES else "yes"
    elif dataset_name == "DDA":
        presence_value = attrs.get("inferencechemicalname", "")
        attrs["inference_chemical_present"] = "no" if presence_value in MISSING_VALUES else "yes"

    if score_low is not None and score_high is not None:
        try:
            score = float(attrs.get("inferencescore", ""))
            attrs["inference_score_bin"] = "low" if score <= score_low else "medium" if score <= score_high else "high"
        except (TypeError, ValueError):
            attrs["inference_score_bin"] = "missing"
    else:
        attrs.setdefault("inference_score_bin", "missing")

    if "ml_similarity_pred" not in attrs:
        raw_pred = attrs.get("similarity_pred")
        if raw_pred in {"0", "1"}:
            attrs["ml_similarity_pred"] = "yes" if raw_pred == "1" else "no"
        elif attrs.get("similarity_score"):
            try:
                attrs["ml_similarity_pred"] = (
                    "yes" if float(attrs["similarity_score"]) >= similarity_threshold else "no"
                )
            except ValueError:
                pass
    return attrs


def load_pattern_instances(path: Path) -> dict[int, list[dict]]:
    """Group the main miner's saved embeddings by their stable pattern id."""

    result: dict[int, list[dict]] = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pattern_id = int(item["pattern_id"])
            result.setdefault(pattern_id, []).append(item)
    return result


def _value_from_mapping(item: object, keys: tuple[str, ...]) -> str:
    if not isinstance(item, dict):
        return normalize_value(item)
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return normalize_value(value)
    return ""


def _node_id_from_instance_value(value: object) -> str:
    return _value_from_mapping(value, ("index", "node_index", "id", "node_id", "src_index", "dst_index"))


def _edge_endpoint(edge_info: dict, endpoint: str) -> str:
    if endpoint == "src":
        return _value_from_mapping(edge_info, ("src_index", "src_id", "src", "source", "from"))
    return _value_from_mapping(edge_info, ("dst_index", "dst_id", "dst", "target", "to"))


def infer_pattern_schema_from_instances(instances_by_pattern: dict[int, list[dict]]) -> dict[int, dict[str, tuple[str, str]]]:
    """Recover edge_var -> (src_node_var, dst_node_var) from saved pattern embeddings.

    The miner stores node bindings and edge endpoint ids in each instance.  We
    reverse the node id mapping and use it to turn every e0/e1/... edge endpoint
    back into v0/v1/... variables.  If one instance is incomplete, another
    instance of the same pattern may still recover the schema.
    """

    schemas: dict[int, dict[str, tuple[str, str]]] = {}
    for pattern_id, instances in instances_by_pattern.items():
        for instance in instances:
            nodes = instance.get("nodes") or {}
            edges = instance.get("edges") or {}
            if not isinstance(nodes, dict) or not isinstance(edges, dict):
                continue

            id_to_node_var: dict[str, str] = {}
            for node_var, node_value in nodes.items():
                node_id = _node_id_from_instance_value(node_value)
                if node_id:
                    id_to_node_var[node_id] = str(node_var)
            if not id_to_node_var:
                continue

            schema: dict[str, tuple[str, str]] = {}
            for edge_var, edge_info in edges.items():
                if not isinstance(edge_info, dict):
                    continue
                src_id = _edge_endpoint(edge_info, "src")
                dst_id = _edge_endpoint(edge_info, "dst")
                src_var = id_to_node_var.get(src_id) or (src_id if src_id in nodes else None)
                dst_var = id_to_node_var.get(dst_id) or (dst_id if dst_id in nodes else None)
                if src_var and dst_var:
                    schema[str(edge_var)] = (src_var, dst_var)
            if schema:
                schemas[pattern_id] = schema
                break
    return schemas


def group_literals_by_entity(literals: tuple[str, ...]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for literal in literals:
        key, _op, _expected = split_literal(literal)
        if "." not in key:
            continue
        entity, _attr = key.split(".", 1)
        grouped.setdefault(entity, []).append(literal)
    return grouped


def rule_expandable_by_body_rematch(
    rule: NegativeExpansionRule,
    schema: dict[str, tuple[str, str]],
    target_edge_var: str,
) -> bool:
    if rule.negative_label != "negative":
        return False
    if consequent_target_edge(rule) != target_edge_var:
        return False
    if target_edge_var not in schema:
        return False

    body_edge_vars = [edge_var for edge_var in schema if edge_var != target_edge_var]
    if not body_edge_vars:
        return False

    target_src_var, target_dst_var = schema[target_edge_var]
    bound_by_body = {node_var for edge_var in body_edge_vars for node_var in schema[edge_var]}
    if target_src_var not in bound_by_body or target_dst_var not in bound_by_body:
        return False

    grouped = group_literals_by_entity(rule.antecedent)
    has_context_edge_literal = any(
        entity != target_edge_var and re.fullmatch(r"e[1-9]\d*", entity)
        for entity in grouped
    )
    if not has_context_edge_literal:
        return False

    for literal in grouped.get(target_edge_var, []):
        key, _op, _expected = split_literal(literal)
        _entity, attr = key.split(".", 1)
        if normalize_key(attr) not in COMPUTABLE_VIRTUAL_E0_ATTRS:
            return False
    return True


def build_edge_attr_index(
    rows: list[dict[str, str]],
    src_column: str,
    dst_column: str,
    dataset_name: str,
) -> dict[tuple[str, str], dict[str, str]]:
    """Look up original interaction attributes by the endpoint ids saved in an instance."""

    score_low, score_high = compute_score_bins(rows, src_column, dst_column)
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        src, dst = endpoint_ids(row, src_column, dst_column)
        if not src or not dst:
            continue
        attrs = build_enriched_edge_attrs(row, dataset_name, score_low, score_high)
        index[(src, dst)] = attrs
        if dataset_name == "PPI" and src != dst:
            index[(dst, src)] = attrs
    return index


def build_edge_records(
    rows: list[dict[str, str]],
    src_column: str,
    dst_column: str,
    dataset_name: str,
) -> list[dict[str, object]]:
    """Build real graph edge records used by body rematching.

    For PPI we treat the graph as undirected and insert the reverse direction so
    the matcher can satisfy either orientation stored in a pattern schema.
    """

    attr_index = build_edge_attr_index(rows, src_column, dst_column, dataset_name)
    records: list[dict[str, object]] = []
    for row in rows:
        src, dst = endpoint_ids(row, src_column, dst_column)
        if not src or not dst:
            continue
        attrs = dict(attr_index.get((src, dst), {}))
        records.append({"src": src, "dst": dst, "attrs": attrs, "row": row})
        if dataset_name == "PPI" and src != dst:
            records.append({"src": dst, "dst": src, "attrs": attrs, "row": row})
    return records


def build_adjacency_records(
    rows: list[dict[str, str]],
    src_column: str,
    dst_column: str,
    dataset_name: str,
    score_low: Optional[float],
    score_high: Optional[float],
) -> tuple[list[dict], dict[str, list[dict]], dict[str, list[dict]]]:
    edge_records: list[dict] = []
    out_adj: dict[str, list[dict]] = {}
    in_adj: dict[str, list[dict]] = {}

    def add_record(src: str, dst: str, attrs: dict[str, str], row: dict[str, str]) -> None:
        record = {"src": src, "dst": dst, "attrs": attrs, "row": row}
        edge_records.append(record)
        out_adj.setdefault(src, []).append(record)
        in_adj.setdefault(dst, []).append(record)

    for row in rows:
        src, dst = endpoint_ids(row, src_column, dst_column)
        if not src or not dst:
            continue
        attrs = build_enriched_edge_attrs(row, dataset_name, score_low, score_high)
        add_record(src, dst, attrs, row)
        if dataset_name == "PPI" and src != dst:
            add_record(dst, src, attrs, row)
    return edge_records, out_adj, in_adj


def pattern_context_from_instance(
    instance: dict,
    edge_attr_index: dict[tuple[str, str], dict[str, str]],
    source_node_attrs: Optional[dict[str, dict[str, str]]] = None,
    target_node_attrs: Optional[dict[str, dict[str, str]]] = None,
    dataset_name: str = "",
) -> dict[str, dict[str, str]]:
    """Build e0/e1/... context by joining saved endpoints back to interaction rows."""

    context: dict[str, dict[str, str]] = {}
    for edge_var, edge_info in instance.get("edges", {}).items():
        src = normalize_value(edge_info.get("src_index") or edge_info.get("src"))
        dst = normalize_value(edge_info.get("dst_index") or edge_info.get("dst"))
        attrs = dict(edge_attr_index.get((src, dst), {}))
        if not attrs:
            attrs = {
                "src": src,
                "dst": dst,
                "src_index": src,
                "dst_index": dst,
                "is_missing_edge": "yes",
                "interaction_label": "",
            }
            if dataset_name in {"DDA", "TI"}:
                attrs.update((target_node_attrs or {}).get(dst, {}))
            elif dataset_name == "PPI":
                attrs.update((source_node_attrs or {}).get(src, {}))
        attrs.setdefault("src", src)
        attrs.setdefault("dst", dst)
        attrs.setdefault("src_index", src)
        attrs.setdefault("dst_index", dst)
        context[edge_var] = attrs

    for node_var, node_id in instance.get("nodes", {}).items():
        node_index = normalize_value(node_id)
        context[node_var] = {"index": node_index, "node_index": node_index}
    return context


def consequent_target_edge(rule: NegativeExpansionRule) -> str:
    lhs = rule.consequent.split("=", 1)[0].strip()
    if "." not in lhs:
        return "e0"
    entity, _attribute = lhs.split(".", 1)
    return entity


def build_existing_edge_set(
    rows: list[dict[str, str]],
    src_column: str,
    dst_column: str,
    dataset_name: str,
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        src, dst = endpoint_ids(row, src_column, dst_column)
        if not src or not dst:
            continue
        pairs.add(tuple(sorted((src, dst))) if dataset_name == "PPI" else (src, dst))
    return pairs


def generate_candidate_non_edges_from_instances(
    instances_by_pattern: dict[int, list[dict]],
    rows: list[dict[str, str]],
    source_node_attrs: dict[str, dict[str, str]],
    target_node_attrs: dict[str, dict[str, str]],
    src_column: str,
    dst_column: str,
    dataset_name: str,
    max_candidates_per_anchor: int,
    min_src_degree: int,
    min_dst_degree: int,
    pattern_ids: Optional[set[int]] = None,
):
    """Yield endpoint-replacement non-edges without materializing a Cartesian product."""

    source_degree = Counter()
    target_degree = Counter()
    for row in rows:
        src, dst = endpoint_ids(row, src_column, dst_column)
        if src:
            source_degree[src] += 1
        if dst:
            target_degree[dst] += 1
    if dataset_name == "PPI":
        degree = source_degree + target_degree
        source_degree = target_degree = degree
    existing_pairs = build_existing_edge_set(rows, src_column, dst_column, dataset_name)
    source_candidates = sorted(node_id for node_id in source_node_attrs if source_degree[node_id] >= min_src_degree)
    target_candidates = sorted(node_id for node_id in target_node_attrs if target_degree[node_id] >= min_dst_degree)

    for pattern_id, instances in instances_by_pattern.items():
        if pattern_ids is not None and pattern_id not in pattern_ids:
            continue
        for anchor_instance in instances:
            anchor_edge = anchor_instance.get("edges", {}).get("e0")
            if not anchor_edge:
                continue
            src = normalize_value(anchor_edge.get("src_index") or anchor_edge.get("src"))
            dst = normalize_value(anchor_edge.get("dst_index") or anchor_edge.get("dst"))
            if not src or not dst:
                continue
            produced = 0
            for new_src in source_candidates:
                pair = tuple(sorted((new_src, dst))) if dataset_name == "PPI" else (new_src, dst)
                if new_src == src or pair in existing_pairs:
                    continue
                yield pattern_id, anchor_instance, new_src, dst
                produced += 1
                if produced >= max_candidates_per_anchor:
                    break
            if produced >= max_candidates_per_anchor:
                continue
            for new_dst in target_candidates:
                pair = tuple(sorted((src, new_dst))) if dataset_name == "PPI" else (src, new_dst)
                if new_dst == dst or pair in existing_pairs:
                    continue
                yield pattern_id, anchor_instance, src, new_dst
                produced += 1
                if produced >= max_candidates_per_anchor:
                    break


def make_synthetic_instance_from_anchor(
    anchor_instance: dict,
    target_edge_var: str,
    new_src: str,
    new_dst: str,
) -> dict:
    """Keep the anchor structure intact while replacing only its target edge endpoints."""

    synthetic = deepcopy(anchor_instance)
    target_edge = synthetic.setdefault("edges", {}).setdefault(target_edge_var, {})
    target_edge.update({
        "src": new_src,
        "dst": new_dst,
        "src_index": new_src,
        "dst_index": new_dst,
    })
    synthetic["is_synthetic"] = True
    synthetic["anchor_match_id"] = anchor_instance.get("match_id", "")
    synthetic["candidate_source"] = "replace_e0_endpoint"
    return synthetic


def is_rule_allowed_for_new_edge(rule: NegativeExpansionRule, require_pair_or_context: bool = True) -> bool:
    if consequent_target_edge(rule) != "e0" or rule.negative_label != "negative":
        return False
    antecedent = rule.antecedent
    if not antecedent:
        return False
    only_e0_disease_name = all(
        literal.startswith("e0.diseasename") or literal.startswith("e0.disease_name")
        for literal in antecedent
    )
    if only_e0_disease_name:
        return False
    if not require_pair_or_context:
        return True
    has_context_edge = any(re.match(r"e[1-9]\d*\.", literal) for literal in antecedent)
    has_pair_feature = any(
        token in literal.lower()
        for literal in antecedent
        for token in ("similarity", "common_neighbor", "direct_evidence", "inference")
    )
    return has_context_edge or has_pair_feature


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
    dataset_name: str = "",
    score_low: Optional[float] = None,
    score_high: Optional[float] = None,
) -> dict[str, dict[str, str]]:
    """构造规则匹配时使用的 e0/v0/v1 命名空间。

    `e0.xxx` 来自 interaction CSV 的边属性。
    `v0.xxx` 来自源端点属性：PPI 是 index_A，DDA 是 Drug，TI 是 Gene。
    `v1.xxx` 来自目标端点属性：PPI 是 index_B，DDA/TI 是 Disease。
    `v*.xxx` 在 literal_matches 中解释成 v0 或 v1 任意一端满足。
    """

    src, dst = endpoint_ids(row, src_column, dst_column)
    edge_attrs = build_enriched_edge_attrs(row, dataset_name, score_low, score_high, similarity_threshold)
    edge_attrs.setdefault("src", src)
    edge_attrs.setdefault("dst", dst)
    edge_attrs.setdefault("src_index", src)
    edge_attrs.setdefault("dst_index", dst)

    left = dict(source_node_attrs.get(src, {}))
    right = dict(target_node_attrs.get(dst, {}))
    for node_id, attrs in ((src, left), (dst, right)):
        degree_bin = node_degree_bins.get(node_id, "")
        attrs.setdefault("degree_bin", degree_bin)
        attrs.setdefault("degree_bucket", degree_bin)

    return {"e0": edge_attrs, "v0": left, "v1": right, "v*": {"_src": src, "_dst": dst}}


def literal_matches(literal: str, context: dict[str, dict[str, str]]) -> bool:
    """检查一个前件 literal，例如 `e0.ml_similarity_pred=yes`。"""

    key, op, expected = split_literal(literal)
    if "." not in key:
        return False
    entity, attr = key.split(".", 1)
    attr = normalize_key(attr)
    if entity == "v*":
        actual_values = []
        for vertex, attrs in context.items():
            if not re.fullmatch(r"v\d+", vertex):
                continue
            value = attrs.get(attr, "")
            if value != "":
                actual_values.append(value)
        if not actual_values:
            return False
        if op == "=":
            return any(value == expected for value in actual_values)
        if op == "!=":
            return all(value != expected for value in actual_values)
        return False
    actual = context.get(entity, {}).get(attr, "")
    if op == "=":
        return actual == expected
    if op == "!=":
        return actual != "" and actual != expected
    return False


def rule_matches(rule: NegativeExpansionRule, context: dict[str, dict[str, str]]) -> bool:
    """一条规则只有在所有前件 literal 都成立时才命中。"""

    return all(literal_matches(literal, context) for literal in rule.antecedent)


def rule_usable_for_existing_edge_labeling(rule: NegativeExpansionRule) -> bool:
    """Existing-row labeling only has e0/v0/v1/v* context available.

    Structural literals such as e1.xxx need anchored body rematching and are
    intentionally skipped in this first mode.
    """

    if rule.negative_label != "negative":
        return False
    if consequent_target_edge(rule) != "e0":
        return False
    allowed_entities = {"e0", "v0", "v1", "v*"}
    for literal in rule.antecedent:
        try:
            key, _op, _expected = split_literal(literal)
        except ValueError:
            return False
        if "." not in key:
            return False
        entity, _attr = key.split(".", 1)
        if entity not in allowed_entities:
            return False
    return True


def rule_usable_for_anchored_existing_edge_labeling(rule: NegativeExpansionRule) -> bool:
    if rule.negative_label != "negative":
        return False
    if consequent_target_edge(rule) != "e0":
        return False
    for literal in rule.antecedent:
        try:
            key, _op, _expected = split_literal(literal)
        except ValueError:
            return False
        if "." not in key:
            return False
        entity, _attr = key.split(".", 1)
        if entity == "v*":
            continue
        if re.fullmatch(r"e\d+", entity) or re.fullmatch(r"v\d+", entity):
            continue
        return False
    return True


def rule_has_structural_edge_literal(rule: NegativeExpansionRule) -> bool:
    for literal in rule.antecedent:
        try:
            key, _op, _expected = split_literal(literal)
        except ValueError:
            continue
        if "." not in key:
            continue
        entity, _attr = key.split(".", 1)
        if re.fullmatch(r"e[1-9]\d*", entity):
            return True
    return False


def is_structural_negative_e0_rule(rule: NegativeExpansionRule) -> bool:
    if rule.negative_label != "negative" or consequent_target_edge(rule) != "e0":
        return False
    for literal in rule.antecedent:
        try:
            key, _op, _expected = split_literal(literal)
        except ValueError:
            continue
        if "." not in key:
            continue
        entity, _attr = key.split(".", 1)
        if re.fullmatch(r"e[1-9]\d*", entity):
            return True
    return False


def _record_attr_matches(attrs: dict[str, str], literal: str) -> bool:
    key, op, expected = split_literal(literal)
    if "." not in key:
        return False
    _entity, attr = key.split(".", 1)
    actual = attrs.get(normalize_key(attr), "")
    if op == "=":
        return actual == expected
    if op == "!=":
        return actual != "" and actual != expected
    return False


def edge_record_matches_literals(edge_record: dict[str, object], literals: list[str]) -> bool:
    attrs = edge_record.get("attrs")
    if not isinstance(attrs, dict):
        return False
    return all(_record_attr_matches(attrs, literal) for literal in literals)


def node_satisfies_literals(
    node_var: str,
    node_id: str,
    node_literals: dict[str, list[str]],
    source_node_attrs: dict[str, dict[str, str]],
    target_node_attrs: dict[str, dict[str, str]],
    node_degree_bins: Optional[dict[str, str]] = None,
) -> bool:
    attrs = node_attrs_for_binding(node_id, source_node_attrs, target_node_attrs, node_degree_bins or {})
    for literal in node_literals.get(node_var, []):
        if not _record_attr_matches(attrs, literal):
            return False
    return True


def node_attrs_for_binding(
    node_id: str,
    source_node_attrs: dict[str, dict[str, str]],
    target_node_attrs: dict[str, dict[str, str]],
    node_degree_bins: dict[str, str],
) -> dict[str, str]:
    node_id = normalize_value(node_id)
    attrs = {"index": node_id, "node_index": node_id}
    for table in (source_node_attrs, target_node_attrs):
        for key, value in table.get(node_id, {}).items():
            attrs.setdefault(key, value)
    degree_bin = node_degree_bins.get(node_id, "")
    attrs.setdefault("degree_bin", degree_bin)
    attrs.setdefault("degree_bucket", degree_bin)
    return attrs


def build_anchored_context(
    row: dict[str, str],
    node_binding: dict[str, str],
    edge_binding: dict[str, dict],
    source_node_attrs: dict[str, dict[str, str]],
    target_node_attrs: dict[str, dict[str, str]],
    node_degree_bins: dict[str, str],
    src_column: str,
    dst_column: str,
    dataset_name: str,
    similarity_threshold: float,
    score_low: Optional[float],
    score_high: Optional[float],
) -> dict[str, dict[str, str]]:
    context: dict[str, dict[str, str]] = {
        "e0": build_enriched_edge_attrs(row, dataset_name, score_low, score_high, similarity_threshold),
        "v*": {},
    }
    src, dst = endpoint_ids(row, src_column, dst_column)
    context["e0"].setdefault("src", src)
    context["e0"].setdefault("dst", dst)
    context["e0"].setdefault("src_index", src)
    context["e0"].setdefault("dst_index", dst)
    for edge_var, record in edge_binding.items():
        attrs = record.get("attrs", {})
        context[edge_var] = dict(attrs) if isinstance(attrs, dict) else {}
        context[edge_var].setdefault("src", normalize_value(record.get("src")))
        context[edge_var].setdefault("dst", normalize_value(record.get("dst")))
    for node_var, node_id in node_binding.items():
        context[node_var] = node_attrs_for_binding(node_id, source_node_attrs, target_node_attrs, node_degree_bins)
    return context


def anchored_body_match_exists(
    rule: NegativeExpansionRule,
    schema: dict[str, tuple[str, str]],
    row: dict[str, str],
    edge_records: list[dict],
    out_adj: dict[str, list[dict]],
    in_adj: dict[str, list[dict]],
    source_node_attrs: dict[str, dict[str, str]],
    target_node_attrs: dict[str, dict[str, str]],
    node_degree_bins: dict[str, str],
    src_column: str,
    dst_column: str,
    dataset_name: str,
    similarity_threshold: float,
    score_low: Optional[float],
    score_high: Optional[float],
    max_partial_matches: int = 1000,
    debug_stats: Optional[Counter[str]] = None,
) -> Optional[dict[str, dict[str, str]]]:
    if debug_stats is not None:
        debug_stats["anchored_match_calls"] += 1
    target_edge_var = consequent_target_edge(rule)
    if target_edge_var != "e0" or "e0" not in schema:
        return None

    src, dst = endpoint_ids(row, src_column, dst_column)
    if not src or not dst:
        return None

    grouped = group_literals_by_entity(rule.antecedent)
    node_literals = {entity: literals for entity, literals in grouped.items() if re.fullmatch(r"v\d+", entity)}
    body_edge_vars = [edge_var for edge_var in schema if edge_var != "e0"]
    has_structural_literal = any(re.fullmatch(r"e[1-9]\d*", entity) for entity in grouped)

    e0_src_var, e0_dst_var = schema["e0"]
    initial_bindings = [{e0_src_var: src, e0_dst_var: dst}]
    if dataset_name == "PPI" and src != dst:
        initial_bindings.append({e0_src_var: dst, e0_dst_var: src})

    def binding_nodes_satisfy(binding: dict[str, str]) -> bool:
        for node_var, node_id in binding.items():
            if not node_satisfies_literals(
                node_var,
                node_id,
                node_literals,
                source_node_attrs,
                target_node_attrs,
                node_degree_bins,
            ):
                return False
        return True

    def candidate_edges_for(edge_var: str, binding: dict[str, str]) -> list[dict]:
        src_var, dst_var = schema[edge_var]
        src_bound = binding.get(src_var)
        dst_bound = binding.get(dst_var)
        if src_bound and dst_bound:
            candidates = [record for record in out_adj.get(src_bound, []) if normalize_value(record.get("dst")) == dst_bound]
        elif src_bound:
            candidates = list(out_adj.get(src_bound, []))
        elif dst_bound:
            candidates = list(in_adj.get(dst_bound, []))
        else:
            candidates = edge_records
        literals = grouped.get(edge_var, [])
        return [
            record for record in candidates
            if record.get("row") is not row and edge_record_matches_literals(record, literals)
        ]

    def bind_node(binding: dict[str, str], node_var: str, node_id: str) -> Optional[dict[str, str]]:
        node_id = normalize_value(node_id)
        if not node_id:
            return None
        existing = binding.get(node_var)
        if existing is not None:
            return binding if existing == node_id else None
        if not node_satisfies_literals(
            node_var,
            node_id,
            node_literals,
            source_node_attrs,
            target_node_attrs,
            node_degree_bins,
        ):
            return None
        next_binding = dict(binding)
        next_binding[node_var] = node_id
        return next_binding

    def fast_context_if_nonstructural(binding: dict[str, str]) -> Optional[dict[str, dict[str, str]]]:
        if has_structural_literal:
            return None
        context = build_anchored_context(
            row,
            binding,
            {},
            source_node_attrs,
            target_node_attrs,
            node_degree_bins,
            src_column,
            dst_column,
            dataset_name,
            similarity_threshold,
            score_low,
            score_high,
        )
        return context if rule_matches(rule, context) else None

    for initial_binding in initial_bindings:
        if not binding_nodes_satisfy(initial_binding):
            continue
        fast_context = fast_context_if_nonstructural(initial_binding)
        if fast_context is not None:
            if debug_stats is not None:
                debug_stats["anchored_fast_hits"] += 1
            return fast_context
        if not body_edge_vars:
            continue

        partial_matches = 0

        def dfs(
            remaining_edge_vars: list[str],
            node_binding: dict[str, str],
            edge_binding: dict[str, dict],
        ) -> Optional[dict[str, dict[str, str]]]:
            nonlocal partial_matches
            partial_matches += 1
            if partial_matches > max_partial_matches:
                if debug_stats is not None:
                    debug_stats["anchored_partial_match_cutoffs"] += 1
                return None
            if not remaining_edge_vars:
                context = build_anchored_context(
                    row,
                    node_binding,
                    edge_binding,
                    source_node_attrs,
                    target_node_attrs,
                    node_degree_bins,
                    src_column,
                    dst_column,
                    dataset_name,
                    similarity_threshold,
                    score_low,
                    score_high,
                )
                return context if rule_matches(rule, context) else None

            ranked: list[tuple[tuple[int, int], str, list[dict]]] = []
            for edge_var in remaining_edge_vars:
                src_var, dst_var = schema[edge_var]
                bound_count = int(src_var in node_binding) + int(dst_var in node_binding)
                candidates = candidate_edges_for(edge_var, node_binding)
                ranked.append(((-bound_count, len(candidates)), edge_var, candidates))
            ranked.sort(key=lambda item: item[0])
            _rank, edge_var, candidates = ranked[0]
            if not candidates:
                return None

            src_var, dst_var = schema[edge_var]
            next_remaining = [item for item in remaining_edge_vars if item != edge_var]
            for record in candidates:
                next_nodes = bind_node(node_binding, src_var, normalize_value(record.get("src")))
                if next_nodes is None:
                    continue
                next_nodes = bind_node(next_nodes, dst_var, normalize_value(record.get("dst")))
                if next_nodes is None:
                    continue
                next_edges = dict(edge_binding)
                next_edges[edge_var] = record
                result = dfs(next_remaining, next_nodes, next_edges)
                if result is not None:
                    return result
                if partial_matches > max_partial_matches:
                    return None
            return None

        result = dfs(body_edge_vars, dict(initial_binding), {})
        if result is not None:
            if debug_stats is not None:
                debug_stats["anchored_dfs_hits"] += 1
            return result
    return None


def body_match_backtracking(
    schema: dict[str, tuple[str, str]],
    target_edge_var: str,
    rule: NegativeExpansionRule,
    edge_records: list[dict[str, object]],
    source_node_attrs: dict[str, dict[str, str]],
    target_node_attrs: dict[str, dict[str, str]],
    dataset_name: str,
    max_body_matches_per_rule: int = 200000,
) -> list[dict[str, dict]]:
    """Match rule body after removing the head edge from the pattern schema."""

    _dataset_name = dataset_name
    grouped = group_literals_by_entity(rule.antecedent)
    body_edge_vars = [edge_var for edge_var in schema if edge_var != target_edge_var]
    edge_literals = {edge_var: grouped.get(edge_var, []) for edge_var in body_edge_vars}
    node_literals = {entity: literals for entity, literals in grouped.items() if entity.startswith("v")}

    candidates_by_edge_var: dict[str, list[dict[str, object]]] = {}
    for edge_var in body_edge_vars:
        literals = edge_literals.get(edge_var, [])
        candidates_by_edge_var[edge_var] = [
            record for record in edge_records if edge_record_matches_literals(record, literals)
        ]
        if not candidates_by_edge_var[edge_var]:
            return []

    ordered_edge_vars = sorted(body_edge_vars, key=lambda edge_var: len(candidates_by_edge_var[edge_var]))
    matches: list[dict[str, dict]] = []

    def bind_node(node_binding: dict[str, str], node_var: str, node_id: str) -> Optional[dict[str, str]]:
        node_id = normalize_value(node_id)
        if not node_id:
            return None
        existing = node_binding.get(node_var)
        if existing is not None:
            return node_binding if existing == node_id else None
        if not node_satisfies_literals(node_var, node_id, node_literals, source_node_attrs, target_node_attrs):
            return None
        next_binding = dict(node_binding)
        next_binding[node_var] = node_id
        return next_binding

    def dfs(position: int, node_binding: dict[str, str], edge_binding: dict[str, dict]) -> None:
        if len(matches) >= max_body_matches_per_rule:
            return
        if position >= len(ordered_edge_vars):
            matches.append({"node_binding": dict(node_binding), "edge_binding": dict(edge_binding)})
            return

        edge_var = ordered_edge_vars[position]
        src_var, dst_var = schema[edge_var]
        for record in candidates_by_edge_var[edge_var]:
            src = normalize_value(record.get("src"))
            dst = normalize_value(record.get("dst"))
            next_nodes = bind_node(node_binding, src_var, src)
            if next_nodes is None:
                continue
            next_nodes = bind_node(next_nodes, dst_var, dst)
            if next_nodes is None:
                continue
            next_edges = dict(edge_binding)
            next_edges[edge_var] = record
            dfs(position + 1, next_nodes, next_edges)
            if len(matches) >= max_body_matches_per_rule:
                return

    dfs(0, {}, {})
    return matches


def normalize_pair_for_dataset(src: str, dst: str, dataset_name: str) -> tuple[str, str]:
    return tuple(sorted((src, dst))) if dataset_name == "PPI" else (src, dst)


def virtual_e0_literals_hold(
    rule: NegativeExpansionRule,
    src: str,
    dst: str,
    dataset_name: str,
    existing_pair_features: Optional[dict[tuple[str, str], dict[str, str]]] = None,
) -> bool:
    grouped = group_literals_by_entity(rule.antecedent)
    e0_literals = grouped.get(consequent_target_edge(rule), [])
    if not e0_literals:
        return True

    pair = normalize_pair_for_dataset(src, dst, dataset_name)
    features = (existing_pair_features or {}).get(pair, {})
    for literal in e0_literals:
        key, op, expected = split_literal(literal)
        if "." not in key:
            return False
        _entity, attr = key.split(".", 1)
        attr = normalize_key(attr)
        if attr not in COMPUTABLE_VIRTUAL_E0_ATTRS:
            return False
        actual = features.get(attr, "")
        if actual == "":
            return False
        if op == "=" and actual != expected:
            return False
        if op == "!=" and actual == expected:
            return False
    return True


def can_export(row: dict[str, str], label_column: str, allowed_existing: set[str], overwrite_existing: bool) -> bool:
    if overwrite_existing:
        return True
    return normalize_value(row.get(label_column)) in allowed_existing


def expand_candidate_non_edges(
    config: ExpansionConfig,
    rows: list[dict[str, str]],
    rules: list[NegativeExpansionRule],
) -> dict[str, int]:
    """Apply suitably contextual rules to endpoint-replacement non-edge candidates."""

    instances_by_pattern = load_pattern_instances(config.pattern_instances_file)
    source_node_attrs = load_node_attrs(str(config.source_node_csv) if config.source_node_csv else None, config.source_node_index_column)
    target_node_attrs = load_node_attrs(str(config.target_node_csv) if config.target_node_csv else None, config.target_node_index_column)
    edge_attr_index = build_edge_attr_index(rows, config.src_column, config.dst_column, config.dataset_name)
    existing_pairs = build_existing_edge_set(rows, config.src_column, config.dst_column, config.dataset_name)

    allowed_rules_by_pattern: dict[int, list[tuple[int, NegativeExpansionRule]]] = {}
    skipped_rule_not_allowed = 0
    for rule_index, rule in enumerate(rules):
        if rule.negative_label != config.negative_value or not is_rule_allowed_for_new_edge(
            rule, config.require_rule_has_pair_or_context
        ):
            skipped_rule_not_allowed += 1
            continue
        allowed_rules_by_pattern.setdefault(rule.pattern_id, []).append((rule_index, rule))

    candidate_pairs = 0
    candidate_rule_checked = 0
    matched_synthetic_instances = 0
    exported_new_pairs = 0
    skipped_existing_pair = 0
    skipped_node_limit = 0
    seen_pairs: set[tuple[str, str]] = set()
    node_new_counts: Counter[str] = Counter()
    output_rows: list[dict[str, str]] = []

    for pattern_id, anchor_instance, new_src, new_dst in generate_candidate_non_edges_from_instances(
        instances_by_pattern,
        rows,
        source_node_attrs,
        target_node_attrs,
        config.src_column,
        config.dst_column,
        config.dataset_name,
        config.max_candidates_per_anchor,
        config.min_src_degree,
        config.min_dst_degree,
        set(allowed_rules_by_pattern),
    ):
        if config.max_new_neg_total is not None and exported_new_pairs >= config.max_new_neg_total:
            break
        candidate_pairs += 1
        pair = tuple(sorted((new_src, new_dst))) if config.dataset_name == "PPI" else (new_src, new_dst)
        if pair in existing_pairs:
            skipped_existing_pair += 1
            continue
        for rule_index, rule in allowed_rules_by_pattern.get(pattern_id, []):
            candidate_rule_checked += 1
            synthetic = make_synthetic_instance_from_anchor(anchor_instance, "e0", new_src, new_dst)
            context = pattern_context_from_instance(
                synthetic,
                edge_attr_index,
                source_node_attrs,
                target_node_attrs,
                config.dataset_name,
            )
            if not rule_matches(rule, context):
                continue
            matched_synthetic_instances += 1
            if pair in seen_pairs:
                break
            if (
                node_new_counts[new_src] >= config.max_new_neg_per_node
                or node_new_counts[new_dst] >= config.max_new_neg_per_node
            ):
                skipped_node_limit += 1
                break
            seen_pairs.add(pair)
            node_new_counts[new_src] += 1
            node_new_counts[new_dst] += 1
            output_rows.append(
                {
                    config.src_column: new_src,
                    config.dst_column: new_dst,
                    "predicted_label": config.negative_value,
                    "negative_rule_pattern_id": str(rule.pattern_id),
                    "negative_rule_index": str(rule_index),
                    "anchor_match_id": str(anchor_instance.get("match_id", "")),
                    "target_edge_var": "e0",
                    "candidate_source": synthetic["candidate_source"],
                    "negative_rule_antecedent": " & ".join(rule.antecedent),
                }
            )
            exported_new_pairs += 1
            break

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with Path(config.output_csv).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                config.src_column,
                config.dst_column,
                "predicted_label",
                "negative_rule_pattern_id",
                "negative_rule_index",
                "anchor_match_id",
                "target_edge_var",
                "candidate_source",
                "negative_rule_antecedent",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    return {
        "rows": len(rows),
        "rules": len(rules),
        "patterns_with_instances": len(instances_by_pattern),
        "candidate_pairs": candidate_pairs,
        "candidate_rule_checked": candidate_rule_checked,
        "matched_synthetic_instances": matched_synthetic_instances,
        "exported_new_pairs": exported_new_pairs,
        "skipped_existing_pair": skipped_existing_pair,
        "skipped_rule_not_allowed": skipped_rule_not_allowed,
        "skipped_node_limit": skipped_node_limit,
    }


def add_degree_bins_to_node_attrs(
    rows: list[dict[str, str]],
    src_column: str,
    dst_column: str,
    source_node_attrs: dict[str, dict[str, str]],
    target_node_attrs: dict[str, dict[str, str]],
) -> None:
    node_degree_bins = degree_bins(rows, src_column, dst_column)
    for table in (source_node_attrs, target_node_attrs):
        for node_id, attrs in table.items():
            degree_bin = node_degree_bins.get(node_id, "")
            attrs.setdefault("degree_bin", degree_bin)
            attrs.setdefault("degree_bucket", degree_bin)


def expand_existing_edges_as_negative(config: ExpansionConfig) -> dict[str, object]:
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
    node_degree_bins = degree_bins(rows, config.src_column, config.dst_column)
    score_low, score_high = compute_score_bins(rows, config.src_column, config.dst_column)

    usable_rule_items: list[tuple[int, NegativeExpansionRule]] = []
    skipped_structural_rule = 0
    for rule_index, rule in enumerate(rules):
        if rule_usable_for_existing_edge_labeling(rule):
            usable_rule_items.append((rule_index, rule))
        elif is_structural_negative_e0_rule(rule):
            skipped_structural_rule += 1

    checked_rows = 0
    matched_rows = 0
    exported_rows = 0
    skipped_positive = 0
    skipped_existing_negative = 0
    skipped_label_not_allowed = 0
    skipped_missing_endpoint = 0
    duplicate_pairs = 0
    label_counts: Counter[str] = Counter()
    seen_pairs: set[tuple[str, str]] = set()
    output_rows: list[dict[str, str]] = []

    for row in rows:
        label = normalize_value(row.get(config.label_column))
        label_counts[label] += 1
        src, dst = endpoint_ids(row, config.src_column, config.dst_column)
        if not src or not dst:
            skipped_missing_endpoint += 1
            continue

        if (
            not config.overwrite_existing
            and not config.allow_positive_relabel
            and label == "positive"
        ):
            skipped_positive += 1
            continue
        if (
            not config.overwrite_existing
            and not config.allow_existing_negative_relabel
            and label == config.negative_value
        ):
            skipped_existing_negative += 1
            continue
        if config.only_labels is not None and label not in config.only_labels:
            skipped_label_not_allowed += 1
            continue

        checked_rows += 1
        context = edge_context(
            row=row,
            source_node_attrs=source_node_attrs,
            target_node_attrs=target_node_attrs,
            node_degree_bins=node_degree_bins,
            src_column=config.src_column,
            dst_column=config.dst_column,
            similarity_threshold=config.similarity_threshold,
            dataset_name=config.dataset_name,
            score_low=score_low,
            score_high=score_high,
        )

        matched_rules: list[tuple[int, NegativeExpansionRule]] = []
        for rule_index, rule in usable_rule_items:
            if rule_matches(rule, context):
                matched_rules.append((rule_index, rule))

        if not matched_rules:
            continue
        matched_rows += 1

        pair = normalize_pair_for_dataset(src, dst, config.dataset_name)
        if pair in seen_pairs:
            duplicate_pairs += 1
            continue
        seen_pairs.add(pair)

        first_rule_index, first_rule = matched_rules[0]
        output_rows.append(
            {
                config.src_column: src,
                config.dst_column: dst,
                "predicted_label": config.negative_value,
                "negative_rule_pattern_id": str(first_rule.pattern_id),
                "negative_rule_index": str(first_rule_index),
                "matched_rule_count": str(len(matched_rules)),
                "candidate_source": "existing_edge_labeling",
                "original_label": label,
                "negative_rule_antecedent": " & ".join(first_rule.antecedent),
            }
        )
        exported_rows += 1

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with Path(config.output_csv).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                config.src_column,
                config.dst_column,
                "predicted_label",
                "negative_rule_pattern_id",
                "negative_rule_index",
                "matched_rule_count",
                "candidate_source",
                "original_label",
                "negative_rule_antecedent",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    return {
        "rows": len(rows),
        "rules": len(rules),
        "usable_rules": len(usable_rule_items),
        "checked_rows": checked_rows,
        "matched_rows": matched_rows,
        "exported_rows": exported_rows,
        "skipped_positive": skipped_positive,
        "skipped_existing_negative": skipped_existing_negative,
        "skipped_label_not_allowed": skipped_label_not_allowed,
        "skipped_missing_endpoint": skipped_missing_endpoint,
        "skipped_structural_rule": skipped_structural_rule,
        "duplicate_pairs": duplicate_pairs,
        "observed_labels": ",".join(f"{label or '<empty>'}:{count}" for label, count in label_counts.most_common(12)),
    }


def expand_existing_edges_as_negative_anchored(config: ExpansionConfig) -> dict[str, object]:
    total_start = time.perf_counter()
    step_start = total_start
    debug_log(config, f"anchored_start dataset={config.dataset_name} input={config.input_csv}")

    rows, _fields = read_rows(str(config.input_csv))
    debug_log(config, f"loaded_rows rows={len(rows)} seconds={time.perf_counter() - step_start:.2f}")

    step_start = time.perf_counter()
    rules = load_rules(config.rules_file)
    debug_log(config, f"loaded_rules rules={len(rules)} seconds={time.perf_counter() - step_start:.2f}")

    step_start = time.perf_counter()
    instances_by_pattern = load_pattern_instances(config.pattern_instances_file)
    pattern_schemas = infer_pattern_schema_from_instances(instances_by_pattern)
    debug_log(
        config,
        f"loaded_pattern_instances patterns={len(instances_by_pattern)} schemas={len(pattern_schemas)} "
        + f"seconds={time.perf_counter() - step_start:.2f}",
    )

    step_start = time.perf_counter()
    source_node_attrs = load_node_attrs(
        str(config.source_node_csv) if config.source_node_csv else None,
        config.source_node_index_column,
    )
    debug_log(
        config,
        f"loaded_source_nodes nodes={len(source_node_attrs)} path={config.source_node_csv} "
        + f"seconds={time.perf_counter() - step_start:.2f}",
    )

    step_start = time.perf_counter()
    target_node_attrs = load_node_attrs(
        str(config.target_node_csv) if config.target_node_csv else None,
        config.target_node_index_column,
    )
    debug_log(
        config,
        f"loaded_target_nodes nodes={len(target_node_attrs)} path={config.target_node_csv} "
        + f"seconds={time.perf_counter() - step_start:.2f}",
    )

    step_start = time.perf_counter()
    node_degree_bins = degree_bins(rows, config.src_column, config.dst_column)
    score_low, score_high = compute_score_bins(rows, config.src_column, config.dst_column)
    debug_log(
        config,
        f"computed_bins node_degree_bins={len(node_degree_bins)} score_low={score_low} score_high={score_high} "
        + f"seconds={time.perf_counter() - step_start:.2f}",
    )

    step_start = time.perf_counter()
    edge_records, out_adj, in_adj = build_adjacency_records(
        rows,
        config.src_column,
        config.dst_column,
        config.dataset_name,
        score_low,
        score_high,
    )
    debug_log(
        config,
        f"built_adjacency edge_records={len(edge_records)} out_nodes={len(out_adj)} in_nodes={len(in_adj)} "
        + f"seconds={time.perf_counter() - step_start:.2f}",
    )

    step_start = time.perf_counter()
    usable_rule_items: list[tuple[int, NegativeExpansionRule, dict[str, tuple[str, str]]]] = []
    simple_rule_items: list[tuple[int, NegativeExpansionRule, dict[str, tuple[str, str]]]] = []
    structural_rule_items: list[tuple[int, NegativeExpansionRule, dict[str, tuple[str, str]]]] = []
    skipped_no_schema = 0
    for rule_index, rule in enumerate(rules):
        if not rule_usable_for_anchored_existing_edge_labeling(rule):
            continue
        schema = pattern_schemas.get(rule.pattern_id)
        if not schema or "e0" not in schema:
            skipped_no_schema += 1
            continue
        item = (rule_index, rule, schema)
        usable_rule_items.append(item)
        if rule_has_structural_edge_literal(rule):
            structural_rule_items.append(item)
        else:
            simple_rule_items.append(item)
    debug_log(
        config,
        f"filtered_rules usable_rules={len(usable_rule_items)} skipped_no_schema={skipped_no_schema} "
        + f"seconds={time.perf_counter() - step_start:.2f}",
    )
    if config.debug_usable_rules:
        print(
            "[UsableRuleSummary] "
            + f"usable={len(usable_rule_items)} "
            + f"structural={len(structural_rule_items)} "
            + f"simple={len(simple_rule_items)} "
            + f"skipped_no_schema={skipped_no_schema}",
            flush=True,
        )
        for rule_index, rule, schema in usable_rule_items[: config.debug_max_print_rules]:
            print(
                "[UsableAnchoredRule] "
                + f"rule_index={rule_index} "
                + f"pattern_id={rule.pattern_id} "
                + f"structural={rule_has_structural_edge_literal(rule)} "
                + f"antecedent={rule.antecedent} "
                + f"schema={schema}",
                flush=True,
            )

    checked_rows = 0
    matched_rows = 0
    exported_rows = 0
    skipped_positive = 0
    skipped_existing_negative = 0
    skipped_label_not_allowed = 0
    skipped_missing_endpoint = 0
    duplicate_pairs = 0
    duplicate_input_pairs = 0
    exported_pair_skips = 0
    simple_rule_calls = 0
    exported_pairs: set[tuple[str, str]] = set()
    input_seen_pairs: set[tuple[str, str]] = set()
    rule_match_counts: Counter[int] = Counter()
    label_counts: Counter[str] = Counter()
    anchored_stats: Counter[str] = Counter()
    output_fieldnames = [
        config.src_column,
        config.dst_column,
        "predicted_label",
        "negative_rule_pattern_id",
        "negative_rule_index",
        "matched_rule_count",
        "candidate_source",
        "original_label",
        "negative_rule_antecedent",
    ]
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_handle = Path(config.output_csv).open("w", encoding="utf-8-sig", newline="")
    output_writer = csv.DictWriter(output_handle, fieldnames=output_fieldnames, extrasaction="ignore")
    output_writer.writeheader()
    output_handle.flush()
    pending_flush_exports = 0
    debug_log(
        config,
        f"opened_incremental_output path={config.output_csv} "
        + f"incremental={config.incremental_write_output} flush_every={config.flush_every_exports}",
    )

    step_start = time.perf_counter()
    try:
        for row_number, row in enumerate(rows, start=1):
            label = normalize_value(row.get(config.label_column))
            label_counts[label] += 1
            src, dst = endpoint_ids(row, config.src_column, config.dst_column)
            if not src or not dst:
                skipped_missing_endpoint += 1
                continue
            if (
                not config.overwrite_existing
                and not config.allow_positive_relabel
                and label == "positive"
            ):
                skipped_positive += 1
                continue
            if (
                not config.overwrite_existing
                and not config.allow_existing_negative_relabel
                and label == config.negative_value
            ):
                skipped_existing_negative += 1
                continue
            if config.only_labels is not None and label not in config.only_labels:
                skipped_label_not_allowed += 1
                continue

            pair = normalize_pair_for_dataset(src, dst, config.dataset_name)
            if pair in exported_pairs:
                duplicate_pairs += 1
                exported_pair_skips += 1
                continue
            if config.use_first_row_per_pair and pair in input_seen_pairs:
                duplicate_input_pairs += 1
                continue
            input_seen_pairs.add(pair)

            checked_rows += 1
            matched_rules: list[tuple[int, NegativeExpansionRule]] = []
            simple_context: Optional[dict[str, dict[str, str]]] = None
            if simple_rule_items:
                simple_context = edge_context(
                    row=row,
                    source_node_attrs=source_node_attrs,
                    target_node_attrs=target_node_attrs,
                    node_degree_bins=node_degree_bins,
                    src_column=config.src_column,
                    dst_column=config.dst_column,
                    similarity_threshold=config.similarity_threshold,
                    dataset_name=config.dataset_name,
                    score_low=score_low,
                    score_high=score_high,
                )
            for rule_index, rule, _schema in simple_rule_items:
                simple_rule_calls += 1
                if simple_context is not None and rule_matches(rule, simple_context):
                    matched_rules.append((rule_index, rule))
                    rule_match_counts[rule_index] += 1
                    if config.early_stop_on_first_match:
                        break

            if not (matched_rules and config.early_stop_on_first_match):
                for rule_index, rule, schema in structural_rule_items:
                    context = anchored_body_match_exists(
                        rule,
                        schema,
                        row,
                        edge_records,
                        out_adj,
                        in_adj,
                        source_node_attrs,
                        target_node_attrs,
                        node_degree_bins,
                        config.src_column,
                        config.dst_column,
                        config.dataset_name,
                        config.similarity_threshold,
                        score_low,
                        score_high,
                        config.max_anchored_partial_matches,
                        anchored_stats,
                    )
                    if context is not None:
                        matched_rules.append((rule_index, rule))
                        rule_match_counts[rule_index] += 1
                        if config.early_stop_on_first_match:
                            break

            if (
                config.debug_progress
                and (
                    row_number % config.debug_every_rows == 0
                    or (checked_rows > 0 and checked_rows % config.debug_every_checked_rows == 0)
                )
            ):
                debug_log(
                    config,
                    f"scan_progress row={row_number}/{len(rows)} checked={checked_rows} matched={matched_rows} "
                    + f"exported={exported_rows} skipped_positive={skipped_positive} "
                    + f"skipped_negative={skipped_existing_negative} skipped_label={skipped_label_not_allowed} "
                    + f"simple_calls={simple_rule_calls} "
                    + f"anchored_match_calls={anchored_stats.get('anchored_match_calls', 0)} "
                    + f"structural_calls={anchored_stats.get('anchored_match_calls', 0)} "
                    + f"exported_pair_skips={exported_pair_skips} "
                    + f"duplicate_input_pairs={duplicate_input_pairs} "
                    + f"usable_simple_rules={len(simple_rule_items)} "
                    + f"usable_structural_rules={len(structural_rule_items)} "
                    + f"cutoffs={anchored_stats.get('anchored_partial_match_cutoffs', 0)} "
                    + f"elapsed={time.perf_counter() - step_start:.2f}",
                )

            if not matched_rules:
                continue
            matched_rows += 1

            exported_pairs.add(pair)

            first_rule_index, first_rule = matched_rules[0]
            output_writer.writerow({
                config.src_column: src,
                config.dst_column: dst,
                "predicted_label": config.negative_value,
                "negative_rule_pattern_id": str(first_rule.pattern_id),
                "negative_rule_index": str(first_rule_index),
                "matched_rule_count": str(len(matched_rules)),
                "candidate_source": "anchored_existing_edge_labeling",
                "original_label": label,
                "negative_rule_antecedent": " & ".join(first_rule.antecedent),
            })
            exported_rows += 1
            pending_flush_exports += 1
            if config.incremental_write_output and pending_flush_exports >= max(1, config.flush_every_exports):
                output_handle.flush()
                debug_log(config, f"incremental_flush exported_rows={exported_rows} path={config.output_csv}")
                pending_flush_exports = 0
    finally:
        output_handle.flush()
        output_handle.close()

    debug_log(
        config,
        f"scan_done checked={checked_rows} matched={matched_rows} exported={exported_rows} "
        + f"simple_calls={simple_rule_calls} "
        + f"anchored_match_calls={anchored_stats.get('anchored_match_calls', 0)} "
        + f"structural_calls={anchored_stats.get('anchored_match_calls', 0)} "
        + f"fast_hits={anchored_stats.get('anchored_fast_hits', 0)} "
        + f"dfs_hits={anchored_stats.get('anchored_dfs_hits', 0)} "
        + f"cutoffs={anchored_stats.get('anchored_partial_match_cutoffs', 0)} "
        + f"exported_pair_skips={exported_pair_skips} "
        + f"duplicate_input_pairs={duplicate_input_pairs} "
        + f"seconds={time.perf_counter() - step_start:.2f}",
    )

    debug_log(config, f"closed_incremental_output rows={exported_rows} path={config.output_csv}")

    return {
        "rows": len(rows),
        "rules": len(rules),
        "schemas": len(pattern_schemas),
        "usable_rules": len(usable_rule_items),
        "usable_simple_rules": len(simple_rule_items),
        "usable_structural_rules": len(structural_rule_items),
        "checked_rows": checked_rows,
        "matched_rows": matched_rows,
        "exported_rows": exported_rows,
        "skipped_positive": skipped_positive,
        "skipped_existing_negative": skipped_existing_negative,
        "skipped_label_not_allowed": skipped_label_not_allowed,
        "skipped_missing_endpoint": skipped_missing_endpoint,
        "duplicate_pairs": duplicate_pairs,
        "simple_rule_calls": simple_rule_calls,
        "structural_rule_calls": anchored_stats.get("anchored_match_calls", 0),
        "exported_pair_skips": exported_pair_skips,
        "duplicate_input_pairs": duplicate_input_pairs,
        "early_stop_on_first_match": config.early_stop_on_first_match,
        "use_first_row_per_pair": config.use_first_row_per_pair,
        "incremental_write_output": config.incremental_write_output,
        "flush_every_exports": config.flush_every_exports,
        "skipped_no_schema": skipped_no_schema,
        "anchored_match_calls": anchored_stats.get("anchored_match_calls", 0),
        "anchored_fast_hits": anchored_stats.get("anchored_fast_hits", 0),
        "anchored_dfs_hits": anchored_stats.get("anchored_dfs_hits", 0),
        "anchored_partial_match_cutoffs": anchored_stats.get("anchored_partial_match_cutoffs", 0),
        "top_rule_matches": ",".join(f"{rule_index}:{count}" for rule_index, count in rule_match_counts.most_common(10)),
        "observed_labels": ",".join(f"{label or '<empty>'}:{count}" for label, count in label_counts.most_common(12)),
        "elapsed_seconds": f"{time.perf_counter() - total_start:.2f}",
    }


def expand_body_rematch_non_edges(
    config: ExpansionConfig,
    rows: list[dict[str, str]],
    rules: list[NegativeExpansionRule],
) -> dict[str, int]:
    instances_by_pattern = load_pattern_instances(config.pattern_instances_file)
    pattern_schemas = infer_pattern_schema_from_instances(instances_by_pattern)
    source_node_attrs = load_node_attrs(str(config.source_node_csv) if config.source_node_csv else None, config.source_node_index_column)
    target_node_attrs = load_node_attrs(str(config.target_node_csv) if config.target_node_csv else None, config.target_node_index_column)
    add_degree_bins_to_node_attrs(rows, config.src_column, config.dst_column, source_node_attrs, target_node_attrs)

    edge_records = build_edge_records(rows, config.src_column, config.dst_column, config.dataset_name)
    existing_pairs = build_existing_edge_set(rows, config.src_column, config.dst_column, config.dataset_name)

    output_rows: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    node_new_counts: Counter[str] = Counter()
    debug_rules: list[tuple[int, int, tuple[str, ...], dict[str, tuple[str, str]], str]] = []

    body_matches = 0
    inferred_new_negative_pairs = 0
    skipped_existing_pair = 0
    skipped_no_schema = 0
    skipped_not_expandable = 0
    skipped_node_limit = 0

    for rule_index, rule in enumerate(rules):
        if config.max_new_neg_total is not None and inferred_new_negative_pairs >= config.max_new_neg_total:
            break

        target_edge_var = consequent_target_edge(rule)
        schema = pattern_schemas.get(rule.pattern_id)
        if not schema:
            skipped_no_schema += 1
            continue
        if not rule_expandable_by_body_rematch(rule, schema, target_edge_var):
            skipped_not_expandable += 1
            continue
        if len(debug_rules) < 5:
            debug_rules.append((rule.pattern_id, rule_index, rule.antecedent, schema, target_edge_var))

        target_src_var, target_dst_var = schema[target_edge_var]
        matches = body_match_backtracking(
            schema,
            target_edge_var,
            rule,
            edge_records,
            source_node_attrs,
            target_node_attrs,
            config.dataset_name,
            config.max_body_matches_per_rule,
        )
        body_matches += len(matches)

        for match in matches:
            if config.max_new_neg_total is not None and inferred_new_negative_pairs >= config.max_new_neg_total:
                break
            node_binding = match.get("node_binding", {})
            src = normalize_value(node_binding.get(target_src_var))
            dst = normalize_value(node_binding.get(target_dst_var))
            if not src or not dst or src == dst:
                continue

            pair = normalize_pair_for_dataset(src, dst, config.dataset_name)
            if pair in existing_pairs:
                skipped_existing_pair += 1
                continue
            if pair in seen_pairs:
                continue
            if (
                node_new_counts[src] >= config.max_new_neg_per_node
                or node_new_counts[dst] >= config.max_new_neg_per_node
            ):
                skipped_node_limit += 1
                continue
            if not virtual_e0_literals_hold(rule, src, dst, config.dataset_name):
                continue

            seen_pairs.add(pair)
            node_new_counts[src] += 1
            node_new_counts[dst] += 1
            output_rows.append(
                {
                    config.src_column: src,
                    config.dst_column: dst,
                    "predicted_label": config.negative_value,
                    "negative_rule_pattern_id": str(rule.pattern_id),
                    "negative_rule_index": str(rule_index),
                    "target_edge_var": target_edge_var,
                    "candidate_source": "body_rematch",
                    "negative_rule_antecedent": " & ".join(rule.antecedent),
                }
            )
            inferred_new_negative_pairs += 1

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with Path(config.output_csv).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                config.src_column,
                config.dst_column,
                "predicted_label",
                "negative_rule_pattern_id",
                "negative_rule_index",
                "target_edge_var",
                "candidate_source",
                "negative_rule_antecedent",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    for pattern_id, rule_index, antecedent, schema, target_edge_var in debug_rules:
        print(
            "[BodyRematchRule] "
            + f"pattern_id={pattern_id} "
            + f"rule_index={rule_index} "
            + f"target_edge_var={target_edge_var} "
            + f"antecedent={antecedent} "
            + f"schema={schema}"
        )

    return {
        "rows": len(rows),
        "rules": len(rules),
        "patterns_with_instances": len(instances_by_pattern),
        "schemas": len(pattern_schemas),
        "body_matches": body_matches,
        "inferred_new_negative_pairs": inferred_new_negative_pairs,
        "skipped_existing_pair": skipped_existing_pair,
        "skipped_no_schema": skipped_no_schema,
        "skipped_not_expandable": skipped_not_expandable,
        "skipped_node_limit": skipped_node_limit,
    }


def expand_negative_edges(config: ExpansionConfig) -> dict[str, int]:
    """导出会被负规则扩展为 negative 的 interaction 端点索引。

    扩展流程：
    1. 读取 interaction CSV。
    2. 读取主流程写出的 deduped_rules.txt。
    3. 对每条 interaction 构造 e0/v0/v1 属性上下文。
    4. 如果满足某条 consequent=negative 的规则，就输出该 interaction 的端点索引。
    """

    if config.expansion_mode == "anchored_existing_edge_labeling":
        return expand_existing_edges_as_negative_anchored(config)
    if config.expansion_mode == "existing_edge_labeling":
        return expand_existing_edges_as_negative(config)

    rows, _fields = read_rows(str(config.input_csv))
    rules = load_rules(config.rules_file)
    if config.expansion_mode == "body_rematch_non_edges":
        return expand_body_rematch_non_edges(config, rows, rules)
    if config.expansion_mode == "candidate_non_edges":
        return expand_candidate_non_edges(config, rows, rules)
    if config.expansion_mode != "matched_existing":
        raise ValueError(f"unsupported expansion_mode: {config.expansion_mode}")
    instances_by_pattern = load_pattern_instances(config.pattern_instances_file)
    edge_attr_index = build_edge_attr_index(rows, config.src_column, config.dst_column, config.dataset_name)
    allowed_existing = set(config.only_labels or MISSING_LABELS)

    matched = 0
    exported = 0
    skipped_existing = 0
    output_rows: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for rule_index, rule in enumerate(rules):
        if rule.negative_label != config.negative_value:
            continue
        for instance in instances_by_pattern.get(rule.pattern_id, []):
            context = pattern_context_from_instance(instance, edge_attr_index)
            if not rule_matches(rule, context):
                continue
            matched += 1

            target_edge_var = consequent_target_edge(rule)
            target_edge = instance.get("edges", {}).get(target_edge_var)
            if not target_edge:
                continue
            src = normalize_value(target_edge.get("src_index") or target_edge.get("src"))
            dst = normalize_value(target_edge.get("dst_index") or target_edge.get("dst"))
            if not src or not dst:
                continue

            target_attrs = edge_attr_index.get((src, dst), {})
            label = normalize_value(target_attrs.get(normalize_key(config.label_column)))
            if not config.overwrite_existing and label not in allowed_existing:
                skipped_existing += 1
                continue

            pair = tuple(sorted((src, dst))) if config.dataset_name == "PPI" else (src, dst)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            output_rows.append(
                {
                    config.src_column: src,
                    config.dst_column: dst,
                    "negative_rule_pattern_id": str(rule.pattern_id),
                    "negative_rule_index": str(rule_index),
                    "match_id": str(instance.get("match_id", "")),
                    "target_edge_var": target_edge_var,
                    "negative_rule_antecedent": " & ".join(rule.antecedent),
                }
            )
            exported += 1

    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with Path(config.output_csv).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                config.src_column,
                config.dst_column,
                "negative_rule_pattern_id",
                "negative_rule_index",
                "match_id",
                "target_edge_var",
                "negative_rule_antecedent",
            ],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    return {
        "rows": len(rows),
        "rules": len(rules),
        "patterns_with_instances": len(instances_by_pattern),
        "matched_instances": matched,
        "exported_pairs": exported,
        "skipped_existing_label": skipped_existing,
    }


def main() -> None:
    summary = expand_negative_edges(CONFIG)
    print(
        "[NegativeEdgeExpansion] "
        + f"dataset={CONFIG.dataset_name} "
        + f"mode={CONFIG.expansion_mode} "
        + " ".join(f"{key}={value}" for key, value in summary.items())
        + f" output={CONFIG.output_csv}"
    )


if __name__ == "__main__":
    main()
