"""Build matched Baseline, LLM-augmented, and GAR+-augmented GNN edge files.

Only labels 1 (positive) and 2 (negative) are written. The downstream
``prepare_data`` routine is responsible for sampling label-0 non-edges.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# =========================
# Edit these paths/settings.
# =========================
LLM_EDGE_PATH = "/home/yyyy/codework/GARplus/experiments/exp1_accuracy/DDA_test/data_signed/edges_labeled_with_reason.csv"
GAR_NEG_EDGE_PATH = "/home/yyyy/codework/GARplus/enumeration-discovery/processed/dda/rule_negative_pairs_0626.csv"
OUTPUT_DIR = "/home/yyyy/codework/GARplus/experiments/exp1_accuracy/DDA_test/data_signed"

NODE_CSV_PATH = "/home/yyyy/codework/GARplus/experiments/exp1_accuracy/DDA_test/data_signed/node_labeled.csv"
NODE_ID_COL = "node_id"  # e.g. node_id, index, original_index, or source_node_id


LLM_NODE_ID_COL = "node_id"
GAR_NODE_ID_COL = "old_index"
# TI commonly stores Disease old_index as disease_index + 1_000_000_000.
GAR_SRC_NODE_ID_OFFSET = 0
GAR_DST_NODE_ID_OFFSET = 0

# Build a fresh node table from the union of LLM and GAR endpoint entities.
# This is the appropriate TI mode when GAR candidate endpoints are outside the
# old GNN node subset. LLM's current node_id is resolved through old_index.
BUILD_TYPED_UNIFIED_NODE_CSV = True
UNIFIED_NODE_CSV_NAME = "unified_node.csv"
LLM_CANONICAL_ID_COL = "old_index"
LLM_SRC_NODE_TYPE = "Gene"
LLM_DST_NODE_TYPE = "Disease"
GAR_SRC_NODE_TYPE = "Gene"
GAR_DST_NODE_TYPE = "Disease"

SRC_COL = "src"
DST_COL = "dst"
LABEL_COL = "label"

 #Leave as None to detect GAR expander outputs automatically:
# TI: gene_index,disease_index; DDA: chemical_index,disease_index;
# PPI: index_A,index_B. Set both explicitly for a custom GAR CSV schema.
GAR_SRC_COL = None
GAR_DST_COL = None

DIRECTED = False
RANDOM_SEED = 42


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str


def require_input_file(raw_path: str, description: str) -> Path:
    if not raw_path or raw_path.startswith("TODO/"):
        raise ValueError(f"{description} is not configured: {raw_path!r}")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def normalize_id(value: object) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
        return text[:-2]
    return text


def load_node_id_map(path: Path, node_id_column: str) -> tuple[dict[str, int], int]:
    if not node_id_column or node_id_column == "TODO":
        raise ValueError("A node id column is not configured. Set NODE_ID_COL or a source-specific override.")
    rows = read_csv_rows(path, (node_id_column,))
    id_map: dict[str, int] = {}
    for dgl_node_id, row in enumerate(rows):
        raw_node_id = normalize_id(row.get(node_id_column))
        if not raw_node_id:
            raise ValueError(f"{path} has an empty {node_id_column!r} value at row {dgl_node_id}")
        if raw_node_id in id_map:
            raise ValueError(
                f"{path} has duplicate {node_id_column!r} value {raw_node_id!r}; raw-to-DGL mapping would be ambiguous."
            )
        id_map[raw_node_id] = dgl_node_id
    if not id_map:
        raise ValueError(f"{path} contains no nodes")
    return id_map, len(rows)


def add_id_offset(raw_id: str, offset: int, description: str) -> str:
    if not offset:
        return raw_id
    try:
        return str(int(raw_id) + offset)
    except ValueError as exc:
        raise ValueError(f"Cannot apply {description}={offset} to non-integer node id {raw_id!r}") from exc


def map_edges_to_dgl_ids(
    edges: Iterable[Edge],
    id_map: dict[str, int],
    src_offset: int = 0,
    dst_offset: int = 0,
) -> tuple[list[Edge], int, int, int]:
    mapped: list[Edge] = []
    missing_node_count = 0
    missing_src_count = 0
    missing_dst_count = 0
    for edge in edges:
        src = id_map.get(add_id_offset(edge.src, src_offset, "src_offset"))
        dst = id_map.get(add_id_offset(edge.dst, dst_offset, "dst_offset"))
        if src is None or dst is None:
            missing_node_count += 1
            missing_src_count += src is None
            missing_dst_count += dst is None
            continue
        mapped.append(Edge(str(src), str(dst)))
    return mapped, missing_node_count, missing_src_count, missing_dst_count


def _typed_node_sort_key(item: tuple[str, str]) -> tuple[str, int, str]:
    node_type, raw_id = item
    try:
        return node_type, 0, f"{int(raw_id):020d}"
    except ValueError:
        return node_type, 1, raw_id


def build_typed_unified_edges(
    node_path: Path,
    llm_positive: list[Edge],
    llm_negative: list[Edge],
    gar_negative: list[Edge],
) -> tuple[list[Edge], list[Edge], list[Edge], list[dict[str, str]], dict[str, int]]:
    """Map LLM and GAR into a shared `(node_type, raw_index)` identifier space."""

    lookup_column = LLM_NODE_ID_COL or NODE_ID_COL
    node_rows = read_csv_rows(node_path, (lookup_column, LLM_CANONICAL_ID_COL))
    llm_crosswalk: dict[str, str] = {}
    for row in node_rows:
        old_id = normalize_id(row.get(lookup_column))
        canonical_id = normalize_id(row.get(LLM_CANONICAL_ID_COL))
        if not old_id or not canonical_id:
            continue
        if old_id in llm_crosswalk and llm_crosswalk[old_id] != canonical_id:
            raise ValueError(f"Ambiguous LLM crosswalk id {old_id!r} in {node_path}")
        llm_crosswalk[old_id] = canonical_id

    missing = {"llm_positive": 0, "llm_negative": 0, "gar_negative": 0}
    typed_llm_positive: list[tuple[tuple[str, str], tuple[str, str]]] = []
    typed_llm_negative: list[tuple[tuple[str, str], tuple[str, str]]] = []
    typed_gar_negative: list[tuple[tuple[str, str], tuple[str, str]]] = []

    def convert_llm(edges: list[Edge], key: str) -> list[tuple[tuple[str, str], tuple[str, str]]]:
        converted = []
        for edge in edges:
            src_raw = llm_crosswalk.get(edge.src)
            dst_raw = llm_crosswalk.get(edge.dst)
            if src_raw is None or dst_raw is None:
                missing[key] += 1
                continue
            converted.append(((LLM_SRC_NODE_TYPE, src_raw), (LLM_DST_NODE_TYPE, dst_raw)))
        return converted

    typed_llm_positive = convert_llm(llm_positive, "llm_positive")
    typed_llm_negative = convert_llm(llm_negative, "llm_negative")
    for edge in gar_negative:
        typed_gar_negative.append(((GAR_SRC_NODE_TYPE, edge.src), (GAR_DST_NODE_TYPE, edge.dst)))

    typed_nodes = {
        node
        for typed_edges in (typed_llm_positive, typed_llm_negative, typed_gar_negative)
        for edge in typed_edges
        for node in edge
    }
    typed_node_to_dgl = {node: index for index, node in enumerate(sorted(typed_nodes, key=_typed_node_sort_key))}

    def materialize(typed_edges: list[tuple[tuple[str, str], tuple[str, str]]]) -> list[Edge]:
        return [Edge(str(typed_node_to_dgl[src]), str(typed_node_to_dgl[dst])) for src, dst in typed_edges]

    unified_nodes = [
        {"node_id": str(node_id), "node_type": node_type, "raw_index": raw_index}
        for (node_type, raw_index), node_id in sorted(typed_node_to_dgl.items(), key=lambda item: item[1])
    ]
    return (
        materialize(typed_llm_positive),
        materialize(typed_llm_negative),
        materialize(typed_gar_negative),
        unified_nodes,
        missing,
    )


def edge_key(edge: Edge) -> tuple[str, str]:
    if DIRECTED:
        return edge.src, edge.dst
    return tuple(sorted((edge.src, edge.dst)))


def is_self_loop(edge: Edge) -> bool:
    return edge.src == edge.dst


def read_csv_rows(path: Path, required_columns: Iterable[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = [column for column in required_columns if column not in fields]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}; found: {sorted(fields)}")
        return list(reader)


def read_llm_edges(path: Path) -> tuple[list[Edge], list[Edge]]:
    rows = read_csv_rows(path, (SRC_COL, DST_COL, LABEL_COL))
    positives: list[Edge] = []
    negatives: list[Edge] = []
    invalid_labels: set[str] = set()
    for row in rows:
        label = normalize_id(row.get(LABEL_COL))
        edge = Edge(normalize_id(row.get(SRC_COL)), normalize_id(row.get(DST_COL)))
        if not edge.src or not edge.dst:
            continue
        if label == "1":
            positives.append(edge)
        elif label == "2":
            negatives.append(edge)
        else:
            invalid_labels.add(label)
    if invalid_labels:
        raise ValueError(
            f"{path} contains labels other than 1/2: {sorted(invalid_labels)}. "
            "This builder expects LLM_EDGE_PATH to contain only labeled positive/negative edges."
        )
    return positives, negatives


def read_gar_negative_edges(path: Path) -> list[Edge]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        endpoint_candidates = (
            (GAR_SRC_COL, GAR_DST_COL),
            (SRC_COL, DST_COL),
            ("gene_index", "disease_index"),
            ("chemical_index", "disease_index"),
            ("index_A", "index_B"),
        )
        src_column = dst_column = None
        for candidate_src, candidate_dst in endpoint_candidates:
            if candidate_src and candidate_dst and candidate_src in fields and candidate_dst in fields:
                src_column, dst_column = candidate_src, candidate_dst
                break
        if src_column is None or dst_column is None:
            raise ValueError(
                f"{path} has no recognized GAR endpoint columns. Expected src/dst, "
                "gene_index/disease_index, chemical_index/disease_index, or index_A/index_B; "
                f"found: {sorted(fields)}. Set GAR_SRC_COL/GAR_DST_COL for a custom schema."
            )
        rows = list(reader)
    if rows and LABEL_COL in rows[0]:
        invalid_labels = {normalize_id(row.get(LABEL_COL)) for row in rows if normalize_id(row.get(LABEL_COL)) != "2"}
        if invalid_labels:
            raise ValueError(
                f"{path} has a {LABEL_COL!r} column but contains non-negative labels: {sorted(invalid_labels)}"
            )
    return [
        Edge(normalize_id(row.get(src_column)), normalize_id(row.get(dst_column)))
        for row in rows
        if normalize_id(row.get(src_column)) and normalize_id(row.get(dst_column))
    ]


def clean_and_dedupe(edges: Iterable[Edge]) -> dict[tuple[str, str], Edge]:
    """Drop loops and retain the first original orientation for each logical edge."""

    result: dict[tuple[str, str], Edge] = {}
    for edge in edges:
        if is_self_loop(edge):
            continue
        result.setdefault(edge_key(edge), edge)
    return result


def sample_edges(edges: Iterable[Edge], count: int, rng: random.Random, description: str) -> list[Edge]:
    pool = list(edges)
    if len(pool) < count:
        raise ValueError(f"Cannot sample {count} {description} edges from a pool of {len(pool)}")
    return rng.sample(pool, count)


def possible_pair_count(node_count: int) -> int:
    return node_count * (node_count - 1) if DIRECTED else node_count * (node_count - 1) // 2


def sample_pseudo_negative_edges(
    nodes: list[str],
    forbidden: set[tuple[str, str]],
    count: int,
    rng: random.Random,
) -> list[Edge]:
    """Sample non-edges reproducibly; fall back to deterministic enumeration if needed."""

    available_upper_bound = possible_pair_count(len(nodes)) - len(forbidden)
    if available_upper_bound < count:
        raise ValueError(
            f"Cannot sample {count} pseudo-negative edges: at most {available_upper_bound} pairs remain. "
            "The node set may be too small or forbidden edges may be too numerous."
        )

    selected: dict[tuple[str, str], Edge] = {}
    attempts = 0
    max_attempts = max(10_000, count * 100)
    while len(selected) < count and attempts < max_attempts:
        src, dst = rng.sample(nodes, 2)
        edge = Edge(src, dst)
        key = edge_key(edge)
        attempts += 1
        if key not in forbidden and key not in selected:
            selected[key] = edge

    if len(selected) < count:
        for src_index, src in enumerate(nodes):
            destinations = nodes if DIRECTED else nodes[src_index + 1 :]
            for dst in destinations:
                if src == dst:
                    continue
                edge = Edge(src, dst)
                key = edge_key(edge)
                if key not in forbidden and key not in selected:
                    selected[key] = edge
                    if len(selected) == count:
                        break
            if len(selected) == count:
                break

    if len(selected) != count:
        raise ValueError(
            f"Could only sample {len(selected)}/{count} pseudo-negative edges. "
            "The node set may be too small or forbidden edges may be too numerous."
        )
    return list(selected.values())


def labeled_rows(positives: Iterable[Edge], negatives: Iterable[Edge]) -> list[dict[str, str]]:
    return (
        [{SRC_COL: edge.src, DST_COL: edge.dst, LABEL_COL: "1"} for edge in positives]
        + [{SRC_COL: edge.src, DST_COL: edge.dst, LABEL_COL: "2"} for edge in negatives]
    )


def validate_output_rows(rows: list[dict[str, str]], description: str) -> tuple[set[tuple[str, str]], int]:
    labels = {row[LABEL_COL] for row in rows}
    if not labels.issubset({"1", "2"}):
        raise AssertionError(f"{description} contains labels other than 1/2: {labels}")
    positive_keys = {edge_key(Edge(row[SRC_COL], row[DST_COL])) for row in rows if row[LABEL_COL] == "1"}
    negative_count = sum(row[LABEL_COL] == "2" for row in rows)
    if len(positive_keys) != sum(row[LABEL_COL] == "1" for row in rows):
        raise AssertionError(f"{description} contains duplicate positive edges")
    return positive_keys, negative_count


def assert_valid_node_ids(rows: list[dict[str, str]], num_nodes: int, name: str) -> None:
    if not rows:
        raise ValueError(f"{name} is empty")
    try:
        src_ids = [int(row[SRC_COL]) for row in rows]
        dst_ids = [int(row[DST_COL]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} contains non-integer src/dst ids") from exc
    min_id = min(min(src_ids), min(dst_ids))
    max_id = max(max(src_ids), max(dst_ids))
    if min_id < 0 or max_id >= num_nodes:
        raise ValueError(f"{name} has invalid node ids: min={min_id}, max={max_id}, num_nodes={num_nodes}")


def write_edge_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[SRC_COL, DST_COL, LABEL_COL])
        writer.writeheader()
        writer.writerows(rows)


def write_unified_node_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["node_id", "node_type", "raw_index"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    llm_path = require_input_file(LLM_EDGE_PATH, "LLM_EDGE_PATH")
    gar_path = require_input_file(GAR_NEG_EDGE_PATH, "GAR_NEG_EDGE_PATH")
    node_path = require_input_file(NODE_CSV_PATH, "NODE_CSV_PATH")
    if not OUTPUT_DIR or OUTPUT_DIR.startswith("TODO/"):
        raise ValueError(f"OUTPUT_DIR is not configured: {OUTPUT_DIR!r}")
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_positive_raw, llm_negative_raw = read_llm_edges(llm_path)
    gar_negative_raw = read_gar_negative_edges(gar_path)
    print(f"[Input] LLM positives raw = {len(llm_positive_raw)}")
    print(f"[Input] LLM negatives raw = {len(llm_negative_raw)}")
    print(f"[Input] GAR negatives raw = {len(gar_negative_raw)}")
    llm_positive_raw_count = len(llm_positive_raw)
    llm_negative_raw_count = len(llm_negative_raw)
    gar_negative_raw_count = len(gar_negative_raw)
    llm_node_id_column = LLM_NODE_ID_COL or NODE_ID_COL
    gar_node_id_column = GAR_NODE_ID_COL or NODE_ID_COL
    unified_node_path = None
    if BUILD_TYPED_UNIFIED_NODE_CSV:
        (
            llm_positive_raw,
            llm_negative_raw,
            gar_negative_raw,
            unified_node_rows,
            missing_counts,
        ) = build_typed_unified_edges(node_path, llm_positive_raw, llm_negative_raw, gar_negative_raw)
        num_nodes = len(unified_node_rows)
        unified_node_path = output_dir / UNIFIED_NODE_CSV_NAME
        write_unified_node_csv(unified_node_path, unified_node_rows)
        llm_positive_missing_node = missing_counts["llm_positive"]
        llm_negative_missing_node = missing_counts["llm_negative"]
        gar_negative_missing_node = 0
        gar_missing_src = gar_missing_dst = 0
        print(
            f"[Nodes] typed unified node rows = {num_nodes}, "
            f"LLM lookup={llm_node_id_column}->{LLM_CANONICAL_ID_COL}, "
            f"GAR=( {GAR_SRC_NODE_TYPE}, {GAR_DST_NODE_TYPE} )"
        )
        print(f"[Nodes] wrote unified_node_csv={unified_node_path}")
    else:
        llm_id_map, num_nodes = load_node_id_map(node_path, llm_node_id_column)
        gar_id_map, gar_num_nodes = load_node_id_map(node_path, gar_node_id_column)
        if gar_num_nodes != num_nodes:
            raise AssertionError("Node mapping views disagree on node_labeled row count.")
        llm_positive_raw, llm_positive_missing_node, _llm_positive_missing_src, _llm_positive_missing_dst = map_edges_to_dgl_ids(
            llm_positive_raw, llm_id_map
        )
        llm_negative_raw, llm_negative_missing_node, _llm_negative_missing_src, _llm_negative_missing_dst = map_edges_to_dgl_ids(
            llm_negative_raw, llm_id_map
        )
        gar_negative_raw, gar_negative_missing_node, gar_missing_src, gar_missing_dst = map_edges_to_dgl_ids(
            gar_negative_raw,
            gar_id_map,
            src_offset=GAR_SRC_NODE_ID_OFFSET,
            dst_offset=GAR_DST_NODE_ID_OFFSET,
        )
        print(
            f"[Nodes] node_labeled rows = {num_nodes}, "
            f"LLM id_column = {llm_node_id_column}, GAR id_column = {gar_node_id_column}"
        )
    print(
        f"[Map] filtered missing nodes: LLM positive={llm_positive_missing_node}, "
        f"LLM negative={llm_negative_missing_node}, GAR negative={gar_negative_missing_node} "
        f"(GAR src_missing={gar_missing_src}, dst_missing={gar_missing_dst})"
    )

    positive_by_key = clean_and_dedupe(llm_positive_raw)
    llm_negative_by_key = clean_and_dedupe(llm_negative_raw)
    gar_negative_by_key = clean_and_dedupe(gar_negative_raw)
    llm_positive_after_dedup = len(positive_by_key)
    llm_negative_after_dedup = len(llm_negative_by_key)
    gar_negative_after_dedup = len(gar_negative_by_key)

    positive_keys = set(positive_by_key)
    llm_conflicts = set(llm_negative_by_key) & positive_keys
    gar_conflicts = set(gar_negative_by_key) & positive_keys
    for key in llm_conflicts:
        del llm_negative_by_key[key]
    for key in gar_conflicts:
        del gar_negative_by_key[key]
    print(f"[Clean] LLM negatives conflict removed = {len(llm_conflicts)}")
    print(f"[Clean] GAR negatives conflict removed = {len(gar_conflicts)}")

    n_negative = min(len(llm_negative_by_key), len(gar_negative_by_key))
    if not positive_by_key or n_negative == 0:
        empty = []
        if not positive_by_key:
            empty.append("LLM positive")
        if not llm_negative_by_key:
            empty.append("LLM negative")
        if not gar_negative_by_key:
            empty.append("GAR negative")
        raise ValueError(f"Cannot build datasets because these cleaned edge sets are empty: {', '.join(empty)}")

    rng = random.Random(RANDOM_SEED)
    shared_positive = list(positive_by_key.values())
    sampled_llm_negative = sample_edges(llm_negative_by_key.values(), n_negative, rng, "LLM negative")
    sampled_gar_negative = sample_edges(gar_negative_by_key.values(), n_negative, rng, "GAR negative")
    print(f"[Build] shared positives = {len(shared_positive)}, negatives per setting = {n_negative}")

    all_nodes = [str(node_id) for node_id in range(num_nodes)]
    if num_nodes < 2:
        raise ValueError("At least two distinct nodes are required to sample baseline pseudo-negative edges.")
    all_observed = set(positive_by_key) | set(llm_negative_by_key) | set(gar_negative_by_key)
    forbidden = all_observed | {edge_key(edge) for edge in shared_positive} | {
        edge_key(edge) for edge in sampled_llm_negative
    } | {edge_key(edge) for edge in sampled_gar_negative}
    baseline_negative = sample_pseudo_negative_edges(all_nodes, forbidden, n_negative, rng)

    baseline_rows = labeled_rows(shared_positive, baseline_negative)
    llm_rows = labeled_rows(shared_positive, sampled_llm_negative)
    gar_rows = labeled_rows(shared_positive, sampled_gar_negative)
    baseline_positive, baseline_negative_count = validate_output_rows(baseline_rows, "baseline_edges.csv")
    llm_positive, llm_negative_count = validate_output_rows(llm_rows, "llm_augmented_edges.csv")
    gar_positive, gar_negative_count = validate_output_rows(gar_rows, "gar_augmented_edges.csv")
    if not (baseline_positive == llm_positive == gar_positive):
        raise AssertionError("The three outputs do not share exactly the same positive edge set.")
    if len({baseline_negative_count, llm_negative_count, gar_negative_count}) != 1:
        raise AssertionError("The three outputs do not contain the same number of negative edges.")
    assert_valid_node_ids(baseline_rows, num_nodes, "baseline_edges.csv")
    assert_valid_node_ids(llm_rows, num_nodes, "llm_augmented_edges.csv")
    assert_valid_node_ids(gar_rows, num_nodes, "gar_augmented_edges.csv")

    outputs = {
        "baseline_edges.csv": baseline_rows,
        "llm_augmented_edges.csv": llm_rows,
        "gar_augmented_edges.csv": gar_rows,
    }
    for filename, rows in outputs.items():
        write_edge_csv(output_dir / filename, rows)
        positives = sum(row[LABEL_COL] == "1" for row in rows)
        negatives = sum(row[LABEL_COL] == "2" for row in rows)
        print(f"[Output] {filename}: label1={positives}, label2={negatives}")

    stats = {
        "num_llm_positive_raw": llm_positive_raw_count,
        "num_llm_negative_raw": llm_negative_raw_count,
        "num_gar_negative_raw": gar_negative_raw_count,
        "num_llm_positive_filtered_missing_node": llm_positive_missing_node,
        "num_llm_negative_filtered_missing_node": llm_negative_missing_node,
        "num_gar_negative_filtered_missing_node": gar_negative_missing_node,
        "num_llm_positive_after_dedup": llm_positive_after_dedup,
        "num_llm_negative_after_dedup": llm_negative_after_dedup,
        "num_gar_negative_after_dedup": gar_negative_after_dedup,
        "num_llm_negative_conflict_with_positive_removed": len(llm_conflicts),
        "num_gar_negative_conflict_with_positive_removed": len(gar_conflicts),
        "n_shared": len(shared_positive),
        "n_negative_per_setting": n_negative,
        "baseline": {"positive": len(shared_positive), "negative": len(baseline_negative)},
        "llm_augmented": {"positive": len(shared_positive), "negative": len(sampled_llm_negative)},
        "gar_augmented": {"positive": len(shared_positive), "negative": len(sampled_gar_negative)},
        "same_positive_across_three_settings": True,
        "same_negative_count_across_three_settings": True,
        "directed": DIRECTED,
        "random_seed": RANDOM_SEED,
        "num_nodes": num_nodes,
        "llm_node_id_column": llm_node_id_column,
        "gar_node_id_column": gar_node_id_column,
        "gar_src_node_id_offset": GAR_SRC_NODE_ID_OFFSET,
        "gar_dst_node_id_offset": GAR_DST_NODE_ID_OFFSET,
        "num_gar_negative_src_filtered_missing_node": gar_missing_src,
        "num_gar_negative_dst_filtered_missing_node": gar_missing_dst,
        "typed_unified_node_mapping": BUILD_TYPED_UNIFIED_NODE_CSV,
        "unified_node_csv": str(unified_node_path) if unified_node_path else None,
    }
    with (output_dir / "dataset_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[Done] files saved to {output_dir}")
    print("These three CSV files can be used directly as edge_csv inputs to the current GNN code; prepare_data samples label=0 no-edges.")


if __name__ == "__main__":
    main()
