import csv
import json
import os
import pickle
import signal
import time

import networkx as nx
import pandas as pd
from networkx.algorithms.isomorphism import GraphMatcher

from inspect_graph import SelectedPPIDataset, to_original_edge_list


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

SELECTED_PATH = os.path.join(CURRENT_DIR, "processed", "ppi", "ppi_selected.pt")
PROTEIN_CSV = os.path.join(CURRENT_DIR, "data", "protein.csv")
PROTEIN_PROTEIN_CSV = os.path.join(CURRENT_DIR, "data", "protein_protein.csv")
BIG_GRAPH_CACHE = os.path.join(CURRENT_DIR, "processed", "ppi", "ppi_big_graph_with_attrs.pkl")

OUTPUT_DIR = os.path.join(CURRENT_DIR, "processed", "ppi", "bn_tables")
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "pattern_hit_summary.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "pattern_hit_summary.json")

# Leave these lists empty first, then fill with the columns you want from protein.csv / protein_protein.csv.
# Example node columns: ["ProteinName"]
# Example edge columns: []
#哪些当literal？
NODE_ATTR_COLUMNS = ["location","pathway","domain","length","Ab"]
EDGE_ATTR_COLUMNS = []

# Matching / export config
MATCH_GRAPH_INDEX = None
MATCH_START_INDEX = 0
MATCH_END_INDEX = None
MIN_HIT_COUNT = 2
MAX_HITS = None
TIMEOUT_SECONDS = None


class MatchTimeout(Exception):
    pass


class TimeLimit:
    def __init__(self, seconds):
        self.seconds = int(seconds) if seconds is not None else 0
        self.old_handler = signal.SIG_DFL

    def __enter__(self):
        if self.seconds > 0 and hasattr(signal, "SIGALRM"):
            self.old_handler = signal.getsignal(signal.SIGALRM)

            def handler(signum, frame):
                raise MatchTimeout()

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.seconds > 0 and hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.old_handler)


def load_protein_tables():
    protein_df = pd.read_csv(PROTEIN_CSV)
    edge_df = pd.read_csv(PROTEIN_PROTEIN_CSV)
    return protein_df, edge_df


def build_big_graph_with_attrs(protein_df, edge_df):
    graph = nx.Graph()

    protein_df = protein_df.copy()
    protein_df.columns = [str(c).strip() for c in protein_df.columns]
    edge_df = edge_df.copy()
    edge_df.columns = [str(c).strip() for c in edge_df.columns]

    if "index" not in protein_df.columns:
        raise ValueError(f"'index' column not found in {PROTEIN_CSV}")
    if "src" not in edge_df.columns or "dst" not in edge_df.columns:
        raise ValueError(f"'src'/'dst' columns not found in {PROTEIN_PROTEIN_CSV}")

    protein_attr_map = {}
    for _, row in protein_df.iterrows():
        node_id = int(row["index"])
        attr = row.to_dict()
        attr["match_label"] = "protein"
        protein_attr_map[node_id] = attr
        graph.add_node(node_id, **attr)

    for _, row in edge_df.iterrows():
        src = int(row["src"])
        dst = int(row["dst"])
        if src == dst:
            continue
        if src not in protein_attr_map or dst not in protein_attr_map:
            continue
        edge_attr = row.to_dict()
        edge_attr["match_label"] = "protein_protein"
        graph.add_edge(src, dst, **edge_attr)

    print(
        f"[BigGraph] Loaded big PPI graph: {graph.number_of_nodes()} nodes, "
        f"{graph.number_of_edges()} edges"
    )
    return graph


def load_or_build_big_graph():
    os.makedirs(os.path.dirname(BIG_GRAPH_CACHE), exist_ok=True)
    if os.path.exists(BIG_GRAPH_CACHE):
        print(f"[Cache] Loading big graph from: {BIG_GRAPH_CACHE}")
        with open(BIG_GRAPH_CACHE, "rb") as f:
            graph = pickle.load(f)
        print(
            f"[Cache] Loaded big graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )
        return graph

    print("[Cache] Big graph cache not found. Building from protein tables...")
    protein_df, edge_df = load_protein_tables()
    graph = build_big_graph_with_attrs(protein_df, edge_df)
    with open(BIG_GRAPH_CACHE, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Cache] Saved big graph cache to: {BIG_GRAPH_CACHE}")
    return graph


def selected_data_to_small_graph(data):
    graph = nx.Graph()
    orig_ids = [int(v) for v in data.orig_node_ids.tolist()]

    for idx, orig_id in enumerate(orig_ids):
        graph.add_node(idx, orig_id=orig_id, match_label="protein")

    seen = set()
    for eid in range(data.edge_index.size(1)):
        src = int(data.edge_index[0, eid])
        dst = int(data.edge_index[1, eid])
        if src == dst:
            continue
        key = tuple(sorted((src, dst)))
        if key in seen:
            continue
        seen.add(key)
        graph.add_edge(src, dst, match_label="protein_protein")

    return graph


def node_match(n1, n2):
    return n1.get("match_label") == n2.get("match_label")


def edge_match(e1, e2):
    return e1.get("match_label") == e2.get("match_label")


def _normalized_mapping(mapping, small_nodes):
    key_set = set(mapping.keys())
    if key_set == small_nodes:
        return {int(k): int(v) for k, v in mapping.items()}
    return {int(v): int(k) for k, v in mapping.items()}


def enumerate_hits(big_graph, small_graph, max_hits=None, timeout_seconds=None):
    matcher = GraphMatcher(big_graph, small_graph, node_match=node_match, edge_match=edge_match)
    small_nodes = set(small_graph.nodes())
    unique_hits = {}
    timed_out = False

    try:
        with TimeLimit(timeout_seconds):
            for mapping in matcher.subgraph_isomorphisms_iter():
                normalized = _normalized_mapping(mapping, small_nodes)
                node_set = frozenset(normalized.values())
                if node_set in unique_hits:
                    continue
                unique_hits[node_set] = normalized
                if max_hits is not None and len(unique_hits) >= max_hits:
                    break
    except MatchTimeout:
        timed_out = True

    return list(unique_hits.values()), timed_out


def build_relation_rows(big_graph, small_graph, hit_mappings, selected_graph_index):
    rows = []
    pattern_nodes = sorted(small_graph.nodes())
    pattern_edges = sorted(tuple(sorted(edge)) for edge in small_graph.edges())

    for hit_id, mapping in enumerate(hit_mappings):
        row = {
            "selected_graph_index": selected_graph_index,
            "hit_id": hit_id,
        }

        for pattern_node in pattern_nodes:
            big_node = mapping[pattern_node]
            row[f"v{pattern_node}_orig_id"] = int(big_node)

            for attr_name in NODE_ATTR_COLUMNS:
                row[f"v{pattern_node}_{attr_name}"] = big_graph.nodes[big_node].get(attr_name)

        for src, dst in pattern_edges:
            big_src = mapping[src]
            big_dst = mapping[dst]
            row[f"e_{src}_{dst}_src_orig_id"] = int(big_src)
            row[f"e_{src}_{dst}_dst_orig_id"] = int(big_dst)

            for attr_name in EDGE_ATTR_COLUMNS:
                row[f"e_{src}_{dst}_{attr_name}"] = big_graph[big_src][big_dst].get(attr_name)

        rows.append(row)

    return rows


def save_pattern_rows(selected_graph_index, rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_csv = os.path.join(OUTPUT_DIR, f"pattern_{selected_graph_index}_hits.csv")
    output_json = os.path.join(OUTPUT_DIR, f"pattern_{selected_graph_index}_hits.json")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"[Saved] CSV -> {output_csv}")
    print(f"[Saved] JSON -> {output_json}")


def save_summary(summary_rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "selected_graph_index",
        "num_pattern_nodes",
        "num_pattern_edges",
        "hit_count",
        "timed_out",
        "elapsed_seconds",
        "exported",
    ]
    with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"[Saved] summary CSV -> {SUMMARY_CSV}")
    print(f"[Saved] summary JSON -> {SUMMARY_JSON}")


def parse_graph_indices(dataset_size):
    if MATCH_GRAPH_INDEX is not None:
        return [MATCH_GRAPH_INDEX]

    start = 0 if MATCH_START_INDEX is None else MATCH_START_INDEX
    end = dataset_size if MATCH_END_INDEX is None else min(MATCH_END_INDEX, dataset_size)
    return list(range(start, end))


def main():
    print(f"[Config] match_graph_index={MATCH_GRAPH_INDEX}")
    print(f"[Config] match_start_index={MATCH_START_INDEX}")
    print(f"[Config] match_end_index={MATCH_END_INDEX}")
    print(f"[Config] min_hit_count={MIN_HIT_COUNT}")
    print(f"[Config] node_attr_columns={NODE_ATTR_COLUMNS}")
    print(f"[Config] edge_attr_columns={EDGE_ATTR_COLUMNS}")
    print(f"[Config] max_hits={MAX_HITS}")
    print(f"[Config] timeout_seconds={TIMEOUT_SECONDS}")

    big_graph = load_or_build_big_graph()
    dataset = SelectedPPIDataset(SELECTED_PATH)
    graph_indices = parse_graph_indices(len(dataset))
    if not graph_indices:
        raise RuntimeError("No selected graphs chosen for export.")

    summary_rows = []
    for selected_graph_index in graph_indices:
        data = dataset.get(selected_graph_index)
        small_graph = selected_data_to_small_graph(data)
        print(
            f"[SmallGraph] graph_index={selected_graph_index}, "
            f"nodes={small_graph.number_of_nodes()}, edges={small_graph.number_of_edges()}"
        )
        print(f"[SmallGraph] original_edges={to_original_edge_list(data)}")

        start_time = time.time()
        hit_mappings, timed_out = enumerate_hits(
            big_graph=big_graph,
            small_graph=small_graph,
            max_hits=MAX_HITS,
            timeout_seconds=TIMEOUT_SECONDS,
        )
        elapsed = time.time() - start_time
        hit_count = len(hit_mappings)
        exported = hit_count >= MIN_HIT_COUNT

        print(
            f"[Match] graph_index={selected_graph_index} hit_count={hit_count} "
            f"timed_out={timed_out} elapsed_seconds={elapsed:.2f} exported={exported}"
        )

        summary_rows.append(
            {
                "selected_graph_index": selected_graph_index,
                "num_pattern_nodes": small_graph.number_of_nodes(),
                "num_pattern_edges": small_graph.number_of_edges(),
                "hit_count": hit_count,
                "timed_out": bool(timed_out),
                "elapsed_seconds": float(elapsed),
                "exported": bool(exported),
            }
        )

        if not exported:
            continue

        rows = build_relation_rows(
            big_graph=big_graph,
            small_graph=small_graph,
            hit_mappings=hit_mappings,
            selected_graph_index=selected_graph_index,
        )
        save_pattern_rows(selected_graph_index, rows)

    save_summary(summary_rows)


if __name__ == "__main__":
    main()
