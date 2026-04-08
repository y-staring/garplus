import csv
import json
import os
import pickle
import signal
import time

import networkx as nx
from networkx.algorithms.isomorphism import GraphMatcher

from inspect_graph import SelectedPPIDataset, summarize_graph, to_original_edge_list
from pick_patterns import DEFAULT_EDGE_CSV, DEFAULT_NODE_CSV, build_ppi_graph


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DEFAULT_SELECTED_PATH = os.path.join(CURRENT_DIR, "processed", "ppi", "ppi_selected.pt")
DEFAULT_BIG_GRAPH_CACHE = os.path.join(CURRENT_DIR, "processed", "ppi", "ppi_big_graph.pkl")
DEFAULT_OUTPUT_CSV = os.path.join(CURRENT_DIR, "processed", "ppi", "selected_match_counts.csv")
DEFAULT_OUTPUT_JSON = os.path.join(CURRENT_DIR, "processed", "ppi", "selected_match_counts.json")

# Run config
SELECTED_PATH = DEFAULT_SELECTED_PATH
EDGE_CSV = DEFAULT_EDGE_CSV
NODE_CSV = DEFAULT_NODE_CSV
BIG_GRAPH_CACHE = DEFAULT_BIG_GRAPH_CACHE
OUTPUT_CSV = DEFAULT_OUTPUT_CSV
OUTPUT_JSON = DEFAULT_OUTPUT_JSON

# Match the whole dataset by default. Set MATCH_GRAPH_INDEX = 0 to test one graph first.
MATCH_GRAPH_INDEX = None
MATCH_START_INDEX = 0
MATCH_END_INDEX = None
MAX_MATCHES = None
TIMEOUT_SECONDS = None

# Matching standard:
# "topology_only" means both big graph and small graph are treated as unlabeled protein-protein graphs.
MATCH_MODE = "topology_only"


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


def selected_data_to_nx(data):
    graph = nx.Graph()
    orig_ids = [int(v) for v in data.orig_node_ids.tolist()]
    for idx in range(len(orig_ids)):
        graph.add_node(idx, match_label="protein")

    edge_index = data.edge_index
    seen = set()
    for eid in range(edge_index.size(1)):
        src = int(edge_index[0, eid])
        dst = int(edge_index[1, eid])
        if src == dst:
            continue
        key = tuple(sorted((src, dst)))
        if key in seen:
            continue
        seen.add(key)
        graph.add_edge(src, dst, match_label="protein_protein")

    return graph


def _mapping_node_set(mapping, small_nodes):
    key_set = set(mapping.keys())
    if key_set == small_nodes:
        return frozenset(int(v) for v in mapping.values())
    return frozenset(int(k) for k in mapping.keys())


def node_match(n1, n2):
    if MATCH_MODE == "topology_only":
        return True
    return n1.get("match_label") == n2.get("match_label")


def edge_match(e1, e2):
    if MATCH_MODE == "topology_only":
        return True
    return e1.get("match_label") == e2.get("match_label")


def count_subgraph_matches(big_graph, small_graph, max_matches=None, timeout_seconds=None):
    matcher = GraphMatcher(big_graph, small_graph, node_match=node_match, edge_match=edge_match)
    unique_matches = set()
    timed_out = False

    try:
        with TimeLimit(timeout_seconds):
            for mapping in matcher.subgraph_isomorphisms_iter():
                match_nodes = _mapping_node_set(mapping, set(small_graph.nodes()))
                unique_matches.add(match_nodes)
                if max_matches is not None and len(unique_matches) >= max_matches:
                    break
    except MatchTimeout:
        timed_out = True

    return len(unique_matches), timed_out


def load_or_build_big_graph(edge_csv, node_csv, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if os.path.exists(cache_path):
        print(f"[Cache] Loading cached big graph from: {cache_path}")
        with open(cache_path, "rb") as f:
            big_graph = pickle.load(f)
        print(
            f"[Cache] Loaded big graph: {big_graph.number_of_nodes()} nodes, "
            f"{big_graph.number_of_edges()} edges"
        )
        return big_graph

    print("[Cache] Cached big graph not found. Building from raw csv...")
    big_graph = build_ppi_graph(edge_csv=edge_csv, node_csv=node_csv)
    for node_id in big_graph.nodes():
        big_graph.nodes[node_id]["match_label"] = "protein"
    for src, dst in big_graph.edges():
        big_graph[src][dst]["match_label"] = "protein_protein"

    with open(cache_path, "wb") as f:
        pickle.dump(big_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Cache] Saved big graph cache to: {cache_path}")
    return big_graph


def match_dataset(
    selected_path,
    edge_csv,
    node_csv,
    big_graph_cache,
    graph_indices=None,
    max_matches=None,
    timeout_seconds=None,
):
    dataset = SelectedPPIDataset(selected_path)
    big_graph = load_or_build_big_graph(edge_csv=edge_csv, node_csv=node_csv, cache_path=big_graph_cache)

    if graph_indices is None:
        graph_indices = list(range(len(dataset)))

    results = []
    for graph_index in graph_indices:
        data = dataset.get(graph_index)
        small_graph = selected_data_to_nx(data)
        stats = summarize_graph(data)
        center_orig = int(data.center_orig_id.item()) if hasattr(data, "center_orig_id") else None

        start_time = time.time()
        match_count, timed_out = count_subgraph_matches(
            big_graph=big_graph,
            small_graph=small_graph,
            max_matches=max_matches,
            timeout_seconds=timeout_seconds,
        )
        elapsed = time.time() - start_time

        result = {
            "graph_index": int(graph_index),
            "center_orig_id": center_orig,
            "num_nodes": int(stats["num_nodes"]),
            "num_edges_undirected": int(stats["num_edges_undirected"]),
            "match_count": int(match_count),
            "timed_out": bool(timed_out),
            "elapsed_seconds": float(elapsed),
            "orig_node_ids": [int(v) for v in data.orig_node_ids.tolist()],
            "original_edges": [
                [int(src), int(dst)] for src, dst in to_original_edge_list(data)
            ],
        }
        results.append(result)
        print(
            f"[Match] graph_index={graph_index} "
            f"nodes={result['num_nodes']} edges={result['num_edges_undirected']} "
            f"matches={result['match_count']} timed_out={result['timed_out']} "
            f"elapsed={result['elapsed_seconds']:.2f}s"
        )

    return results


def parse_graph_indices(graph_index, start_index, end_index, dataset_size):
    if graph_index is not None:
        return [graph_index]

    start = 0 if start_index is None else start_index
    end = dataset_size if end_index is None else min(end_index, dataset_size)
    return list(range(start, end))


def save_results(results, output_csv, output_json):
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph_index",
                "center_orig_id",
                "num_nodes",
                "num_edges_undirected",
                "match_count",
                "timed_out",
                "elapsed_seconds",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(
                {
                    "graph_index": row["graph_index"],
                    "center_orig_id": row["center_orig_id"],
                    "num_nodes": row["num_nodes"],
                    "num_edges_undirected": row["num_edges_undirected"],
                    "match_count": row["match_count"],
                    "timed_out": row["timed_out"],
                    "elapsed_seconds": row["elapsed_seconds"],
                }
            )

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[Saved] CSV -> {output_csv}")
    print(f"[Saved] JSON -> {output_json}")


def main():
    dataset = SelectedPPIDataset(SELECTED_PATH)
    graph_indices = parse_graph_indices(
        graph_index=MATCH_GRAPH_INDEX,
        start_index=MATCH_START_INDEX,
        end_index=MATCH_END_INDEX,
        dataset_size=len(dataset),
    )
    if not graph_indices:
        raise RuntimeError("No selected graphs were chosen for matching.")

    print(f"[Config] MATCH_MODE={MATCH_MODE}")
    print(f"[Config] selected_path={SELECTED_PATH}")
    print(f"[Config] graph_count={len(graph_indices)}")
    print(f"[Config] graph_indices={graph_indices[:10]}{'...' if len(graph_indices) > 10 else ''}")
    print(f"[Config] max_matches={MAX_MATCHES}")
    print(f"[Config] timeout_seconds={TIMEOUT_SECONDS}")

    results = match_dataset(
        selected_path=SELECTED_PATH,
        edge_csv=EDGE_CSV,
        node_csv=NODE_CSV,
        big_graph_cache=BIG_GRAPH_CACHE,
        graph_indices=graph_indices,
        max_matches=MAX_MATCHES,
        timeout_seconds=TIMEOUT_SECONDS,
    )
    save_results(results, OUTPUT_CSV, OUTPUT_JSON)


if __name__ == "__main__":
    main()
