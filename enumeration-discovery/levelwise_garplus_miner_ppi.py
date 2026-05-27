from __future__ import annotations

"""
Level-wise GAR+ miner, first coarse version.

This miner follows the high-level idea of SeqDis:
- vertical spawning: level-wise pattern expansion by adding one edge;
- pattern verification: coarse support checking over sampled patterns;
- horizontal spawning: level-wise predicate expansion X -> X ∪ {p};
- GFD/GAR+ validation: coarse support/confidence checking over the predicate table.

The difference is:
- this implementation adds Pattern-BN pruning before accepting structural extensions;
- this implementation adds Predicate-BN pruning before accepting predicate extensions;
- exact graph validation is not implemented in this first version.

Important limitations of this first version:
1. This is a coarse first version.
2. Pattern support is computed over sampled patterns, not exact subgraph isomorphism over G.
3. Predicate support/confidence is computed over global_predicate_table_full.csv.
4. Pattern-BN and Predicate-BN are only used for soft pruning/ranking.
5. Final exact GAR+ validation over the original graph G is left for the next stage.
"""

import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
# Allow running this file from the repo root by ensuring `enumeration-discovery`
# is on sys.path so sibling modules can be imported.
current_dir_str = str(CURRENT_DIR)
if current_dir_str not in sys.path:
    sys.path.insert(0, current_dir_str)

import build_pattern_edge_node_bn as pattern_bn_module

BN_FALLBACK_SCORE = 0.1


def load_predicate_repository(repo_path: str) -> dict:
    with open(repo_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_predicate_table(table_path: str) -> pd.DataFrame:
    return pd.read_csv(table_path)


def load_family_bn_edges(family_bns_dir: str) -> dict[str, dict]:
    """
    Load the learned family-wise Predicate-BNs as undirected adjacency maps.

    Expected layout:
      family_bns/
        family_<name>/
          result.json  (expects {"status": "learned"} when usable)
          edges.csv    (expects columns: source,target)
    """
    family_bns_path = Path(family_bns_dir)
    family_bn_states: dict[str, dict] = {}
    if not family_bns_path.exists():
        return family_bn_states

    for family_dir in sorted(family_bns_path.glob("family_*")):
        if not family_dir.is_dir():
            continue

        family_name = family_dir.name[len("family_") :]
        result_path = family_dir / "result.json"
        edges_path = family_dir / "edges.csv"

        status = "skipped"
        if result_path.exists():
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                status = str(result.get("status", "skipped"))
            except Exception:
                status = "skipped"

        neighbors: dict[str, set[str]] = {}
        nodes: set[str] = set()

        if status == "learned" and edges_path.exists():
            try:
                edges_df = pd.read_csv(edges_path)
                for _, row in edges_df.iterrows():
                    src = str(row["source"])
                    dst = str(row["target"])
                    nodes.add(src)
                    nodes.add(dst)
                    neighbors.setdefault(src, set()).add(dst)
                    neighbors.setdefault(dst, set()).add(src)
            except Exception:
                status = "skipped"
                neighbors = {}
                nodes = set()

        family_bn_states[family_name] = {
            "neighbors": neighbors,
            "nodes": nodes,
            "status": status,
        }

    return family_bn_states


def get_predicate_family_map(repository: dict) -> dict[str, str]:
    pred_family: dict[str, str] = {}
    for pred in repository.get("predicates", []):
        pid = pred.get("pid")
        family = pred.get("family", "unknown")
        if pid is not None:
            pred_family[str(pid)] = str(family)
    return pred_family


def get_candidate_predicates(
    repository: dict,
    table: pd.DataFrame,
    exclude_families: set[str] | None = None,
    exclude_sources: set[str] | None = None,
    min_support: int = 1,
    max_predicates: int | None = None,
) -> list[str]:
    exclude_families = set() if exclude_families is None else set(exclude_families)
    exclude_sources = set() if exclude_sources is None else set(exclude_sources)

    table_columns = set(table.columns) - {"pattern_id"}
    support_series = table.drop(columns=["pattern_id"], errors="ignore").sum().sort_values(ascending=False)

    candidates: list[tuple[str, int]] = []
    for pred in repository.get("predicates", []):
        pid = str(pred.get("pid"))
        family = str(pred.get("family", "unknown"))
        source = str(pred.get("source", "unknown"))

        if pid not in table_columns:
            continue
        if family in exclude_families:
            continue
        if source in exclude_sources:
            continue

        support = int(support_series.get(pid, 0))
        if support < min_support:
            continue
        candidates.append((pid, support))

    candidates.sort(key=lambda x: (-x[1], x[0]))
    if max_predicates is not None:
        candidates = candidates[:max_predicates]
    return [pid for pid, _ in candidates]


def compute_body_mask(table: pd.DataFrame, body: frozenset[str]) -> pd.Series:
    if not body:
        return pd.Series(True, index=table.index)
    return table[list(body)].all(axis=1)


def compute_rule_stats(table: pd.DataFrame, body: frozenset[str], head: str) -> dict:
    num_patterns = int(table.shape[0])
    body_mask = compute_body_mask(table, body)
    body_support = int(body_mask.sum())
    head_support = int(table[head].sum())

    if body_support == 0:
        support = 0
        confidence = 0.0
    else:
        support = int((body_mask & (table[head] == 1)).sum())
        confidence = float(support / body_support)

    head_prob = float(head_support / num_patterns) if num_patterns > 0 else 0.0
    lift = 0.0 if head_prob <= 0 else float(confidence / head_prob)

    return {
        "body_support": body_support,
        "support": support,
        "confidence": confidence,
        "head_support": head_support,
        "lift": lift,
    }


def bn_neighbor_score(
    body: frozenset[str],
    candidate: str,
    family_bn_states: dict,
    predicate_family: dict[str, str],
) -> float:
    """
    Score one candidate predicate extension using the learned family BN graph.

    Signals:
      - direct BN neighbor => 1.0
      - 2-hop neighbor     => 0.5
      - otherwise          => BN_FALLBACK_SCORE
    """
    candidate_family = predicate_family.get(candidate)
    if candidate_family is None:
        return BN_FALLBACK_SCORE

    family_state = family_bn_states.get(candidate_family)
    if not family_state or family_state.get("status") != "learned":
        return BN_FALLBACK_SCORE

    family_nodes = set(family_state.get("nodes", set()))
    if candidate not in family_nodes:
        return BN_FALLBACK_SCORE

    neighbors: dict[str, set[str]] = family_state.get("neighbors", {})
    candidate_neighbors = set(neighbors.get(candidate, set()))
    if not candidate_neighbors:
        return BN_FALLBACK_SCORE

    body_vars = set(body)
    if body_vars & candidate_neighbors:
        return 1.0

    for b in body_vars:
        for hop in neighbors.get(b, set()):
            if candidate in neighbors.get(hop, set()):
                return 0.5

    return BN_FALLBACK_SCORE


def filter_candidate_extensions_by_bn(
    body: frozenset[str],
    candidates: list[str],
    family_bn_states: dict,
    predicate_family: dict[str, str],
    tau_bn: float,
    top_k: int | None,
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    for cand in candidates:
        score = bn_neighbor_score(body, cand, family_bn_states, predicate_family)
        if score >= tau_bn:
            scored.append((cand, float(score)))

    scored.sort(key=lambda x: (-x[1], x[0]))
    if top_k is not None:
        scored = scored[:top_k]
    return scored

#pipeline: pick_patterns.py -> predicate_construction.py
SELECTED_PATH = str(CURRENT_DIR / "processed" / "ppi" / "ppi_selected.pt")
REPO_PATH = str(CURRENT_DIR / "processed" / "ppi" / "global_predicate_repo" / "global_predicate_repository.json")
TABLE_PATH = str(CURRENT_DIR / "processed" / "ppi" / "global_predicate_repo" / "global_predicate_table_full.csv")
FAMILY_BNS_PATH = str(CURRENT_DIR / "processed" / "ppi" / "global_predicate_repo" / "family_bns")
PATTERN_BNS_PATH = str(CURRENT_DIR / "processed" / "ppi" / "pattern_multi_bn")
OUTPUT_PATH = str(CURRENT_DIR / "processed" / "ppi" / "levelwise_garplus_mining")

SIGMA_PATTERN = 5
SIGMA_RULE = 5
DELTA = 0.8
MAX_PATTERN_EDGES = 3
MAX_BODY_SIZE = 3
TAU_PATTERN_BN = 0.0
TAU_PREDICATE_BN = 0.0
TOP_K_PATTERN_EXTENSIONS = 50
TOP_K_PREDICATE_EXTENSIONS = 50
MIN_PREDICATE_SUPPORT = 2
MAX_PREDICATES = 300
EXCLUDE_FAMILIES = {"qualifications", "edge_label_other"}
PATTERN_SUPPORT_MODE = "edge_subset"  # "edge_subset" | "exact_signature"

PATTERN_BN_FALLBACK_SCORE = 0.1


@dataclass(frozen=True)
class PatternState:
    pattern_id: str
    graph: nx.Graph
    edge_count: int
    node_count: int
    parent_id: str | None = None
    added_edge: tuple[Any, Any] | None = None
    support: int | None = None
    bn_score: float | None = None


def canonical_edge(u: Any, v: Any) -> tuple[str, str]:
    left, right = sorted((str(u), str(v)))
    return left, right


def pattern_signature(g: nx.Graph) -> str:
    """
    First-version pattern signature for de-duplication.

    This is not an isomorphism-invariant hash. It simply canonicalizes the
    observed node/edge ids from sampled patterns. Later this can be replaced by
    WL-hash or exact graph isomorphism.
    """
    nodes = sorted(str(n) for n in g.nodes())
    edges = sorted(f"{u}--{v}" for u, v in (canonical_edge(u, v) for u, v in g.edges()))
    return f"nodes:[{'|'.join(nodes)}];edges:[{'|'.join(edges)}]"


def make_pattern_state(g: nx.Graph, parent_id=None, added_edge=None, bn_score=None) -> PatternState:
    sig = pattern_signature(g)
    return PatternState(
        pattern_id=sig,
        graph=g.copy(),
        edge_count=int(g.number_of_edges()),
        node_count=int(g.number_of_nodes()),
        parent_id=parent_id,
        added_edge=added_edge,
        support=None,
        bn_score=bn_score,
    )


def graph_edges_as_set(g: nx.Graph) -> set[tuple[str, str]]:
    return {canonical_edge(u, v) for u, v in g.edges()}


def build_union_graph(pattern_graphs: list[tuple[int, nx.Graph]]) -> nx.Graph:
    union_graph = nx.Graph()
    for _, graph in pattern_graphs:
        union_graph.add_nodes_from(graph.nodes(data=True))
        union_graph.add_edges_from(graph.edges(data=True))
    return union_graph


def initialize_seed_patterns(pattern_graphs, max_seed_edges=1) -> list[PatternState]:
    """
    Initialize the first vertical level P_1.

    Prefer all sampled patterns with exactly one edge.
    If there are none, use the smallest-edge bucket among sampled patterns.
    De-duplicate by pattern_signature.
    """
    if not pattern_graphs:
        return []

    edge_buckets: dict[int, list[nx.Graph]] = {}
    for _, graph in pattern_graphs:
        edge_buckets.setdefault(int(graph.number_of_edges()), []).append(graph)

    if max_seed_edges in edge_buckets:
        seed_graphs = edge_buckets[max_seed_edges]
    else:
        min_edges = min(edge_buckets.keys())
        seed_graphs = edge_buckets[min_edges]

    seen = set()
    seeds: list[PatternState] = []
    for graph in seed_graphs:
        state = make_pattern_state(graph)
        if state.pattern_id in seen:
            continue
        seen.add(state.pattern_id)
        seeds.append(state)
    return seeds


def compute_pattern_match_ids(
    pattern: nx.Graph,
    pattern_graphs: list[tuple[int, nx.Graph]],
    support_mode: str = PATTERN_SUPPORT_MODE,
) -> list[int]:
    """
    Coarse pattern verification over sampled patterns.

    edge_subset:
        Q is considered supported by a sampled pattern if all edges of Q appear
        in that sampled pattern.

    exact_signature:
        only identical signatures count.
    """
    query_sig = pattern_signature(pattern)
    query_edges = graph_edges_as_set(pattern)
    match_ids = []

    for sampled_id, sampled_graph in pattern_graphs:
        if support_mode == "exact_signature":
            if pattern_signature(sampled_graph) == query_sig:
                match_ids.append(int(sampled_id))
            continue

        sampled_edges = graph_edges_as_set(sampled_graph)
        if query_edges.issubset(sampled_edges):
            match_ids.append(int(sampled_id))

    return match_ids


def compute_pattern_support(
    pattern: nx.Graph,
    pattern_graphs: list[tuple[int, nx.Graph]],
    support_mode: str = PATTERN_SUPPORT_MODE,
) -> int:
    return len(compute_pattern_match_ids(pattern, pattern_graphs, support_mode=support_mode))


def generate_pattern_extensions(
    pattern: nx.Graph,
    union_graph: nx.Graph,
    max_new_edges: int = 1,
) -> list[tuple[nx.Graph, tuple[Any, Any]]]:
    """
    First-version vertical spawning (VSpawn): add one edge.

    Allowed:
    - add one edge between two existing nodes
    - add one edge from an existing node to one new node

    Not allowed:
    - adding one edge whose two endpoints are both new nodes, because that
      would disconnect the new pattern from the current one.
    """
    if max_new_edges != 1:
        raise ValueError("First version only supports add-one-edge expansion.")

    pattern_nodes = set(pattern.nodes())
    pattern_edges = graph_edges_as_set(pattern)
    extensions = []
    seen = set()

    for u, v in union_graph.edges():
        edge_key = canonical_edge(u, v)
        if edge_key in pattern_edges:
            continue
        if u not in pattern_nodes and v not in pattern_nodes:
            continue

        new_graph = pattern.copy()
        if u not in new_graph:
            new_graph.add_node(u)
        if v not in new_graph:
            new_graph.add_node(v)
        new_graph.add_edge(u, v)

        sig = pattern_signature(new_graph)
        if sig in seen:
            continue
        seen.add(sig)
        extensions.append((new_graph, (u, v)))

    return extensions


def load_pattern_bn_state(pattern_bn_dir: str) -> dict:
    """
    Load Pattern-BN outputs from processed/ppi/pattern_multi_bn.

    We reuse the saved node_family_values and group edge files. The loaded state
    is sufficient for:
    - extracting pattern BN variables
    - scoring delta variables by direct / 2-hop connectivity
    """
    root = Path(pattern_bn_dir)
    if not root.exists():
        return {}

    node_family_path = root / "node_family_values.json"
    if not node_family_path.exists():
        return {}

    with open(node_family_path, "r", encoding="utf-8") as f:
        raw_node_family_values = json.load(f)

    node_family_values = {}
    for node, values in raw_node_family_values.items():
        try:
            node_key = int(node)
        except Exception:
            node_key = node
        node_family_values[node_key] = values

    group_states: dict[str, dict] = {}
    global_light_variables: list[str] = []

    for group_dir in sorted(root.iterdir()):
        if not group_dir.is_dir():
            continue
        result_path = group_dir / "result.json"
        edges_path = group_dir / "edges.csv"
        table_path = group_dir / "table.csv"

        status = "skipped"
        if result_path.exists():
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    result = json.load(f)
                status = str(result.get("status", "skipped"))
            except Exception:
                status = "skipped"

        neighbors: dict[str, set[str]] = {}
        nodes: set[str] = set()
        if edges_path.exists():
            try:
                edges_df = pd.read_csv(edges_path)
                for _, row in edges_df.iterrows():
                    src = str(row["source"])
                    dst = str(row["target"])
                    nodes.add(src)
                    nodes.add(dst)
                    neighbors.setdefault(src, set()).add(dst)
                    neighbors.setdefault(dst, set()).add(src)
            except Exception:
                neighbors = {}
                nodes = set()

        if group_dir.name == "global_light" and table_path.exists():
            try:
                global_light_variables = pd.read_csv(table_path).columns.tolist()
            except Exception:
                global_light_variables = []

        group_states[group_dir.name] = {
            "status": status,
            "neighbors": neighbors,
            "nodes": nodes,
        }

    # Reuse extract_multi_bn_vars from the existing pattern BN module.
    pattern_bn_module.MULTI_BN_STATE = {
        "node_family_values": node_family_values,
        "global_light_variables": global_light_variables,
    }

    return {
        "node_family_values": node_family_values,
        "group_states": group_states,
        "global_light_variables": global_light_variables,
    }


def extract_pattern_vars_for_scoring(pattern: nx.Graph, pattern_bn_state: dict) -> dict[str, set[str]]:
    """
    Reuse the multi-BN variable extraction when Pattern-BN state is available.

    If not available, fall back to simple NODE/EDGE variables.
    """
    if pattern_bn_state and pattern_bn_state.get("node_family_values"):
        try:
            return pattern_bn_module.extract_multi_bn_vars(pattern)
        except Exception:
            pass

    node_vars = {f"NODE:{node}" for node in pattern.nodes()}
    edge_vars = {f"EDGE:{u}-{v}" for u, v in sorted(graph_edges_as_set(pattern))}
    return {"fallback": node_vars | edge_vars}


def score_pattern_extension_by_pattern_bn(
    current_pattern: nx.Graph,
    candidate_pattern: nx.Graph,
    pattern_bn_state: dict,
) -> float:
    """
    Score Q -> Q' using Pattern-BN on delta variables only.

    We compare:
        current_vars = vars(Q)
        candidate_vars = vars(Q')
        delta_vars = vars(Q') - vars(Q)

    This avoids inflating the score by re-counting variables that already
    existed in Q.
    """
    if not pattern_bn_state:
        return PATTERN_BN_FALLBACK_SCORE

    current_vars = extract_pattern_vars_for_scoring(current_pattern, pattern_bn_state)
    candidate_vars = extract_pattern_vars_for_scoring(candidate_pattern, pattern_bn_state)
    group_states = pattern_bn_state.get("group_states", {})

    best_score = PATTERN_BN_FALLBACK_SCORE
    has_any_delta = False

    for group_name, cand_group_vars in candidate_vars.items():
        cur_group_vars = current_vars.get(group_name, set())
        delta_vars = set(cand_group_vars) - set(cur_group_vars)
        if not delta_vars:
            continue
        has_any_delta = True

        group_state = group_states.get(group_name)
        if not group_state or group_state.get("status") not in {"ok", "learned"}:
            best_score = max(best_score, PATTERN_BN_FALLBACK_SCORE)
            continue

        neighbors = group_state.get("neighbors", {})
        direct_hit = False
        two_hop_hit = False

        for delta_var in delta_vars:
            delta_neighbors = set(neighbors.get(delta_var, set()))
            for cur_var in cur_group_vars:
                if cur_var in delta_neighbors:
                    direct_hit = True
                    break
                cur_neighbors = set(neighbors.get(cur_var, set()))
                for hop in cur_neighbors:
                    if delta_var in neighbors.get(hop, set()):
                        two_hop_hit = True
                if direct_hit:
                    break
            if direct_hit:
                break

        if direct_hit:
            best_score = max(best_score, 1.0)
        elif two_hop_hit:
            best_score = max(best_score, 0.5)
        else:
            best_score = max(best_score, PATTERN_BN_FALLBACK_SCORE)

    if not has_any_delta:
        return 0.0
    return best_score


def filter_pattern_extensions_by_bn(
    current_pattern: nx.Graph,
    candidate_extensions: list[tuple[nx.Graph, tuple[Any, Any]]],
    pattern_bn_state: dict,
    tau_pattern_bn: float,
    top_k: int | None,
) -> list[tuple[nx.Graph, tuple[Any, Any], float]]:
    scored = []
    for candidate_graph, added_edge in candidate_extensions:
        score = score_pattern_extension_by_pattern_bn(current_pattern, candidate_graph, pattern_bn_state)
        if score >= tau_pattern_bn:
            scored.append((candidate_graph, added_edge, score))

    scored.sort(key=lambda x: (-x[2], pattern_signature(x[0])))
    if top_k is not None:
        scored = scored[:top_k]
    return scored


def predicate_level_search_for_pattern(
    pattern_state: PatternState,
    predicate_table: pd.DataFrame,
    repository: dict,
    family_bn_states: dict,
    sigma_rule: int,
    delta: float,
    max_body_size: int,
    tau_predicate_bn: float,
    top_k_predicate_extensions: int | None,
    min_predicate_support: int,
    max_predicates: int | None,
    exclude_families: set[str] | None,
) -> tuple[list[dict], dict]:
    """
    Horizontal spawning inside one fixed pattern Q.

    First coarse version:
    - we restrict the predicate table to sampled patterns that coarsely match Q
      (edge-subset or exact-signature, depending on support mode)
    - support / confidence are then computed over this restricted match table
    """
    start_time = time.time()
    # The caller already passes the coarse match table for the current pattern.
    # So this horizontal miner can focus only on predicate-level expansion.
    table_q = predicate_table.copy()

    if table_q.empty:
        return [], {
            "pattern_id": pattern_state.pattern_id,
            "num_candidate_predicates": 0,
            "num_rules": 0,
            "elapsed_seconds": float(time.time() - start_time),
            "num_level_candidates": {},
            "num_level_survivors": {},
            "num_match_patterns": 0,
        }

    working_table = table_q.drop(columns=["pattern_id"], errors="ignore").copy()
    predicate_family = get_predicate_family_map(repository)
    candidate_predicates = get_candidate_predicates(
        repository=repository,
        table=table_q,
        exclude_families=exclude_families,
        exclude_sources=None,
        min_support=min_predicate_support,
        max_predicates=max_predicates,
    )

    rules: list[dict] = []
    level = {frozenset([p]) for p in candidate_predicates}
    seen_bodies = set(level)
    num_level_candidates: dict[str, int] = {}
    num_level_survivors: dict[str, int] = {}

    for body_size in range(1, max_body_size + 1):
        if not level:
            break

        current_level = sorted(level, key=lambda b: tuple(sorted(b)))
        num_level_candidates[str(body_size)] = len(current_level)
        next_level = set()

        for body in current_level:
            body_mask = compute_body_mask(working_table, body)
            body_support = int(body_mask.sum())
            if body_support < sigma_rule:
                continue

            for head in candidate_predicates:
                if head in body:
                    continue

                stats = compute_rule_stats(working_table, body, head)
                if stats["support"] >= sigma_rule and stats["confidence"] >= delta:
                    rules.append(
                        {
                            "pattern_id": pattern_state.pattern_id,
                            "pattern_signature": pattern_state.pattern_id,
                            "pattern_edges": " | ".join(f"{u}--{v}" for u, v in sorted(graph_edges_as_set(pattern_state.graph))),
                            "pattern_nodes": " | ".join(sorted(str(n) for n in pattern_state.graph.nodes())),
                            "pattern_support": int(pattern_state.support or 0),
                            "body": " && ".join(sorted(body)),
                            "head": head,
                            "body_size": len(body),
                            "support": stats["support"],
                            "body_support": stats["body_support"],
                            "head_support": stats["head_support"],
                            "confidence": stats["confidence"],
                            "lift": stats["lift"],
                        }
                    )

            if body_size < max_body_size:
                extension_candidates = [p for p in candidate_predicates if p not in body]
                bn_ranked = filter_candidate_extensions_by_bn(
                    body=body,
                    candidates=extension_candidates,
                    family_bn_states=family_bn_states,
                    predicate_family=predicate_family,
                    tau_bn=tau_predicate_bn,
                    top_k=top_k_predicate_extensions,
                )
                for candidate, _ in bn_ranked:
                    new_body = frozenset(set(body) | {candidate})
                    if len(new_body) != body_size + 1:
                        continue
                    if new_body in seen_bodies:
                        continue
                    seen_bodies.add(new_body)
                    next_level.add(new_body)

        num_level_survivors[str(body_size)] = len(next_level)
        level = next_level

    return rules, {
        "pattern_id": pattern_state.pattern_id,
        "num_candidate_predicates": len(candidate_predicates),
        "num_rules": len(rules),
        "elapsed_seconds": float(time.time() - start_time),
        "num_level_candidates": num_level_candidates,
        "num_level_survivors": num_level_survivors,
        "num_match_patterns": int(table_q.shape[0]),
    }


def save_mining_outputs(
    output_dir: str,
    rules_df: pd.DataFrame,
    summary: dict,
    patterns_summary_df: pd.DataFrame,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not rules_df.empty:
        rules_df = rules_df.sort_values(
            ["confidence", "support", "lift", "pattern_support", "body_size", "head", "body"],
            ascending=[False, False, False, False, True, True, True],
        )

    rules_df.to_csv(output_path / "mined_rules.csv", index=False)
    patterns_summary_df.to_csv(output_path / "patterns_summary.csv", index=False)
    with open(output_path / "mining_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def levelwise_garplus_mine(
    pattern_graphs: list[tuple[int, nx.Graph]],
    predicate_table: pd.DataFrame,
    repository: dict,
    pattern_bn_state: dict | None,
    family_bn_states: dict,
    sigma_pattern: int,
    sigma_rule: int,
    delta: float,
    max_pattern_edges: int = 3,
    max_body_size: int = 3,
    tau_pattern_bn: float = 0.0,
    tau_predicate_bn: float = 0.0,
    top_k_pattern_extensions: int | None = 50,
    top_k_predicate_extensions: int | None = 50,
    min_predicate_support: int = 2,
    max_predicates: int | None = 300,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    start_time = time.time()
    union_graph = build_union_graph(pattern_graphs)

    # Vertical level 1 initialization.
    pattern_level = initialize_seed_patterns(pattern_graphs, max_seed_edges=1)
    seen_patterns = {state.pattern_id for state in pattern_level}

    all_rules: list[dict] = []
    all_pattern_states: list[PatternState] = []
    predicate_level_stats: dict[str, Any] = {}
    pattern_level_stats: dict[str, Any] = {}

    num_verified_patterns = 0
    num_generated_patterns = len(pattern_level)

    for edge_level in range(1, max_pattern_edges + 1):
        if not pattern_level:
            print(f"[Pattern-Level {edge_level}] level is empty, stop early.")
            break

        print(f"[Pattern-Level {edge_level}] current_patterns={len(pattern_level)}")
        next_pattern_level: list[PatternState] = []
        verified_this_level = 0
        generated_this_level = 0

        for state in pattern_level:
            match_ids = compute_pattern_match_ids(state.graph, pattern_graphs, support_mode=PATTERN_SUPPORT_MODE)
            verified_state = replace(state, support=len(match_ids))
            all_pattern_states.append(verified_state)

            if (verified_state.support or 0) < sigma_pattern:
                continue

            verified_this_level += 1
            num_verified_patterns += 1

            # Coarse horizontal predicate mining over the sampled patterns that
            # contain the current pattern under edge-subset semantics.
            if "pattern_id" in predicate_table.columns:
                predicate_table_q = predicate_table[predicate_table["pattern_id"].isin(match_ids)].copy()
            else:
                predicate_table_q = predicate_table.copy()

            rules_q, pred_summary_q = predicate_level_search_for_pattern(
                pattern_state=verified_state,
                predicate_table=predicate_table_q,
                repository=repository,
                family_bn_states=family_bn_states,
                sigma_rule=sigma_rule,
                delta=delta,
                max_body_size=max_body_size,
                tau_predicate_bn=tau_predicate_bn,
                top_k_predicate_extensions=top_k_predicate_extensions,
                min_predicate_support=min_predicate_support,
                max_predicates=max_predicates,
                exclude_families=EXCLUDE_FAMILIES,
            )
            predicate_level_stats[verified_state.pattern_id] = pred_summary_q
            all_rules.extend(rules_q)

            # Vertical spawning: add one edge, then let Pattern-BN soft-rank
            # which structural extensions should enter the next pattern level.
            if verified_state.edge_count < max_pattern_edges:
                candidates = generate_pattern_extensions(verified_state.graph, union_graph)
                scored_candidates = filter_pattern_extensions_by_bn(
                    current_pattern=verified_state.graph,
                    candidate_extensions=candidates,
                    pattern_bn_state=pattern_bn_state or {},
                    tau_pattern_bn=tau_pattern_bn,
                    top_k=top_k_pattern_extensions,
                )

                for cand_graph, added_edge, score in scored_candidates:
                    new_state = make_pattern_state(
                        cand_graph,
                        parent_id=verified_state.pattern_id,
                        added_edge=added_edge,
                        bn_score=score,
                    )
                    if new_state.pattern_id in seen_patterns:
                        continue
                    seen_patterns.add(new_state.pattern_id)
                    next_pattern_level.append(new_state)
                    generated_this_level += 1

        num_generated_patterns += generated_this_level
        pattern_level_stats[str(edge_level)] = {
            "current_patterns": len(pattern_level),
            "verified_patterns": verified_this_level,
            "generated_next_patterns": generated_this_level,
        }

        print(
            f"[Pattern-Level {edge_level}] verified={verified_this_level} "
            f"next_patterns={generated_this_level} cumulative_rules={len(all_rules)}"
        )
        pattern_level = next_pattern_level

    rules_df = pd.DataFrame(all_rules)
    if not rules_df.empty:
        rules_df = rules_df.sort_values(
            ["confidence", "support", "lift", "pattern_support", "body_size", "head", "body"],
            ascending=[False, False, False, False, True, True, True],
        ).reset_index(drop=True)

    patterns_summary_rows = []
    for state in all_pattern_states:
        patterns_summary_rows.append(
            {
                "pattern_id": state.pattern_id,
                "parent_id": state.parent_id,
                "edge_count": state.edge_count,
                "node_count": state.node_count,
                "support": state.support,
                "bn_score": state.bn_score,
                "edges": " | ".join(f"{u}--{v}" for u, v in sorted(graph_edges_as_set(state.graph))),
                "signature": state.pattern_id,
            }
        )
    patterns_summary_df = pd.DataFrame(patterns_summary_rows)

    summary = {
        "num_seed_patterns": len(initialize_seed_patterns(pattern_graphs, max_seed_edges=1)),
        "num_verified_patterns": num_verified_patterns,
        "num_generated_patterns": num_generated_patterns,
        "num_rules": int(len(rules_df)),
        "elapsed_seconds": float(time.time() - start_time),
        "sigma_pattern": int(sigma_pattern),
        "sigma_rule": int(sigma_rule),
        "delta": float(delta),
        "max_pattern_edges": int(max_pattern_edges),
        "max_body_size": int(max_body_size),
        "tau_pattern_bn": float(tau_pattern_bn),
        "tau_predicate_bn": float(tau_predicate_bn),
        "top_k_pattern_extensions": None if top_k_pattern_extensions is None else int(top_k_pattern_extensions),
        "top_k_predicate_extensions": None if top_k_predicate_extensions is None else int(top_k_predicate_extensions),
        "pattern_level_stats": pattern_level_stats,
        "predicate_level_stats": predicate_level_stats,
        "notes": [
            "This is a coarse first version.",
            "Pattern support is computed over sampled patterns, not exact subgraph isomorphism over G.",
            "Predicate support/confidence is computed over global_predicate_table_full.csv.",
            "Pattern-BN and Predicate-BN are only used for soft pruning/ranking.",
            "Final exact GAR+ validation over the original graph G is left for the next stage.",
        ],
    }
    return rules_df, summary, patterns_summary_df


def main():
    start_time = time.time()

    pattern_graphs = pattern_bn_module.load_selected_pattern_graphs(SELECTED_PATH)
    repository = load_predicate_repository(REPO_PATH)
    predicate_table = load_predicate_table(TABLE_PATH)
    family_bn_states = load_family_bn_edges(FAMILY_BNS_PATH)
    pattern_bn_state = load_pattern_bn_state(PATTERN_BNS_PATH)

    print(f"[Info] loaded_patterns={len(pattern_graphs)}")
    print(f"[Info] loaded_predicates={len(repository.get('predicates', []))}")
    print(f"[Info] predicate_table patterns={predicate_table.shape[0]} predicates={len([c for c in predicate_table.columns if c != 'pattern_id'])}")
    print(f"[Info] loaded_family_bns={sum(1 for s in family_bn_states.values() if s.get('status') == 'learned')}")
    print(f"[Info] pattern_bn_available={bool(pattern_bn_state)}")

    rules_df, summary, patterns_summary_df = levelwise_garplus_mine(
        pattern_graphs=pattern_graphs,
        predicate_table=predicate_table,
        repository=repository,
        pattern_bn_state=pattern_bn_state,
        family_bn_states=family_bn_states,
        sigma_pattern=SIGMA_PATTERN,
        sigma_rule=SIGMA_RULE,
        delta=DELTA,
        max_pattern_edges=MAX_PATTERN_EDGES,
        max_body_size=MAX_BODY_SIZE,
        tau_pattern_bn=TAU_PATTERN_BN,
        tau_predicate_bn=TAU_PREDICATE_BN,
        top_k_pattern_extensions=TOP_K_PATTERN_EXTENSIONS,
        top_k_predicate_extensions=TOP_K_PREDICATE_EXTENSIONS,
        min_predicate_support=MIN_PREDICATE_SUPPORT,
        max_predicates=MAX_PREDICATES,
    )

    save_mining_outputs(
        output_dir=OUTPUT_PATH,
        rules_df=rules_df,
        summary=summary,
        patterns_summary_df=patterns_summary_df,
    )

    elapsed = time.time() - start_time
    print(f"[Done] patterns={summary['num_verified_patterns']}, rules={summary['num_rules']}, elapsed={elapsed:.2f}s")


if __name__ == "__main__":
    main()
