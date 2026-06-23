from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from pprint import pprint
from typing import Callable, List, Optional

from graph_types import DataGraph, FrequentPattern, GraphInstance, PatternOptions
from garplus_ml_predicates import MLPredicateConfig, inject_ml_predicates
from predicate_enrichment import PredicateEnrichmentConfig, enrich_numeric_bin_predicates
from pattern_bn import PatternBayesianNetwork, PatternBNConfig
from pattern_extension import GraphSpawn, topology_pattern_code
from vf3_linux import find_matches_with_limit
from vf3_like import find_matches_with_limit as find_matches_with_limit_like
from predicate_bn import PredicateBayesianNetwork, PredicateBNConfig
from predicate_selection import DecisionTreePredicateSelector, FPGrowthPredicateSelector, Rule
from rulegeneration import (
    FreqCount,
    PatternView,
    RuleGenerationStatus,
    RuleSender,
    RuleStatistics,
    SegmentRuleSet,
    ZLRule,
    send_zl_rules,
    update_status_after_generate_rules,
    zl_rule_filter,
)
from sampled_frequent_patterns import (
    build_directed_frequent_patterns,
    edge_priors_from_frequent_patterns,
    mine_sampled_frequent_patterns,
)


GraphLoader = Callable[..., object]
SeedBuilder = Callable[[object], FrequentPattern]


def build_undirected_rematch_graph(graph: DataGraph) -> tuple[DataGraph, dict[object, object]]:
    """Build a temporary bidirectional graph for undirected-pattern rematching."""

    rematch_graph = deepcopy(graph)
    reverse_edge_id_map: dict[object, object] = {}
    existing = {(edge.src, edge.dst, edge.label) for edge in rematch_graph.all_edges()}
    for edge in list(graph.all_edges()):
        reverse_key = (edge.dst, edge.src, edge.label)
        if reverse_key in existing:
            continue
        synthetic_edge_id = ("_undirected_reverse", edge.edge_id)
        reverse_edge_id_map[synthetic_edge_id] = edge.edge_id
        reverse_attrs = dict(edge.attrs)
        reverse_attrs["_undirected_reverse"] = True
        reverse_attrs["_original_edge_id"] = edge.edge_id
        reverse_edge = replace(
            edge,
            edge_id=synthetic_edge_id,
            src=edge.dst,
            dst=edge.src,
            attrs=reverse_attrs,
        )
        rematch_graph.out_edges.setdefault(reverse_edge.src, []).append(reverse_edge)
        rematch_graph.in_edges.setdefault(reverse_edge.dst, []).append(reverse_edge)
        rematch_graph.edges_by_id[synthetic_edge_id] = reverse_edge
        existing.add(reverse_key)
    return rematch_graph, reverse_edge_id_map


def normalize_undirected_rematch_instances(
    instances: List[GraphInstance],
    original_graph: DataGraph,
    reverse_edge_id_map: dict[object, object],
) -> List[GraphInstance]:
    """Map synthetic reverse edges back to their original edge ids."""

    normalized: List[GraphInstance] = []
    for instance in instances:
        edge_bindings = {
            pattern_edge_index: reverse_edge_id_map.get(edge_id, edge_id)
            for pattern_edge_index, edge_id in instance.edge_bindings.items()
        }
        edge_triplets = []
        for edge_id in edge_bindings.values():
            edge = original_graph.edges_by_id.get(edge_id)
            if edge is not None:
                edge_triplets.append((edge.src, edge.dst, edge.label))
        normalized.append(
            GraphInstance(
                node_map=dict(instance.node_map),
                edge_ids=tuple(sorted(edge_triplets)),
                pivot=instance.pivot,
                edge_bindings=edge_bindings,
            )
        )
    return normalized


def dedupe_rematch_instances(instances: List[GraphInstance]) -> List[GraphInstance]:
    """Remove duplicate embeddings introduced by temporary reverse edges."""

    seen = set()
    deduped: List[GraphInstance] = []
    for instance in instances:
        key = (
            tuple(sorted(instance.node_map.items())),
            tuple(sorted(instance.edge_bindings.items(), key=lambda item: (item[0], str(item[1])))),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(instance)
    return deduped


@dataclass
class GarplusRunConfig:
    dataset_name: str
    interaction_csv_path: str
    sampled_pt_path: Optional[str]
    sampled_graph_loader: Optional[GraphLoader]
    seed_builder: SeedBuilder
    node_csv_path: Optional[str] = None
    node_csv_label: str = "node_csv"
    csv_graph_loader: Optional[GraphLoader] = None
    auto_discover_if_missing: bool = False
    fallback_interaction_name: str = "edges.csv"
    fallback_node_name: str = "node.csv"
    use_sampled_pt_graph: bool = True
    force_edge_label: Optional[str] = None
    edge_label_column: str = "Experimental System"
    augment_negative_edges: bool = True
    negative_edge_limit: int = 2000
    balance_edge_labels: bool = True
    mode: str = "decision-tree"  # fp-growth
    y_key: str = "e0.interaction_label"
    target_y_keys: Optional[List[str]] = None
    include_ml_predicate_targets: bool = True
    pattern_extension_only: bool = False
    pattern_extension_debug: bool = False
    pattern_extension_debug_limit: int = 500
    global_rematch_patterns: bool = True
    global_vspawn_instances: bool = False
    decision_tree_max_depth: int = 3
    max_rows: int = 50
    undirected: bool = True
    undirected_pattern: bool = True
    topology_only_pattern_dedup: bool = False
    topology_dedupe_respect_direction: bool = False
    full_solution: bool = False
    pattern_support: int = 5
    min_support: int = 50
    min_confidence: float = 0.6
    min_value_support_count: int = 20
    max_radius: int = 4
    max_add_edge: int = 4
    node_max_add_edge: int = 4
    max_multi_support: Optional[int] = 10000
    pattern_dedup_prefer_target_value: Optional[str] = None
    print_rule_limit: int = 10
    print_deduped_rule_limit: int = 50
    deduped_rules_output_path: Optional[str] = None
    print_full_payload: bool = False
    print_payload_snapshot: bool = False
    print_instances: bool = False
    print_instance_limit: int = 5
    print_bn_stats: bool = True
    enable_sampled_frequent_patterns: bool = True
    sampled_frequent_min_graph_support: int = 5
    sampled_frequent_print_limit: int = 20
    inject_sampled_frequent_patterns: bool = True
    sampled_frequent_pattern_limit: int = 8
    include_sampled_frequent_edge_pattern: bool = True
    materialize_sampled_frequent_patterns: bool = True
    drop_unknown_target_rows: bool = True
    ignored_target_values: tuple[str, ...] = ("unknown",)
    drop_ignored_target_edges: bool = False
    filter_degree_predicates: bool = False
    ignored_predicate_key_tokens: tuple[str, ...] = ("degree", "high_degree")
    enable_pattern_bn: bool = True
    tau_p: float = 0.5
    pattern_bn_top_k_per_spawn_node: Optional[int] = None
    pattern_bn_min_keep_per_spawn_node: int = 1
    pattern_bn_frequent_prior_weight: float = 0.25
    pattern_bn_cache_path: Optional[str] = None
    retrain_pattern_bn: bool = True
    enable_predicate_bn: bool = True
    tau_x: float = 0.5
    predicate_bn_top_k_features: Optional[int] = None
    predicate_bn_min_keep_features: int = 8
    predicate_bn_max_parent_features: int = 12
    predicate_bn_max_feature_cardinality: int = 50
    predicate_bn_focus_target: str = "e0.interaction_label=negative"
    predicate_bn_focus_targets: Optional[tuple[str, ...]] = None
    predicate_bn_feature_score: str = "bic"
    predicate_bn_estimator: str = "maximum_likelihood"
    predicate_bn_cache_path: Optional[str] = None
    retrain_predicate_bn: bool = True
    ml_predicates: MLPredicateConfig = MLPredicateConfig()
    predicate_enrichment: PredicateEnrichmentConfig = PredicateEnrichmentConfig()


def resolve_path(raw_path: Optional[str], fallback_name: str, auto_discover: bool) -> str:
    if raw_path:
        return raw_path
    if not auto_discover:
        raise FileNotFoundError(f"{fallback_name} is empty and auto discovery is disabled")
    search_root = Path(__file__).resolve().parents[2]
    matches = list(search_root.rglob(fallback_name))
    if not matches:
        raise FileNotFoundError(f"Could not auto-discover {fallback_name}")
    matches.sort(key=lambda candidate: len(str(candidate)))
    return str(matches[0])


def normalize_rule_literal_for_dedupe(item: str) -> str:
    key, value = item.split("=", 1) if "=" in item else (item, "")
    if "." in key:
        entity, attr = key.split(".", 1)
        if entity.startswith("v") and entity[1:].isdigit():
            key = f"v*.{attr}"
    return f"{key}={value}" if value else key

def parse_rule_literal(item: str) -> tuple[str, str, str]:
    if "!=" in item:
        key, value = item.split("!=", 1)
        return key, "!=", value
    if "=" in item:
        key, value = item.split("=", 1)
        return key, "=", value
    return item, "", ""

def rule_semantic_key(rule: Rule):
    normalized_antecedent = tuple(sorted(normalize_rule_literal_for_dedupe(item) for item in rule.antecedent))
    normalized_consequent = normalize_rule_literal_for_dedupe(rule.consequent)
    return normalized_antecedent, normalized_consequent


def dedupe_rules_semantically(pattern_rules):
    best = {}
    for pattern_id, rule in pattern_rules:
        key = rule_semantic_key(rule)
        current = best.get(key)
        candidate_rank = (float(rule.confidence), float(rule.support), float(rule.lift))
        if current is None or candidate_rank > current[0]:
            best[key] = (candidate_rank, pattern_id, rule)
    rows = [(pattern_id, rule, key) for key, (_rank, pattern_id, rule) in best.items()]
    rows.sort(key=lambda item: (float(item[1].confidence), float(item[1].support), float(item[1].lift)), reverse=True)
    return rows


def rule_is_entailed(rule_key, other_rule_keys) -> bool:
    antecedent_key, consequent_key = rule_key
    closure = set(antecedent_key)
    if consequent_key in closure:
        return True
    changed = True
    while changed:
        changed = False
        for other_antecedent, other_consequent in other_rule_keys:
            if set(other_antecedent).issubset(closure) and other_consequent not in closure:
                closure.add(other_consequent)
                changed = True
                if consequent_key in closure:
                    return True
    return False


def compute_rule_cover(deduped_rows):
    """Return an irredundant rule subset equivalent under forward implication."""

    cover = list(deduped_rows)
    weakest_first = sorted(
        deduped_rows,
        key=lambda item: (float(item[1].confidence), float(item[1].support), float(item[1].lift)),
    )
    for candidate in weakest_first:
        if candidate not in cover:
            continue
        candidate_key = candidate[2]
        other_keys = [row[2] for row in cover if row is not candidate]
        if rule_is_entailed(candidate_key, other_keys):
            cover.remove(candidate)
    cover.sort(key=lambda item: (float(item[1].confidence), float(item[1].support), float(item[1].lift)), reverse=True)
    return cover


def format_deduped_rule_row(pattern_id, rule, key) -> str:
    antecedent_key, consequent_key = key
    return (
        "  cover_rule "
        f"pattern_id={pattern_id} antecedent={antecedent_key} consequent={consequent_key} "
        f"raw_antecedent={rule.antecedent} raw_consequent={rule.consequent} "
        f"support={int(rule.support)} confidence={rule.confidence:.3f} lift={rule.lift:.3f}"
    )


def _print_deduped_rule_row(pattern_id, rule, key) -> None:
    print(format_deduped_rule_row(pattern_id, rule, key))


def write_deduped_rules(path: str, deduped_rows) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for pattern_id, rule, key in deduped_rows:
            handle.write(format_deduped_rule_row(pattern_id, rule, key).strip() + "\n")


def print_deduped_rules(pattern_rules, limit: int, output_path: Optional[str] = None):
    deduped = dedupe_rules_semantically(pattern_rules)
    cover = compute_rule_cover(deduped)
    grouped = {"positive": [], "negative": [], "other": []}
    for row in cover:
        _pattern_id, rule, _key = row
        value = _rule_consequent_value(rule)
        if value == "positive":
            grouped["positive"].append(row)
        elif value == "negative":
            grouped["negative"].append(row)
        else:
            grouped["other"].append(row)

    print(
        f"[RuleCover] raw={len(pattern_rules)} deduped={len(deduped)} cover={len(cover)} "
        f"redundant_removed={len(deduped) - len(cover)} limit_per_group={limit} "
        f"positive={len(grouped['positive'])} negative={len(grouped['negative'])} other={len(grouped['other'])}"
    )
    for group_name in ("positive", "negative", "other"):
        rows = grouped[group_name]
        print(f"[RuleCover/{group_name}] count={len(rows)} shown={min(len(rows), limit)}")
        for pattern_id, rule, key in rows[:limit]:
            _print_deduped_rule_row(pattern_id, rule, key)
    if output_path:
        write_deduped_rules(output_path, cover)
        print(f"[RuleCover] wrote={len(cover)} path={output_path}")
    return cover


def rule_consequent_distribution(rules) -> dict:
    counts = {}
    for rule in rules:
        value = rule.consequent.split("=", 1)[1] if "=" in rule.consequent else rule.consequent
        counts[value] = counts.get(value, 0) + 1
    return counts


def _rule_consequent_value(rule: Rule) -> str:
    return rule.consequent.split("=", 1)[1] if "=" in rule.consequent else rule.consequent


def discovered_rule_table_stats(deduped_rows, pattern_size_by_id: dict) -> dict:
    rules = [rule for _pattern_id, rule, _key in deduped_rows]
    total = len(rules)
    positive = sum(1 for rule in rules if _rule_consequent_value(rule) == "positive")
    negative = sum(1 for rule in rules if _rule_consequent_value(rule) == "negative")
    avg_conf = sum(float(rule.confidence) for rule in rules) / total if total else 0.0
    avg_pattern_size = (
        sum(float(pattern_size_by_id.get(pattern_id, 0)) for pattern_id, _rule, _key in deduped_rows) / total
        if total
        else 0.0
    )
    return {
        "positive": positive,
        "negative": negative,
        "negative_ratio": negative / total if total else 0.0,
        "avg_pattern_size": avg_pattern_size,
        "avg_confidence": avg_conf,
        "total": total,
    }


def print_discovered_rule_stats_table(dataset_name: str, stats: dict) -> None:
    print("[DiscoveredRuleStats]")
    print("Dataset | |Sigma_p| | |Sigma_n| | Neg. Ratio | Avg. |Q| | Avg. Conf. | #Total")
    print(
        f"{dataset_name} | {stats['positive']} | {stats['negative']} | "
        f"{stats['negative_ratio']:.4f} | {stats['avg_pattern_size']:.3f} | "
        f"{stats['avg_confidence']:.3f} | {stats['total']}"
    )


def trim_instances(payload: dict, cfg: GarplusRunConfig) -> dict:
    if cfg.print_instances:
        for key in ("x_instance", "y_instance"):
            if key in payload:
                payload[key] = payload[key][: cfg.print_instance_limit]
        return payload
    for key in ("x_instance", "y_instance"):
        if key in payload:
            payload[key] = f"<{len(payload[key])} instances hidden; set print_instances=True to show>"
    return payload


def print_bn_summary(pattern_bn, predicate_bn, cfg: GarplusRunConfig) -> None:
    if not cfg.print_bn_stats:
        return
    if pattern_bn is not None:
        summary = pattern_bn.pruning_summary()
        print(
            "[PatternBN] "
            f"backend={summary.get('backend', 'unknown')} "
            f"rank_calls={summary['rank_calls']} seen={summary['candidates_seen']} "
            f"kept={summary['candidates_kept']} pruned={summary['candidates_pruned']} "
            f"tau_p={summary.get('tau_p')} threshold_pruned={summary.get('threshold_pruned')} "
            f"topk_pruned={summary.get('topk_pruned')} min_keep_rescued={summary.get('min_keep_rescued')} "
            f"freq_priors={summary.get('frequent_edge_prior_count')} prior_weight={summary.get('frequent_prior_weight')}"
        )
        for score, desc in summary["top_snapshot"]:
            print(f"  pattern_bn_top score={score:.6f} candidate={desc}")
    if predicate_bn is not None:
        summary = predicate_bn.pruning_summary()
        print(
            "[PredicateBN] "
            f"backend={summary.get('backend', 'unknown')} trained={summary.get('trained')} "
            f"rows={summary['rows']} train_features={summary.get('training_feature_count')} "
            f"feature_calls={summary['feature_rank_calls']} "
            f"seen={summary['features_seen']} kept={summary['features_kept']} "
            f"pruned={summary['features_pruned']} rules_ranked={summary['rules_ranked']}"
        )
        print(
            f"  tau_x={summary.get('tau_x')} tau_pruned={summary.get('tau_pruned')} "
            f"feature_limit_pruned={summary.get('feature_limit_pruned')} "
            f"topk_pruned={summary.get('topk_pruned')} "
            f"min_keep_rescued={summary.get('min_keep_rescued')}"
        )
        print(f"  target_values={summary['target_values']}")
        print(f"  focus_target_item={summary.get('focus_target_item')}")
        print(
            f"  feature_score={summary.get('feature_score')} "
            f"target_cardinality={summary.get('target_cardinality')} "
            f"estimated_target_cpd_cells={summary.get('estimated_target_cpd_cells')} "
            f"max_cpd_cells={summary.get('max_cpd_cells')}"
        )
        print(
            f"  sparse_candidates={summary.get('candidate_features_after_sparse_filter')} "
            f"sparse_skipped={summary.get('sparse_features_skipped')} "
            f"skip_reasons={summary.get('sparse_skip_reasons')}"
        )
        print(f"  training_features={summary.get('training_features')}")
        if summary.get("last_scored_features"):
            print(f"  scored_features_top={summary.get('last_scored_features')}")
        if summary.get("last_unranked_features"):
            print(f"  unranked_by_bn_feature_limit_sample={summary.get('last_unranked_features')}")
        if summary.get("sparse_skip_examples"):
            print("  sparse_skip_examples:")
            for reason, examples in summary.get("sparse_skip_examples", {}).items():
                print(f"    {reason}: {examples}")
        if summary.get("skipped_cpd_budget"):
            print(f"  skipped_cpd_budget={summary.get('skipped_cpd_budget')}")
        for score, key in summary["top_features"]:
            print(f"  predicate_bn_top score={score:.6f} feature={key}")


def print_interaction_label_distribution(graph) -> None:
    counts = {}
    for edge in graph.all_edges():
        label = edge.attrs.get("interaction_label", "<missing>")
        counts[label] = counts.get(label, 0) + 1
    if counts:
        print(f"[Graph] interaction_label_distribution={counts}")



def build_target_y_list(graph, cfg: GarplusRunConfig) -> List[str]:
    """Build Go-style yList: each item is mined as one independent RHS."""

    candidates = list(cfg.target_y_keys) if cfg.target_y_keys else [cfg.y_key]
    if cfg.include_ml_predicate_targets:
        edge_keys = {key for edge in graph.all_edges() for key in edge.attrs.keys()}
        for key in ("ml_equivalence_pred", "ml_similarity_pred", "ml_pred_ppi", "ml_pred_not_ppi", "ml_offline_predicate_name"):
            if key in edge_keys:
                candidates.append(f"e0.{key}")
    seen = set()
    y_list: List[str] = []
    for key in candidates:
        if not key or key in seen:
            continue
        seen.add(key)
        y_list.append(key)
    return y_list


def predicate_focus_for_y_key(cfg: GarplusRunConfig, y_key: str) -> Optional[str]:
    if y_key == cfg.y_key:
        return cfg.predicate_bn_focus_target
    if (
        y_key.endswith("ml_equivalence_pred")
        or y_key.endswith("ml_similarity_pred")
        or y_key.endswith("ml_pred_ppi")
        or y_key.endswith("ml_pred_not_ppi")
    ):
        return f"{y_key}=yes"
    if y_key.endswith("ml_offline_predicate_name"):
        return f"{y_key}=ml_pred_ppi"
    return None

def predicate_focus_items_for_y_key(cfg: GarplusRunConfig, y_key: str) -> List[Optional[str]]:
    if y_key == cfg.y_key and cfg.predicate_bn_focus_targets:
        items: List[Optional[str]] = []
        for value in cfg.predicate_bn_focus_targets:
            text = str(value).strip()
            if not text:
                continue
            items.append(text if "=" in text else f"{y_key}={text}")
        return items or [predicate_focus_for_y_key(cfg, y_key)]
    return [predicate_focus_for_y_key(cfg, y_key)]


def focus_cache_suffix(focus_item: Optional[str]) -> str:
    if not focus_item:
        return "none"
    return "".join(ch if ch.isalnum() else "_" for ch in focus_item).strip("_") or "none"


def predicate_bn_cache_for_y_key(cfg: GarplusRunConfig, y_key: str, focus_item: Optional[str] = None) -> Optional[str]:
    if not cfg.predicate_bn_cache_path:
        return None
    path = Path(cfg.predicate_bn_cache_path)
    safe_key = "".join(ch if ch.isalnum() else "_" for ch in y_key).strip("_")
    if focus_item is not None:
        safe_key = f"{safe_key}_{focus_cache_suffix(focus_item)}"
    return str(path.with_name(f"{path.stem}_{safe_key}{path.suffix}"))


def drop_edges_by_target_values(
    graph: DataGraph,
    ignored_values: tuple[str, ...],
    label_column: str = "interaction_label",
) -> dict[str, object]:
    ignored = {str(value).strip().lower() for value in ignored_values if str(value).strip()}
    if not ignored:
        return {"enabled": False, "removed": 0, "values": []}
    kept_edges = []
    removed_counts = {}
    for edge in graph.all_edges():
        value = str(edge.attrs.get(label_column, "")).strip().lower()
        if value in ignored:
            removed_counts[value] = removed_counts.get(value, 0) + 1
        else:
            kept_edges.append(edge)
    if not removed_counts:
        return {"enabled": True, "removed": 0, "values": sorted(ignored), "removed_counts": {}}

    graph.out_edges = {}
    graph.in_edges = {}
    graph.edges_by_id = {}
    for edge in kept_edges:
        graph.out_edges.setdefault(edge.src, []).append(edge)
        graph.in_edges.setdefault(edge.dst, []).append(edge)
        graph.edges_by_id[edge.edge_id] = edge
    return {
        "enabled": True,
        "removed": sum(removed_counts.values()),
        "values": sorted(ignored),
        "removed_counts": dict(sorted(removed_counts.items())),
        "remaining_edges": len(graph.edges_by_id),
    }


def print_rule_consequent_distribution(rules: List[Rule]) -> None:
    print(f"[PredicateSelection] consequent_distribution={rule_consequent_distribution(rules)}")


def print_target_rule_diagnostics(selector, target_value: str, limit: int = 20) -> None:
    method_name = f"{target_value}_diagnostics"
    diagnostics = getattr(selector, method_name)(limit=limit) if hasattr(selector, method_name) else []
    if not diagnostics:
        print(f"[PredicateSelection] {target_value}_candidate_diagnostics=[]")
        return
    print(f"[PredicateSelection] {target_value}_candidate_diagnostics top={len(diagnostics)}")
    for item in diagnostics:
        print(
            f"  {target_value}_candidate "
            f"antecedent={item['antecedent']} consequent={item['consequent']} "
            f"support={item['support']} confidence={item['confidence']:.4f} "
            f"lift={item['lift']:.4f} reason={item['reason']}"
        )


def print_rule_candidate_diagnostics(selector, limit: int = 20) -> None:
    print_target_rule_diagnostics(selector, "negative", limit=limit)
    print_target_rule_diagnostics(selector, "positive", limit=limit)


def print_target_row_diagnostics(selector, focus_item: Optional[str] = None) -> None:
    summary = getattr(selector, "target_stage_summary", None) or {}
    if not summary:
        return
    print(
        "[TargetRows] "
        f"y_key={summary.get('y_key')} raw_rows={summary.get('raw_rows')} "
        f"raw_counts={summary.get('raw_counts')} "
        f"after_value_rows={summary.get('after_value_rows')} "
        f"after_value_counts={summary.get('after_value_counts')} "
        f"missing_target_after_value_pruning={summary.get('missing_target_after_value_pruning')} "
        f"ignored_counts={summary.get('ignored_counts')} "
        f"after_ignored_rows={summary.get('after_ignored_rows')} "
        f"after_ignored_counts={summary.get('after_ignored_counts')}"
    )
    if not focus_item or "=" not in focus_item:
        return
    _focus_key, focus_value = focus_item.split("=", 1)
    raw_count = int(summary.get("raw_counts", {}).get(focus_value, 0))
    after_value_count = int(summary.get("after_value_counts", {}).get(focus_value, 0))
    after_ignored_count = int(summary.get("after_ignored_counts", {}).get(focus_value, 0))
    if raw_count == 0:
        reason = "no_matched_instances_with_target_value"
    elif after_value_count == 0:
        reason = "target_value_removed_by_min_value_support"
    elif after_ignored_count == 0:
        reason = "target_value_removed_by_ignored_target_values"
    else:
        reason = "target_value_available_for_itemset_mining"
    print(
        f"[TargetFocusAvailability] focus={focus_item} raw={raw_count} "
        f"after_value={after_value_count} after_ignored={after_ignored_count} reason={reason}"
    )


def print_negative_rule_diagnostics(selector, limit: int = 20) -> None:
    print_target_rule_diagnostics(selector, "negative", limit=limit)

def _rule_matches_row(rule: Rule, row: dict) -> bool:
    for antecedent in rule.antecedent:
        key, op, value = parse_rule_literal(antecedent)
        row_value = str(row.get(key))
        if op == "=" and row_value != value:
            return False
        if op == "!=" and row_value == value:
            return False
    return True

def _rule_matches_row(rule: Rule, row: dict) -> bool:
    for antecedent in rule.antecedent:
        key, value = antecedent.split("=", 1)
        if str(row.get(key)) != value:
            return False
    return True


def _instance_edge_identity(instance, pattern_edge_index: int = 0):
    edge_id = instance.get_edge_id(pattern_edge_index)
    if edge_id is not None:
        return ("edge_id", edge_id)
    if instance.edge_ids:
        return ("edge_tuple", instance.edge_ids[pattern_edge_index if pattern_edge_index < len(instance.edge_ids) else 0])
    return ("instance", tuple(sorted(instance.node_map.items())))


def evaluate_negative_rule_recall(selector, graph, frequent_pattern: FrequentPattern, rules: List[Rule], y_key: str, negative_value: str = "negative") -> dict:
    rows = selector.prune_rows_by_value_support(selector.build_instance_rows(graph, frequent_pattern))
    rows = [row for row in rows if y_key in row]
    rows = selector.filter_target_rows(rows, y_key)
    negative_rules = [rule for rule in rules if rule.consequent == f"{y_key}={negative_value}"]
    negative_edges = set()
    covered_edges = set()
    covered_instances = 0
    negative_instances = 0
    for index, row in enumerate(rows):
        if str(row.get(y_key)) != negative_value:
            continue
        negative_instances += 1
        if index >= len(frequent_pattern.instances):
            edge_identity = ("row", index)
        else:
            edge_identity = _instance_edge_identity(frequent_pattern.instances[index])
        negative_edges.add(edge_identity)
        if any(_rule_matches_row(rule, row) for rule in negative_rules):
            covered_instances += 1
            covered_edges.add(edge_identity)
    denominator = len(negative_edges)
    covered = len(covered_edges)
    return {
        "negative_rules": len(negative_rules),
        "negative_instances": negative_instances,
        "covered_negative_instances": covered_instances,
        "total_negative_edges": denominator,
        "covered_negative_edges": covered,
        "edge_recall": covered / denominator if denominator else 0.0,
        "instance_recall": covered_instances / negative_instances if negative_instances else 0.0,
    }



def predicate_rule_to_zl(rule: Rule, frequent_pattern: FrequentPattern) -> ZLRule:
    general_keys = []
    values = []
    semantics = []
    for antecedent in rule.antecedent:
        key, op, value = parse_rule_literal(antecedent)
        general_keys.append(key)
        values.append([value])
        semantics.append("not" if op == "!=" else "or")
    y_key, _ = rule.consequent.split("=", 1)
    xy_support = max(0, int(rule.support))
    xy_support_single = min(frequent_pattern.single_support(), xy_support)
    xy_support_multiple = min(frequent_pattern.multi_support(), xy_support)
    segment = SegmentRuleSet(
        keys=[y_key],
        intervals=[(float("-inf"), float("inf"))],
        is_nans=[False],
        statistics=RuleStatistics(
            freq_antecedent=FreqCount(frequent_pattern.single_support(), frequent_pattern.multi_support()),
            freq_union=FreqCount(xy_support_single, xy_support_multiple),
            confidence=rule.confidence,
            lift=rule.lift,
            status=1,
        ),
    )
    x_instance = [
        {
            "v": [{"id": node_idx, "data_id": data_id} for node_idx, data_id in instance.node_map.items()],
            "e": list(instance.edge_ids),
        }
        for instance in frequent_pattern.instances
    ]
    return ZLRule(
        segment_rules=segment,
        general_keys=general_keys,
        values=values,
        semantics=semantics,
        is_nans=[False] * len(general_keys),
        instances=(x_instance, x_instance),
        y_literal=rule.consequent,
    )


def load_graph(cfg: GarplusRunConfig, interaction_csv_path: str, node_csv_path: Optional[str]):
    if cfg.use_sampled_pt_graph:
        if not cfg.sampled_pt_path:
            raise ValueError("use_sampled_pt_graph=True but sampled_pt_path is empty")
        if cfg.sampled_graph_loader is None:
            raise ValueError("use_sampled_pt_graph=True but sampled_graph_loader is not configured")
        print(f"[Input] sampled_pt={cfg.sampled_pt_path}")
        return cfg.sampled_graph_loader(
            cfg.sampled_pt_path,
            interaction_path=interaction_csv_path,
            protein_path=node_csv_path,
            protein_index_column="index",
            edge_label_column=cfg.edge_label_column,
            force_edge_label=cfg.force_edge_label,
            augment_negative_edges=cfg.augment_negative_edges,
            negative_edge_limit=cfg.negative_edge_limit,
            interaction_label_column="interaction_label",
            balance_edge_labels=cfg.balance_edge_labels,
        )
    if cfg.csv_graph_loader is None:
        raise ValueError("use_sampled_pt_graph=False but csv_graph_loader is not configured")
    return cfg.csv_graph_loader(
        interaction_csv_path,
        max_rows=cfg.max_rows,
        undirected=cfg.undirected,
        protein_path=node_csv_path,
        protein_index_column="index",
    )


def run_demo(cfg: GarplusRunConfig) -> None:
    print(f"=== GAR {cfg.dataset_name} Demo ===")
    print(f"[RunStart] dataset={cfg.dataset_name}")
    interaction_csv_path = resolve_path(cfg.interaction_csv_path, cfg.fallback_interaction_name, cfg.auto_discover_if_missing)
    node_csv_path = (
        resolve_path(cfg.node_csv_path, cfg.fallback_node_name, cfg.auto_discover_if_missing)
        if cfg.node_csv_path or cfg.auto_discover_if_missing
        else None
    )
    print(f"[Input] interaction_csv={interaction_csv_path}")
    print(f"[Input] {cfg.node_csv_label}={node_csv_path}")
    print(f"[Config] dataset={cfg.dataset_name} mode={cfg.mode} max_rows={cfg.max_rows} y_key={cfg.y_key} min_value_support_count={cfg.min_value_support_count}")
    print(f"[Pruning] tau_p={cfg.tau_p} tau_x={cfg.tau_x} predicate_focus={cfg.predicate_bn_focus_target}")
    print(
        f"[PatternConfig] support={cfg.pattern_support} max_radius={cfg.max_radius} "
        f"max_add_edge={cfg.max_add_edge} node_max_add_edge={cfg.node_max_add_edge} "
        f"pattern_top_k={cfg.pattern_bn_top_k_per_spawn_node} pattern_min_keep={cfg.pattern_bn_min_keep_per_spawn_node}"
    )
    print(f"[BN] pattern_bn={cfg.enable_pattern_bn} predicate_bn={cfg.enable_predicate_bn}")
    print(
        f"[PatternDebug] only={cfg.pattern_extension_only} enabled={cfg.pattern_extension_debug} "
        f"event_limit={cfg.pattern_extension_debug_limit}"
    )
    print(
        f"[PatternMatching] global_rematch_patterns={cfg.global_rematch_patterns} "
        f"global_vspawn_instances={cfg.global_vspawn_instances}"
    )
    print(f"[Targets] include_ml_predicate_targets={cfg.include_ml_predicate_targets}")
    print(f"[PatternMode] undirected_pattern={cfg.undirected_pattern}")

    graph = load_graph(cfg, interaction_csv_path, node_csv_path)
    if cfg.drop_ignored_target_edges:
        drop_summary = drop_edges_by_target_values(graph, cfg.ignored_target_values)
        print(f"[TargetEdgeFilter] {drop_summary}")
    target_y_list = []
    if not cfg.pattern_extension_only:
        ml_summary = inject_ml_predicates(graph, cfg.dataset_name, cfg.ml_predicates)
        if ml_summary.get("enabled"):
            print(f"[MLPredicate] {ml_summary}")
        enrichment_summary = enrich_numeric_bin_predicates(graph, cfg.predicate_enrichment)
        if enrichment_summary.get("enabled"):
            print(f"[PredicateEnrichment] {enrichment_summary}")
        target_y_list = build_target_y_list(graph, cfg)
        print(f"[YList] targets={target_y_list}")
    isolated_vertices = sum(1 for node_id in graph.vertices if not graph.out_edges.get(node_id) and not graph.in_edges.get(node_id))
    print(
        f"[Graph] vertices={len(graph.vertices)} out_edge_lists={sum(len(v) for v in graph.out_edges.values())} "
        f"isolated_vertices={isolated_vertices}"
    )
    print_interaction_label_distribution(graph)

    frequent_sampled = []
    frequent_edge_priors = {}
    if cfg.enable_sampled_frequent_patterns:
        frequent_sampled = mine_sampled_frequent_patterns(
            graph,
            min_graph_support=cfg.sampled_frequent_min_graph_support,
            max_patterns=cfg.sampled_frequent_print_limit,
        )
        frequent_edge_priors = edge_priors_from_frequent_patterns(frequent_sampled)
        print(
            f"[SampledFSM] min_graph_support={cfg.sampled_frequent_min_graph_support} "
            f"frequent={len(frequent_sampled)} edge_priors={len(frequent_edge_priors)}"
        )
        for signature, support in frequent_sampled[: cfg.sampled_frequent_print_limit]:
            print(f"  sampled_frequent support={support} signature={signature}")
        for prior_key, prior_value in sorted(frequent_edge_priors.items(), key=lambda item: (-item[1], str(item[0])))[:10]:
            print(f"  sampled_edge_prior score={prior_value:.4f} key={prior_key}")

    pattern_bn = None
    if cfg.enable_pattern_bn:
        pattern_bn = PatternBayesianNetwork.fit_graph(
            graph,
            PatternBNConfig(
                enabled=True,
                top_k_per_spawn_node=cfg.pattern_bn_top_k_per_spawn_node,
                min_score=cfg.tau_p,
                min_keep_per_spawn_node=cfg.pattern_bn_min_keep_per_spawn_node,
                frequent_edge_priors=frequent_edge_priors,
                frequent_prior_weight=cfg.pattern_bn_frequent_prior_weight,
                cache_path=cfg.pattern_bn_cache_path,
                retrain=cfg.retrain_pattern_bn,
            ),
        )

    predicate_bns = {}


    seed = cfg.seed_builder(graph)
    spawn = GraphSpawn(
        graph,
        [seed],
        options=PatternOptions(
            pattern_support_threshold=cfg.pattern_support,
            max_radius=cfg.max_radius,
            max_add_edge=cfg.max_add_edge,
            node_max_add_edge=cfg.node_max_add_edge,
            full_solution=cfg.full_solution,
            max_multi_support=cfg.max_multi_support,
            undirected_pattern=cfg.undirected_pattern,
            topology_only_dedup=cfg.topology_only_pattern_dedup,
            topology_dedupe_respect_direction=cfg.topology_dedupe_respect_direction,
            global_vspawn_instances=cfg.global_vspawn_instances,
            extension_debug=cfg.pattern_extension_debug,
            extension_debug_limit=cfg.pattern_extension_debug_limit,
        ),
        pattern_bn=pattern_bn,
    )

    generated = []
    round_index = 0
    while spawn.unstoppable():
        round_generated = spawn.vspawn()
        round_index += 1
        generated.extend(round_generated)
        print(f"[VSpawn] round={round_index} generated={len(round_generated)} total={len(generated)}")

    sampled_structural_patterns = []
    if cfg.inject_sampled_frequent_patterns and frequent_sampled:
        sampled_structural_patterns = build_directed_frequent_patterns(
            graph,
            frequent_sampled,
            edge_label=cfg.force_edge_label or "candidate_interaction",
            min_support=cfg.pattern_support,
            max_multi_support=cfg.max_multi_support,
            start_pattern_id=100000,
            include_edge=cfg.include_sampled_frequent_edge_pattern,
            materialize_instances=cfg.materialize_sampled_frequent_patterns,
        )[: cfg.sampled_frequent_pattern_limit]
        print(
            f"[SampledFSMInject] injected={len(sampled_structural_patterns)} "
            f"limit={cfg.sampled_frequent_pattern_limit} include_edge={cfg.include_sampled_frequent_edge_pattern} "
            f"materialized={cfg.materialize_sampled_frequent_patterns}"
        )
        for item in sampled_structural_patterns:
            edges = [(edge.src, edge.dst, edge.label) for edge in item.pattern.edges]
            print(
                f"  injected_pattern id={item.pattern.pattern_id} edges={edges} "
                f"single_support={item.single_support()} multi_support={item.multi_support()}"
            )

    generated.extend(sampled_structural_patterns)
    if not generated:
        raise RuntimeError("No pattern generated. Try lowering pattern_support or increasing max_radius/max_add_edge.")

    if cfg.global_rematch_patterns:
        # VSpawn may use capped parent embeddings for fast structural exploration.
        # Rebuild every candidate's instances globally before rule mining.
        rematch_limit = None if (cfg.full_solution or cfg.global_vspawn_instances) else cfg.max_multi_support
        if cfg.dataset_name.upper() == "PPI" and cfg.undirected_pattern:
            rematch_graph, reverse_edge_id_map = build_undirected_rematch_graph(graph)
            print(
                "[GlobalRematchGraph] mode=undirected_bidirectional_copy "
                f"synthetic_reverse_edges={len(reverse_edge_id_map)}"
            )
        else:
            rematch_graph = graph
            reverse_edge_id_map = {}
        globally_matched = []
        for item in generated:
            previous_multi_support = item.multi_support()
            matches = find_matches_with_limit(item.pattern, rematch_graph, rematch_limit)
            match_backend = "vf3_linux"
            # The installed vf3py backend has returned empty results for valid
            # multi-edge PPI embeddings. Use the pure-Python matcher only as a
            # correctness fallback; it is slower but follows our VSpawn semantics.
            if (
                not matches
                and item.pattern.edge_count() > 1
                and cfg.dataset_name.upper() == "PPI"
                and cfg.undirected_pattern
            ):
                matches = find_matches_with_limit_like(item.pattern, rematch_graph, rematch_limit)
                match_backend = "vf3_like_fallback"
            if reverse_edge_id_map:
                matches = normalize_undirected_rematch_instances(matches, graph, reverse_edge_id_map)
                matches = dedupe_rematch_instances(matches)
            single_support = len({match.pivot for match in matches if match.pivot is not None}) or len(matches)
            if single_support < cfg.pattern_support:
                print(
                    f"[GlobalRematch] pattern_id={item.pattern.pattern_id} kept=False "
                    f"backend={match_backend} "
                    f"incremental_multi={previous_multi_support} global_multi={len(matches)} "
                    f"global_single={single_support} reason=support<{cfg.pattern_support}"
                )
                continue
            rematched = FrequentPattern(
                pattern=item.pattern,
                instances=matches,
                sampled=not cfg.full_solution,
                total_single_support=single_support,
                total_multi_support=len(matches),
            )
            globally_matched.append(rematched)
            print(
                f"[GlobalRematch] pattern_id={item.pattern.pattern_id} kept=True "
                f"backend={match_backend} "
                f"incremental_multi={previous_multi_support} global_multi={len(matches)} "
                f"global_single={single_support}"
            )
        generated = globally_matched
        if not generated:
            raise RuntimeError("No globally matched pattern passed pattern_support.")

    def pattern_representative_rank(item):
        if not cfg.pattern_dedup_prefer_target_value:
            return item.single_support(), item.multi_support()
        preferred_count = 0
        for instance in item.instances:
            edge_id = instance.get_edge_id(0)
            edge = graph.edges_by_id.get(edge_id) if edge_id is not None else None
            if edge is not None and str(edge.attrs.get("interaction_label")) == cfg.pattern_dedup_prefer_target_value:
                preferred_count += 1
        return preferred_count, item.single_support(), item.multi_support()

    unique_patterns = {}
    for item in generated:
        if cfg.topology_only_pattern_dedup:
            key = topology_pattern_code(item.pattern, cfg.topology_dedupe_respect_direction)
        else:
            key = item.pattern.undirected_canonical_code() if cfg.undirected_pattern else item.pattern.canonical_code()
        current = unique_patterns.get(key)
        if current is None or pattern_representative_rank(item) > pattern_representative_rank(current):
            unique_patterns[key] = item
    patterns_to_mine = sorted(
        unique_patterns.values(),
        key=lambda item: (item.pattern.edge_count(), item.single_support(), item.multi_support()),
        reverse=True,
    )
    print(
        f"[Patterns] generated_total={len(generated)} unique_total={len(patterns_to_mine)} "
        f"deduped={len(generated) - len(patterns_to_mine)} undirected={cfg.undirected_pattern} "
        f"topology_only={cfg.topology_only_pattern_dedup} "
        f"respect_direction={cfg.topology_dedupe_respect_direction}"
    )
    if cfg.pattern_extension_only:
        print(f"\n=== Pattern Extension Results dataset={cfg.dataset_name} ===")
        for index, item in enumerate(patterns_to_mine, start=1):
            edges = [(edge.src, edge.dst, edge.label) for edge in item.pattern.edges]
            print(
                f"[PatternExtension/Result] index={index}/{len(patterns_to_mine)} "
                f"pattern_id={item.pattern.pattern_id} nodes={item.pattern.node_labels} edges={edges} "
                f"radius={item.pattern.radius} single={item.single_support()} multi={item.multi_support()}"
            )
        print_bn_summary(pattern_bn, None, cfg)
        return

    total_sent = 0
    all_pattern_rules = []
    pattern_size_by_id = {}
    for pattern_index, target_pattern in enumerate(patterns_to_mine, start=1):
        edges = [(edge.src, edge.dst, edge.label) for edge in target_pattern.pattern.edges]
        print(
            f"\n=== Pattern {pattern_index}/{len(patterns_to_mine)} "
            f"dataset={cfg.dataset_name} pattern_id={target_pattern.pattern.pattern_id} ==="
        )
        print(
            f"[PatternStart] dataset={cfg.dataset_name} pattern_index={pattern_index}/{len(patterns_to_mine)} "
            f"pattern_id={target_pattern.pattern.pattern_id} labels={target_pattern.pattern.node_labels} "
            f"edges={edges} single_support={target_pattern.single_support()} "
            f"multi_support={target_pattern.multi_support()}"
        )
        pattern_size_by_id[target_pattern.pattern.pattern_id] = target_pattern.pattern.node_count()

        if cfg.mode == "pattern-only":
            continue

        for y_key in target_y_list:
            print(f"\n--- PredicateMining dataset={cfg.dataset_name} pattern_id={target_pattern.pattern.pattern_id} y_key={y_key} ---")
            print(f"[YTarget] dataset={cfg.dataset_name} pattern_id={target_pattern.pattern.pattern_id} y_key={y_key}")
            focus_items = predicate_focus_items_for_y_key(cfg, y_key) if cfg.enable_predicate_bn else [None]
            focus_results = []
            for focus_item in focus_items:
                print(
                    f"[PredicateFocus] dataset={cfg.dataset_name} pattern_id={target_pattern.pattern.pattern_id} "
                    f"y_key={y_key} focus={focus_item}"
                )
                predicate_bn = None
                if cfg.enable_predicate_bn:
                    bn_key = (y_key, focus_item or "")
                    if bn_key not in predicate_bns:
                        predicate_bns[bn_key] = PredicateBayesianNetwork(
                            PredicateBNConfig(
                                enabled=True,
                                target_key=y_key,
                                top_k_features=cfg.predicate_bn_top_k_features,
                                min_score=cfg.tau_x,
                                focus_target_item=focus_item,
                                min_keep_features=cfg.predicate_bn_min_keep_features,
                                feature_score=cfg.predicate_bn_feature_score,
                                estimator=cfg.predicate_bn_estimator,
                                max_parent_features=cfg.predicate_bn_max_parent_features,
                                max_feature_cardinality=cfg.predicate_bn_max_feature_cardinality,
                                cache_path=predicate_bn_cache_for_y_key(cfg, y_key, focus_item),
                                retrain=cfg.retrain_predicate_bn,
                            )
                        )
                    predicate_bn = predicate_bns[bn_key]

                if cfg.mode == "decision-tree":
                    selector = DecisionTreePredicateSelector(
                        min_support=cfg.min_support,
                        min_confidence=cfg.min_confidence,
                        min_value_support_count=cfg.min_value_support_count,
                        predicate_bn=predicate_bn,
                        drop_target_values=set(cfg.ignored_target_values) if cfg.drop_unknown_target_rows else None,
                        drop_feature_key_tokens=cfg.ignored_predicate_key_tokens if cfg.filter_degree_predicates else None,
                        max_depth=cfg.decision_tree_max_depth,
                    )
                    focus_rules = selector.generate_rules(graph, target_pattern, y_key)
                    print(
                        f"[PredicateSelection/DecisionTree] pattern_id={target_pattern.pattern.pattern_id} "
                        f"focus={focus_item} rules={len(focus_rules)} y_key={y_key}"
                    )
                elif cfg.mode == "fp-growth":
                    selector = FPGrowthPredicateSelector(
                        min_support=cfg.min_support,
                        min_confidence=cfg.min_confidence,
                        min_value_support_count=cfg.min_value_support_count,
                        predicate_bn=predicate_bn,
                        drop_target_values=set(cfg.ignored_target_values) if cfg.drop_unknown_target_rows else None,
                        drop_feature_key_tokens=cfg.ignored_predicate_key_tokens if cfg.filter_degree_predicates else None,
                    )
                    focus_rules = selector.generate_rules(graph, target_pattern, y_key)
                    print(
                        f"[PredicateSelection/FPGrowth] pattern_id={target_pattern.pattern.pattern_id} "
                        f"focus={focus_item} rules={len(focus_rules)} y_prefix={y_key}"
                    )
                else:
                    raise ValueError(f"Unsupported mode: {cfg.mode}")

                if getattr(selector, "filtered_feature_keys", None):
                    dropped_keys = sorted(selector.filtered_feature_keys)
                    print(
                        f"[PredicateFilter] pattern_id={target_pattern.pattern.pattern_id} y_key={y_key} "
                        f"focus={focus_item} tokens={cfg.ignored_predicate_key_tokens} dropped_keys={len(dropped_keys)} "
                        f"sample={dropped_keys[:20]}"
                    )
                print_target_row_diagnostics(selector, focus_item)
                print_rule_consequent_distribution(focus_rules)
                print_rule_candidate_diagnostics(selector)
                focus_results.append((focus_item, selector, focus_rules))

            rule_by_key = {}
            for _focus_item, _selector, focus_rules in focus_results:
                for rule in focus_rules:
                    key = (tuple(rule.antecedent), rule.consequent)
                    current = rule_by_key.get(key)
                    if current is None or (rule.confidence, rule.support, rule.lift) > (current.confidence, current.support, current.lift):
                        rule_by_key[key] = rule
            rules = sorted(rule_by_key.values(), key=lambda item: (item.confidence, item.support, item.lift), reverse=True)
            active_selector = focus_results[-1][1] if focus_results else None
            print(
                f"[PredicateFocusMerge] pattern_id={target_pattern.pattern.pattern_id} y_key={y_key} "
                f"focus_runs={len(focus_results)} merged_rules={len(rules)} raw_focus_rules={sum(len(item[2]) for item in focus_results)}"
            )

            all_pattern_rules.extend((target_pattern.pattern.pattern_id, rule) for rule in rules)
            print_rule_consequent_distribution(rules)
            if y_key.endswith("interaction_label"):
                recall_negative_value = "negative"
            elif y_key.endswith("ml_offline_predicate_name"):
                recall_negative_value = "ml_pred_not_ppi"
            else:
                recall_negative_value = "no"
            target_recall = evaluate_negative_rule_recall(
                active_selector,
                graph,
                target_pattern,
                rules,
                y_key,
                negative_value=recall_negative_value,
            )
            print(
                "[TargetRecall] "
                f"pattern_id={target_pattern.pattern.pattern_id} y_key={y_key} "
                f"target_value={recall_negative_value} "
                f"target_rules={target_recall['negative_rules']} "
                f"covered_edges={target_recall['covered_negative_edges']}/{target_recall['total_negative_edges']} "
                f"edge_recall={target_recall['edge_recall']:.4f} "
                f"covered_instances={target_recall['covered_negative_instances']}/{target_recall['negative_instances']} "
                f"instance_recall={target_recall['instance_recall']:.4f}"
            )
            for rule in rules[: cfg.print_rule_limit]:
                print(
                    f"  antecedent={rule.antecedent} consequent={rule.consequent} "
                    f"support={int(rule.support)} confidence={rule.confidence:.3f} lift={rule.lift:.3f}"
                )

            if not rules:
                print(
                    f"[RuleGeneration] pattern_id={target_pattern.pattern.pattern_id} "
                    f"y_key={y_key} skipped because no predicate rules were generated"
                )
                continue

            zl_rules = [predicate_rule_to_zl(rule, target_pattern) for rule in rules]
            filtered = zl_rule_filter(zl_rules, filter_flag=True, min_confidence=cfg.min_confidence)
            sender = RuleSender()
            pattern_view = PatternView.from_pattern(target_pattern.pattern)
            sent = send_zl_rules(pattern_view, filtered, y_literal=rules[0].consequent, sender=sender)
            total_sent += sent
            status = RuleGenerationStatus()
            update_status_after_generate_rules(status, pattern_view.pattern_id, sent, max(0, len(zl_rules) - sent))
            print(f"[RuleGeneration] pattern_id={target_pattern.pattern.pattern_id} y_key={y_key} filtered={len(filtered)} sent={sent}")
            print(
                f"  status: discovered_rules={status.discovered_rule_num} "
                f"abandon_rules={status.abandon_rule_num} abandon_patterns={status.abandon_pattern_num}"
            )
            if cfg.print_payload_snapshot and sender.sent_rules:
                print("[RuleGeneration] first payload snapshot:")
                snapshot = dict(sender.sent_rules[0])
                if cfg.print_full_payload:
                    pprint(snapshot)
                else:
                    pprint(trim_instances(snapshot, cfg))
    print(f"\n=== BN Summary dataset={cfg.dataset_name} ===")
    print_bn_summary(pattern_bn, None, cfg)
    for bn_key, predicate_bn in predicate_bns.items():
        if isinstance(bn_key, tuple):
            y_key, focus_item = bn_key
        else:
            y_key, focus_item = bn_key, ""
        print(f"[PredicateBNTarget] dataset={cfg.dataset_name} y_key={y_key} focus={focus_item or None}")
        print_bn_summary(None, predicate_bn, cfg)

    print(f"\n=== Final Results dataset={cfg.dataset_name} ===")
    deduped_rows = print_deduped_rules(all_pattern_rules, cfg.print_deduped_rule_limit, cfg.deduped_rules_output_path)
    raw_rule_distribution = rule_consequent_distribution([rule for _pattern_id, rule in all_pattern_rules])
    deduped_rule_distribution = rule_consequent_distribution([rule for _pattern_id, rule, _key in deduped_rows])
    table_stats = discovered_rule_table_stats(deduped_rows, pattern_size_by_id)
    print(f"[RuleConsequentDistribution] raw={raw_rule_distribution} deduped={deduped_rule_distribution}")
    print_discovered_rule_stats_table(cfg.dataset_name, table_stats)
    print(
        f"[Summary] dataset={cfg.dataset_name} patterns_mined={len(patterns_to_mine)} "
        f"raw_rules={len(all_pattern_rules)} deduped_rules={len(deduped_rows)} "
        f"positive_rules={table_stats['positive']} negative_rules={table_stats['negative']} "
        f"negative_ratio={table_stats['negative_ratio']:.4f} "
        f"avg_pattern_size={table_stats['avg_pattern_size']:.3f} "
        f"avg_confidence={table_stats['avg_confidence']:.3f} "
        f"raw_consequents={raw_rule_distribution} deduped_consequents={deduped_rule_distribution} "
        f"total_sent={total_sent}"
    )
