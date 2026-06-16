from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Callable, List, Optional

from graph_types import FrequentPattern, PatternOptions
from garplus_ml_predicates import MLPredicateConfig, inject_ml_predicates
from pattern_bn import PatternBayesianNetwork, PatternBNConfig
from pattern_extension import GraphSpawn
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
    mode: str = "decision-tree"
    y_key: str = "e0.interaction_label"
    max_rows: int = 50
    undirected: bool = True
    undirected_pattern: bool = True
    full_solution: bool = False
    pattern_support: int = 5
    min_support: int = 100
    min_confidence: float = 0.6
    min_value_support_count: int = 20
    max_radius: int = 4
    max_add_edge: int = 4
    node_max_add_edge: int = 4
    max_multi_support: int = 10000
    print_rule_limit: int = 10
    print_deduped_rule_limit: int = 50
    print_full_payload: bool = False
    print_instances: bool = False
    print_instance_limit: int = 5
    print_bn_stats: bool = True
    enable_sampled_frequent_patterns: bool = True
    sampled_frequent_min_graph_support: int = 5
    sampled_frequent_print_limit: int = 20
    inject_sampled_frequent_patterns: bool = True
    sampled_frequent_pattern_limit: int = 8
    materialize_sampled_frequent_patterns: bool = True
    drop_unknown_target_rows: bool = True
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
    predicate_bn_feature_score: str = "bic"
    predicate_bn_estimator: str = "maximum_likelihood"
    predicate_bn_cache_path: Optional[str] = None
    retrain_predicate_bn: bool = True
    ml_predicates: MLPredicateConfig = MLPredicateConfig()


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


def print_deduped_rules(pattern_rules, limit: int):
    deduped = dedupe_rules_semantically(pattern_rules)
    print(f"[DedupedRules] raw={len(pattern_rules)} unique={len(deduped)} limit={limit}")
    for pattern_id, rule, key in deduped[:limit]:
        antecedent_key, consequent_key = key
        print(
            "  deduped_rule "
            f"pattern_id={pattern_id} antecedent={antecedent_key} consequent={consequent_key} "
            f"raw_antecedent={rule.antecedent} raw_consequent={rule.consequent} "
            f"support={int(rule.support)} confidence={rule.confidence:.3f} lift={rule.lift:.3f}"
        )
    return deduped


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


def print_rule_consequent_distribution(rules: List[Rule]) -> None:
    print(f"[PredicateSelection] consequent_distribution={rule_consequent_distribution(rules)}")


def print_negative_rule_diagnostics(selector, limit: int = 20) -> None:
    diagnostics = selector.negative_diagnostics(limit=limit) if hasattr(selector, "negative_diagnostics") else []
    if not diagnostics:
        print("[PredicateSelection] negative_candidate_diagnostics=[]")
        return
    print(f"[PredicateSelection] negative_candidate_diagnostics top={len(diagnostics)}")
    for item in diagnostics:
        print(
            "  negative_candidate "
            f"antecedent={item['antecedent']} consequent={item['consequent']} "
            f"support={item['support']} confidence={item['confidence']:.4f} "
            f"lift={item['lift']:.4f} reason={item['reason']}"
        )


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
        key, value = antecedent.split("=", 1)
        general_keys.append(key)
        values.append([value])
        semantics.append("or")
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
    print(f"[PatternMode] undirected_pattern={cfg.undirected_pattern}")

    graph = load_graph(cfg, interaction_csv_path, node_csv_path)
    ml_summary = inject_ml_predicates(graph, cfg.dataset_name, cfg.ml_predicates)
    if ml_summary.get("enabled"):
        print(f"[MLPredicate] {ml_summary}")
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

    predicate_bn = None
    if cfg.enable_predicate_bn:
        predicate_bn = PredicateBayesianNetwork(
            PredicateBNConfig(
                enabled=True,
                target_key=cfg.y_key,
                top_k_features=cfg.predicate_bn_top_k_features,
                min_score=cfg.tau_x,
                focus_target_item=cfg.predicate_bn_focus_target,
                min_keep_features=cfg.predicate_bn_min_keep_features,
                feature_score=cfg.predicate_bn_feature_score,
                estimator=cfg.predicate_bn_estimator,
                max_parent_features=cfg.predicate_bn_max_parent_features,
                max_feature_cardinality=cfg.predicate_bn_max_feature_cardinality,
                cache_path=cfg.predicate_bn_cache_path,
                retrain=cfg.retrain_predicate_bn,
            )
        )

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
            include_edge=False,
            materialize_instances=cfg.materialize_sampled_frequent_patterns,
        )[: cfg.sampled_frequent_pattern_limit]
        print(
            f"[SampledFSMInject] injected={len(sampled_structural_patterns)} "
            f"limit={cfg.sampled_frequent_pattern_limit} materialized={cfg.materialize_sampled_frequent_patterns}"
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

    unique_patterns = {}
    for item in generated:
        key = item.pattern.undirected_canonical_code() if cfg.undirected_pattern else item.pattern.canonical_code()
        current = unique_patterns.get(key)
        if current is None or (item.single_support(), item.multi_support()) > (current.single_support(), current.multi_support()):
            unique_patterns[key] = item
    patterns_to_mine = sorted(
        unique_patterns.values(),
        key=lambda item: (item.pattern.edge_count(), item.single_support(), item.multi_support()),
        reverse=True,
    )
    print(
        f"[Patterns] generated_total={len(generated)} unique_total={len(patterns_to_mine)} "
        f"deduped={len(generated) - len(patterns_to_mine)} undirected={cfg.undirected_pattern}"
    )

    total_sent = 0
    all_pattern_rules = []
    pattern_size_by_id = {}
    for pattern_index, target_pattern in enumerate(patterns_to_mine, start=1):
        edges = [(edge.src, edge.dst, edge.label) for edge in target_pattern.pattern.edges]
        print(
            f"[Pattern {pattern_index}/{len(patterns_to_mine)}] "
            f"id={target_pattern.pattern.pattern_id} labels={target_pattern.pattern.node_labels} "
            f"edges={edges} single_support={target_pattern.single_support()} "
            f"multi_support={target_pattern.multi_support()}"
        )
        pattern_size_by_id[target_pattern.pattern.pattern_id] = target_pattern.pattern.node_count()

        if cfg.mode == "pattern-only":
            continue
        if cfg.mode == "decision-tree":
            selector = DecisionTreePredicateSelector(
                min_support=cfg.min_support,
                min_confidence=cfg.min_confidence,
                min_value_support_count=cfg.min_value_support_count,
                predicate_bn=predicate_bn,
                drop_target_values={"unknown"} if cfg.drop_unknown_target_rows else None,
            )
            rules = selector.generate_rules(graph, target_pattern, cfg.y_key)
            print(f"[PredicateSelection/DecisionTree] pattern_id={target_pattern.pattern.pattern_id} rules={len(rules)} y_key={cfg.y_key}")
            active_selector = selector
        elif cfg.mode == "fp-growth":
            selector = FPGrowthPredicateSelector(
                min_support=cfg.min_support,
                min_confidence=cfg.min_confidence,
                min_value_support_count=cfg.min_value_support_count,
                predicate_bn=predicate_bn,
                drop_target_values={"unknown"} if cfg.drop_unknown_target_rows else None,
            )
            rules = selector.generate_rules(graph, target_pattern, cfg.y_key)
            print(f"[PredicateSelection/FPGrowth] pattern_id={target_pattern.pattern.pattern_id} rules={len(rules)} y_prefix={cfg.y_key}")
            active_selector = selector
        else:
            raise ValueError(f"Unsupported mode: {cfg.mode}")

        all_pattern_rules.extend((target_pattern.pattern.pattern_id, rule) for rule in rules)
        print_rule_consequent_distribution(rules)
        print_negative_rule_diagnostics(active_selector)
        negative_recall = evaluate_negative_rule_recall(active_selector, graph, target_pattern, rules, cfg.y_key)
        print(
            "[NegativeRecall] "
            f"pattern_id={target_pattern.pattern.pattern_id} "
            f"negative_rules={negative_recall['negative_rules']} "
            f"covered_edges={negative_recall['covered_negative_edges']}/{negative_recall['total_negative_edges']} "
            f"edge_recall={negative_recall['edge_recall']:.4f} "
            f"covered_instances={negative_recall['covered_negative_instances']}/{negative_recall['negative_instances']} "
            f"instance_recall={negative_recall['instance_recall']:.4f}"
        )
        for rule in rules[: cfg.print_rule_limit]:
            print(f"  antecedent={rule.antecedent} consequent={rule.consequent} support={int(rule.support)} confidence={rule.confidence:.3f} lift={rule.lift:.3f}")

        if not rules:
            print(f"[RuleGeneration] pattern_id={target_pattern.pattern.pattern_id} skipped because no predicate rules were generated")
            continue

        zl_rules = [predicate_rule_to_zl(rule, target_pattern) for rule in rules]
        filtered = zl_rule_filter(zl_rules, filter_flag=True, min_confidence=cfg.min_confidence)
        sender = RuleSender()
        pattern_view = PatternView.from_pattern(target_pattern.pattern)
        sent = send_zl_rules(pattern_view, filtered, y_literal=rules[0].consequent, sender=sender)
        total_sent += sent
        status = RuleGenerationStatus()
        update_status_after_generate_rules(status, pattern_view.pattern_id, sent, max(0, len(zl_rules) - sent))
        print(f"[RuleGeneration] pattern_id={target_pattern.pattern.pattern_id} filtered={len(filtered)} sent={sent}")
        print(
            f"  status: discovered_rules={status.discovered_rule_num} "
            f"abandon_rules={status.abandon_rule_num} abandon_patterns={status.abandon_pattern_num}"
        )
        if sender.sent_rules:
            print("[RuleGeneration] first payload snapshot:")
            snapshot = dict(sender.sent_rules[0])
            if cfg.print_full_payload:
                pprint(snapshot)
            else:
                pprint(trim_instances(snapshot, cfg))

    deduped_rows = print_deduped_rules(all_pattern_rules, cfg.print_deduped_rule_limit)
    raw_rule_distribution = rule_consequent_distribution([rule for _pattern_id, rule in all_pattern_rules])
    deduped_rule_distribution = rule_consequent_distribution([rule for _pattern_id, rule, _key in deduped_rows])
    table_stats = discovered_rule_table_stats(deduped_rows, pattern_size_by_id)
    print(f"[RuleConsequentDistribution] raw={raw_rule_distribution} deduped={deduped_rule_distribution}")
    print_discovered_rule_stats_table(cfg.dataset_name, table_stats)
    print_bn_summary(pattern_bn, predicate_bn, cfg)
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
