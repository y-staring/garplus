from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import List, Optional

from pattern_extension import GraphSpawn
from pattern_bn import PatternBayesianNetwork, PatternBNConfig
from predicate_bn import PredicateBayesianNetwork, PredicateBNConfig
from predicate_selection import DecisionTreePredicateSelector, FPGrowthPredicateSelector, Rule
from graph_types import FrequentPattern, PatternOptions
from ppi_loader import build_ppi_seed_pattern, load_ppi_csv
from sampled_pt_loader import build_sampled_seed_pattern, load_sampled_pt_graph
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

# =========================
# PPI demo fixed config
# =========================
CSV_PATH: Optional[str] = r"/home/yyyy/codework/GARplus/enumeration-discovery/去病图数据/protein_protein_signed.csv"
PROTEIN_CSV_PATH: Optional[str] = r"/home/yyyy/codework/GARplus/enumeration-discovery/去病图数据/protein.csv"
AUTO_DISCOVER_IF_MISSING = False
USE_SAMPLED_PT_GRAPH = True
SAMPLED_PT_PATH: Optional[str] = "/home/yyyy/codework/GARplus/enumeration-discovery/processed/ppi/ppi_selected.pt"
FORCE_EDGE_LABEL = "candidate_interaction"
AUGMENT_NEGATIVE_EDGES = True
NEGATIVE_EDGE_LIMIT = 2000
BALANCE_EDGE_LABELS = True
MODE = "decision-tree"  # pattern-only | decision-tree | fp-growth
#目标列
#e0.interaction_label
Y_KEY = "e0.interaction_label"
MAX_ROWS = 50
UNDIRECTED = True
FULL_SOLUTION = False

PATTERN_SUPPORT = 5
MIN_SUPPORT = 100
MIN_CONFIDENCE = 0.3
MIN_VALUE_SUPPORT_COUNT =20
MAX_RADIUS = 1
MAX_ADD_EDGE = 1
NODE_MAX_ADD_EDGE = 1
MAX_MULTI_SUPPORT = 10000
PRINT_RULE_LIMIT = 10
PRINT_FULL_PAYLOAD = False
PRINT_INSTANCES = False
PRINT_INSTANCE_LIMIT = 5
PRINT_BN_STATS = True
DROP_UNKNOWN_TARGET_ROWS = True
ENABLE_PATTERN_BN = True
PATTERN_BN_TOP_K_PER_SPAWN_NODE = 8
PATTERN_BN_MIN_SCORE = 0.0

ENABLE_PREDICATE_BN = True
PREDICATE_BN_TOP_K_FEATURES = 20
PREDICATE_BN_MIN_SCORE = 0.0


def resolve_csv_path(raw_path: Optional[str], fallback_name: str) -> str:
    if raw_path:
        return raw_path
    if not AUTO_DISCOVER_IF_MISSING:
        raise FileNotFoundError(f"{fallback_name} is empty and auto discovery is disabled")
    search_root = Path(r"D:\CodeWork\python\GAR+")
    matches = list(search_root.rglob(fallback_name))
    if not matches:
        raise FileNotFoundError(f"Could not auto-discover {fallback_name}")
    matches.sort(key=lambda candidate: len(str(candidate)))
    return str(matches[0])


def pick_edge_pattern(patterns: List[FrequentPattern]) -> FrequentPattern:
    edge_patterns = [item for item in patterns if item.pattern.edge_count() == 1 and item.pattern.node_count() == 2]
    if not edge_patterns:
        raise RuntimeError("No 2-node/1-edge pattern generated. Try lowering support or max_rows.")
    edge_patterns.sort(key=lambda item: item.single_support(), reverse=True)
    return edge_patterns[0]


def trim_instances(payload: dict) -> dict:
    """Avoid dumping all matched instances unless explicitly requested."""

    if PRINT_INSTANCES:
        for key in ("x_instance", "y_instance"):
            if key in payload:
                payload[key] = payload[key][:PRINT_INSTANCE_LIMIT]
        return payload
    for key in ("x_instance", "y_instance"):
        if key in payload:
            payload[key] = f"<{len(payload[key])} instances hidden; set PRINT_INSTANCES=True to show>"
    return payload


def print_bn_summary(pattern_bn, predicate_bn) -> None:
    if not PRINT_BN_STATS:
        return
    if pattern_bn is not None:
        summary = pattern_bn.pruning_summary()
        print(
            "[PatternBN] "
            f"backend={summary.get('backend', 'unknown')} "
            f"rank_calls={summary['rank_calls']} seen={summary['candidates_seen']} "
            f"kept={summary['candidates_kept']} pruned={summary['candidates_pruned']}"
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
        print(f"  target_values={summary['target_values']}")
        print(
            f"  target_cardinality={summary.get('target_cardinality')} "
            f"estimated_target_cpd_cells={summary.get('estimated_target_cpd_cells')} "
            f"max_cpd_cells={summary.get('max_cpd_cells')}"
        )
        print(
            f"  sparse_candidates={summary.get('candidate_features_after_sparse_filter')} "
            f"sparse_skipped={summary.get('sparse_features_skipped')} "
            f"skip_reasons={summary.get('sparse_skip_reasons')}"
        )
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
    counts = {}
    for rule in rules:
        value = rule.consequent.split("=", 1)[1] if "=" in rule.consequent else rule.consequent
        counts[value] = counts.get(value, 0) + 1
    print(f"[PredicateSelection] consequent_distribution={counts}")

def print_interaction_label_distribution(graph) -> None:
    counts = {}
    for edge in graph.all_edges():
        label = edge.attrs.get("interaction_label", "<missing>")
        counts[label] = counts.get(label, 0) + 1
    if counts:
        print(f"[Graph] interaction_label_distribution={counts}")





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
    # `rule.support` follows the paper definition now: |Q(G, X ? p0)|.
    # Do not multiply it by pattern support again.
    xy_support = max(0, int(rule.support))
    xy_support_single = min(frequent_pattern.single_support(), xy_support)
    xy_support_multiple = min(frequent_pattern.multi_support(), xy_support)
    segment = SegmentRuleSet(
        keys=[y_key],
        intervals=[(float("-inf"), float("inf"))],
        is_nans=[False],
        statistics=RuleStatistics(
            freq_antecedent=FreqCount(frequent_pattern.single_support(), frequent_pattern.multi_support()),
            freq_union=FreqCount(
                xy_support_single,
                xy_support_multiple,
            ),
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


def main() -> None:
    print("=== GAR PPI Demo ===")
    csv_path = resolve_csv_path(CSV_PATH, "protein_protein.csv")
    protein_csv_path = resolve_csv_path(PROTEIN_CSV_PATH, "protein.csv") if PROTEIN_CSV_PATH or AUTO_DISCOVER_IF_MISSING else None
    print(f"[Input] interaction_csv={csv_path}")
    print(f"[Input] protein_csv={protein_csv_path}")
    print(f"[Config] mode={MODE} max_rows={MAX_ROWS} y_key={Y_KEY} min_value_support_count={MIN_VALUE_SUPPORT_COUNT}")
    print(f"[BN] pattern_bn={ENABLE_PATTERN_BN} predicate_bn={ENABLE_PREDICATE_BN}")

    if USE_SAMPLED_PT_GRAPH:
        if not SAMPLED_PT_PATH:
            raise ValueError("USE_SAMPLED_PT_GRAPH=True but SAMPLED_PT_PATH is empty")
        print(f"[Input] sampled_pt={SAMPLED_PT_PATH}")
        graph = load_sampled_pt_graph(
            SAMPLED_PT_PATH,
            interaction_path=csv_path,
            protein_path=protein_csv_path,
            protein_index_column="index",
            edge_label_column="Experimental System",
            force_edge_label=FORCE_EDGE_LABEL,
            augment_negative_edges=AUGMENT_NEGATIVE_EDGES,
            negative_edge_limit=NEGATIVE_EDGE_LIMIT,
            interaction_label_column="interaction_label",
            balance_edge_labels=BALANCE_EDGE_LABELS,
        )
    else:
        graph = load_ppi_csv(
            csv_path,
            max_rows=MAX_ROWS,
            undirected=UNDIRECTED,
            protein_path=protein_csv_path,
            protein_index_column='index',
        )
    isolated_vertices = sum(1 for node_id in graph.vertices if not graph.out_edges.get(node_id) and not graph.in_edges.get(node_id))
    print(
        f"[Graph] vertices={len(graph.vertices)} out_edge_lists={sum(len(v) for v in graph.out_edges.values())} "
        f"isolated_vertices={isolated_vertices}"
    )
    print_interaction_label_distribution(graph)

    pattern_bn = None
    if ENABLE_PATTERN_BN:
        pattern_bn = PatternBayesianNetwork.fit_graph(
            graph,
            PatternBNConfig(
                enabled=True,
                top_k_per_spawn_node=PATTERN_BN_TOP_K_PER_SPAWN_NODE,
                min_score=PATTERN_BN_MIN_SCORE,
            ),
        )
    predicate_bn = None
    if ENABLE_PREDICATE_BN:
        predicate_bn = PredicateBayesianNetwork(
            PredicateBNConfig(
                enabled=True,
                target_key=Y_KEY,
                top_k_features=PREDICATE_BN_TOP_K_FEATURES,
                min_score=PREDICATE_BN_MIN_SCORE,
            )
        )

    seed = build_sampled_seed_pattern(graph) if USE_SAMPLED_PT_GRAPH else build_ppi_seed_pattern(graph)
    spawn = GraphSpawn(
        graph,
        [seed],
        options=PatternOptions(
            pattern_support_threshold=PATTERN_SUPPORT,
            max_radius=MAX_RADIUS,
            max_add_edge=MAX_ADD_EDGE,
            node_max_add_edge=NODE_MAX_ADD_EDGE,
            full_solution=FULL_SOLUTION,
            max_multi_support=MAX_MULTI_SUPPORT,
        ),
        pattern_bn=pattern_bn,
    )
    generated = spawn.vspawn()
    print(f"[VSpawn] generated={len(generated)}")
    target_pattern = pick_edge_pattern(generated)
    edges = [(edge.src, edge.dst, edge.label) for edge in target_pattern.pattern.edges]
    print(
        f"[Pattern] id={target_pattern.pattern.pattern_id} labels={target_pattern.pattern.node_labels} "
        f"edges={edges} single_support={target_pattern.single_support()} multi_support={target_pattern.multi_support()}"
    )

    if MODE == "pattern-only":
        return

    if MODE == "decision-tree":
        selector = DecisionTreePredicateSelector(min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE, min_value_support_count=MIN_VALUE_SUPPORT_COUNT, predicate_bn=predicate_bn, drop_target_values={"unknown"} if DROP_UNKNOWN_TARGET_ROWS else None)
        rules = selector.generate_rules(graph, target_pattern, Y_KEY)
        print(f"[PredicateSelection/DecisionTree] rules={len(rules)} y_key={Y_KEY}")
        active_selector = selector
    elif MODE == "fp-growth":
        selector = FPGrowthPredicateSelector(min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE, min_value_support_count=MIN_VALUE_SUPPORT_COUNT, predicate_bn=predicate_bn, drop_target_values={"unknown"} if DROP_UNKNOWN_TARGET_ROWS else None)
        rules = selector.generate_rules(graph, target_pattern, Y_KEY)
        print(f"[PredicateSelection/FPGrowth] rules={len(rules)} y_prefix={Y_KEY}")
        active_selector = selector
    else:
        raise ValueError(f"Unsupported MODE: {MODE}")

    print_rule_consequent_distribution(rules)
    print_negative_rule_diagnostics(active_selector)
    negative_recall = evaluate_negative_rule_recall(active_selector, graph, target_pattern, rules, Y_KEY)
    print(
        "[NegativeRecall] "
        f"negative_rules={negative_recall['negative_rules']} "
        f"covered_edges={negative_recall['covered_negative_edges']}/{negative_recall['total_negative_edges']} "
        f"edge_recall={negative_recall['edge_recall']:.4f} "
        f"covered_instances={negative_recall['covered_negative_instances']}/{negative_recall['negative_instances']} "
        f"instance_recall={negative_recall['instance_recall']:.4f}"
    )
    for rule in rules[:PRINT_RULE_LIMIT]:
        print(f"  antecedent={rule.antecedent} consequent={rule.consequent} support={int(rule.support)} confidence={rule.confidence:.3f} lift={rule.lift:.3f}")
    print_bn_summary(pattern_bn, predicate_bn)

    if not rules:
        print("[RuleGeneration] skipped because no predicate rules were generated")
        return

    zl_rules = [predicate_rule_to_zl(rule, target_pattern) for rule in rules]
    filtered = zl_rule_filter(zl_rules, filter_flag=True, min_confidence=MIN_CONFIDENCE)
    sender = RuleSender()
    pattern_view = PatternView.from_pattern(target_pattern.pattern)
    sent = send_zl_rules(pattern_view, filtered, y_literal=rules[0].consequent, sender=sender)
    status = RuleGenerationStatus()
    update_status_after_generate_rules(status, pattern_view.pattern_id, sent, max(0, len(zl_rules) - sent))
    print(f"[RuleGeneration] filtered={len(filtered)} sent={sent}")
    print(
        f"  status: discovered_rules={status.discovered_rule_num} "
        f"abandon_rules={status.abandon_rule_num} abandon_patterns={status.abandon_pattern_num}"
    )
    if sender.sent_rules:
        print("[RuleGeneration] first payload snapshot:")
        snapshot = dict(sender.sent_rules[0])
        if PRINT_FULL_PAYLOAD:
            pprint(snapshot)
        else:
            pprint(trim_instances(snapshot))


if __name__ == "__main__":
    main()

