from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import List, Optional

from pattern_extension import GraphSpawn
from predicate_selection import DecisionTreePredicateSelector, FPGrowthPredicateSelector, Rule
from graph_types import FrequentPattern, PatternOptions
from ppi_loader import build_ppi_seed_pattern, load_ppi_csv
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
CSV_PATH: Optional[str] = r"D:\CodeWork\python\GAR+\数据\去病图数据\去病图数据\protein_protein.csv"
PROTEIN_CSV_PATH: Optional[str] = r"D:\CodeWork\python\GAR+\数据\去病图数据\去病图数据\protein.csv"
AUTO_DISCOVER_IF_MISSING = False
MODE = "decision-tree"  # pattern-only | decision-tree | fp-growth
#目标列
Y_KEY = "v0.high_degree"
MAX_ROWS = 50
UNDIRECTED = True
FULL_SOLUTION = False

PATTERN_SUPPORT = 5
MIN_SUPPORT = 0.1
MIN_CONFIDENCE = 0.6
MIN_VALUE_SUPPORT_COUNT = 5
MAX_RADIUS = 2
MAX_ADD_EDGE = 2
NODE_MAX_ADD_EDGE = 1
MAX_MULTI_SUPPORT = 5000
PRINT_RULE_LIMIT = 10


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
    segment = SegmentRuleSet(
        keys=[y_key],
        intervals=[(float("-inf"), float("inf"))],
        is_nans=[False],
        statistics=RuleStatistics(
            freq_antecedent=FreqCount(frequent_pattern.single_support(), frequent_pattern.multi_support()),
            freq_union=FreqCount(
                max(1, int(rule.support * frequent_pattern.single_support())),
                max(1, int(rule.support * frequent_pattern.multi_support())),
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

    seed = build_ppi_seed_pattern(graph)
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
        selector = DecisionTreePredicateSelector(min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE, min_value_support_count=MIN_VALUE_SUPPORT_COUNT)
        rules = selector.generate_rules(graph, target_pattern, Y_KEY)
        print(f"[PredicateSelection/DecisionTree] rules={len(rules)} y_key={Y_KEY}")
    elif MODE == "fp-growth":
        selector = FPGrowthPredicateSelector(min_support=MIN_SUPPORT, min_confidence=MIN_CONFIDENCE, min_value_support_count=MIN_VALUE_SUPPORT_COUNT)
        rules = selector.generate_rules(graph, target_pattern, Y_KEY)
        print(f"[PredicateSelection/FPGrowth] rules={len(rules)} y_prefix={Y_KEY}")
    else:
        raise ValueError(f"Unsupported MODE: {MODE}")

    for rule in rules[:PRINT_RULE_LIMIT]:
        print(f"  antecedent={rule.antecedent} consequent={rule.consequent} conf={rule.confidence:.3f} lift={rule.lift:.3f}")

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
        pprint(sender.sent_rules[0])


if __name__ == "__main__":
    main()
