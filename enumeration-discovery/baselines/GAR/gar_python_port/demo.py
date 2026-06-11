from __future__ import annotations

from pprint import pprint
from typing import List, Sequence, Tuple

from graph_types import (
    DataGraph,
    FrequentPattern,
    GraphInstance,
    GraphPattern,
    PatternOptions,
    Vertex,
)
from pattern_extension import GraphSpawn
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


def build_demo_graph() -> DataGraph:
    graph = DataGraph(
        vertices={
            1: Vertex(1, "Person", {"age_group": "young", "city": "SZ", "target": 1}),
            2: Vertex(2, "Movie", {"genre": "Action", "target": 1}),
            3: Vertex(3, "Person", {"age_group": "young", "city": "SZ", "target": 1}),
            4: Vertex(4, "Movie", {"genre": "Action", "target": 1}),
            5: Vertex(5, "Person", {"age_group": "adult", "city": "GZ", "target": 0}),
            6: Vertex(6, "Movie", {"genre": "Drama", "target": 0}),
        }
    )
    graph.add_edge(1, 2, "likes")
    graph.add_edge(3, 4, "likes")
    graph.add_edge(5, 6, "likes")
    return graph


def build_seed_pattern(graph: DataGraph) -> FrequentPattern:
    seed_pattern = GraphPattern(node_labels=["Person"])
    instances = [
        GraphInstance(node_map={0: node_id}, edge_ids=(), pivot=node_id)
        for node_id, vertex in graph.vertices.items()
        if vertex.label == "Person"
    ]
    return FrequentPattern(pattern=seed_pattern, instances=instances)


def choose_person_movie_pattern(patterns: Sequence[FrequentPattern]) -> FrequentPattern:
    for item in patterns:
        labels = item.pattern.node_labels
        edges = [(edge.src, edge.dst, edge.label) for edge in item.pattern.edges]
        if labels == ["Person", "Movie"] and edges == [(0, 1, "likes")]:
            return item
    raise RuntimeError("Did not generate expected Person-likes-Movie pattern")


def predicate_rule_to_zl(rule: Rule, frequent_pattern: FrequentPattern) -> ZLRule:
    general_keys: List[str] = []
    values: List[List[object]] = []
    semantics: List[str] = []
    for antecedent in rule.antecedent:
        key, value = antecedent.split("=", 1)
        general_keys.append(key)
        values.append([value])
        semantics.append("or")

    y_literal = rule.consequent
    y_key, y_value = y_literal.split("=", 1)
    segment = SegmentRuleSet(
        keys=[y_key],
        intervals=[(float("-inf"), float("inf"))],
        is_nans=[False],
        statistics=RuleStatistics(
            freq_antecedent=FreqCount(frequent_pattern.single_support(), frequent_pattern.multi_support()),
            freq_union=FreqCount(max(1, int(rule.support * frequent_pattern.single_support())), max(1, int(rule.support * frequent_pattern.multi_support()))),
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
        y_literal=y_literal,
    )


def run_vspawn_demo(graph: DataGraph) -> FrequentPattern:
    seed = build_seed_pattern(graph)
    spawn = GraphSpawn(
        graph,
        [seed],
        options=PatternOptions(
            pattern_support_threshold=1,
            max_radius=2,
            max_add_edge=4,
            node_max_add_edge=2,
        ),
    )
    generated = spawn.vspawn()
    print(f"[VSpawn] generated {len(generated)} patterns")
    for item in generated:
        edges = [(edge.src, edge.dst, edge.label) for edge in item.pattern.edges]
        print(
            f"  pattern_id={item.pattern.pattern_id} labels={item.pattern.node_labels} "
            f"edges={edges} single_support={item.single_support()} multi_support={item.multi_support()}"
        )
    return choose_person_movie_pattern(generated)


def run_predicate_selection_demo(graph: DataGraph, frequent_pattern: FrequentPattern) -> Tuple[List[Rule], List[Rule]]:
    dt_selector = DecisionTreePredicateSelector(min_support=0.3, min_confidence=0.6)
    dt_rules = dt_selector.generate_rules(graph, frequent_pattern, "v0.target")
    print(f"[PredicateSelection/DecisionTree] generated {len(dt_rules)} rules")
    for rule in dt_rules:
        print(f"  antecedent={rule.antecedent} consequent={rule.consequent} conf={rule.confidence:.3f} lift={rule.lift:.3f}")

    fp_selector = FPGrowthPredicateSelector(min_support=0.3, min_confidence=0.6)
    fp_rules = fp_selector.generate_rules(graph, frequent_pattern, "v0.target")
    print(f"[PredicateSelection/FPGrowth] generated {len(fp_rules)} rules")
    for rule in fp_rules:
        print(f"  antecedent={rule.antecedent} consequent={rule.consequent} conf={rule.confidence:.3f} lift={rule.lift:.3f}")
    return dt_rules, fp_rules


def run_rulegeneration_demo(frequent_pattern: FrequentPattern, predicate_rules: Sequence[Rule]) -> None:
    pattern_view = PatternView.from_pattern(frequent_pattern.pattern)
    zl_rules = [predicate_rule_to_zl(rule, frequent_pattern) for rule in predicate_rules]
    filtered = zl_rule_filter(zl_rules, filter_flag=True, min_confidence=0.6)
    sender = RuleSender()
    sent = send_zl_rules(pattern_view, filtered, y_literal="v0.target=1", sender=sender)
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


def main() -> None:
    print("=== GAR Python Port Demo ===")
    graph = build_demo_graph()
    frequent_pattern = run_vspawn_demo(graph)
    dt_rules, fp_rules = run_predicate_selection_demo(graph, frequent_pattern)
    preferred_rules = dt_rules or fp_rules
    if not preferred_rules:
        raise RuntimeError("No predicate-selection rules generated in demo")
    run_rulegeneration_demo(frequent_pattern, preferred_rules)


if __name__ == "__main__":
    main()
