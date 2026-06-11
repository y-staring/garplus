from __future__ import annotations

"""GAR-like rule serialization for the Python port.

Conceptually:
- `PatternView` describes the structural pattern
- `ZLRule` stores a rule in an intermediate, GAR-shaped representation
- `serialize_rule(...)` converts it into a JSON-friendly payload
"""

from dataclasses import dataclass, field
from itertools import count
from math import isinf, isnan
from typing import Dict, List, Optional, Sequence, Tuple

from graph_types import GraphPattern, Label


_RULE_ID_COUNTER = count(1)


@dataclass
class FreqCount:
    """Single-support / multi-support pair."""

    single_support: int = 0
    multi_support: int = 0


@dataclass
class RuleStatistics:
    """Statistics attached to one rule."""

    freq_antecedent: FreqCount = field(default_factory=FreqCount)
    freq_consequent: FreqCount = field(default_factory=FreqCount)
    freq_union: FreqCount = field(default_factory=FreqCount)
    confidence: float = 0.0
    lift: float = 0.0
    max_confidence: float = 0.0
    added_value: float = 0.0
    filter_flag: int = 0
    status: int = 0


@dataclass
class SegmentRuleSet:
    """GAR-style segmentation view of the rule's Y side."""

    keys: List[str] = field(default_factory=list)
    intervals: List[Tuple[float, float]] = field(default_factory=list)
    is_nans: List[bool] = field(default_factory=list)
    statistics: RuleStatistics = field(default_factory=RuleStatistics)

    def to_dict(self, pattern_view: "PatternView", x_length: int = 0) -> Optional[Dict[str, object]]:
        """Convert segmentation metadata into the output payload format."""

        if not self.keys:
            return None
        segmentation: List[Dict[str, object]] = []
        for index, key in enumerate(self.keys):
            item: Dict[str, object] = {"placeholder": pattern_view.placeholder(key)}
            lower, upper = self.intervals[index]
            if lower != float("-inf"):
                item["lower"] = lower
            if upper != float("inf"):
                item["upper"] = upper
            if self.is_nans[index]:
                item["isnull"] = True
            segmentation.append(item)
        stats = self.statistics
        measurement = {
            "x_support_single": int(stats.freq_antecedent.single_support),
            "x_support_multiple": int(stats.freq_antecedent.multi_support),
            "y_support_single": int(stats.freq_union.single_support),
            "y_support_multiple": int(stats.freq_union.multi_support),
            "confidence": stats.confidence,
            "lift": stats.lift,
            "rule_weight": [float(len(pattern_view.edges)), float(len(pattern_view.nodes)), stats.confidence, float(x_length + len(segmentation)), stats.lift],
        }
        return {"segmentation": segmentation, "measurement": measurement}


@dataclass
class ZLRule:
    """Intermediate rule object close to the Go project's ZLRule."""

    segment_rules: SegmentRuleSet = field(default_factory=SegmentRuleSet)
    general_keys: List[str] = field(default_factory=list)
    values: List[List[object]] = field(default_factory=list)
    semantics: List[str] = field(default_factory=list)
    is_nans: List[bool] = field(default_factory=list)
    old_confidence: float = 0.0
    instances: Tuple[List[Dict[str, object]], List[Dict[str, object]]] = field(default_factory=lambda: ([], []))
    y_literal: Optional[str] = None

    @property
    def statistics(self) -> RuleStatistics:
        return self.segment_rules.statistics

    @statistics.setter
    def statistics(self, value: RuleStatistics) -> None:
        self.segment_rules.statistics = value

    def zl_literals(self, pattern_view: "PatternView") -> Optional[List[Dict[str, object]]]:
        """Return the X-side literals (`zl_col`) used by the serialized payload."""

        if not self.general_keys:
            return None
        result: List[Dict[str, object]] = []
        for index, key in enumerate(self.general_keys):
            item: Dict[str, object] = {"placeholder": pattern_view.placeholder(key)}
            item["value"] = self.values[index]
            item["semantics"] = self.semantics[index]
            if self.is_nans[index]:
                item["isnull"] = True
            result.append(item)
        return result


@dataclass
class RelatedRuleInfo:
    pattern_id: int
    node_id: int
    rule_id: int


@dataclass
class PatternView:
    """A minimal structural view used during output serialization."""

    pattern_id: int
    nodes: List[Label]
    edges: List[Tuple[int, int, Label]]

    @classmethod
    def from_pattern(cls, pattern: GraphPattern) -> "PatternView":
        return cls(pattern_id=pattern.pattern_id, nodes=list(pattern.node_labels), edges=[(edge.src, edge.dst, edge.label) for edge in pattern.edges])

    def placeholder(self, raw_key: str) -> Dict[str, object]:
        """Map a literal key such as `v0.attr` into a structured output placeholder."""

        if "." in raw_key:
            entity, key = raw_key.split(".", 1)
            if entity.startswith("v") and entity[1:].isdigit():
                return {"type": "vertex", "id": int(entity[1:]), "key": key}
            if entity.startswith("e") and entity[1:].isdigit():
                edge_index = int(entity[1:])
                if 0 <= edge_index < len(self.edges):
                    src, dst, label = self.edges[edge_index]
                    return {"type": "edge", "src": src, "dst": dst, "label": label, "key": key}
        return {"type": "attr", "key": raw_key}


@dataclass
class RuleGenerationStatus:
    """Counters used by the simplified send/filter pipeline."""

    discovered_rule_num: int = 0
    abandon_rule_num: int = 0
    discovered_pattern_num: int = 0
    abandon_pattern_num: int = 0

    def merge_rule_status(self, discovered: int, abandoned: int) -> "RuleGenerationStatus":
        self.discovered_rule_num += discovered
        self.abandon_rule_num += abandoned
        return self

    def merge_pattern_status(self, discovered: int, abandoned: int) -> "RuleGenerationStatus":
        self.discovered_pattern_num += discovered
        self.abandon_pattern_num += abandoned
        return self


class RuleSender:
    """A local stand-in for the original downstream sender / queue."""

    def __init__(self) -> None:
        self.sent_rules: List[Dict[str, object]] = []

    def send_rule(self, payload: Dict[str, object]) -> None:
        self.sent_rules.append(payload)


def zl_rule_equals(left: ZLRule, right: ZLRule) -> bool:
    return _normalize_rule(left) == _normalize_rule(right)


def zl_rule_filter(rules: Sequence[ZLRule], filter_flag: bool = True, min_confidence: float = 0.0) -> List[ZLRule]:
    """Deduplicate rules and keep only those above the confidence threshold."""

    filtered = rule_deduplicate(rules) if filter_flag else list(rules)
    return [rule for rule in filtered if rule.statistics.confidence >= min_confidence]


def rule_deduplicate(rules: Sequence[ZLRule]) -> List[ZLRule]:
    result: List[ZLRule] = []
    seen: set = set()
    for rule in rules:
        key = _normalize_rule(rule)
        if key in seen:
            continue
        seen.add(key)
        result.append(rule)
    return result


def _normalize_rule(rule: ZLRule) -> Tuple[object, ...]:
    stats = rule.statistics
    return (
        tuple(rule.general_keys),
        tuple(tuple(_normalize_value(v) for v in values) for values in rule.values),
        tuple(rule.semantics),
        tuple(rule.is_nans),
        tuple(rule.segment_rules.keys),
        tuple((_normalize_value(low), _normalize_value(high)) for low, high in rule.segment_rules.intervals),
        tuple(rule.segment_rules.is_nans),
        stats.confidence,
        stats.lift,
        stats.freq_union.single_support,
        stats.freq_union.multi_support,
        stats.freq_antecedent.single_support,
        stats.freq_antecedent.multi_support,
    )


def _normalize_value(value: object) -> object:
    if isinstance(value, float):
        if isnan(value):
            return "NaN"
        if isinf(value):
            return "INF" if value > 0 else "-INF"
    return value


def generate_json_graph_info(pattern_view: PatternView, items: Optional[Sequence[str]] = None, delete_items: Optional[Dict[int, RelatedRuleInfo]] = None, delete_edges: Optional[Dict[int, RelatedRuleInfo]] = None) -> Dict[str, object]:
    """Render a pattern plus selected literals into the GAR-style graph view."""

    node_infos: List[Dict[str, object]] = []
    edge_infos: List[Dict[str, object]] = []
    node_attrs: Dict[int, List[Dict[str, object]]] = {idx: [] for idx in range(len(pattern_view.nodes))}
    edge_attrs: Dict[int, List[Dict[str, object]]] = {idx: [] for idx in range(len(pattern_view.edges))}

    for item_index, raw_item in enumerate(items or []):
        placeholder = pattern_view.placeholder(raw_item)
        delete_info = delete_items.get(item_index) if delete_items else None
        if placeholder["type"] == "vertex":
            payload: Dict[str, object] = {"key": placeholder["key"], "value": raw_item}
            if delete_info:
                payload["link"] = f"({delete_info.pattern_id},{delete_info.node_id},{delete_info.rule_id})"
            node_attrs[placeholder["id"]].append(payload)
        elif placeholder["type"] == "edge":
            payload = {"key": placeholder["key"], "value": raw_item}
            if delete_info:
                payload["link"] = f"({delete_info.pattern_id},{delete_info.node_id},{delete_info.rule_id})"
            for edge_index, edge in enumerate(pattern_view.edges):
                if edge == (placeholder["src"], placeholder["dst"], placeholder["label"]):
                    edge_attrs[edge_index].append(payload)
                    break

    for node_index, label in enumerate(pattern_view.nodes):
        node_infos.append({"id": node_index, "label": label, "attribute": node_attrs[node_index]})
    for edge_index, (src, dst, label) in enumerate(pattern_view.edges):
        edge_info: Dict[str, object] = {"src": src, "dst": dst, "label": label, "attribute": edge_attrs[edge_index]}
        if delete_edges and edge_index in delete_edges:
            info = delete_edges[edge_index]
            edge_info["link"] = f"({info.pattern_id},{info.node_id},{info.rule_id})"
        edge_infos.append(edge_info)
    return {"v": node_infos, "e": edge_infos}


def serialize_rule(pattern_view: PatternView, rule: ZLRule, rule_index: int, x_info: Optional[Dict[str, object]] = None, y_info: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Convert one `ZLRule` into the payload printed by the demo."""

    stats = rule.statistics
    segmentation = rule.segment_rules.to_dict(pattern_view, 0)
    return {
        "id": next(_RULE_ID_COUNTER),
        "pattern_index": pattern_view.pattern_id,
        "node_index": 1,
        "rule_index": rule_index,
        "edge_num": len(pattern_view.edges),
        "attribute_num": len(rule.general_keys),
        "x_support_single": stats.freq_antecedent.single_support,
        "y_support_single": stats.freq_union.single_support,
        "x_support_multiple": stats.freq_antecedent.multi_support,
        "y_support_multiple": stats.freq_union.multi_support,
        "confidence": stats.confidence,
        "lift": stats.lift,
        "status": stats.status,
        "x_info": x_info if x_info is not None else generate_json_graph_info(pattern_view, rule.general_keys),
        "y_info": y_info if y_info is not None else generate_json_graph_info(pattern_view, [rule.y_literal] if rule.y_literal else []),
        "x_instance": rule.instances[0] if rule.instances[0] else None,
        "y_instance": rule.instances[1] if rule.instances[1] else None,
        "rule_weight": [float(len(pattern_view.edges)), float(len(pattern_view.nodes)), 0.0, stats.confidence, stats.lift],
        "segmentation": [segmentation] if segmentation else [],
        "zl_col": rule.zl_literals(pattern_view) or [],
    }


def send_zl_rules(pattern_view: PatternView, rules: Sequence[ZLRule], y_literal: Optional[str] = None, sender: Optional[RuleSender] = None) -> int:
    """Serialize and collect all valid rules for one pattern."""

    sender = sender or RuleSender()
    sent_count = 0
    x_info = generate_json_graph_info(pattern_view, None)
    base_y_info = generate_json_graph_info(pattern_view, [y_literal] if y_literal else [])
    for index, rule in enumerate(rules):
        if not rule.instances[0]:
            continue
        payload = serialize_rule(pattern_view, rule, index, x_info=x_info, y_info=base_y_info if y_literal or rule.y_literal else {})
        sender.send_rule(payload)
        sent_count += 1
    return sent_count


def update_status_after_generate_rules(status: RuleGenerationStatus, pattern_id: int, discovered_rule_num: int, abandon_rule_num: int) -> int:
    """Update simplified counters after one pattern's rules are handled."""

    if discovered_rule_num == 0:
        status.merge_pattern_status(0, 1)
    status.merge_rule_status(discovered_rule_num, abandon_rule_num)
    return status.discovered_rule_num
