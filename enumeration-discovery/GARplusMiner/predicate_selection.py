from __future__ import annotations

"""Predicate mining for the Python GAR port.

For one fixed frequent pattern we:
1. expand every matched instance into literals
2. arrange those literals into either a table or a transaction list
3. prune low-support values / columns
4. mine `X -> Y` rules with either a decision-tree-like heuristic or FP-Growth-like logic
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from graph_types import DataGraph, FrequentPattern, instance_literals


@dataclass
class Rule:
    """A mined predicate rule in `X -> Y` form."""

    antecedent: Tuple[str, ...]
    consequent: str
    support: float
    confidence: float
    lift: float


class PredicateTableMixin:
    """Shared utilities for converting matched instances into mining-ready rows."""

    def __init__(
        self,
        min_value_support_count: int = 1,
        drop_target_values: Optional[set] = None,
        drop_feature_key_tokens: Optional[tuple[str, ...]] = None,
    ) -> None:
        self.min_value_support_count = max(1, int(min_value_support_count))
        self.drop_target_values = set(drop_target_values or [])
        self.drop_feature_key_tokens = tuple(token.lower() for token in (drop_feature_key_tokens or ()) if token)
        self.filtered_feature_keys: set[str] = set()
        self.diagnostics: List[Dict[str, object]] = []

    def min_support_count(self, total_rows: int) -> int:
        """Convert configured support threshold to paper-style absolute support."""

        if self.min_support <= 1:
            return max(1, int(total_rows * self.min_support))
        return max(1, int(self.min_support))

    def reset_diagnostics(self) -> None:
        self.diagnostics = []
        self.filtered_feature_keys = set()

    def record_diagnostic(self, antecedent: Tuple[str, ...], consequent: str, support: int, confidence: float, lift: float, reason: str) -> None:
        self.diagnostics.append(
            {
                "antecedent": antecedent,
                "consequent": consequent,
                "support": support,
                "confidence": confidence,
                "lift": lift,
                "reason": reason,
            }
        )

    def target_value_diagnostics(self, target_value: str, limit: int = 20) -> List[Dict[str, object]]:
        suffix = f"={target_value}"
        rows = [item for item in self.diagnostics if str(item.get("consequent", "")).endswith(suffix)]
        rows.sort(key=lambda item: (float(item["confidence"]), int(item["support"]), float(item["lift"])), reverse=True)
        return rows[:limit]

    def negative_diagnostics(self, limit: int = 20) -> List[Dict[str, object]]:
        return self.target_value_diagnostics("negative", limit=limit)

    def positive_diagnostics(self, limit: int = 20) -> List[Dict[str, object]]:
        return self.target_value_diagnostics("positive", limit=limit)

    def build_instance_rows(self, graph: DataGraph, frequent_pattern: FrequentPattern) -> List[Dict[str, object]]:
        """Convert all matched instances of one pattern into row dictionaries."""

        rows: List[Dict[str, object]] = []
        for instance in frequent_pattern.instances:
            row: Dict[str, object] = {}
            for record in instance_literals(graph, frequent_pattern.pattern, instance):
                literal_key = f"{record.entity}.{record.key}"
                if literal_key not in row:
                    row[literal_key] = record.value
                else:
                    existing = row[literal_key]
                    if isinstance(existing, list):
                        if record.value not in existing:
                            existing.append(record.value)
                    elif existing != record.value:
                        row[literal_key] = [existing, record.value]
            normalized = {key: ("|".join(str(v) for v in value) if isinstance(value, list) else value) for key, value in row.items()}
            rows.append(normalized)
        return rows

    def filter_target_rows(self, rows: List[Dict[str, object]], y_key: str) -> List[Dict[str, object]]:
        """Drop rows whose target value should not participate in rule mining."""

        if not self.drop_target_values:
            return rows
        return [row for row in rows if row.get(y_key) not in self.drop_target_values]

    def is_dropped_feature_key(self, key: str, y_key: Optional[str] = None) -> bool:
        if key == y_key or not self.drop_feature_key_tokens:
            return False
        lowered = key.lower()
        return any(token in lowered for token in self.drop_feature_key_tokens)

    def filter_feature_keys(self, rows: List[Dict[str, object]], y_key: Optional[str] = None) -> List[Dict[str, object]]:
        """Remove configured shortcut/noisy feature columns from mining rows."""

        if not self.drop_feature_key_tokens:
            return rows
        filtered_rows: List[Dict[str, object]] = []
        for row in rows:
            filtered_row: Dict[str, object] = {}
            for key, value in row.items():
                if self.is_dropped_feature_key(key, y_key):
                    self.filtered_feature_keys.add(key)
                    continue
                filtered_row[key] = value
            filtered_rows.append(filtered_row)
        return filtered_rows

    def support_ranked_feature_keys(self, rows: List[Dict[str, object]], y_key: str, exclude: Optional[set] = None) -> List[str]:
        exclude = set(exclude or set())
        counts: Counter = Counter()
        for row in rows:
            for key in row.keys():
                if key != y_key and key not in exclude and not self.is_dropped_feature_key(key, y_key):
                    counts[key] += 1
        return [key for key, _ in counts.most_common(self.extra_candidate_key_count)]

    def prune_rows_by_value_support(self, rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
        """Drop low-support values first, then let empty columns disappear naturally."""

        if not rows:
            return []
        value_counts: Dict[str, Counter] = {}
        for row in rows:
            for key, value in row.items():
                value_counts.setdefault(key, Counter())[value] += 1

        allowed_values: Dict[str, set] = {}
        for key, counts in value_counts.items():
            kept = {value for value, count in counts.items() if count >= self.min_value_support_count}
            if kept:
                allowed_values[key] = kept

        pruned_rows: List[Dict[str, object]] = []
        for row in rows:
            pruned = {key: value for key, value in row.items() if key in allowed_values and value in allowed_values[key]}
            pruned_rows.append(pruned)
        return pruned_rows


class DecisionTreePredicateSelector(PredicateTableMixin):
    """A lightweight decision-tree-style selector."""

    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5, min_value_support_count: int = 1, predicate_bn: Optional[Any] = None, drop_target_values: Optional[set] = None, drop_feature_key_tokens: Optional[tuple[str, ...]] = None) -> None:
        super().__init__(min_value_support_count=min_value_support_count, drop_target_values=drop_target_values, drop_feature_key_tokens=drop_feature_key_tokens)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.predicate_bn = predicate_bn

    def generate_literal_df(self, graph: DataGraph, frequent_pattern: FrequentPattern, y_key: str) -> List[Dict[str, object]]:
        """Build the per-pattern table and keep only rows that still contain the target."""

        rows = self.build_instance_rows(graph, frequent_pattern)
        rows = self.prune_rows_by_value_support(rows)
        rows = [row for row in rows if y_key in row]
        rows = self.filter_target_rows(rows, y_key)
        return self.filter_feature_keys(rows, y_key)

    def corr_analysis(self, rows: List[Dict[str, object]], y_key: str) -> List[str]:
        """A simple feature-screening heuristic based on equality rate to `y_key`."""

        if not rows:
            return []
        selected: List[str] = []
        for key in rows[0].keys():
            if key == y_key or self.is_dropped_feature_key(key, y_key):
                continue
            same = sum(1 for row in rows if row.get(key) == row.get(y_key))
            score = same / len(rows)
            if score >= self.min_support:
                selected.append(key)
        return selected

    def generate_rules(self, graph: DataGraph, frequent_pattern: FrequentPattern, y_key: str) -> List[Rule]:
        """Generate simple `one antecedent -> one target` rules."""

        self.reset_diagnostics()
        rows = self.generate_literal_df(graph, frequent_pattern, y_key)
        if not rows:
            return []
        if self.predicate_bn is not None:
            self.predicate_bn.fit_rows(rows, y_key)
            all_feature_keys = sorted({key for row in rows for key in row.keys() if key != y_key})
            candidate_keys = [key for _, key in self.predicate_bn.rank_feature_keys(rows, all_feature_keys)]
            for dropped_key in sorted(set(all_feature_keys) - set(candidate_keys)):
                for row in rows:
                    if row.get(y_key) == "negative" and dropped_key in row:
                        self.record_diagnostic((f"{dropped_key}={row[dropped_key]}",), f"{y_key}=negative", 0, 0.0, 0.0, "filtered_by_predicate_bn")
                        break
        else:
            candidate_keys = self.corr_analysis(rows, y_key)
        if not candidate_keys:
            return []
        rules: List[Rule] = []
        support_threshold = self.min_support_count(len(rows))
        for key in candidate_keys:
            grouped: Dict[Tuple[object, object], int] = Counter((row.get(key), row.get(y_key)) for row in rows)
            x_count = Counter(row.get(key) for row in rows)
            y_count = Counter(row.get(y_key) for row in rows)
            for (x_value, y_value), pair_count in grouped.items():
                support = pair_count
                confidence = pair_count / x_count[x_value]
                lift = confidence / (y_count[y_value] / len(rows))
                consequent = f"{y_key}={y_value}"
                antecedent = (f"{key}={x_value}",)
                failed_reasons = []
                if support < support_threshold:
                    failed_reasons.append(f"support<{support_threshold}")
                if confidence < self.min_confidence:
                    failed_reasons.append(f"confidence<{self.min_confidence}")
                if failed_reasons:
                    self.record_diagnostic(antecedent, consequent, support, confidence, lift, ";".join(failed_reasons))
                else:
                    rules.append(Rule(antecedent=antecedent, consequent=consequent, support=support, confidence=confidence, lift=lift))
        if self.predicate_bn is not None:
            rules = self.predicate_bn.rank_rules(rules)
        return rules


class FPGrowthPredicateSelector(PredicateTableMixin):
    """A lightweight frequent-itemset selector."""

    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5, min_value_support_count: int = 1, predicate_bn: Optional[Any] = None, drop_target_values: Optional[set] = None, drop_feature_key_tokens: Optional[tuple[str, ...]] = None) -> None:
        super().__init__(min_value_support_count=min_value_support_count, drop_target_values=drop_target_values, drop_feature_key_tokens=drop_feature_key_tokens)
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.predicate_bn = predicate_bn

    def get_transaction_list(self, graph: DataGraph, frequent_pattern: FrequentPattern) -> List[List[str]]:
        """Convert one pattern's matched instances into transactions."""

        rows = self.build_instance_rows(graph, frequent_pattern)
        rows = self.prune_rows_by_value_support(rows)
        rows = self.filter_feature_keys(rows)
        transactions: List[List[str]] = []
        for row in rows:
            transaction = [f"{key}={value}" for key, value in row.items()]
            if transaction:
                transactions.append(sorted(set(transaction)))
        return transactions

    def frequent_itemsets(self, transactions: List[List[str]]) -> Dict[Tuple[str, ...], int]:
        """Count small frequent itemsets. This is not a production FP-tree."""

        if not transactions:
            return {}
        threshold = self.min_support_count(len(transactions))
        counts: Counter = Counter()
        for transaction in transactions:
            for item in transaction:
                counts[(item,)] += 1
        frequent = {items: count for items, count in counts.items() if count >= threshold}
        pairs: Counter = Counter()
        triples: Counter = Counter()
        for transaction in transactions:
            for i in range(len(transaction)):
                for j in range(i + 1, len(transaction)):
                    pairs[(transaction[i], transaction[j])] += 1
                    for k in range(j + 1, len(transaction)):
                        triples[(transaction[i], transaction[j], transaction[k])] += 1
        frequent.update({items: count for items, count in pairs.items() if count >= threshold})
        frequent.update({items: count for items, count in triples.items() if count >= threshold})
        return frequent

    def generate_rules(self, graph: DataGraph, frequent_pattern: FrequentPattern, y_prefix: str) -> List[Rule]:
        """Emit association rules whose consequent belongs to the requested target prefix."""

        self.reset_diagnostics()
        rows = self.prune_rows_by_value_support(self.build_instance_rows(graph, frequent_pattern))
        rows = [row for row in rows if y_prefix in row]
        rows = self.filter_target_rows(rows, y_prefix)
        rows = self.filter_feature_keys(rows, y_prefix)
        if self.predicate_bn is not None:
            self.predicate_bn.fit_rows(rows, y_prefix)
            all_feature_keys = sorted({key for row in rows for key in row.keys() if key != y_prefix})
            kept_feature_keys = {key for _, key in self.predicate_bn.rank_feature_keys(rows, all_feature_keys)}
            for dropped_key in sorted(set(all_feature_keys) - set(kept_feature_keys)):
                for row in rows:
                    if row.get(y_prefix) == "negative" and dropped_key in row:
                        self.record_diagnostic((f"{dropped_key}={row[dropped_key]}",), f"{y_prefix}=negative", 0, 0.0, 0.0, "filtered_by_predicate_bn")
                        break
        else:
            kept_feature_keys = None
        transactions: List[List[str]] = []
        for row in rows:
            transaction = []
            for key, value in row.items():
                if key == y_prefix or kept_feature_keys is None or key in kept_feature_keys:
                    transaction.append(f"{key}={value}")
            if transaction:
                transactions.append(sorted(set(transaction)))
        itemsets = self.frequent_itemsets(transactions)
        total = len(transactions) or 1
        support_threshold = self.min_support_count(total)
        single_counts = {items[0]: count for items, count in itemsets.items() if len(items) == 1}
        rules: List[Rule] = []
        for items, itemset_count in itemsets.items():
            if len(items) < 2:
                continue
            y_items = [item for item in items if item.startswith(y_prefix)]
            if not y_items:
                continue
            for consequent in y_items:
                antecedent_items = tuple(item for item in items if item != consequent and not item.startswith(y_prefix))
                if not antecedent_items:
                    continue
                antecedent_count = single_counts.get(antecedent_items[0], itemset_count) if len(antecedent_items) == 1 else itemsets.get(tuple(sorted(antecedent_items)), itemset_count)
                consequent_count = single_counts.get(consequent, itemset_count)
                confidence = itemset_count / antecedent_count
                support = itemset_count
                lift = confidence / (consequent_count / total)
                antecedent = tuple(sorted(antecedent_items))
                failed_reasons = []
                if support < support_threshold:
                    failed_reasons.append(f"support<{support_threshold}")
                if confidence < self.min_confidence:
                    failed_reasons.append(f"confidence<{self.min_confidence}")
                if failed_reasons:
                    self.record_diagnostic(antecedent, consequent, support, confidence, lift, ";".join(failed_reasons))
                else:
                    rules.append(Rule(antecedent=antecedent, consequent=consequent, support=support, confidence=confidence, lift=lift))
        unique = {}
        for rule in rules:
            unique[(rule.antecedent, rule.consequent)] = rule
        rules = list(unique.values())
        if self.predicate_bn is not None:
            rules = self.predicate_bn.rank_rules(rules)
        return rules

