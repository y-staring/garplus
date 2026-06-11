from __future__ import annotations

"""Predicate mining for the Python GAR port.

For one fixed frequent pattern we:
1. expand every matched instance into literals
2. arrange those literals into either a table or a transaction list
3. prune low-support values / columns
4. mine `X -> Y` rules with either a decision-tree-like heuristic or FP-Growth-like logic
"""

from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

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

    def __init__(self, min_value_support_count: int = 1) -> None:
        self.min_value_support_count = max(1, int(min_value_support_count))

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

    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5, min_value_support_count: int = 1) -> None:
        super().__init__(min_value_support_count=min_value_support_count)
        self.min_support = min_support
        self.min_confidence = min_confidence

    def generate_literal_df(self, graph: DataGraph, frequent_pattern: FrequentPattern, y_key: str) -> List[Dict[str, object]]:
        """Build the per-pattern table and keep only rows that still contain the target."""

        rows = self.build_instance_rows(graph, frequent_pattern)
        rows = self.prune_rows_by_value_support(rows)
        return [row for row in rows if y_key in row]

    def corr_analysis(self, rows: List[Dict[str, object]], y_key: str) -> List[str]:
        """A simple feature-screening heuristic based on equality rate to `y_key`."""

        if not rows:
            return []
        selected: List[str] = []
        for key in rows[0].keys():
            if key == y_key:
                continue
            same = sum(1 for row in rows if row.get(key) == row.get(y_key))
            score = same / len(rows)
            if score >= self.min_support:
                selected.append(key)
        return selected

    def generate_rules(self, graph: DataGraph, frequent_pattern: FrequentPattern, y_key: str) -> List[Rule]:
        """Generate simple `one antecedent -> one target` rules."""

        rows = self.generate_literal_df(graph, frequent_pattern, y_key)
        candidate_keys = self.corr_analysis(rows, y_key)
        if not rows or not candidate_keys:
            return []
        rules: List[Rule] = []
        for key in candidate_keys:
            grouped: Dict[Tuple[object, object], int] = Counter((row.get(key), row.get(y_key)) for row in rows)
            x_count = Counter(row.get(key) for row in rows)
            y_count = Counter(row.get(y_key) for row in rows)
            for (x_value, y_value), pair_count in grouped.items():
                support = pair_count / len(rows)
                confidence = pair_count / x_count[x_value]
                lift = confidence / (y_count[y_value] / len(rows))
                if support >= self.min_support and confidence >= self.min_confidence:
                    rules.append(Rule(antecedent=(f"{key}={x_value}",), consequent=f"{y_key}={y_value}", support=support, confidence=confidence, lift=lift))
        return rules


class FPGrowthPredicateSelector(PredicateTableMixin):
    """A lightweight frequent-itemset selector."""

    def __init__(self, min_support: float = 0.1, min_confidence: float = 0.5, min_value_support_count: int = 1) -> None:
        super().__init__(min_value_support_count=min_value_support_count)
        self.min_support = min_support
        self.min_confidence = min_confidence

    def get_transaction_list(self, graph: DataGraph, frequent_pattern: FrequentPattern) -> List[List[str]]:
        """Convert one pattern's matched instances into transactions."""

        rows = self.build_instance_rows(graph, frequent_pattern)
        rows = self.prune_rows_by_value_support(rows)
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
        threshold = max(1, int(len(transactions) * self.min_support))
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

        transactions = self.get_transaction_list(graph, frequent_pattern)
        itemsets = self.frequent_itemsets(transactions)
        total = len(transactions) or 1
        single_counts = {items[0]: count for items, count in itemsets.items() if len(items) == 1}
        rules: List[Rule] = []
        for items, itemset_count in itemsets.items():
            if len(items) < 2:
                continue
            y_items = [item for item in items if item.startswith(y_prefix)]
            if not y_items:
                continue
            for consequent in y_items:
                antecedent_items = tuple(item for item in items if item != consequent)
                if not antecedent_items:
                    continue
                antecedent_count = single_counts.get(antecedent_items[0], itemset_count) if len(antecedent_items) == 1 else itemsets.get(tuple(sorted(antecedent_items)), itemset_count)
                consequent_count = single_counts.get(consequent, itemset_count)
                confidence = itemset_count / antecedent_count
                support = itemset_count / total
                lift = confidence / (consequent_count / total)
                if support >= self.min_support and confidence >= self.min_confidence:
                    rules.append(Rule(antecedent=tuple(sorted(antecedent_items)), consequent=consequent, support=support, confidence=confidence, lift=lift))
        unique = {}
        for rule in rules:
            unique[(rule.antecedent, rule.consequent)] = rule
        return list(unique.values())
