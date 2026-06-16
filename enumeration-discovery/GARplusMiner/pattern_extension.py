from __future__ import annotations

"""Pattern extension (VSpawn) for the Python GAR port.

This module is the structural-mining half of GAR:
1. choose eligible nodes in the current pattern
2. propose candidate edges around those nodes
3. grow the pattern by one edge
4. match the grown pattern back to the large graph
5. keep only the frequent candidates
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set, Any

from graph_types import DataGraph, EdgePattern, FrequentPattern, GraphPattern, PatternEdge, PatternOptions, SpawnEdge, derive_edge_types
# from vf3_like import find_matches_with_limit
from vf3_linux import find_matches_with_limit

@dataclass
class SpawnStats:
    """Reserved statistics container for debugging or reporting VSpawn."""

    generated_pattern: int = 0
    local_instance_num: int = 0
    global_instance_num: int = 0


@dataclass
class GraphSpawn:
    """Top-level VSpawn controller for one mining run."""

    data_graph: DataGraph
    pattern_set: List[FrequentPattern]
    spawn_radius: int = 0
    options: PatternOptions = field(default_factory=PatternOptions)
    _known_codes: Set[object] = field(default_factory=set)
    _next_pattern_id: int = 1
    pattern_bn: Optional[Any] = None

    def __post_init__(self) -> None:
        for item in self.pattern_set:
            self._known_codes.add(_pattern_code(item.pattern, getattr(self.options, "undirected_pattern", False)))
            if item.pattern.pattern_id < 0:
                item.pattern.pattern_id = self._request_pattern_id()

    def _request_pattern_id(self) -> int:
        pattern_id = self._next_pattern_id
        self._next_pattern_id += 1
        return pattern_id

    def unstoppable(self) -> bool:
        """Whether another radius-expansion round is still allowed."""

        return self.spawn_radius < self.options.max_radius

    def vspawn(self) -> List[FrequentPattern]:
        """Run one VSpawn round at the current radius."""

        generated: List[FrequentPattern] = []
        base_patterns = list(self.pattern_set)
        for frequent_pattern in base_patterns:
            spawn_nodes = get_spawn_nodes(
                frequent_pattern.pattern,
                self.spawn_radius,
                self.options.node_max_add_edge,
                self.options.max_add_edge,
            )
            if not spawn_nodes:
                continue
            spawner = GraphSpawner(
                data_graph=self.data_graph,
                base_pattern=frequent_pattern,
                spawn_nodes=spawn_nodes,
                options=self.options,
                known_codes=self._known_codes,
                pattern_id_factory=self._request_pattern_id,
                pattern_bn=self.pattern_bn,
            )
            generated.extend(spawner.grow())
        self.pattern_set.extend(generated)
        self.spawn_radius += 1
        return generated


def _pattern_code(pattern: GraphPattern, undirected: bool):
    if undirected and hasattr(pattern, "undirected_canonical_code"):
        return pattern.undirected_canonical_code()
    if undirected:
        edges = []
        for edge in pattern.edges:
            left = (pattern.node_labels[edge.src], min(edge.src, edge.dst))
            right = (pattern.node_labels[edge.dst], max(edge.src, edge.dst))
            if str(left) > str(right):
                left, right = right, left
            edges.append((left, right, edge.label))
        return tuple(pattern.node_labels), tuple(sorted(edges, key=str))
    return pattern.canonical_code()

def get_spawn_nodes(pattern: GraphPattern, spawn_radius: int, node_max_add_edge: int, max_add_edge: int) -> List[int]:
    """Choose which pattern nodes may still spawn new edges at this radius."""

    radii = pattern.bfs_radius(0)
    total_added_edge = 0
    spawn_nodes: List[int] = []
    for node_idx, radius in enumerate(radii):
        if radius != spawn_radius:
            continue
        edge_num = sum(1 for neighbor in pattern.undirected_adj(node_idx) if radii[neighbor] >= spawn_radius)
        total_added_edge += edge_num
        if edge_num < node_max_add_edge:
            spawn_nodes.append(node_idx)
    if total_added_edge >= max_add_edge:
        return []
    return sorted(spawn_nodes)


@dataclass
class GraphSpawner:
    """Expand one base frequent pattern by repeatedly adding one edge."""

    data_graph: DataGraph
    base_pattern: FrequentPattern
    spawn_nodes: List[int]
    options: PatternOptions
    known_codes: Set[object]
    pattern_id_factory: Callable[[], int]
    edge_types_map: Dict[object, List[EdgePattern]] = field(init=False)
    pattern_bn: Optional[Any] = None

    def __post_init__(self) -> None:
        self.edge_types_map = derive_edge_types(self.data_graph)

    def grow(self) -> List[FrequentPattern]:
        """Enumerate all frequent descendants reachable from the base pattern."""

        generated: List[FrequentPattern] = []
        queue: List[FrequentPattern] = [self.base_pattern]
        for spawn_node in self.spawn_nodes:
            pending = list(queue)
            while pending:
                freq = pending.pop(0)
                pattern = freq.pattern
                spawn_label = pattern.node_labels[spawn_node]
                spawn_edges: List[SpawnEdge] = []
                for edge_type in self.edge_types_map.get(spawn_label, []):
                    for target_node in find_target_label_nodes(pattern, spawn_node, edge_type):
                        spawn_edges.append(
                            SpawnEdge(
                                from_node=spawn_node,
                                to_node=target_node,
                                edge_label=edge_type.edge_label,
                                target_label=edge_type.target_label,
                                direction=edge_type.direction,
                                external=target_node == -1,
                            )
                        )
                if self.pattern_bn is not None:
                    ranked_spawn_edges = [edge for _, edge in self.pattern_bn.rank_spawn_edges(pattern, spawn_node, spawn_edges)]
                else:
                    ranked_spawn_edges = spawn_edges
                for spawn_edge in ranked_spawn_edges:
                    candidate = self.graph_pattern_add_spawn_edge(freq, spawn_edge)
                    if candidate is None:
                        continue
                    generated.append(candidate)
                    pending.append(candidate)
                    queue.append(candidate)
        return generated

    def graph_pattern_add_spawn_edge(self, base: FrequentPattern, spawn_edge: SpawnEdge) -> Optional[FrequentPattern]:
        """Grow one candidate pattern and re-match it to test support."""

        grown_pattern = pattern_grow(base.pattern, spawn_edge)
        grown_pattern.refresh_radius()
        code = _pattern_code(grown_pattern, self.options.undirected_pattern)
        if code in self.known_codes:
            return None
        if not check_add_edge_constraints(grown_pattern, self.options):
            return None
        matches = find_matches_with_limit(
            grown_pattern,
            self.data_graph,
            None if self.options.full_solution else self.options.max_multi_support,
            pivot_candidates=sorted(base.pivots()) or None,
        )
        if not matches:
            return None
        single_support = len({m.pivot for m in matches if m.pivot is not None}) or len(matches)
        if single_support < self.options.pattern_support_threshold:
            return None
        grown_pattern.pattern_id = self.pattern_id_factory()
        frequent = FrequentPattern(
            pattern=grown_pattern,
            instances=matches,
            sampled=not self.options.full_solution,
            total_single_support=single_support,
            total_multi_support=len(matches),
        )
        self.known_codes.add(code)
        return frequent


def pattern_grow(pattern: GraphPattern, spawn_edge: SpawnEdge) -> GraphPattern:
    """Clone a pattern and add one new edge, possibly introducing a new node."""

    grown = pattern.clone()
    if spawn_edge.external:
        new_node_idx = grown.node_count()
        grown.node_labels.append(spawn_edge.target_label)
        to_node = new_node_idx
    else:
        to_node = spawn_edge.to_node
    if spawn_edge.direction == "out":
        grown.edges.append(PatternEdge(src=spawn_edge.from_node, dst=to_node, label=spawn_edge.edge_label))
    else:
        grown.edges.append(PatternEdge(src=to_node, dst=spawn_edge.from_node, label=spawn_edge.edge_label))
    return grown


def find_target_label_nodes(pattern: GraphPattern, node_index: int, edge_type: EdgePattern) -> List[int]:
    """Try to reuse an existing node first; `-1` means create a new node."""

    connected: Set[int] = set()
    if edge_type.direction == "out":
        for edge in pattern.out_adj(node_index):
            if pattern.node_labels[edge.dst] == edge_type.target_label and edge.label == edge_type.edge_label:
                connected.add(edge.dst)
    else:
        for edge in pattern.in_adj(node_index):
            if pattern.node_labels[edge.src] == edge_type.target_label and edge.label == edge_type.edge_label:
                connected.add(edge.src)
    reuse = [idx for idx, label in enumerate(pattern.node_labels) if idx != node_index and label == edge_type.target_label and idx not in connected]
    reuse.sort()
    reuse.append(-1)
    return reuse


def check_add_edge_constraints(pattern: GraphPattern, options: PatternOptions) -> bool:
    """Apply the local structural constraints used by this simplified VSpawn."""

    radii = pattern.bfs_radius(0)
    for node_idx in range(pattern.node_count()):
        local_edges = 0
        edge_counter: Counter = Counter()
        for edge in pattern.edges:
            if edge.src == node_idx or edge.dst == node_idx:
                other = edge.dst if edge.src == node_idx else edge.src
                if radii[other] >= radii[node_idx]:
                    local_edges += 1
                    edge_counter[(min(node_idx, other), max(node_idx, other), edge.label)] += 1
        if local_edges > options.node_max_add_edge:
            return False
        if options.insipid_edge_limit and any(count > options.insipid_edge_limit for count in edge_counter.values()):
            return False
    radius_edge_count: Counter = Counter()
    for edge in pattern.edges:
        radius_edge_count[radii[edge.src]] += 1
        radius_edge_count[radii[edge.dst]] += 1
    return all(count <= options.max_add_edge for count in radius_edge_count.values())

