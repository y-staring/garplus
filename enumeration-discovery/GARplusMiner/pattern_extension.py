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
import time
from typing import Callable, Dict, List, Optional, Set, Any
from itertools import permutations
from graph_types import DataGraph, EdgePattern, FrequentPattern, GraphInstance, GraphPattern, PatternEdge, PatternOptions, SpawnEdge, derive_edge_types
# from vf3_like import find_matches_with_limit
from vf3_linux import find_matches_with_limit

@dataclass
class SpawnStats:
    """Reserved statistics container for debugging or reporting VSpawn."""

    generated_pattern: int = 0
    local_instance_num: int = 0
    global_instance_num: int = 0
    candidates_seen: int = 0
    bn_pruned: int = 0
    duplicate_pruned: int = 0
    constraint_pruned: int = 0
    no_match_pruned: int = 0
    support_pruned: int = 0


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
    stats: SpawnStats = field(default_factory=SpawnStats)

    def __post_init__(self) -> None:
        for item in self.pattern_set:
            if self.options.topology_only_dedup:
                code = topology_pattern_code(item.pattern, self.options.topology_dedupe_respect_direction)
            else:
                code = _pattern_code(item.pattern, self.options.undirected_pattern)
            self._known_codes.add(code)
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
        if self.options.extension_debug:
            print(
                f"[PatternExtension/RoundStart] radius={self.spawn_radius} "
                f"frontier={len(base_patterns)} known={len(self._known_codes)}"
            )
        for frequent_pattern in base_patterns:
            if self.options.global_vspawn_instances:
                matches = find_matches_with_limit(frequent_pattern.pattern, self.data_graph, None)
                single_support = len({match.pivot for match in matches if match.pivot is not None}) or len(matches)
                frequent_pattern.instances = matches
                frequent_pattern.total_single_support = single_support
                frequent_pattern.total_multi_support = len(matches)
                frequent_pattern.sampled = False
                if self.options.extension_debug:
                    print(
                        f"[PatternExtension/GlobalParentMatch] pattern_id={frequent_pattern.pattern.pattern_id} "
                        f"single={single_support} multi={len(matches)}"
                    )
                if single_support < self.options.pattern_support_threshold:
                    continue
            spawn_nodes = get_spawn_nodes(
                frequent_pattern.pattern,
                self.spawn_radius,
                self.options.node_max_add_edge,
                self.options.max_add_edge,
            )
            if not spawn_nodes:
                if self.options.extension_debug:
                    print(
                        f"[PatternExtension/BaseSkip] radius={self.spawn_radius} "
                        f"pattern={_pattern_description(frequent_pattern.pattern)} reason=no_spawn_nodes"
                    )
                continue
            if self.options.extension_debug:
                print(
                    f"[PatternExtension/Base] radius={self.spawn_radius} spawn_nodes={spawn_nodes} "
                    f"single={frequent_pattern.single_support()} multi={frequent_pattern.multi_support()} "
                    f"pattern={_pattern_description(frequent_pattern.pattern)}"
                )
            spawner = GraphSpawner(
                data_graph=self.data_graph,
                base_pattern=frequent_pattern,
                spawn_nodes=spawn_nodes,
                options=self.options,
                known_codes=self._known_codes,
                pattern_id_factory=self._request_pattern_id,
                pattern_bn=self.pattern_bn,
                stats=self.stats,
            )
            generated.extend(spawner.grow())
        # Go appendToPatternSet only carries patterns that reached the next radius.
        frontier = [
            item
            for item in generated
            if item.pattern.radius > self.spawn_radius and item.pattern.radius < self.options.max_radius
        ]
        self.pattern_set = frontier
        if self.options.extension_debug:
            print(
                f"[PatternExtension/RoundEnd] radius={self.spawn_radius} generated={len(generated)} "
                f"next_frontier={len(frontier)} stats={self.stats}"
            )
        self.spawn_radius += 1
        return generated


# def _pattern_code(pattern: GraphPattern, undirected: bool):
#     if undirected and hasattr(pattern, "undirected_canonical_code"):
#         return pattern.undirected_canonical_code()
#     if undirected:
#         edges = []
#         for edge in pattern.edges:
#             left = (pattern.node_labels[edge.src], min(edge.src, edge.dst))
#             right = (pattern.node_labels[edge.dst], max(edge.src, edge.dst))
#             if str(left) > str(right):
#                 left, right = right, left
#             edges.append((left, right, edge.label))
#         return tuple(pattern.node_labels), tuple(sorted(edges, key=str))
#     return pattern.canonical_code()

def _pattern_code(
    pattern: GraphPattern,
    undirected: bool,
    rooted: bool = True,
    ignore_node_labels: bool = False,
    ignore_edge_labels: bool = False,
):
    """
    Canonical code for duplicate pruning.

    rooted=True means node 0 is treated as the root/pivot and must stay node 0.
    This is safer for your current code, because bfs_radius(0) and pivot support
    both make node 0 special.

    If you want to merge purely structural duplicates regardless of root position,
    set rooted=False.
    """

    n = pattern.node_count()
    labels = tuple("*" for _ in pattern.node_labels) if ignore_node_labels else tuple(str(label) for label in pattern.node_labels)

    best_code = None

    for perm in permutations(range(n)):
        # perm[new_idx] = old_idx
        # rooted=True: old node 0 must still be new node 0
        if rooted and n > 0 and perm[0] != 0:
            continue

        old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(perm)}

        new_labels = tuple(labels[old_idx] for old_idx in perm)

        new_edges = []
        for edge in pattern.edges:
            src = old_to_new[edge.src]
            dst = old_to_new[edge.dst]
            edge_label = "*" if ignore_edge_labels else str(edge.label)

            if undirected and src > dst:
                src, dst = dst, src

            new_edges.append((src, dst, edge_label))

        code = (new_labels, tuple(sorted(new_edges)))

        if best_code is None or code < best_code:
            best_code = code

    return best_code


def topology_pattern_code(pattern: GraphPattern, respect_direction: bool = False):
    """Exact topology key: ignore labels and generation order, optionally direction."""

    return _pattern_code(
        pattern,
        undirected=not respect_direction,
        rooted=False,
        ignore_node_labels=True,
        ignore_edge_labels=True,
    )

def _pattern_description(pattern: GraphPattern) -> str:
    edges = [(edge.src, edge.dst, str(edge.label)) for edge in pattern.edges]
    return f"id={pattern.pattern_id} nodes={list(map(str, pattern.node_labels))} edges={edges} radius={pattern.radius}"

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
    stats: SpawnStats = field(default_factory=SpawnStats)
    _debug_count: int = 0
    _debug_suppressed: bool = False

    def __post_init__(self) -> None:
        self.edge_types_map = derive_edge_types(self.data_graph)

    def _debug(self, message: str) -> None:
        if not self.options.extension_debug:
            return
        if self._debug_count < self.options.extension_debug_limit:
            print(message)
            self._debug_count += 1
        elif not self._debug_suppressed:
            print(f"[PatternExtension/DebugLimit] limit={self.options.extension_debug_limit} further_events_suppressed=True")
            self._debug_suppressed = True

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
                self.stats.candidates_seen += len(spawn_edges)
                scored_edges = []
                if self.pattern_bn is not None:
                    scored_edges = []
                    for edge in spawn_edges:
                        if hasattr(self.pattern_bn, "score_spawn_edge_components"):
                            components = self.pattern_bn.score_spawn_edge_components(pattern, spawn_node, edge)
                        else:
                            score = self.pattern_bn.score_spawn_edge(pattern, spawn_node, edge)
                            components = {
                                "edge_prob": score,
                                "dst_prob": 1.0,
                                "bn_score": score,
                                "frequent_prior": 0.0,
                                "final_score": score,
                            }
                        scored_edges.append((components, edge))
                    ranked = self.pattern_bn.rank_spawn_edges(pattern, spawn_node, spawn_edges)
                    ranked_spawn_edges = [edge for _, edge in ranked]
                    self.stats.bn_pruned += len(spawn_edges) - len(ranked_spawn_edges)
                else:
                    ranked_spawn_edges = spawn_edges
                kept = set(ranked_spawn_edges)
                self._debug(
                    f"[PatternExtension/Candidates] spawn_node={spawn_node} raw={len(spawn_edges)} "
                    f"bn_kept={len(ranked_spawn_edges)} base={_pattern_description(pattern)}"
                )
                if self.pattern_bn is not None:
                    for components, edge in sorted(scored_edges, key=lambda item: item[0]["final_score"], reverse=True):
                        self._debug(
                            f"[PatternExtension/BN] score={components['final_score']:.8f} "
                            f"bn={components['bn_score']:.8f} edge_prob={components['edge_prob']:.8f} "
                            f"dst_prob={components['dst_prob']:.8f} prior={components['frequent_prior']:.8f} "
                            f"kept={edge in kept} "
                            f"tau_p={self.pattern_bn.config.min_score} edge={edge}"
                        )
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
        if self.options.topology_only_dedup:
            code = topology_pattern_code(grown_pattern, self.options.topology_dedupe_respect_direction)
        else:
            code = _pattern_code(grown_pattern, self.options.undirected_pattern)
        if code in self.known_codes:
            self.stats.duplicate_pruned += 1
            self._debug(f"[PatternExtension/Reject] reason=duplicate edge={spawn_edge} pattern={_pattern_description(grown_pattern)}")
            return None
        if not check_add_edge_constraints(grown_pattern, self.options):
            self.stats.constraint_pruned += 1
            self._debug(f"[PatternExtension/Reject] reason=add_edge_constraint edge={spawn_edge} pattern={_pattern_description(grown_pattern)}")
            return None
        match_limit = None if (self.options.full_solution or self.options.global_vspawn_instances) else self.options.max_multi_support
        match_started = time.perf_counter()
        if base.instances:
            matches = extend_instances_by_spawn_edge(
                base,
                grown_pattern,
                spawn_edge,
                self.data_graph,
                limit=match_limit,
            )
            match_backend = "incremental_add_edge"
        else:
            matches = find_matches_with_limit(
                grown_pattern,
                self.data_graph,
                match_limit,
                pivot_candidates=sorted(base.pivots()) or None,
            )
            match_backend = "full_match_fallback"
        self._debug(
            f"[PatternExtension/Match] backend={match_backend} base_instances={len(base.instances)} "
            f"matches={len(matches)} seconds={time.perf_counter() - match_started:.6f} "
            f"edge={spawn_edge}"
        )
        if not matches:
            self.stats.no_match_pruned += 1
            self._debug(f"[PatternExtension/Reject] reason=no_matches edge={spawn_edge} pattern={_pattern_description(grown_pattern)}")
            return None
        single_support = len({m.pivot for m in matches if m.pivot is not None}) or len(matches)
        if single_support < self.options.pattern_support_threshold:
            self.stats.support_pruned += 1
            self._debug(
                f"[PatternExtension/Reject] reason=support<{self.options.pattern_support_threshold} "
                f"single={single_support} multi={len(matches)} edge={spawn_edge} pattern={_pattern_description(grown_pattern)}"
            )
            return None
        grown_pattern.pattern_id = self.pattern_id_factory()
        frequent = FrequentPattern(
            pattern=grown_pattern,
            instances=matches,
            sampled=not (self.options.full_solution or self.options.global_vspawn_instances),
            total_single_support=single_support,
            total_multi_support=len(matches),
        )
        self.known_codes.add(code)
        self.stats.generated_pattern += 1
        self.stats.local_instance_num += len(matches)
        self.stats.global_instance_num += len(matches)
        self._debug(
            f"[PatternExtension/Accept] single={single_support} multi={len(matches)} "
            f"edge={spawn_edge} pattern={_pattern_description(grown_pattern)}"
        )
        return frequent


def extend_instances_by_spawn_edge(
    base: FrequentPattern,
    grown_pattern: GraphPattern,
    spawn_edge: SpawnEdge,
    graph: DataGraph,
    limit: Optional[int] = None,
) -> List[GraphInstance]:
    """Extend parent embeddings by one edge, matching Go's instancesAddEdge path."""

    results: List[GraphInstance] = []
    new_pattern_edge_index = len(base.pattern.edges)
    new_pattern_node = base.pattern.node_count() if spawn_edge.external else spawn_edge.to_node
    for instance in base.instances:
        from_data_node = instance.node_map.get(spawn_edge.from_node)
        if from_data_node is None:
            continue
        if spawn_edge.direction == "in":
            candidate_edges = graph.in_neighbors(from_data_node)
        else:
            candidate_edges = graph.out_neighbors(from_data_node)
        used_edge_ids = set(instance.edge_bindings.values())
        used_data_nodes = set(instance.node_map.values())
        for edge in candidate_edges:
            if edge.edge_id in used_edge_ids or edge.label != spawn_edge.edge_label:
                continue
            target_data_node = edge.src if spawn_edge.direction == "in" else edge.dst
            target_vertex = graph.vertices.get(target_data_node)
            if target_vertex is None or target_vertex.label != spawn_edge.target_label:
                continue
            if spawn_edge.external:
                if target_data_node in used_data_nodes:
                    continue
            elif instance.node_map.get(spawn_edge.to_node) != target_data_node:
                continue

            node_map = dict(instance.node_map)
            if spawn_edge.external:
                node_map[new_pattern_node] = target_data_node
            edge_bindings = dict(instance.edge_bindings)
            edge_bindings[new_pattern_edge_index] = edge.edge_id
            edge_ids = tuple(sorted(instance.edge_ids + ((edge.src, edge.dst, edge.label),)))
            results.append(
                GraphInstance(
                    node_map=node_map,
                    edge_ids=edge_ids,
                    pivot=instance.pivot if instance.pivot is not None else node_map.get(0),
                    edge_bindings=edge_bindings,
                )
            )
            if limit is not None and len(results) >= limit:
                return results
    return results


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
    # Match Go checkAddEdgeNumConstraint: count an edge at a radius only from
    # endpoints whose neighbor is at the same or a greater radius.
    radius_edge_count: Counter = Counter()
    for node_idx, radius in enumerate(radii):
        for edge in pattern.edges:
            if edge.src != node_idx and edge.dst != node_idx:
                continue
            other = edge.dst if edge.src == node_idx else edge.src
            if radii[other] >= radius:
                radius_edge_count[radius] += 1
    return all(count <= options.max_add_edge for count in radius_edge_count.values())
