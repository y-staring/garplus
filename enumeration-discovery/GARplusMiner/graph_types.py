from __future__ import annotations

"""Core in-memory graph and pattern data structures used by the Python GAR port.

This module is the bridge between:
1. graph loading
2. structural pattern mining
3. predicate mining
4. rule serialization

Overall GAR flow in this Python version:
- load CSV files into `DataGraph`
- run VSpawn / pattern extension to obtain `FrequentPattern`
- expand each pattern instance into literals
- mine predicates / rules on top of those literals
- serialize rules into a GAR-like payload
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union


NodeId = int
EdgeId = int
PatternNodeId = int
Label = Union[int, str]


@dataclass(frozen=True)
class Edge:
    """A data-graph edge with optional attributes."""

    edge_id: EdgeId
    src: NodeId
    dst: NodeId
    label: Label
    attrs: Dict[str, object] = field(default_factory=dict)


@dataclass
class Vertex:
    """A data-graph vertex. In PPI mode each vertex is a `Protein` node."""

    id: NodeId
    label: Label
    attrs: Dict[str, object] = field(default_factory=dict)


@dataclass
class DataGraph:
    """The large input graph on which GAR is mined."""

    vertices: Dict[NodeId, Vertex]
    out_edges: Dict[NodeId, List[Edge]] = field(default_factory=dict)
    in_edges: Dict[NodeId, List[Edge]] = field(default_factory=dict)
    edges_by_id: Dict[EdgeId, Edge] = field(default_factory=dict)
    _next_edge_id: int = 0

    def add_edge(self, src: NodeId, dst: NodeId, label: Label, attrs: Optional[Dict[str, object]] = None) -> EdgeId:
        """Insert one data-graph edge and return its internal edge id."""

        edge = Edge(edge_id=self._next_edge_id, src=src, dst=dst, label=label, attrs=dict(attrs or {}))
        self._next_edge_id += 1
        self.out_edges.setdefault(src, []).append(edge)
        self.in_edges.setdefault(dst, []).append(edge)
        self.edges_by_id[edge.edge_id] = edge
        return edge.edge_id

    def out_neighbors(self, node_id: NodeId) -> List[Edge]:
        return self.out_edges.get(node_id, [])

    def in_neighbors(self, node_id: NodeId) -> List[Edge]:
        return self.in_edges.get(node_id, [])

    def get_edge_by_id(self, edge_id: EdgeId) -> Edge:
        return self.edges_by_id[edge_id]

    def find_edges(self, src: NodeId, dst: NodeId, label: Label) -> List[Edge]:
        """Return all edges matching `(src, dst, label)`."""

        return [edge for edge in self.out_neighbors(src) if edge.dst == dst and edge.label == label]

    def all_edges(self) -> List[Edge]:
        return list(self.edges_by_id.values())


@dataclass(frozen=True)
class PatternEdge:
    """An edge inside a candidate pattern graph."""

    src: PatternNodeId
    dst: PatternNodeId
    label: Label


@dataclass
class GraphPattern:
    """A structural pattern discovered during VSpawn."""

    node_labels: List[Label]
    edges: List[PatternEdge] = field(default_factory=list)
    pattern_id: int = -1
    radius: int = 0

    def node_count(self) -> int:
        return len(self.node_labels)

    def edge_count(self) -> int:
        return len(self.edges)

    def clone(self) -> "GraphPattern":
        return GraphPattern(node_labels=list(self.node_labels), edges=list(self.edges), pattern_id=self.pattern_id, radius=self.radius)

    def out_adj(self, node_idx: PatternNodeId) -> List[PatternEdge]:
        return [edge for edge in self.edges if edge.src == node_idx]

    def in_adj(self, node_idx: PatternNodeId) -> List[PatternEdge]:
        return [edge for edge in self.edges if edge.dst == node_idx]

    def undirected_adj(self, node_idx: PatternNodeId) -> List[PatternNodeId]:
        neighbors: List[PatternNodeId] = []
        for edge in self.edges:
            if edge.src == node_idx:
                neighbors.append(edge.dst)
            elif edge.dst == node_idx:
                neighbors.append(edge.src)
        return neighbors

    def edge_signature(self) -> Tuple[Tuple[int, int, Label], ...]:
        """Simple canonical edge signature used for duplicate detection."""

        return tuple(sorted((edge.src, edge.dst, edge.label) for edge in self.edges))

    def canonical_code(self) -> Tuple[Tuple[Label, ...], Tuple[Tuple[int, int, Label], ...]]:
        return tuple(self.node_labels), self.edge_signature()

    def has_edge(self, src: int, dst: int, label: Label) -> bool:
        return any(edge.src == src and edge.dst == dst and edge.label == label for edge in self.edges)

    def bfs_radius(self, root: int = 0) -> List[int]:
        """Compute distances to the pivot/root node inside the pattern."""

        if not self.node_labels:
            return []
        radius = [-1] * self.node_count()
        radius[root] = 0
        queue = [root]
        while queue:
            node = queue.pop(0)
            for nxt in self.undirected_adj(node):
                if radius[nxt] == -1:
                    radius[nxt] = radius[node] + 1
                    queue.append(nxt)
        return radius

    def refresh_radius(self, root: int = 0) -> None:
        radii = self.bfs_radius(root)
        self.radius = max(radii) if radii else 0


@dataclass(frozen=True)
class EdgePattern:
    """One reusable edge type seen around a node label in the data graph."""

    edge_label: Label
    target_label: Label
    direction: str = "out"


@dataclass(frozen=True)
class SpawnEdge:
    """A concrete pattern-expansion action proposed by VSpawn."""

    from_node: int
    to_node: int
    edge_label: Label
    target_label: Label
    direction: str
    external: bool


@dataclass
class GraphInstance:
    """One matched embedding of a pattern in the data graph."""

    node_map: Dict[PatternNodeId, NodeId]
    edge_ids: Tuple[Tuple[NodeId, NodeId, Label], ...] = field(default_factory=tuple)
    pivot: Optional[NodeId] = None
    edge_bindings: Dict[int, EdgeId] = field(default_factory=dict)

    def contains_data_node(self, node_id: NodeId) -> bool:
        return node_id in self.node_map.values()

    def contains_data_edge(self, src: NodeId, dst: NodeId, label: Label) -> bool:
        return (src, dst, label) in self.edge_ids

    def get_edge_id(self, pattern_edge_index: int) -> Optional[EdgeId]:
        return self.edge_bindings.get(pattern_edge_index)


@dataclass
class FrequentPattern:
    """A frequent pattern together with all currently collected matches."""

    pattern: GraphPattern
    instances: List[GraphInstance]
    sampled: bool = False
    total_single_support: int = 0
    total_multi_support: int = 0

    def single_support(self) -> int:
        """Support after deduplicating by pivot node."""

        return len({instance.pivot for instance in self.instances if instance.pivot is not None}) or len(self.instances)

    def multi_support(self) -> int:
        """Raw number of matched instances, without pivot deduplication."""

        return len(self.instances)

    def pivots(self) -> Set[NodeId]:
        return {instance.pivot for instance in self.instances if instance.pivot is not None}


@dataclass
class PatternOptions:
    """A small subset of the original Go pattern-mining parameters."""

    pattern_support_threshold: int = 3
    max_add_edge: int = 4
    max_radius: int = 1
    insipid_edge_limit: int = 1
    node_max_add_edge: int = 2
    full_solution: bool = True
    max_multi_support: int = 20_000_000
    timeout_seconds: int = 1
    timeout_vf3_seconds: int = 15
    parallel_edge: bool = True


@dataclass
class LiteralRecord:
    """A single literal extracted from one matched instance."""

    key: str
    value: object
    entity: str


def _iter_literal_values(value: object) -> Iterable[object]:
    """Normalize scalars / lists into a clean literal stream."""

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        items = [item for item in value if item not in (None, '', '-', 'NA', 'N/A')]
        return items
    if value in ('', '-', 'NA', 'N/A'):
        return []
    return [value]


def instance_literals(graph: DataGraph, pattern: GraphPattern, instance: GraphInstance) -> List[LiteralRecord]:
    """Bridge structural matching and predicate mining.

    After pattern matching finds one concrete `(pattern -> graph)` mapping, this function
    turns the mapped nodes / edges into attribute literals such as `v0.organism_name` or
    `e0.throughput`.
    """

    records: List[LiteralRecord] = []
    for pattern_idx, data_node in instance.node_map.items():
        vertex = graph.vertices[data_node]
        for key, value in vertex.attrs.items():
            for one in _iter_literal_values(value):
                records.append(LiteralRecord(key=key, value=one, entity=f"v{pattern_idx}"))

    for pattern_edge_index, pattern_edge in enumerate(pattern.edges):
        edge_id = instance.get_edge_id(pattern_edge_index)
        edge = graph.get_edge_by_id(edge_id) if edge_id is not None else None
        if edge is None:
            src = instance.node_map.get(pattern_edge.src)
            dst = instance.node_map.get(pattern_edge.dst)
            if src is None or dst is None:
                continue
            matches = graph.find_edges(src, dst, pattern_edge.label)
            if not matches:
                continue
            edge = matches[0]
        for key, value in edge.attrs.items():
            for one in _iter_literal_values(value):
                records.append(LiteralRecord(key=key, value=one, entity=f"e{pattern_edge_index}"))
    return records


def derive_edge_types(data_graph: DataGraph) -> Dict[Label, List[EdgePattern]]:
    """Summarize which edge types can be used to expand a node label."""

    result: Dict[Label, Set[EdgePattern]] = {}
    for src, edges in data_graph.out_edges.items():
        src_label = data_graph.vertices[src].label
        bucket = result.setdefault(src_label, set())
        for edge in edges:
            bucket.add(EdgePattern(edge_label=edge.label, target_label=data_graph.vertices[edge.dst].label, direction="out"))
    for dst, edges in data_graph.in_edges.items():
        dst_label = data_graph.vertices[dst].label
        bucket = result.setdefault(dst_label, set())
        for edge in edges:
            bucket.add(EdgePattern(edge_label=edge.label, target_label=data_graph.vertices[edge.src].label, direction="in"))
    return {label: sorted(list(edges), key=lambda x: (str(x.direction), str(x.edge_label), str(x.target_label))) for label, edges in result.items()}
