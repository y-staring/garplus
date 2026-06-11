# GAR Python Port

这个目录是把你刚才关心的几块 Go 代码，按“便于单机阅读和继续开发”的方式，整理成一个独立 Python 版本。

## 文件对应关系

- `graph_types.py`
  - 对应 Go 里的图、pattern、instance、support 相关数据结构
- `vf3_like.py`
  - 对应 `vf3/*` 与 `structure/pattern/*` 里的子图匹配主逻辑
- `pattern_extension.py`
  - 对应 `patternmining/fsm/graph_vspawn.go`
  - 对应 `patternmining/fsm/graph_spawner.go`
- `predicate_selection.py`
  - 对应 `rulemining/procedure.go`
  - 对应 `rulemining/decisiontreemining/mining.go`
  - 对应 `rulemining/fpgrowth/fp_producer.go`
- `rulegeneration.py`
  - 对应 `rulegeneration/rule.go`
  - 对应 `rulegeneration/rule_filter.go`
  - 对应 `rulegeneration/rule_proc.go`
  - 对应 `rulegeneration/rule_send.go`

## 重点说明

- 这里保留的是 **算法主干**，不是完整分布式移植版。
- Go 版本里的 RPC、Kafka、ETCD、分布式 worker 同步，这里没有搬。
- `vf3_like.py` 是一个便于阅读的纯 Python 回溯匹配器，作用上对应 VF3，但不是逐行机械翻译。
- `predicate_selection.py` 里给了两个入口：
  - `DecisionTreePredicateSelector`
  - `FPGrowthPredicateSelector`
- `rulegeneration.py` 里给了规则对象、过滤、序列化和发送接口

## 最小使用方式

```python
from gar_python_port.graph_types import DataGraph, FrequentPattern, GraphPattern, PatternOptions, Vertex
from gar_python_port.pattern_extension import GraphSpawn

graph = DataGraph(
    vertices={
        1: Vertex(1, "Person", {"age": 18, "city": "SZ"}),
        2: Vertex(2, "Movie", {"genre": "Action"}),
    }
)
graph.add_edge(1, 2, "likes")

seed = FrequentPattern(
    pattern=GraphPattern(node_labels=["Person"]),
    instances=[],
)

vspawn = GraphSpawn(graph, [seed], options=PatternOptions(pattern_support_threshold=1, max_radius=2))
new_patterns = vspawn.vspawn()
```

## 运行完整 demo

```bash
python -m gar_python_port.demo
```

这个 demo 会依次演示：

- 构建一个小型属性图
- 执行 `VSpawn` 生成 `Person -likes-> Movie` pattern
- 分别运行决策树风格和 FP-Growth 风格的谓词选择
- 把选出的规则转成 `ZLRule`
- 调用 `rulegeneration.py` 做过滤、序列化和发送

## 你接下来最适合补的部分

- 把属性约束也纳入 pattern matching
- 把 FP-Growth 从“两项集规则”扩成完整条件树版本
- 把决策树版从启发式筛选，扩成真正树训练
- 给这套 Python 版补一个真实数据集驱动的 demo


## Run on PPI CSV

```bash
python -m gar_python_port.ppi_demo --mode pattern-only
python -m gar_python_port.ppi_demo --mode decision-tree --max-rows 1000 --y-key v0.high_degree
python -m gar_python_port.ppi_demo --mode fp-growth --max-rows 1000 --y-key v0.high_degree
```

Notes:
- `protein_protein.csv` is an interaction edge table, so this loader turns every protein into a `Protein` vertex.
- The current pure-Python matcher is expensive on large graphs, so start with `--max-rows 500` or `--max-rows 1000`.
- To make predicate/rule mining runnable on PPI, the loader derives `degree`, `degree_bucket`, and `high_degree` as node attributes.


## GARplusMiner BN version

This folder is the BN-guided GAR+ implementation. It intentionally stays separate from `baselines/GAR`.

New files:
- `pattern_bn.py`: learns a lightweight Pattern BN from graph edge/node-label co-occurrence and ranks VSpawn expansion edges.
- `predicate_bn.py`: learns a lightweight Predicate BN from the flattened match table and ranks/prunes predicate features before rule mining.

Main switches in `ppi_demo.py`:
- `ENABLE_PATTERN_BN`: enable Pattern-BN-guided VSpawn.
- `PATTERN_BN_TOP_K_PER_SPAWN_NODE`: keep only the top-K structural expansion actions per spawn node.
- `ENABLE_PREDICATE_BN`: enable Predicate-BN-guided predicate selection.
- `PREDICATE_BN_TOP_K_FEATURES`: keep only the top-K predicate columns associated with the target.
- `Y_KEY`: the rule consequent key. For link prediction with a positive/negative interaction column, use the normalized edge key, e.g. `e0.interaction_label`.

Recommended link-prediction setup:
- Keep the interaction type / experimental system as the structural edge label.
- Put positive/negative in an edge attribute column, not as the pattern edge label.
- Set `Y_KEY` to that edge attribute, so generated rules become `X -> e0.xxx=positive` or `X -> e0.xxx=negative`.
