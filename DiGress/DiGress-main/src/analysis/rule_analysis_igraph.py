#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import networkx as nx
from igraph import Graph
from support_loader import _read_signed_digraph


##########################################################################
# 1) NetworkX → iGraph 转换函数
##########################################################################

def nx_to_igraph(G):
    """
    将 NetworkX 无向图转换为 iGraph 无向图，
    保留节点/边的 label，并构造一个用于颜色剪枝的 color_label。
    注意：NetworkX 节点 ID 可能不是 0..n-1，这里会重新映射。
    """
    # 固定一个节点顺序，并建立 id 映射
    nx_nodes = list(G.nodes())
    n = len(nx_nodes)
    node_index = {nid: i for i, nid in enumerate(nx_nodes)}

    g = Graph()
    g.add_vertices(n)

    # ===== Node attributes =====
    v_label = []
    v_color_label = []
    for nid in nx_nodes:
        lbl = G.nodes[nid].get("label", 0)
        deg = G.degree(nid)
        # 度做个小bin，避免太细导致颜色过多
        deg_bin = min(deg, 9)
        color_lbl = lbl * 10 + deg_bin

        v_label.append(lbl)
        v_color_label.append(color_lbl)

    g.vs["label"] = v_label          # 原始标签（可留作调试）
    g.vs["color_label"] = v_color_label  # 用于 VF2 颜色剪枝

    # ===== Edge structure + attributes =====
    nx_edges = list(G.edges())
    ig_edges = [(node_index[u], node_index[v]) for (u, v) in nx_edges]
    g.add_edges(ig_edges)

    e_label = []
    for (u, v) in nx_edges:
        el = G[u][v].get("label", 1)
        e_label.append(el)
    g.es["label"] = e_label

    return g


##########################################################################
# 2) 子图解析（与原始版本一致）
##########################################################################

def parse_generated_txt(filepath):
    graphs = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("N="):
            n = int(line.split("=")[1])
            i += 1

            # === X ===
            assert lines[i].strip().startswith("X")
            i += 1

            # 1 行节点标签
            node_labels = list(map(int, lines[i].strip().split()))
            i += 1

            # === E ===
            while not lines[i].strip().startswith("E"):
                i += 1
            i += 1

            edge_matrix = []
            for _ in range(n):
                edge_matrix.append(list(map(int, lines[i].strip().split())))
                i += 1

            graphs.append((node_labels, edge_matrix))

        i += 1

    return graphs


##########################################################################
# 3) 生成子图（无向，只添加上三角）
##########################################################################

def to_nx(node_labels, edge_matrix):
    """
    子图始终用 0..n-1 的节点编号，这样转 igraph 会非常干净。
    """
    n = len(node_labels)
    G = nx.Graph()

    # 节点属性
    for i, lbl in enumerate(node_labels):
        G.add_node(i, label=lbl)

    # 边（只遍历 i<j，避免重复）
    for i in range(n):
        for j in range(i + 1, n):
            et = edge_matrix[i][j]
            if et > 0:
                G.add_edge(i, j, label=et)

    return G


##########################################################################
# 4) iGraph 支持度计算（核心）
##########################################################################

def compute_support_igraph(g_sub, g_big):
    """
    使用 igraph 的 VF2++ 引擎计算支持度。

    - 使用 color_label 作为节点颜色（包含原始label+度信息）
    - 使用 label 作为边颜色

    这样所有匹配和剪枝都在 C 里完成，不需要 Python 回调。
    """
    return g_big.count_subisomorphisms_vf2(
        g_sub,
        color1="color_label",   # big graph vertex color attr
        color2="color_label",   # sub graph vertex color attr
        edge_color1="label",
        edge_color2="label",
    )


##########################################################################
# 5) 主流程
##########################################################################

if __name__ == "__main__":

    # ===== 1) 读取子图模式 =====
    graphs = parse_generated_txt("analysis/generated_samples1.txt")

    # ===== 2) 读大图 =====
    bigG = _read_signed_digraph("../data/epinions/raw/soc-sign-epinions.txt")

    # 如果读取的是 DiGraph，统一转无向
    if bigG.is_directed():
        bigG = bigG.to_undirected()

    print("Loaded big graph nodes:", bigG.number_of_nodes())
    print("Loaded big graph edges:", bigG.number_of_edges())

    # ===== 3) 添加 NetworkX 端的 label 属性 =====
    num_bins = 10
    deg = {n: bigG.degree(n) for n in bigG.nodes()}
    for n, d in deg.items():
        bigG.nodes[n]["label"] = min(d, num_bins - 1)

    # 边 label（正=2，负=1）
    for u, v, data in bigG.edges(data=True):
        s = data.get("sign", 1)
        data["label"] = 2 if s >= 0 else 1

    # ===== 4) 转为 iGraph（大图只转一次） =====
    print("Converting big graph to igraph...")
    g_big = nx_to_igraph(bigG)
    print("Convert done. igraph big graph:",
          g_big.vcount(), "nodes,", g_big.ecount(), "edges")

    # ===== 5) 遍历子图计算支持度 + cache =====
    pattern_cache = {}
    results = []

    for idx, (nodes, edge_matrix) in enumerate(graphs):
        # 用 (节点标签, 边矩阵) 作为模式的 key，避免重复计算
        key = (
            tuple(nodes),
            tuple(tuple(row) for row in edge_matrix)
        )

        if key in pattern_cache:
            sup = pattern_cache[key]
        else:
            subG_nx = to_nx(nodes, edge_matrix)
            g_sub = nx_to_igraph(subG_nx)

            sup = compute_support_igraph(g_sub, g_big)
            pattern_cache[key] = sup

        results.append((idx, sup))
        print(f"[Graph {idx}] Support = {sup}")

    # ===== 6) 输出 top-k =====
    results.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 10 most frequent patterns:")
    for item in results[:10]:
        print(item)
