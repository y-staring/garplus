import networkx as nx
import time
import signal
import matplotlib.pyplot as plt
from networkx.algorithms.isomorphism import GraphMatcher
import os
import traceback  # <--- 新增

# ================= 配置区 =================
MATCH_LIMIT = 500
TIME_LIMIT = 5
TARGET_FILE = "/home/yyyy/codework/GARplus/DiGress/outputs/2025-12-11/21-57-13-epinions/generated_samples1.txt"
RAW_DATA_FILE = "../data/epinions/raw/epinions_neg_focused.txt" 
# =========================================

class TimeoutException(Exception): pass
def timeout_handler(signum, frame): raise TimeoutException()
signal.signal(signal.SIGALRM, timeout_handler)

def edge_match(d1, d2):
    return d1.get('label') == d2.get('label')

def to_nx_undirected_simple(edge_matrix):
    n = len(edge_matrix)
    G = nx.Graph()
    for i in range(n): G.add_node(i) 
    for i in range(n):
        for j in range(i + 1, n):
            val = edge_matrix[i][j]
            if val != 0: 
                G.add_edge(i, j, label=val)
    return G

def parse_graph_txt(filepath):
    graphs = []
    if not os.path.exists(filepath): return []
    with open(filepath, "r") as f: lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("N="):
            try:
                n = int(line.split("=")[1])
                i += 1
                while i < len(lines) and not lines[i].strip().startswith("X"): i += 1
                if i < len(lines): i += 1
                while i < len(lines) and not lines[i].strip().startswith("E"): i += 1
                edge_matrix = []
                if i < len(lines):
                    i += 1
                    for _ in range(n):
                        if i >= len(lines): break
                        row_vals = list(map(int, lines[i].strip().replace(',', ' ').split()))
                        edge_matrix.append(row_vals)
                        i += 1
                    if len(edge_matrix) == n:
                        graphs.append(edge_matrix)
            except: i += 1
        else:
            i += 1
    return graphs

def load_digress_as_big_graph(filepath):
    BigG = nx.DiGraph()
    if not os.path.exists(filepath):
        print(f"[Error] File not found: {filepath}")
        return BigG
    print(f"Parsing BigGraph from: {filepath} ...")
    matrices = parse_graph_txt(filepath)
    node_offset = 0
    total_edges = 0
    for mat in matrices:
        n = len(mat)
        for r in range(n):
            for c in range(n):
                val = mat[r][c]
                if val != 0:
                    BigG.add_edge(r + node_offset, c + node_offset, label=val)
                    total_edges += 1
        node_offset += n
    print(f"BigGraph Constructed. Nodes: {node_offset}, Edges: {total_edges}")
    return BigG

def compute_structural_confidence(subG, bigG, match_limit, time_limit):
    BIG_GRAPH_NEGATIVE_LABEL = 2
    edges_to_test = list(subG.edges(data=True))
    if not edges_to_test: return [] 

    results = []
    target_candidates = [
        (u, v) for u, v, d in edges_to_test 
        if d.get('label') == BIG_GRAPH_NEGATIVE_LABEL
    ]
    if not target_candidates:
        target_candidates = [(u, v) for u, v, d in edges_to_test][:3]

    for target_u, target_v in target_candidates:
        premiseG = subG.copy()
        premiseG.remove_edge(target_u, target_v)
        if premiseG.number_of_edges() == 0: continue
        # GM = GraphMatcher(premiseG,bigG , node_match=None, edge_match=edge_match)

        GM = GraphMatcher(bigG, premiseG, node_match=None, edge_match=edge_match)
        supp_premise = 0
        supp_negative = 0
        signal.alarm(time_limit)
        status = "Finished"
        edges_status = 0
        
        try:
            for mapping in GM.subgraph_isomorphisms_iter():
                
                # =================================================
                # ★★★ 终极修复：统一修正映射方向 (u 和 v 同时处理) ★★★
                # =================================================
                
                # 定义一个变量存放最终可用的字典
                valid_mapping = None

                # 情况 A: mapping 是正向的 {SubNode -> BigNode}
                # 检查 target_u 和 target_v 是否都是 mapping 的 Key
                if (target_u in mapping) and (target_v in mapping):
                    valid_mapping = mapping

                # 情况 B: mapping 是反向的 {BigNode -> SubNode} (你遇到的情况)
                else:
                    # 创建反转字典: {SubNode: BigNode}
                    inv_map = {v: k for k, v in mapping.items()}
                    
                    # 检查 target_u 和 target_v 是否都是 inv_map 的 Key
                    if (target_u in inv_map) and (target_v in inv_map):
                        valid_mapping = inv_map
                
                # 如果正向反向都找不到这两个点，说明匹配异常，跳过此次循环
                if valid_mapping is None:
                    # print(f"[Skipping] Nodes {target_u},{target_v} not found in mapping keys/values.")
                    continue
                
                # --- 现在我们可以安全地获取 ID 了 ---
                supp_premise += 1
                real_u = valid_mapping[target_u]
                real_v = valid_mapping[target_v]
                # =================================================

                # 检查大图是否存在这条边
                if bigG.has_edge(real_u, real_v):
                    edge_data = bigG.get_edge_data(real_u, real_v)
                    edges_status +=1
                    if edge_data.get('label') == BIG_GRAPH_NEGATIVE_LABEL:
                        supp_negative += 1
                
                if supp_premise >= match_limit:
                    status = "Limit"
                    break
            
            signal.alarm(0)

        except TimeoutException:
            status = "TimeOut"
            
        except Exception as e:
            signal.alarm(0)
            status = "Error"
            # 这里的打印可以保留，万一还有其他错能看到
            print(f"\n[Error] Edge: {target_u}-{target_v} | {type(e).__name__}: {e}")
            # traceback.print_exc() 
            continue
        # print(supp_premise)
        print(edges_status)
        if supp_premise > 0:
            confidence = supp_negative / supp_premise
            results.append({
                'target_edge': (target_u, target_v),
                'confidence': confidence,
                'support_negative': supp_negative,
                'support_shape': supp_premise,
                'status': status
            })
            
    return results

if __name__ == "__main__":
    bigG_directed = load_digress_as_big_graph(RAW_DATA_FILE)
    bigG = bigG_directed.to_undirected()
    
    # ★★★ 检查 BigGraph 是否为空 ★★★
    print(f"DEBUG: BigGraph Nodes: {bigG.number_of_nodes()}")
    print(f"DEBUG: BigGraph Edges: {bigG.number_of_edges()}")
    if bigG.number_of_nodes() == 0:
        print("EXITING: BigGraph is empty.")
        exit()

    edge_matrices = parse_graph_txt(TARGET_FILE)
    print(f"\n{'ID':<4} | {'Sub':<3} | {'Conf':<7} | {'Supp':<7} | {'ShapeHits':<9} | {'Status'}")
    print("-" * 65)
    LABEL_NEG = 2
    for idx, edges in enumerate(edge_matrices):
        full_G = to_nx_undirected_simple(edges)
        components = [full_G.subgraph(c).copy() for c in nx.connected_components(full_G)]
        valid_components = [g for g in components if g.number_of_edges() >= 2]

        for sub_idx, subG in enumerate(valid_components):
            has_negative_edge = False
            for u, v, d in subG.edges(data=True):
                if d.get('label') == LABEL_NEG:
                    has_negative_edge = True
                    break
            
            # 如果全是正边，直接跳过这个分量
            if not has_negative_edge:
                continue
            metrics = compute_structural_confidence(subG, bigG, MATCH_LIMIT, TIME_LIMIT)
            if not metrics: continue
            best_m = max(metrics, key=lambda x: x['confidence'])
            print(f"{idx:<4} | {sub_idx:<3} | {best_m['confidence']:.4f}  | {best_m['support_negative']:<7} | {best_m['support_shape']:<9} | {best_m['status']}")