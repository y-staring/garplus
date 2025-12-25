import networkx as nx
import time
import signal
import matplotlib.pyplot as plt
from networkx.algorithms.isomorphism import GraphMatcher

# ================= 配置区 =================
MATCH_LIMIT = 500
TIME_LIMIT = 5
GENERATED_FILE = "/home/yyyy/codework/GARplus/DiGress/outputs/2025-12-11/21-57-13-epinions/generated_samples1.txt"
RAW_DATA_FILE = "../data/epinions/raw/soc-sign-epinions.txt"
# =========================================

class TimeoutException(Exception): pass
def timeout_handler(signum, frame): raise TimeoutException()
signal.signal(signal.SIGALRM, timeout_handler)

# 纯拓扑匹配
node_match = None
edge_match = None 

import networkx as nx

def split_connected_subgraphs(G: nx.Graph, min_edges: int = 2):
    """
    把 G 按连通分量拆成若干子图。
    - 默认过滤掉边数 < min_edges 的分量（比如只有1条边或全是孤点），
      因为你的算法需要移除目标边后仍有前提边，否则会被跳过。
    """
    subgraphs = []
    for nodes in nx.connected_components(G):
        SG = G.subgraph(nodes).copy()
        if SG.number_of_edges() >= min_edges:
            subgraphs.append(SG)
    return subgraphs




def to_nx_undirected_simple(node_labels, edge_matrix):
    n = len(node_labels)
    G = nx.Graph()
    for i in range(n): G.add_node(i) 
    for i in range(n):
        for j in range(i + 1, n):
            if edge_matrix[i][j] != 0: 
                G.add_edge(i, j)
    return G

def compute_structural_confidence(subG, bigG, match_limit, time_limit):
    BIG_GRAPH_NEGATIVE_LABEL = 1 
    edges_to_test = list(subG.edges())
    
    if not edges_to_test: return [] 

    results = []
    
    # 限制测试边的数量以提速
    for target_u, target_v in edges_to_test[:3]:
        premiseG = subG.copy()
        premiseG.remove_edge(target_u, target_v)
        
        # 🛡️ 修复核心：如果去边后变成散点图（无边），直接跳过
        if premiseG.number_of_edges() == 0:
            continue

        GM = GraphMatcher(bigG, premiseG, node_match=None, edge_match=None)
        
        supp_premise = 0
        supp_negative = 0
        
        signal.alarm(time_limit)
        
        try:
            for mapping in GM.subgraph_isomorphisms_iter():
                supp_premise += 1
                
                # 🛡️ 修复核心：捕获 KeyError
                try:
                    real_u = mapping[target_u]
                    real_v = mapping[target_v]
                except KeyError:
                    # 如果映射出错，说明该样本无效，跳过
                    continue
                
                if bigG.has_edge(real_u, real_v):
                    edge_data = bigG.get_edge_data(real_u, real_v)
                    if edge_data.get('label') == BIG_GRAPH_NEGATIVE_LABEL:
                        supp_negative += 1
                
                if supp_premise >= match_limit:
                    break
            
            signal.alarm(0)
            status = "Limit" if supp_premise >= match_limit else "Exact"

        except TimeoutException:
            status = "HardTimeOut"
        except Exception as e:
            signal.alarm(0)
            status = f"Error" # 忽略具体错误，反正是不重要的图

        if supp_premise > 0:
            confidence = supp_negative / supp_premise
            results.append({
                'target_edge': (target_u, target_v),
                'confidence': confidence,
                'support_negative': supp_negative,
                'support_shape': supp_premise,
                'status': status
            })
            
        if status == "HardTimeOut": break
            
    return results

# ... (Loader 和 parse 函数保持不变) ...
def parse_generated_txt(filepath):
    graphs = []
    try:
        with open(filepath, "r") as f: lines = f.readlines()
    except: return []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("N="):
            n = int(line.split("=")[1]); i+=1
            while i<len(lines) and not lines[i].strip().startswith("X"): i+=1
            if i<len(lines): i+=1; node_labels=list(map(int,lines[i].strip().split())); i+=1
            while i<len(lines) and not lines[i].strip().startswith("E"): i+=1
            if i<len(lines):
                i+=1; edge_matrix=[]
                for _ in range(n): edge_matrix.append(list(map(int,lines[i].strip().split()))); i+=1
                graphs.append((node_labels, edge_matrix))
        else: i+=1
    return graphs

try:
    from support_loader import _read_signed_digraph
except ImportError:
    def _read_signed_digraph(path): return nx.DiGraph()

if __name__ == "__main__":
    print(f"Loading Data...")
    graphs = parse_generated_txt(GENERATED_FILE)
    bigG_directed = _read_signed_digraph(RAW_DATA_FILE)
    bigG = bigG_directed.to_undirected()
    
    NEG_LBL = 1
    POS_LBL = 2
    # 预处理大图标签
    for u, v, data in bigG.edges(data=True):
        if "label" not in data:
            s = data.get("sign", 1)
            data["label"] = NEG_LBL if s < 0 else POS_LBL

    print(f"Ready. Computing Topological Confidence (Split Connected Components)...")
    # 修改表头，增加 SubID
    print(f"{'ID':<4} | {'Sub':<3} | {'Conf':<7} | {'Supp':<7} | {'ShapeHits':<9} | {'Status'}")
    print("-" * 65)

    best_conf = -1
    best_target_edge = None
    best_subG = None  # 直接存储最好的那个子图对象，方便画图
    best_info_str = "" # 用于画图标题

    for idx, (nodes, edges) in enumerate(graphs):
        # 1. 先构建原始的大图（可能包含不连通部分）
        full_G = to_nx_undirected_simple(nodes, edges)
        
        # 2. 核心修改：获取所有连通分量
        # nx.connected_components 返回节点的集合生成器
        # 我们对每个集合通过 subgraph(c).copy() 创建独立的子图对象
        components = [full_G.subgraph(c).copy() for c in nx.connected_components(full_G)]
        
        for sub_idx, subG in enumerate(components):
            # 过滤掉边太少的简单结构（比如单条边或孤立点）
            if subG.number_of_edges() < 2: 
                continue
            
            # 3. 对该连通分量计算置信度
            metrics = compute_structural_confidence(subG, bigG, MATCH_LIMIT, TIME_LIMIT)
            if not metrics: 
                continue
            
            # 找到该分量中表现最好的边
            best_metric = max(metrics, key=lambda x: x['confidence'])
            
            # 打印结果，增加 sub_idx 区分不同连通块
            print(f"{idx:<4} | {sub_idx:<3} | {best_metric['confidence']:.4f}  | {best_metric['support_negative']:<7} | {best_metric['support_shape']:<9} | {best_metric['status']}")

            # 4. 更新全局最优
            if best_metric['confidence'] > best_conf:
                best_conf = best_metric['confidence']
                best_target_edge = best_metric['target_edge']
                best_subG = subG  # 保存这个特定的连通子图
                best_info_str = f"Graph {idx} (Part {sub_idx})"

    # === 可视化最强规则 ===
    if best_subG is not None:
        print(f"\nPlotting Best Rule: {best_info_str} (Conf={best_conf:.2f})...")
        
        plt.figure(figsize=(6, 6))
        # 使用保存的 best_subG 进行布局和绘制
        pos = nx.spring_layout(best_subG)
        
        # 画所有边（实线）
        nx.draw(best_subG, pos, with_labels=True, node_color='lightblue', edge_color='black')
        
        # 高亮预测的负边（红色虚线）
        u, v = best_target_edge
        # 确保这条边在 best_subG 中（理论上一定在，但如果是 removed 的边需要特殊处理）
        # 注意：compute_structural_confidence 中 target_edge 是 subG 里的边
        # 我们画图时，直接画这条虚拟边即可
        nx.draw_networkx_edges(best_subG, pos, edgelist=[(u, v)], edge_color='red', style='dashed', width=2)
        
        plt.title(f"Best Rule: {best_info_str}\nRed Dashed = Predicted Negative\nConfidence = {best_conf:.2f}")
        plt.savefig("best_rule_connected.png")
        print("Saved visualization to 'best_rule_connected.png'")
    else:
        print("\nNo valid rules found.")
