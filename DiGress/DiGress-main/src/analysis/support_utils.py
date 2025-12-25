# src/analysis/support_utils.py
import networkx as nx

def compute_support(G_large, G_sub):
    """
    return: int = number of matches
    """

    # 1) 匹配规则：边 sign 相同才算匹配
    def edge_match(d1, d2):
        # d1,d2: {'sign': +1/-1}
        return d1.get('sign') == d2.get('sign')

    # 2) 调用 VF2 匹配算法
    matcher = nx.algorithms.isomorphism.DiGraphMatcher(
        G_large, G_sub, edge_match=edge_match
    )

    # 3) 所有匹配个数 = 支持度
    matches = list(matcher.subgraph_isomorphisms_iter())
    return len(matches)


def degree_filter(G_large, G_sub):
    deg_large = sorted([d for _,d in G_large.degree()])
    deg_sub = sorted([d for _,d in G_sub.degree()])
    return max(deg_sub) <= max(deg_large)


def sign_distribution(G):
    pos = 0
    neg = 0
    for (_,_,d) in G.edges(data=True):
        if d.get("sign",1) > 0:
            pos += 1
        else:
            neg += 1
    return pos,neg


# src/analysis/support_utils.py
def compute_support_fast(G_large, G_sub):

    # Quick filters
    if not degree_filter(G_large, G_sub):
        return 0

    pos_L, neg_L = sign_distribution(G_large)
    pos_S, neg_S = sign_distribution(G_sub)

    if (pos_S > pos_L) or (neg_S > neg_L):
        return 0

    # VF2 match
    return compute_support(G_large, G_sub)


def batch_support(G_large, list_of_subgraphs, top_k=None):
    """
    list_of_subgraphs = list of networkx graphs
    return list of supports
    """
    supports = []
    for G_sub in list_of_subgraphs:
        s = compute_support_fast(G_large, G_sub)
        supports.append(s)

    # 排序（选 top_k 最频繁子图）
    if top_k is not None:
        supports = sorted(supports, reverse=True)[:top_k]

    return supports


def quick_filter(G_large, G_sub):
    # sign distribution
    posL = sum(1 for (_,_,d) in G_large.edges(data=True) if d.get("sign",1)>0)
    negL = sum(1 for (_,_,d) in G_large.edges(data=True) if d.get("sign",1)<=0)
    posS = sum(1 for (_,_,d) in G_sub.edges(data=True) if d.get("sign",1)>0)
    negS = sum(1 for (_,_,d) in G_sub.edges(data=True) if d.get("sign",1)<=0)

    if posS > posL or negS > negL:
        return False

    if G_sub.number_of_nodes() > G_large.number_of_nodes():
        return False

    return True


def compute_support_per_graph(G_large, nx_graphs):
    """
    nx_graphs: list of networkx graphs (generated subgraphs)
    return:
    [
      {"id": int, "support": int},
      ...
    ]
    """
    results = []

    for i, G_sub in enumerate(nx_graphs):

        # speed filter
        if not quick_filter(G_large, G_sub):
            results.append({"id": i, "support": 0})
            continue
        
        s = compute_support(G_large, G_sub)
        results.append({"id": i, "support": s})

    return results


