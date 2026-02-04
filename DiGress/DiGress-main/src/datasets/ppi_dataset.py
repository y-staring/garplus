import os
import torch
import pandas as pd
import numpy as np
import networkx as nx
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos
import signal
from networkx.algorithms.isomorphism import GraphMatcher
from collections import Counter

# ==============================================================================
# 0. 全局配置 & GAR+ 谓词编码逻辑
# ==============================================================================

# --- 超时控制 ---
class TimeoutException(Exception): 
    pass

class TimeLimit:
    def __init__(self, seconds):
        self.seconds = seconds
        self.old_handler = signal.SIG_DFL

    def __enter__(self):
        if hasattr(signal, 'SIGALRM'):
            self.old_handler = signal.getsignal(signal.SIGALRM)
            def handler(signum, frame):
                raise TimeoutException()
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, type, value, traceback):
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.old_handler)

# --- 验证相关常量 ---
MATCH_LIMIT = 400  
TIME_LIMIT = 5
ML_THRESHOLD = 0.3



# --- [NEW] 节点特征编码映射 ---
import pandas as pd

# --- [NEW] 节点特征编码映射 ---
NODE_BIT_MAP = {
    'is_kinase': 0,          # +1
    'is_disease_related': 1  # +2
}

# --- [FIXED] 连续位的边特征编码映射（只保留你要的 key） ---
EDGE_BIT_MAP = {
    "is_negative": 0,           # +1  
    "location_match_y": 1,      # +2
    "M_sim": 2,                 # +4
    "physical_interaction": 3,  # +8
    "x_hub_degree": 4,          # +16
    "x_low_betweenness": 5,     # +32
    "Affinity_Capture_MS": 6,   # +64
}


# --- [MODIFIED] 特征维度定义 ---
NUM_NODE_CLASSES = 4
NUM_EDGE_CLASSES = 1 + (2 ** len(EDGE_BIT_MAP))  # 1 + 2^7 = 129


# --- [NEW] 编码函数实现 ---
def encode_node_feature(node_attr: dict) -> int:
    """ 将单个节点的属性字典编码为一个整数特征 N (0-3) """
    n_feature = 0
    # 判定 1: x.is_kinase
    keywords = str(node_attr.get('Keywords', ''))
    families = str(node_attr.get('Protein families', ''))
    if 'Kinase' in keywords or 'ATP-binding' in keywords or 'Kinase' in families:
        n_feature |= (1 << NODE_BIT_MAP['is_kinase'])
    # 判定 2: x.is_disease_related
    if pd.notna(node_attr.get('Involvement in disease')):
        n_feature |= (1 << NODE_BIT_MAP['is_disease_related'])
    return n_feature

def _norm_str(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and pd.isna(v):
        return ""
    return str(v).strip()


def encode_edge_feature(
    id_x,
    id_y,
    node_x_attr: dict,
    node_y_attr: dict,
    edge_row,
    global_stats: dict,
    sim_threshold: float = ML_THRESHOLD
) -> int:
    """
    连续位 bitmask 编码（只使用 EDGE_BIT_MAP 里的 key）：
      - location_match_y
      - M_sim            (edge_row["ml_score"] > sim_threshold)
      - physical_interaction
      - is_negative
      - x_hub_degree
      - x_low_betweenness
      - Affinity_Capture_MS
    """
    e_feature = 0

    # A) location_match_y：canon 位置严格相等
    loc_x = _norm_str(node_x_attr.get(
        "canon_Subcellular_location_CC",
        node_x_attr.get("Subcellular location [CC]", node_x_attr.get("location", ""))
    ))
    loc_y = _norm_str(node_y_attr.get(
        "canon_Subcellular_location_CC",
        node_y_attr.get("Subcellular location [CC]", node_y_attr.get("location", ""))
    ))
    if loc_x and (loc_x == loc_y):
        e_feature |= (1 << EDGE_BIT_MAP["location_match_y"])

    # B) x_hub_degree：x.degree >= global q75(degree)
    try:
        deg_x = float(node_x_attr.get("degree", 0.0))
    except Exception:
        deg_x = 0.0
    q75_deg = float(global_stats.get("degree", {}).get("q75", float("inf")))
    if deg_x >= q75_deg:
        e_feature |= (1 << EDGE_BIT_MAP["x_hub_degree"])

    # C) x_low_betweenness：x.betweenness <= global q25(betweenness)
    try:
        bet_x = float(node_x_attr.get("betweenness_centrality", 0.0))
    except Exception:
        bet_x = 0.0
    q25_bet = float(global_stats.get("betweenness_centrality", {}).get("q25", float("-inf")))
    if bet_x <= q25_bet:
        e_feature |= (1 << EDGE_BIT_MAP["x_low_betweenness"])

    # D/E) edge_row 相关邮票：M_sim / Affinity Capture-MS / physical / negative
    if edge_row:
        # D) M_sim：看 ml_score 是否过阈值
        try:
            ml_score = float(edge_row.get("ml_score", 0.0))
        except Exception:
            ml_score = 0.0
        if ml_score > sim_threshold:
            e_feature |= (1 << EDGE_BIT_MAP["M_sim"])

        # E1) Affinity Capture-MS
        exp_sys = _norm_str(edge_row.get("Experimental System", ""))
        if exp_sys == "Affinity Capture-MS":
            e_feature |= (1 << EDGE_BIT_MAP["Affinity_Capture_MS"])

        # E2) physical_interaction
        sys_type = str(edge_row.get("Experimental System Type", "")).lower()
        if "physical" in sys_type:
            e_feature |= (1 << EDGE_BIT_MAP["physical_interaction"])

        # E3) is_negative
        raw_type = str(edge_row.get("type", "positive")).lower()
        if ("negative" in raw_type) or (edge_row.get("is_negative") is True):
            e_feature |= (1 << EDGE_BIT_MAP["is_negative"])

    return int(e_feature)



# --- 辅助函数 ---
def map_loc_to_category(loc_str):
    """ 
    [Legacy] 用于保留 cat_idx 以便可视化或调试，不参与训练特征生成。
    """
    s = str(loc_str).lower()
    if s == 'nan' or s == '' or s == '-': return 9
    if 'nucleus' in s or 'nuclear' in s or 'nucleoplasm' in s: return 0
    if 'membrane' in s: return 2
    if 'mitochondri' in s: return 4
    if 'reticulum' in s: return 5
    if 'golgi' in s: return 6
    if 'lysosome' in s or 'peroxisome' in s or 'endosome' in s: return 7
    if 'secreted' in s or 'extracellular' in s: return 3
    if 'cytoplasm' in s or 'cytosol' in s: return 1
    return 8 

# [REMOVED] get_gar_edge_index 函数已删除，因为它与新编码逻辑冲突

# ==============================================================================
# 1. 验证函数 (Validation)
# ==============================================================================

def edge_match(d1, d2):
    return d1.get('label') == d2.get('label')

def node_match_fn(n1, n2):
    # [MODIFIED] 使用细粒度的位掩码特征进行匹配
    return n1.get('feature_val') == n2.get('feature_val')

def calculate_subgraph_metrics(subG, bigG):
    """
    计算子图在生成它的母图(BigG)中的 Support 和 Confidence。
    """
    edges_to_test = list(subG.edges(data=True))
    
    candidates = []
    for u, v, d in edges_to_test:
        val = d.get('label', 0)
        # [MODIFIED] 检查 Bit 7 是否为 1
        is_negative_edge = bool(val & (1 << EDGE_BIT_MAP['is_negative']))
        if is_negative_edge:
            candidates.append((u,v))

    if not candidates: return None

    target_u, target_v = candidates[0]
    
    # 1. 构建前提 (挖掉这条负边)
    premiseG = subG.copy()
    premiseG.remove_edge(target_u, target_v)
    
    if premiseG.number_of_edges() == 0: return None

    # 2. 匹配
    GM = GraphMatcher(bigG, premiseG, node_match=node_match_fn, edge_match=edge_match)
    
    unique_matches = set()
    supp_neg = 0
    
    try:
        with TimeLimit(TIME_LIMIT):
            for mapping in GM.subgraph_isomorphisms_iter():
                real_u = mapping.get(target_u)
                real_v = mapping.get(target_v)
                
                if real_u is None or real_v is None: continue
                
                edge_key = frozenset([real_u, real_v])
                if edge_key in unique_matches: continue
                unique_matches.add(edge_key)
                
                if bigG.has_edge(real_u, real_v):
                    real_label = bigG[real_u][real_v].get('label', 0)
                    # [MODIFIED] 检查大图中的边是否也是负边
                    real_is_neg = bool(real_label & (1 << EDGE_BIT_MAP['is_negative']))
                    if real_is_neg:
                        supp_neg += 1
                
                if len(unique_matches) >= MATCH_LIMIT: break
                    
    except TimeoutException:
        pass
    except Exception:
        return None

    supp_shape = len(unique_matches)
    conf = supp_neg / supp_shape if supp_shape > 0 else 0.0
    
    return {
        'conf': conf,
        'supp_neg': supp_neg,
        'supp_shape': supp_shape,
        'label': subG[target_u][target_v]['label']
    }


# ==============================================================================
# 2. Dataset 类定义
# ==============================================================================

class PPIGraphDataset(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None, 
                 num_subgraphs=2000, min_nodes=3, max_nodes=5):
        self.num_subgraphs = num_subgraphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['protein_protein_with_type.csv', 'protein.csv']

    @property
    def processed_file_names(self):
        return [f'ppi_{self.split}.pt']

    def download(self):
        pass 

    def process(self):
        ppi_path = os.path.join(self.root, 'raw', 'protein_protein_with_type.csv')
        meta_path = os.path.join(self.root, 'raw', 'protein.csv')
        
        # --- A. 加载 Metadata ---
        print(f"Loading metadata from {meta_path}...")
        id_to_attrs = {} 
        try:
            df_meta = pd.read_csv(meta_path, low_memory=False)
            df_meta.columns = df_meta.columns.str.strip()
            id_col = 'biogrid_id'
            
            df_meta[id_col] = pd.to_numeric(df_meta[id_col], errors='coerce')
            df_meta = df_meta.dropna(subset=[id_col])
            
            for _, row in df_meta.iterrows():
                bid = str(int(row[id_col]))
                attr_dict = row.to_dict()
                loc_raw = row.get('location', '')
                attr_dict['cat_idx'] = map_loc_to_category(loc_raw) # 仅用于Legacy显示
                id_to_attrs[bid] = attr_dict
            print(f"Metadata map built. Covered proteins: {len(id_to_attrs)}")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return

                # --- B. 构建大图 ---
        print(f"Loading PPI structure from {ppi_path}...")
        bigG = nx.Graph()
        try:
            df_ppi = pd.read_csv(ppi_path, sep=',' if ',' in open(ppi_path).readline() else '\t')
            df_ppi.columns = df_ppi.columns.str.strip()

            # 第一遍：只加点、加边，边 label 先占位 0
            for _, row in df_ppi.iterrows():
                u_bid = str(row.get('BioGRID ID Interactor A', '')).split('.')[0]
                v_bid = str(row.get('BioGRID ID Interactor B', '')).split('.')[0]
                if not u_bid or not v_bid:
                    continue

                u_attrs = id_to_attrs.get(u_bid, {})
                v_attrs = id_to_attrs.get(v_bid, {})

                # 节点特征编码（你已有逻辑）
                u_enc = encode_node_feature(u_attrs)
                v_enc = encode_node_feature(v_attrs)

                # 添加节点
                bigG.add_node(u_bid, **u_attrs, feature_val=u_enc)
                bigG.add_node(v_bid, **v_attrs, feature_val=v_enc)

                edge_data = row.to_dict()

                # 先占位，第二遍再用全局统计 + GNN + Affinity Capture-MS 规则重算
                bigG.add_edge(u_bid, v_bid, label=0, **edge_data)

            # --- B2. 计算节点统计量与全局分位数（给 x.degree / x.betweenness 的谓词用）---
            deg_dict = dict(bigG.degree())
            nx.set_node_attributes(bigG, deg_dict, "degree")

            bet_dict = nx.betweenness_centrality(bigG,k=256,seed=42)
            nx.set_node_attributes(bigG, bet_dict, "betweenness_centrality")

            deg_vals = np.array(list(deg_dict.values()), dtype=float)
            bet_vals = np.array(list(bet_dict.values()), dtype=float)
            global_stats = {
                "degree": {"q75": float(np.quantile(deg_vals, 0.75))},
                "betweenness_centrality": {"q25": float(np.quantile(bet_vals, 0.25))}
            }

            # --- B3. 准备 pred_scores（没有就先空着，不会影响流程）---
            pred_scores = {"GNN": {}}
            GNN_THRESHOLD = 0.5

            # --- B4. 第二遍：重算边 label（严格按你的谓词策略）---
            for u, v, d in bigG.edges(data=True):
                ux = bigG.nodes[u]
                vy = bigG.nodes[v]
                d["label"] = int(
                    encode_edge_feature(
                        id_x=u,
                        id_y=v,
                        node_x_attr=ux,
                        node_y_attr=vy,
                        edge_row=d,
                        global_stats=global_stats
                    )

                )

        except Exception as e:
            print(f"Error building graph: {e}")
            import traceback
            traceback.print_exc()
            return

        print(f"Big Graph loaded. Nodes: {bigG.number_of_nodes()}, Edges: {bigG.number_of_edges()}")

    
        # --- C. 扫描负边 ---
        neg_edge_nodes = set()
        print("Scanning for negative edges to prioritize sampling...")
        for u, v, d in bigG.edges(data=True):
            val = d.get('label', 0)
            # [MODIFIED] 检查 Bit 7 (Is_Negative)
            if val & (1 << EDGE_BIT_MAP['is_negative']):
                neg_edge_nodes.add(u)
                neg_edge_nodes.add(v)
                
        neg_edge_nodes = list(neg_edge_nodes)
        print(f"Found {len(neg_edge_nodes)} nodes involved in negative edges.")
        
        if len(neg_edge_nodes) == 0:
            print("Warning: No negative edges found! Sampling will be purely random.")

        # --- D. 采样与转换 ---
        data_list = []
        all_nodes = list(bigG.nodes())
        
        print(f"Sampling {self.num_subgraphs} subgraphs (Strategy: Negative-Centric)...")
        
        pbar = tqdm(total=self.num_subgraphs)
        high_support_cnt = 0
        neg_sub_cnt = 0
        
        while len(data_list) < self.num_subgraphs:
            # 80% 概率从负边区域采样
            use_biased_sampling = (len(neg_edge_nodes) > 0) and (np.random.rand() < 0.8)
            seed = np.random.choice(neg_edge_nodes) if use_biased_sampling else np.random.choice(all_nodes)
            
            target_size = np.random.randint(self.min_nodes, self.max_nodes + 1)
            sub_nodes = [seed]
            
            # BFS 扩张
            try:
                bfs_successors = dict(nx.bfs_successors(bigG, seed))
                queue = [seed]
                steps = 0
                while len(sub_nodes) < target_size and queue and steps < 1000:
                    curr = queue.pop(0)
                    steps += 1
                    if curr in bfs_successors:
                        neighbors = bfs_successors[curr]
                        np.random.shuffle(neighbors)
                        for n in neighbors:
                            if n not in sub_nodes:
                                sub_nodes.append(n)
                                queue.append(n)
                                if len(sub_nodes) >= target_size: break
            except: pass
            
            if len(sub_nodes) < self.min_nodes: continue
                
            G_sub = bigG.subgraph(sub_nodes).copy()

            # [MODIFIED] 检查子图是否包含负边 (使用新的位掩码逻辑)
            has_neg = False
            for _, _, d in G_sub.edges(data=True):
                val = d.get('label', 0)
                if val & (1 << EDGE_BIT_MAP['is_negative']):
                    has_neg = True
                    break
            
            if has_neg:
                neg_sub_cnt += 1
                # 仅在包含负边时计算 Support
                metrics = calculate_subgraph_metrics(G_sub, bigG)
                if metrics and metrics['supp_shape'] >= MATCH_LIMIT:
                    high_support_cnt += 1

            # 转 PyG
            G_sub = nx.convert_node_labels_to_integers(G_sub, label_attribute='orig_symbol')
            pyg_data = self._to_pyg_data(G_sub)
            if pyg_data: 
                data_list.append(pyg_data)
                pbar.update(1)
        
        print(f"*Overall neg sub pattern: {neg_sub_cnt}")
        print(f"*Overall high support pattern: {high_support_cnt}")
    
        pbar.close()
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("Done!")

    def _to_pyg_data(self, G):
        # 1) 节点特征：feature_val (0..3) -> 4维 one-hot
        xs = [G.nodes[n].get('feature_val', 0) for n in G.nodes()]
        x_idx = torch.tensor(xs, dtype=torch.long)
        x = F.one_hot(x_idx, num_classes=NUM_NODE_CLASSES).float()  # [n, 4]

        # 2) 构造有向边 + 特征
        src, dst = [], []
        edge_type_ids = []
        edge_bitmasks = []

        # 先收集所有“候选 directed edges”
        cand = []  # (s, t, edge_type, bitmask)
        for u, v, d in G.edges(data=True):
            bitmask = int(d.get('label', 0))     # 0..(2^k-1)
            edge_type = bitmask + 1              # 1.. (预留 0 给 no-edge)

            # 无向 -> 双向
            cand.append((u, v, edge_type, bitmask))
            cand.append((v, u, edge_type, bitmask))

        if not cand:
            return None

        # 2.1) 去 self-loop + 去重 directed edge
        seen = set()
        dup = 0
        self_loops = 0

        for s, t, et, bm in cand:
            s = int(s); t = int(t)
            if s == t:
                self_loops += 1
                continue

            key = (s, t)
            if key in seen:
                dup += 1
                continue
            seen.add(key)

            src.append(s)
            dst.append(t)
            edge_type_ids.append(int(et))
            edge_bitmasks.append(int(bm))

        if len(src) == 0:
            return None

        if self_loops > 0:
            print(f"[WARN] removed self-loops: {self_loops}")
        if dup > 0:
            print(f"[WARN] duplicate directed edges removed: {dup}")

        # 2.2) 组装 PyG 张量
        edge_index = torch.tensor([src, dst], dtype=torch.long)  # [2, E]
        edge_type_ids = torch.tensor(edge_type_ids, dtype=torch.long)  # [E]
        edge_attr = F.one_hot(edge_type_ids, num_classes=NUM_EDGE_CLASSES).float()  # [E, 129] 你这边是129

        edge_label_mask = torch.tensor(edge_bitmasks, dtype=torch.long)  # [E] debug

        # 3) 最后再做一道硬检查：保证不会出现 num_edges > n*(n-1)
        n = G.number_of_nodes()
        if edge_index.size(1) > n * (n - 1):
            # 理论上去重+去自环后不可能发生
            print(f"[FATAL] edges > n*(n-1): n={n}, E={edge_index.size(1)}")
            return None

        y = torch.zeros(1, 0).float()
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_label_mask=edge_label_mask,
            n_nodes=n,
            y=y
        )





# ==============================================================================
# 3. DataModule & Infos (保持基本不变)
# ==============================================================================

class PPIDataModule(AbstractDataModule):
    def __init__(self, cfg):
        current_file_path = os.path.realpath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        abs_datadir = os.path.join(project_root, 'data', 'PPI')
        self.datadir = abs_datadir
        print(f"[Info] Absolute Data Directory: {self.datadir}")
        
        base_dataset = PPIGraphDataset(
            root=self.datadir, 
            num_subgraphs=cfg.dataset.num_subgraphs,
            min_nodes=cfg.dataset.min_nodes,
            max_nodes=cfg.dataset.max_nodes,
            split='train' 
        )
        
        datasets = {'train': base_dataset, 'val': base_dataset, 'test': base_dataset}
        super().__init__(cfg, datasets)

class PPIDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'ppi'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        print(">>> 真实数据节点分布:", self.node_types)
        self.edge_types = self.datamodule.edge_counts()
        print(">>> Edge type distribution:", self.edge_types)
        print(">>> Edge type dim:", len(self.edge_types))

        super().complete_infos(self.n_nodes, self.node_types)