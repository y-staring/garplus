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
ML_THRESHOLD = 0.5



# --- [NEW] 节点特征编码映射 ---
NODE_BIT_MAP = {
    'is_kinase': 0,          # +1
    'is_disease_related': 1  # +2
}

# --- [NEW] 边特征编码映射 ---
EDGE_BIT_MAP = {
    'location_match_y': 0,     # +1
    'process_match_y': 1,      # +2
    'M_GNN': 2,                # +4 (ML High Confidence)
    'physical_interaction': 3, # +8
    'genetic_interaction': 4,  # +16
    'phosphorylation': 5,      # +32
    'ubiquitination': 6,       # +64
    "is_negative": 7           # +128 [NEW]
}


# --- [MODIFIED] 特征维度定义 ---
# 节点特征: 2 bit -> 4 类 (0-3)
NUM_NODE_CLASSES = 4 
# 边特征: 8 bit -> 256 类 (0-127)
NUM_EDGE_CLASSES = 1 + (2 ** len(EDGE_BIT_MAP))  # 1+256=257


def edge_bitmask_to_vector(bitmask: int) -> torch.Tensor:
    """
    将 8-bit bitmask (0..255) 转为 9维向量:
    [no-edge, pred1, pred2, ..., pred8]
    
    对于真实存在的边：no-edge=0
    对于 non-edge：由 encode_no_edge() 在 dense 图里自动填 no-edge=1
    """
    vec = torch.zeros(NUM_EDGE_CLASSES, dtype=torch.float)  # 9维
    vec[0] = 0.0  # 真实边：no-edge 永远是 0
    
    # 注意：EDGE_BIT_MAP 的 key 顺序不保证稳定，建议按 bit 位排序
    # 我们按 bit index 从小到大展开，保证维度语义固定
    #bit 0对应第一维， 
    for name, bit in sorted(EDGE_BIT_MAP.items(), key=lambda x: x[1]):
        if bitmask & (1 << bit):
            vec[1 + bit] = 1.0
    return vec


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

def encode_edge_feature(node_x_attr: dict, node_y_attr: dict, edge_row: dict) -> int:
    """ 编码两个节点 (x, y) 及其边属性为整数特征 E (0-255) """
    e_feature = 0
    
    # --- A. 属性比较谓词 ---
    loc_x_str = str(node_x_attr.get('Subcellular location [CC]', node_x_attr.get('location', '')))
    loc_y_str = str(node_y_attr.get('Subcellular location [CC]', node_y_attr.get('location', '')))
    loc_x = set(loc_x_str.split(';'))
    loc_y = set(loc_y_str.split(';'))
    loc_x.discard(''); loc_y.discard('')
    
    if len(loc_x.intersection(loc_y)) > 0:
        e_feature |= (1 << EDGE_BIT_MAP['location_match_y'])
        
    # 2. process_match_y
    proc_x = set(str(node_x_attr.get('Gene Ontology (biological process)', '')).split(';'))
    proc_y = set(str(node_y_attr.get('Gene Ontology (biological process)', '')).split(';'))
    proc_x.discard(''); proc_y.discard('')
    if len(proc_x.intersection(proc_y)) > 0:
        e_feature |= (1 << EDGE_BIT_MAP['process_match_y'])

    # --- B. ML 谓词 & 实验谓词 (依赖 edge_row) ---
    if edge_row:
        # ML Predicate
        try:
            score = float(edge_row.get('Score', 0.0))
        except: 
            score = 0.0
        if score > ML_THRESHOLD:
            e_feature |= (1 << EDGE_BIT_MAP['M_GNN'])

        raw_type = str(edge_row.get('type', 'positive')).lower() 
        
        # [MODIFIED] 负边判定逻辑
        # 如果标签包含 negative，或者有单独的 is_negative 列
        if 'negative' in raw_type or edge_row.get('is_negative') == True:
            e_feature |= (1 << EDGE_BIT_MAP['is_negative']) # Bit 7 置 1

        # 3. 实验手段 (Physical/Genetic)
        # 注意：即使是负边，通常也意味着“通过某种手段证实为负”，所以手段位依然可以是 1
        # 例如：Y2H 实验显示不互作 -> physical=1, is_negative=1
        sys_type = str(edge_row.get('Experimental System Type', raw_type)).lower()
        if 'physical' in sys_type:
            e_feature |= (1 << EDGE_BIT_MAP['physical_interaction'])
        if 'genetic' in sys_type:
            e_feature |= (1 << EDGE_BIT_MAP['genetic_interaction'])
            
        # PTM 修饰
        mod_text = str(edge_row.get('Modification', '')).lower()
        if 'phosphorylation' in mod_text:
            e_feature |= (1 << EDGE_BIT_MAP['phosphorylation'])
        if 'ubiquitin' in mod_text:
            e_feature |= (1 << EDGE_BIT_MAP['ubiquitination'])
            
    return e_feature

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
                 num_subgraphs=2000, min_nodes=10, max_nodes=50):
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
            
            for _, row in df_ppi.iterrows():
                u_bid = str(row.get('BioGRID ID Interactor A', '')).split('.')[0]
                v_bid = str(row.get('BioGRID ID Interactor B', '')).split('.')[0]
                if not u_bid or not v_bid: continue
                
                u_attrs = id_to_attrs.get(u_bid, {})
                v_attrs = id_to_attrs.get(v_bid, {})
                
                # [MODIFIED] 使用新编码器生成 feature_val
                u_enc = encode_node_feature(u_attrs)
                v_enc = encode_node_feature(v_attrs)
                
                # 添加节点 (feature_val 是关键训练特征)
                bigG.add_node(u_bid, **u_attrs, feature_val=u_enc)
                bigG.add_node(v_bid, **v_attrs, feature_val=v_enc)
                
                edge_data = row.to_dict()
                
                # [MODIFIED] 使用新编码器生成边 label (0-255)
                # 这一步非常关键！原来的 get_gar_edge_index 被替换了
                edge_label_enc = encode_edge_feature(u_attrs, v_attrs, edge_data)
                
                bigG.add_edge(u_bid, v_bid, label=edge_label_enc, **edge_data)
                
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

        # 2) 边特征：bitmask -> categorical edge type (0:no-edge, 1..256: bitmask+1)
        src, dst = [], []
        edge_type_ids = []
        edge_bitmasks = []  # debug：保留原始 bitmask (0..255)

        for u, v, d in G.edges(data=True):
            bitmask = int(d.get('label', 0))          # 0..255 (组合)
            edge_type = bitmask + 1                   # 1..256 (预留 0 给 no-edge)

            # 无向边 -> 双向有向边
            src.extend([u, v])
            dst.extend([v, u])
            edge_type_ids.extend([edge_type, edge_type])
            edge_bitmasks.extend([bitmask, bitmask])

        if not src:
            return None

        edge_index = torch.tensor([src, dst], dtype=torch.long)           # [2, 2E]
        edge_type_ids = torch.tensor(edge_type_ids, dtype=torch.long)     # [2E]
        edge_attr = F.one_hot(edge_type_ids, num_classes=NUM_EDGE_CLASSES).float()  # [2E, 257]

        # 采样得到的 dense E[i,j] 是类别 id（collapse 后）：

        # 0 → no-edge

        # k (1..256) → bitmask = k - 1

        # 然后你可以用你之前的 EDGE_BIT_MAP 反解每一位是不是 1。bitmask=1表示没有谓词组合匹配
        edge_label_mask = torch.tensor(edge_bitmasks, dtype=torch.long)   # [2E] 可选保留 debug

        y = torch.zeros(1, 0).float()
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_label_mask=edge_label_mask,  # 仍然保存原bitmask（方便你还原谓词）
            n_nodes=G.number_of_nodes(),
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