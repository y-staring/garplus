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



ML_THRESHOLD = 0.5
NEGATIVE_LABELS = [3, 4]
TIME_LIMIT = 5


# 定义超时异常
class TimeoutException(Exception): 
    pass

# 定义上下文管理器：只在需要的时候接管信号，用完立刻归还
class TimeLimit:
    def __init__(self, seconds):
        self.seconds = seconds
        self.old_handler = signal.SIG_DFL

    def __enter__(self):
        if hasattr(signal, 'SIGALRM'):
            # 1. 备份系统原有的 Handler (防止干扰 PyTorch)
            self.old_handler = signal.getsignal(signal.SIGALRM)
            
            # 2. 定义临时 Handler
            def handler(signum, frame):
                raise TimeoutException()
            
            # 3. 注册并开启闹钟
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, type, value, traceback):
        if hasattr(signal, 'SIGALRM'):
            # 4. 关闭闹钟
            signal.alarm(0)
            # 5. 归还 Handler 给系统 (关键！)
            signal.signal(signal.SIGALRM, self.old_handler)


# ==================supp & conf validation===========================
MATCH_LIMIT = 400  # 限制匹配次数，防止预处理卡死
NEGATIVE_LABELS = [3, 4] # 假设 3,4 是负边

def edge_match(d1, d2):
    return d1.get('label') == d2.get('label')

def node_match_fn(n1, n2):
    # 注意：在 process 里的 bigG，节点属性通常叫 'cat_idx' 或 'cat'
    # 请根据你实际构建 bigG 时的属性名修改这里
    return n1.get('cat_idx') == n2.get('cat_idx')

# 注册信号 (在 Linux/Mac 上有效，Windows 上可能不支持 SIGALRM)
# 如果是在 Windows 上跑，建议改用 time.time() 在循环内检查


def calculate_subgraph_metrics(subG, bigG):
    """
    计算子图在生成它的母图(BigG)中的 Support 和 Confidence。
    带超时控制。
    """
    edges_to_test = list(subG.edges(data=True))
    
    # 找负边 (Label 3 or 4)
    candidates = [(u, v) for u, v, d in edges_to_test if d.get('label') in NEGATIVE_LABELS]
    if not candidates: return None 

    target_u, target_v = candidates[0]
    
    # 1. 构建前提 (挖掉这条负边)
    premiseG = subG.copy()
    premiseG.remove_edge(target_u, target_v)
    # ==================== 深度调试 START ====================
    print("\n--- DEBUG: Attribute Mismatch Check ---")
    
    # 1. 检查节点 (Node) 属性
    # 我们取 subG 里任意一个节点，去 bigG 里找它的对应项
    check_node = list(premiseG.nodes())[0]
    
    # 获取两边的属性字典
    sub_node_attrs = premiseG.nodes[check_node]
    big_node_attrs = bigG.nodes[check_node] if check_node in bigG else "Node Not Found!"
    
    print(f"Node: '{check_node}'")
    print(f"  [SubG] Attrs: {sub_node_attrs}")
    print(f"  [BigG] Attrs: {big_node_attrs}")
    
    # 模拟 node_match_fn 调用
    try:
        match_res = node_match_fn(sub_node_attrs, big_node_attrs)
        print(f"  -> node_match_fn Result: {match_res} (Expect: True)")
    except Exception as e:
        print(f"  -> node_match_fn CRASHED: {e}")

    # 2. 检查边 (Edge) 属性
    # 如果 premiseG 里还有边，我们检查第一条
    if premiseG.number_of_edges() > 0:
        u, v = list(premiseG.edges())[0]
        
        # 获取两边的属性字典
        sub_edge_attrs = premiseG[u][v]
        big_edge_attrs = bigG[u][v] if bigG.has_edge(u, v) else "Edge Not Found!"
        
        print(f"Edge: {u} -- {v}")
        print(f"  [SubG] Attrs: {sub_edge_attrs}")
        print(f"  [BigG] Attrs: {big_edge_attrs}")
        
        # 模拟 edge_match 调用
        try:
            match_res = edge_match(sub_edge_attrs, big_edge_attrs)
            print(f"  -> edge_match Result: {match_res} (Expect: True)")
        except Exception as e:
            print(f"  -> edge_match CRASHED: {e}")
    else:
        print("Edge: Premise graph has no edges left to check.")

    print("-------------------------------------------\n")
    # ==================== 深度调试 END ====================
   
    
    if premiseG.number_of_edges() == 0: return None

    # 2. 匹配
    # 注意：对于无向图，GraphMatcher 会自动处理对称性
    # GM = GraphMatcher(bigG, premiseG, node_match=node_match_fn, edge_match=edge_match)
    GM = GraphMatcher(bigG, premiseG)

    
    # ★★★ 关键修改：使用 Set 去重，防止对称结构导致重复统计 ★★★
    unique_matches = set() # 存 frozenset({real_u, real_v})
    supp_neg = 0
    
    try:
        with TimeLimit(TIME_LIMIT):
            for mapping in GM.subgraph_isomorphisms_iter():
                # 映射回大图节点
                real_u = mapping.get(target_u)
                real_v = mapping.get(target_v)
                
                if real_u is None or real_v is None: continue
                
                # ★ 无向图的关键：构建唯一键
                # 使用 frozenset 是因为 {u, v} 和 {v, u} 是同一个集合
                edge_key = frozenset([real_u, real_v])
                
                if edge_key in unique_matches:
                    continue # 这个物理位置已经统计过了，跳过
                
                unique_matches.add(edge_key)
                
                # 检查大图里这条边的真实情况
                if bigG.has_edge(real_u, real_v):
                    real_label = bigG[real_u][real_v].get('label', 0)
                    if real_label in NEGATIVE_LABELS:
                        supp_neg += 1
                
                # 检查的是“唯一的物理位置”数量
                if len(unique_matches) >= MATCH_LIMIT:
                    break
                    
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
# 0. GAR+ 逻辑配置 & 清洗映射
# ==============================================================================

# 定义 10 个标准的亚细胞定位类别
LOC_CATEGORIES = [
    "Nucleus",       # 0: 细胞核
    "Cytoplasm",     # 1: 细胞质/胞质溶胶
    "Membrane",      # 2: 膜 (细胞膜/质膜)
    "Secreted",      # 3: 分泌/胞外
    "Mitochondria",  # 4: 线粒体
    "ER",            # 5: 内质网
    "Golgi",         # 6: 高尔基体
    "Lysosome",      # 7: 溶酶体/过氧化物酶体
    "Other",         # 8: 有值但无法归类
    "Unknown"        # 9: 缺失值 (NaN)
]

NUM_NODE_CLASSES = len(LOC_CATEGORIES) # 10类
NUM_EDGE_CLASSES = 5 
ML_THRESHOLD = 0.5

def map_loc_to_category(loc_str):
    """ 
    文本清洗函数：从复杂的 Uniprot 描述中提取核心定位。
    """
    s = str(loc_str).lower()
    
    if s == 'nan' or s == '' or s == '-': return 9 # Unknown
        
    if 'nucleus' in s or 'nuclear' in s or 'nucleoplasm' in s: return 0
    if 'membrane' in s: return 2
    if 'mitochondri' in s: return 4
    if 'reticulum' in s: return 5
    if 'golgi' in s: return 6
    if 'lysosome' in s or 'peroxisome' in s or 'endosome' in s: return 7
    if 'secreted' in s or 'extracellular' in s: return 3
    if 'cytoplasm' in s or 'cytosol' in s: return 1
        
    return 8 # Other

def get_gar_edge_index(edge_type_str, score):
    """ 计算边索引: 1 + (is_neg * 2) + is_high_conf """
    s = str(edge_type_str).lower()
    is_negative = 1 if 'negative' in s else 0
    try:
        score_val = float(score)
    except:
        score_val = 0.0
    is_high_conf = 1 if score_val >= ML_THRESHOLD else 0
    return 1 + (is_negative * 2) + is_high_conf


# ==============================================================================
# 1. Dataset 类定义
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
        pass # 假设用户已把文件放好

    def process(self):
        ppi_path = os.path.join(self.root, 'raw', 'protein_protein_with_type.csv')
        meta_path = os.path.join(self.root, 'raw', 'protein.csv')
        
        # --- A. 加载 Metadata ---
        print(f"Loading metadata from {meta_path}...")
        try:
            df_meta = pd.read_csv(meta_path, low_memory=False)
            df_meta.columns = df_meta.columns.str.strip()
            id_col = 'biogrid_id'
            feat_col = 'location'
            
            # 清洗 ID (.0 问题)
            df_meta[id_col] = pd.to_numeric(df_meta[id_col], errors='coerce')
            df_meta = df_meta.dropna(subset=[id_col])
            df_meta[id_col] = df_meta[id_col].astype(int).astype(str)
            
            # 构建映射
            id_to_cat = {}
            for _, row in df_meta.iterrows():
                bid = row[id_col]
                loc_raw = row.get(feat_col, '')
                id_to_cat[bid] = map_loc_to_category(loc_raw)
            print(f"Metadata map built. Covered proteins: {len(id_to_cat)}")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            return

        # --- B. 构建大图 ---
        print(f"Loading PPI structure from {ppi_path}...")
        bigG = nx.Graph()
        try:
            df_ppi = pd.read_csv(ppi_path, sep=',')
            if len(df_ppi.columns) < 5: df_ppi = pd.read_csv(ppi_path, sep='\t')
            df_ppi.columns = df_ppi.columns.str.strip()
            
            for _, row in df_ppi.iterrows():
                u_sym = str(row.get('Official Symbol Interactor A', row[0])) 
                v_sym = str(row.get('Official Symbol Interactor B', row[1]))
                u_bid = str(row.get('BioGRID ID Interactor A')).split('.')[0]
                v_bid = str(row.get('BioGRID ID Interactor B')).split('.')[0]
                
                u_cat = id_to_cat.get(u_bid, 9)
                v_cat = id_to_cat.get(v_bid, 9)
                
                bigG.add_node(u_sym, cat_idx=u_cat)
                bigG.add_node(v_sym, cat_idx=v_cat)
                
                edge_type = row.get('type', 'positive')
                score = row.get('Score', 0.0)
                if pd.isna(score): score = 0.0
                # bigG.add_edge(u_sym, v_sym, type=edge_type, score=score)
                bigG.add_edge(u_sym, v_sym, type=edge_type, score=score, label=get_gar_edge_index(edge_type, score))
        except Exception as e:
            print(f"Error building graph: {e}")
            return
            
        print(f"Big Graph loaded. Nodes: {bigG.number_of_nodes()}, Edges: {bigG.number_of_edges()}")
    
        
        # ==================== 调试代码 START ====================
        from collections import Counter
        all_labels = [d.get('label') for u, v, d in bigG.edges(data=True)]
        print("★ DEBUG: BigGraph Label Distribution:", Counter(all_labels))
        
        # 看看前几个边的完整属性长什么样
        print("★ DEBUG: First 3 edges data:", list(bigG.edges(data=True))[:3])
        # ==================== 调试代码 END ====================
        # ======================================================================
        # ★★★ 新增：预先找出“含有负边”的重点关注节点 ★★★
        # ======================================================================
        neg_edge_nodes = set()
        # 假设你的负边 Label 对应的是 3 (NegLow) 和 4 (NegHigh)
        # 如果你还没做 Label 映射，确保这里填的是你数据里代表负边的值
        TARGET_NEG_LABELS = [3, 4] 
        
        print("Scanning for negative edges to prioritize sampling...")
        for u, v, d in bigG.edges(data=True):
            # 获取边的类型索引
            # 注意：这里的 d['type'] 或 d['score'] 需要根据你之前的 get_gar_edge_index 逻辑判断
            # 最简单的方法是直接复用你的 get_gar_edge_index 函数算一下
            e_type = d.get('type', 'positive')
            e_score = d.get('score', 0.0)
            e_idx = get_gar_edge_index(e_type, e_score) # 0-4
            
            if e_idx in TARGET_NEG_LABELS:
                neg_edge_nodes.add(u)
                neg_edge_nodes.add(v)
                
        neg_edge_nodes = list(neg_edge_nodes)
        print(f"Found {len(neg_edge_nodes)} nodes involved in negative edges.")
        
        if len(neg_edge_nodes) == 0:
            print("Warning: No negative edges found! Sampling will be purely random.")
            
        # ======================================================================

        # --- C. 采样与转换 ---
        data_list = []
        all_nodes = list(bigG.nodes())
        
        print(f"Sampling {self.num_subgraphs} subgraphs (Strategy: Negative-Centric)...")
        
        pbar = tqdm(total=self.num_subgraphs)
        high_support_cnt = 0
        neg_sub_cnt = 0
        while len(data_list) < self.num_subgraphs:
            
            # ★★★ 修改采样策略：80% 概率从负边节点开始 ★★★
            use_biased_sampling = (len(neg_edge_nodes) > 0) and (np.random.rand() < 0.8)
            
            if use_biased_sampling:
                seed = np.random.choice(neg_edge_nodes) # 强行关注负边区域
            else:
                seed = np.random.choice(all_nodes)      # 随机探索
            
            target_size = np.random.randint(self.min_nodes, self.max_nodes + 1)
            sub_nodes = [seed]
            
            try:
                # BFS 扩张
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

            high_support_cnt = 0
            #===============================================================================
            # 只有当子图包含负边时才计算
            has_neg = any(d.get('label') in [3, 4] for _,_,d in G_sub.edges(data=True))
            # print("has neg:", has_neg)
            if has_neg:
                neg_sub_cnt +=1
                # 调用计算函数
                metrics = calculate_subgraph_metrics(G_sub, bigG)
                
                if metrics:
                    print(f"Sampled Neg Graph | Conf: {metrics['conf']:.2f} | "
                               f"Supp: {metrics['supp_neg']}/{metrics['supp_shape']}")
                    if metrics['supp_shape'] >= MATCH_LIMIT:
                        high_support_cnt +=1
            

                    # [选项 A] 仅仅打印出来看看
                    # tqdm.write 可以在进度条中安全打印
                    # tqdm.write(f"Sampled Neg Graph | Conf: {metrics['conf']:.2f} | "
                    #            f"Supp: {metrics['supp_neg']}/{metrics['supp_shape']}")
                    
                    # [选项 B] 过滤：如果 Conf 太低，直接丢弃这个训练样本？
                    # if metrics['conf'] < 0.1: continue

            #===============================================================================

            
            # ★★★ 二次检查：确保这个子图里真的有负边（可选，为了极致的数据纯度）★★★
            # 如果是 Biased 采样，我们希望这个子图尽量包含负边
            # if use_biased_sampling:
            #     has_neg = any(get_gar_edge_index(d.get('type'), d.get('score')) in TARGET_NEG_LABELS 
            #                   for u, v, d in G_sub.edges(data=True))
            #     if not has_neg: continue # 如果运气不好没带上负边，就扔掉重采
            
            G_sub = nx.convert_node_labels_to_integers(G_sub, label_attribute='orig_symbol')
            pyg_data = self._to_pyg_data(G_sub)
            if pyg_data: 
                data_list.append(pyg_data)
                pbar.update(1)
        


        print("*Overall neg sub pattern", neg_sub_cnt)
        print("*Overall high support pattern:", high_support_cnt)
    
        pbar.close()
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("Done!")

    def _to_pyg_data(self, G):
        # 节点特征 X
        xs = [G.nodes[n].get('cat_idx', 9) for n in G.nodes()]
        x_idx = torch.tensor(xs, dtype=torch.long)
        x = F.one_hot(x_idx, num_classes=NUM_NODE_CLASSES).float()
        
        # 边特征 E
        src, dst, edge_attrs = [], [], []
        for u, v, d in G.edges(data=True):
            attr_idx = get_gar_edge_index(d.get('type'), d.get('score'))
            if attr_idx == 0: continue
            src.extend([u, v])
            dst.extend([v, u])
            edge_attrs.extend([attr_idx, attr_idx])
            
        if not src: return None
        
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        ea_idx = torch.tensor(edge_attrs, dtype=torch.long)
        edge_attr = F.one_hot(ea_idx, num_classes=NUM_EDGE_CLASSES).float()
        y = torch.zeros(1, 0).float()
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                    n_nodes=G.number_of_nodes(), y=y)



# ==============================================================================
# 2. DataModule 定义
# ==============================================================================

class PPIDataModule(AbstractDataModule):
    def __init__(self, cfg):
        # 1. 计算绝对路径 (Anchor to the project root)
        # 获取当前脚本 (src/datasets/ppi_dataset.py) 的绝对路径
        current_file_path = os.path.realpath(__file__)
        # 回退两层找到项目根目录 (Assuming src/datasets/ -> src/ -> root/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        
        # 拼接出数据的绝对路径
        abs_datadir = os.path.join(project_root, 'data', 'PPI')
        
        # 覆盖 cfg 中的相对路径
        self.datadir = abs_datadir
        
        # 打印一下确认路径对不对
        print(f"[Info] Absolute Data Directory: {self.datadir}")
        
        # 创建数据集
        base_dataset = PPIGraphDataset(
            root=self.datadir,  # 传入绝对路径
            num_subgraphs=cfg.dataset.num_subgraphs,
            min_nodes=cfg.dataset.min_nodes,
            max_nodes=cfg.dataset.max_nodes,
            split='train' 
        )
        
        datasets = {
            'train': base_dataset,
            'val': base_dataset, 
            'test': base_dataset
        }
        
        super().__init__(cfg, datasets)


# ==============================================================================
# 3. DatasetInfos 定义
# ==============================================================================

class PPIDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'ppi'
        self.n_nodes = self.datamodule.node_counts()
       
        self.node_types = self.datamodule.node_types()
        print(">>> 真实数据节点分布:", self.node_types)
        self.edge_types = self.datamodule.edge_counts()
        # 调用父类的 complete_infos 来初始化 DistributionNodes
        super().complete_infos(self.n_nodes, self.node_types)