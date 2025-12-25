import os
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
import networkx as nx
import numpy as np
import pandas as pd  # 需要用到 pandas 读取列属性
from tqdm import tqdm
from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

# ==============================================================================
# 0. GAR+ 逻辑配置与辅助函数
# ==============================================================================

# 节点功能分类 (x.A)
GO_CATEGORIES = [
    "Enzyme", "Transcription", "Transporter", "Receptor", 
    "Structural", "Ubiquitin", "Immune", "Metabolic", "Signaling", "Other"
]
NUM_NODE_CLASSES = len(GO_CATEGORIES) # 10

# 边类别定义 (GAR+ Predicates)
# 0: No Edge
# 1: l(x,y) + ¬M(x,y)  (Pos, Low)
# 2: l(x,y) + M(x,y)   (Pos, High)
# 3: ¬l(x,y) + ¬M(x,y) (Neg, Low)
# 4: ¬l(x,y) + M(x,y)  (Neg, High)
NUM_EDGE_CLASSES = 5 
ML_THRESHOLD = 0.5

def map_ontology_to_category(ontology_str):
    """ 将 Ontology 字符串映射为 0-9 的整数 """
    s = str(ontology_str).lower()
    if 'kinase' in s or 'phosphatase' in s or 'enzyme' in s: return 0  #酶-修饰
    if 'transcription' in s: return 1                                  #转录因子
    if 'transporter' in s or 'channel' in s: return 2                  #转运
    if 'receptor' in s: return 3                                       #受体
    if 'structural' in s or 'cytoskeleton' in s: return 4              #结构
    if 'ubiquitin' in s: return 5                                       #泛素化
    if 'immune' in s: return 6                                          #免疫
    if 'metabolic' in s: return 7                                       #代谢
    if 'signaling' in s: return 8
    return 9 # Other


def get_gar_edge_index(edge_type_str, score):
    """ 
    根据边类型和分数计算 DiGress 离散索引 (1-4) 
    edge_type_str: 'positive' 或 'negative'
    score: float
    """
    # 1. 判断 l(x,y) 还是 ¬l(x,y)
    # 根据你的数据：type 列直接包含 'positive' 或 'negative'
    is_negative = 0
    s = str(edge_type_str).lower()
    
    if 'negative' in s:
        is_negative = 1
    # 默认为 positive (0)
        
    # 2. 判断 M(x,y) (ML Predicate)
    try:
        score_val = float(score)
    except:
        score_val = 0.0
        
    is_high_conf = 1 if score_val >= ML_THRESHOLD else 0
    
    # 3. 组合公式: 1 + (IsNeg * 2) + IsHigh
    # Pos/Low  = 1
    # Pos/High = 2
    # Neg/Low  = 3
    # Neg/High = 4
    return 1 + (is_negative * 2) + is_high_conf

# ==============================================================================
# 1. Dataset 类
# ==============================================================================
class PPIGraphDataset(InMemoryDataset):
    def __init__(self, root, split='train', 
                 raw_file_name='ppi_raw.txt',  
                 num_subgraphs=2000,           
                 min_nodes=10, max_nodes=50,   
                 transform=None, pre_transform=None, pre_filter=None):
        
        self.split = split
        self.raw_file_name = raw_file_name
        self.num_subgraphs = num_subgraphs
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.raw_file_name]

    @property
    def processed_file_names(self):
        return [f'ppi_{self.split}.pt']

    def download(self):
        pass

    def process(self):
        # 1. 读取原始大图 (修改为使用 Pandas 读取丰富属性)
        raw_path = self.raw_paths[0]
        print(f"Loading PPI graph from {raw_path}...")
        
        bigG = nx.Graph()
        
        # ★★★ 修改 A: 使用 Pandas 读取带表头的 TSV/CSV ★★★
        # 假设 raw_file 是 TAB 分隔的 BioGRID 格式
        try:
            df = pd.read_csv(raw_path, sep=',') 
            
            # 确保列名没有前后空格
            df.columns = df.columns.str.strip()
            print("\n[DEBUG] CSV 列名列表:", df.columns.tolist())
            print("[DEBUG] 前 5 行数据预览:")
            # 打印你代码里试图读取的那几列
            cols_to_check = [c for c in df.columns if 'Ontology' in c or 'Symbol' in c]
            print(df[cols_to_check].head(5))
            
            # 强制检查第一行的 Ontology 到底是个啥
            first_ont = df.iloc[0].get('Ontology Term Names Interactor A', 'KEY_NOT_FOUND')
            print(f"[DEBUG] 第一行的 Ontology A Raw Value: '{first_ont}'")
            print(f"[DEBUG] 第一行的 Ontology A 类型: {type(first_ont)}")
            # 遍历 DataFrame
            for _, row in df.iterrows():
                # 假设列名如下 (请根据你 CSV 的实际列名微调)
                # 如果没有 Official Symbol，可以用第一列和第二列
                u = str(row.get('Official Symbol Interactor A', row[0])) 
                v = str(row.get('Official Symbol Interactor B', row[1]))
                
                # ★★★ 提取你的 'type' 列 ★★★
                # 这里直接读取你处理好的 type 列 (positive/negative)
                edge_type = row.get('type', 'positive') 
                
                # 提取分数 (如果存在)
                score = row.get('Score', 0.0)
                
                # 提取节点属性 (Ontology)
                ont_A = row.get('Ontology Term Names Interactor A', '')
                ont_B = row.get('Ontology Term Names Interactor B', '')
                
                # 添加节点
                bigG.add_node(u, ontology=ont_A)
                bigG.add_node(v, ontology=ont_B)
                
                # 添加边 (存入 type 和 score)
                bigG.add_edge(u, v, type=edge_type, score=score)
                
        except Exception as e:
            print(f"Error reading PPI file: {e}")
            return

        print(f"Big Graph Loaded. Nodes: {bigG.number_of_nodes()}, Edges: {bigG.number_of_edges()}")

        # 2. 采样 BFS 子图
        data_list = []
        all_nodes = list(bigG.nodes())
        pbar = tqdm(total=self.num_subgraphs, desc=f"Sampling {self.split}")
        
        while len(data_list) < self.num_subgraphs:
            seed = np.random.choice(all_nodes)
            sub_nodes = [seed]
            try:
                bfs_successors = dict(nx.bfs_successors(bigG, seed))
                queue = [seed]
                while len(sub_nodes) < self.max_nodes and queue:
                    curr = queue.pop(0)
                    if curr in bfs_successors:
                        neighbors = bfs_successors[curr]
                        np.random.shuffle(neighbors)
                        for n in neighbors:
                            if n not in sub_nodes:
                                sub_nodes.append(n)
                                queue.append(n)
                                if len(sub_nodes) >= self.max_nodes: break
            except:
                pass 

            if len(sub_nodes) < self.min_nodes:
                continue
                
            subG = bigG.subgraph(sub_nodes).copy()
            
            # 转换为 PyG Data
            data = self._to_pyg_data(subG)
            if data:
                data_list.append(data)
                pbar.update(1)
        
        pbar.close()

        # 3. 保存
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _to_pyg_data(self, G):
        # 重新映射节点 ID 到 0 ~ N-1
        mapping = {n: i for i, n in enumerate(G.nodes())}
        
        # --- ★★★ 修改 B: 构造节点特征 (X) based on Ontology ★★★ ---
        xs_indices = []
        for n in G.nodes():
            # 获取节点属性
            ont_str = G.nodes[n].get('ontology', '')
            # 使用 GAR+ 映射函数
            cat_idx = map_ontology_to_category(ont_str)
            xs_indices.append(cat_idx)
            
        x_idx = torch.tensor(xs_indices, dtype=torch.long)
        # One-Hot Float (Dim = 10)
        x = F.one_hot(x_idx, num_classes=NUM_NODE_CLASSES).float()
        
        # --- ★★★ 修改 C: 构造边特征 (E) based on Type & Score ★★★ ---
        src, dst = [], []
        edge_attr_indices = []
        
        for u, v, d in G.edges(data=True):
            if u not in mapping or v not in mapping: continue
            u_idx, v_idx = mapping[u], mapping[v]
 
            edge_type_str = d.get('type', 'positive') # 从 networkx edge 属性中取
            score = d.get('score', 0)
            
            # 调用辅助函数计算 1-4 的索引
            attr_idx = get_gar_edge_index(edge_type_str, score)
            
            # 双向添加 (无向图)
            src.extend([u_idx, v_idx])
            dst.extend([v_idx, u_idx])
            edge_attr_indices.extend([attr_idx, attr_idx])
            
        if not src: return None
        
        edge_index = torch.tensor([src, dst], dtype=torch.long).contiguous()
        
        # 构造 One-Hot
        ea_idx = torch.tensor(edge_attr_indices, dtype=torch.long)
        edge_attr = F.one_hot(ea_idx, num_classes=NUM_EDGE_CLASSES).float()
        
        # 构造 y
        y = torch.zeros(1, 0).float()
        
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                    n_nodes=G.number_of_nodes(), y=y)
        return data

# ==============================================================================
# 2. DataModule
# ==============================================================================
class PPIDataModule(AbstractDataModule):
    def __init__(self, cfg):
        self.cfg = cfg
        root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/PPI')
        
        # ★★★ 修改这里：优先从 cfg.dataset 读取参数，如果没有才用默认值 ★★★
        num_train = getattr(cfg.dataset, 'num_subgraphs', 2000)
        # 验证集和测试集按比例缩小，或者固定一个小值
        num_val = int(num_train * 0.1) if num_train > 100 else 4
        num_test = int(num_train * 0.1) if num_train > 100 else 4
        
        args = {
            'root': root_path,
            'raw_file_name': 'protein_protein_with_type.csv',
            # 读取 debug.yaml 中的 max_nodes (15)，而不是默认的 50
            'min_nodes': getattr(cfg.dataset, 'min_nodes', 10),
            'max_nodes': getattr(cfg.dataset, 'max_nodes', 50)
        }
        
        # 传入动态计算的数量
        train_ds = PPIGraphDataset(split='train', num_subgraphs=num_train, **args)
        val_ds = PPIGraphDataset(split='val', num_subgraphs=num_val, **args)
        test_ds = PPIGraphDataset(split='test', num_subgraphs=num_test, **args)
        
        super().__init__(cfg, {'train': train_ds, 'val': val_ds, 'test': test_ds})

# ==============================================================================
# 3. DatasetInfos
# ==============================================================================
class PPIDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'ppi'
        
        # 自动统计分布
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()
        print(">>> 真实数据节点分布:", self.node_types)
        super().complete_infos(self.n_nodes, self.node_types)