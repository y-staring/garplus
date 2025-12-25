import networkx as nx
import dgl

import pandas as pd
import csv

import torch


def save_dgl_graph(g, save_path):
    torch.save(g, save_path)


def reload_dgl_graph(save_path):
    return torch.load(save_path)


class CtdLoader:
    def load_nx_graph(self) -> nx.Graph:
        print("=====================using this loader=====================")
        raise "Not implemented"

    def load_homo_dgl_graph(self) -> dgl.DGLGraph:
        G = self.load_nx_graph()
        node_type_to_id = {}
        for node in G.nodes(data=True):
            node_type = node[1]['type']
            if node_type not in node_type_to_id:
                node_type_to_id[node_type] = len(node_type_to_id)
            G.nodes[node[0]]['type_id'] = node_type_to_id[node_type]
        edge_type_to_id = {}
        for u, v, data in G.edges(data=True):
            etype = data['type']
            if etype not in edge_type_to_id:
                edge_type_to_id[etype] = len(edge_type_to_id)
            G.edges[u, v]['type_id'] = edge_type_to_id[etype]

        return dgl.from_networkx(G, node_attrs=['type_id'], edge_attrs=['type_id'])


class CtdLoader1(CtdLoader):
    def __init__(self, node_files, edge_files):
        self.node_files = node_files
        self.edge_files = edge_files

    def load_nx_graph(self):
        # 初始化每种节点类型的全局索引
        global_node_id = 0
        global_to_local_id = {}
        local_to_global_id = {}

        G = nx.Graph()

        # 遍历每个节点文件
        for file in self.node_files:
            df = pd.read_csv(file)  # 读取 CSV 文件
            node_type = file.split('/')[-1].split('.')[0]  # 提取节点类型

            # 遍历每一行，添加节点到图中
            for _, row in df.iterrows():
                local_id = row['index']
                global_id = global_node_id
                global_to_local_id[global_id] = (node_type, local_id)
                local_to_global_id[(node_type, local_id)] = global_id
                node_attributes = row.drop('index').to_dict()
                G.add_node(global_id, type=node_type, **node_attributes)
                global_node_id += 1

        none_num = 0
        none_mapping = 0
        # 添加边到图中
        for file, src_type, dst_type in self.edge_files:
            df = pd.read_csv(file)  # 读取边文件
            visit_set = set({})
            for _, row in df.iterrows():
                # 获取 src 和 dst 对应的 id
                if row['src'] is None or row['dst'] is None:
                    none_num += 1
                    continue

                src_global_id = local_to_global_id.get(('gene', row['src'])) if src_type == 'gene' else \
                    local_to_global_id.get(('protein', row['src'])) if src_type == 'protein' else \
                        local_to_global_id.get(('drug', row['src'])) if src_type == 'drug' else \
                            local_to_global_id.get(('disease', row['src']))

                dst_global_id = local_to_global_id.get(('gene', row['dst'])) if dst_type == 'gene' else \
                    local_to_global_id.get(('protein', row['dst'])) if dst_type == 'protein' else \
                        local_to_global_id.get(('drug', row['dst'])) if dst_type == 'drug' else \
                            local_to_global_id.get(('disease', row['dst']))

                if src_global_id is None or dst_global_id is None:
                    # 如果边没有添加，记录该边的信息
                    none_mapping += 1
                    continue
                if src_global_id == dst_global_id:
                    # 去除自连边
                    none_mapping += 1
                    continue
                if src_type == dst_type:
                    # 去除同label的正向、反向边
                    if (src_global_id, dst_global_id) in visit_set or (dst_global_id, src_global_id) in visit_set:
                        none_mapping += 1
                        continue
                    visit_set.add((src_global_id, dst_global_id))

                G.add_edge(src_global_id, dst_global_id, type=f'{src_type}_{dst_type}')

        # 输出图的信息
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        G.remove_nodes_from(list(nx.isolates(G)))
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        G = G.to_directed()
        return G


class CtdLoader2(CtdLoader):
    def __init__(self, etypes, node_file, edge_file):
        # etypes为需要加载的边类型
        self.etypes = etypes
        self.node_file = node_file
        self.edge_file = edge_file

    def load_nx_graph(self):
        print("=====================using this loader=====================")
        def add_nodes(g, node_file):
            with open(node_file, 'r', encoding="utf-8") as f:
                csv_reader = csv.reader(f, delimiter=',')
                head = next(csv_reader)
                global_id_index = head.index("node_index")
                type_index = head.index("node_type")
                for row in csv_reader:
                    global_id = row[global_id_index].strip()
                    typ = row[type_index].strip()
                    g.add_node(int(global_id), type=typ)

        def add_edges(g, edge_file, total_etypes):
            with open(edge_file, 'r', encoding="utf-8") as f:
                visit_e_map = {}
                csv_reader = csv.reader(f, delimiter=',')
                head = next(csv_reader)
                src_index = head.index("x_index")
                dst_index = head.index("y_index")
                type_index = head.index("relation")
                for row in csv_reader:
                    src = row[src_index].strip()
                    dst = row[dst_index].strip()
                    typ = row[type_index].strip()
                    # 去除自连边
                    if src == dst:
                        continue
                    # 对于src、dst相同label的，去除反向边
                    typs = typ.split("_")
                    if len(typs) == 2 and typs[0] == typs[1]:
                        # src与dst同label
                        visit_set = visit_e_map.get(typ, set({}))
                        if (src, dst) in visit_set or (dst, src) in visit_set:
                            continue
                        visit_set.add((src, dst))
                        visit_e_map[typ] = visit_set

                    if typ in total_etypes:
                        g.add_edge(int(src), int(dst), type=typ)

        G = nx.Graph()
        add_nodes(G, self.node_file)
        add_edges(G, self.edge_file, self.etypes)

        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        G.remove_nodes_from(list(nx.isolates(G)))
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        G = G.to_directed()
        return G
