import pandas as pd
from pathlib import Path
import os

# ========= 0. 原始文件路径 =========
FILE_PATH = Path(r"D:\CodeWork\python\GAR+\数据\去病图数据\去病图数据\protein_protein.csv")

# ========= 1. 规则定义（和你之前的一致） =========

# 1.1 Experimental System：正向
positive_exp_system = {
    "Affinity Capture-MS",
    "Two-hybrid",
    "Affinity Capture-Western",
    "Reconstituted Complex",
    "Proximity Label-MS",
    "Biochemical Activity",
    "Co-purification",
    "Co-fractionation",
    "Co-localization",
    "PCA",
    "Co-crystal Structure",
    "Far Western",
    "Protein-peptide",
    "Protein-RNA",
    "Affinity Capture-RNA",
    "FRET",
    "Affinity Capture-Luminescence",
    "Positive Genetic",
    "Phenotypic Enhancement",
    "Phenotypic Suppression",
    "Dosage Rescue",
    "Synthetic Rescue",
}

# 1.2 Experimental System：负向（遗传互作里典型的负向类型）
negative_exp_system = {
    "Synthetic Lethality",
    "Negative Genetic",
    "Synthetic Growth Defect",
    "Dosage Lethality",
    "Dosage Growth Defect",
}

# 1.3 Modification：正向修饰（通常代表激活或功能增强）
positive_modifications = {
    "Methylation",
    "Ubiquitination",
    "Phosphorylation",
    "Ribosylation",
    "Sumoylation",
    "Acetylation",
    "FAT10ylation",
    "Nedd(Rub1)ylation",
    "Glycosylation",
    "Neddylation",
}

# 1.4 Modification：负向修饰（通常代表去激活/破坏/降解）
negative_modifications = {
    "Deubiquitination",
    "Proteolytic Processing",
    "Desumoylation",
    "Deacetylation",
    "Dephosphorylation",
    "Demethylation",
    "Deneddylation",
}

# 1.5 Ontology Term Types：正向关键词（代表“救回/正常状态”）
positive_onto_type_keywords = ["partial rescue", "wild type"]

# 1.6 Ontology Term Names：负向关键词（扩展 negative 语义的重点）
negative_onto_name_keywords = [
    "abnormal",
    "defect",
    "defective",
    "negative regulation",
    "downregulation",
    "down-regulation",
    "reduced",
    "loss of",
    "decrease",
    "decreased",
    "impaired",
    "resistance",
    "toxicity",
]


# ========= 2. 判定正/负语义 =========
def classify_edge_with_reason(row):
    es = str(row.get("Experimental System", "") or "")
    mod = str(row.get("Modification", "") or "")
    onto_type = str(row.get("Ontology Term Types", "") or "")
    onto_name = str(row.get("Ontology Term Names", "") or "")

    is_pos = False
    is_neg = False

    # 2.1 Experimental System
    if es in positive_exp_system:
        is_pos = True
    if es in negative_exp_system:
        is_neg = True

    # 2.2 Modification
    if mod in positive_modifications:
        is_pos = True
    if mod in negative_modifications:
        is_neg = True

    # 2.3 Ontology Term Types：partial rescue / wild type 视为正向
    lt = onto_type.lower()
    for k in positive_onto_type_keywords:
        if k in lt:
            is_pos = True
            break

    # 2.4 Ontology Term Names：负向关键词
    ln = onto_name.lower()
    for k in negative_onto_name_keywords:
        if k in ln:
            is_neg = True
            break

    # 决策：这里只关心 positive / negative / neutral
    if is_pos and not is_neg:
        return "positive"
    elif is_neg and not is_pos:
        return "negative"
    elif is_pos and is_neg:
        # 有冲突的先当 neutral，后面会过滤掉
        return "neutral"
    else:
        return "neutral"


# ========= 3. 读原始 CSV，打标签 =========
df = pd.read_csv(FILE_PATH, sep=None, engine="python")

# 确保关键列存在
required_cols = [
    "Experimental System",
    "Experimental System Type",
    "Modification",
    "Ontology Term Types",
    "Ontology Term Names",
    "index_A",
    "index_B",
]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"缺少必要字段: {c}")

df["edge_semantic"] = df.apply(classify_edge_with_reason, axis=1)

pos_df = df[df["edge_semantic"] == "positive"][["index_A", "index_B"]].copy()
neg_df = df[df["edge_semantic"] == "negative"][["index_A", "index_B"]].copy()

print("正语义边数量:", len(pos_df))
print("负语义边数量:", len(neg_df))

# 如果你只想用真正有语义的边，可以只保留 pos_df / neg_df
pos_neg_df = pd.concat(
    [
        pos_df.assign(sign=1),
        neg_df.assign(sign=-1),
    ],
    ignore_index=True,
)

# ========= 4. 重新映射节点 ID（node_id 从 0 连续编号） =========
# 只考虑出现在正/负边里的节点
all_nodes = pd.unique(
    pd.concat([pos_neg_df["index_A"], pos_neg_df["index_B"]], ignore_index=True)
)
all_nodes_sorted = sorted(all_nodes)

old2new = {old_id: new_id for new_id, old_id in enumerate(all_nodes_sorted)}

# ========= 5. 生成 node.csv =========
node_df = pd.DataFrame({
    "node_id": [old2new[n] for n in all_nodes_sorted]
})

os.makedirs("data_signed", exist_ok=True)
node_df.to_csv("data_signed/node.csv", index=False)
print("保存 node.csv 到 data_signed/node.csv，节点数：", len(node_df))

# ========= 6. 生成 edges.csv =========
edges_out = pd.DataFrame({
    "src": pos_neg_df["index_A"].map(old2new),
    "dst": pos_neg_df["index_B"].map(old2new),
    "rel": "protein_protein",          # 统一关系名
    "sign": pos_neg_df["sign"],        # 1 / -1
})

edges_out.to_csv("data_signed/edges.csv", index=False)
print("保存 edges.csv 到 data_signed/edges.csv，边数：", len(edges_out))
print(edges_out.shape[0])
print(edges_out.head())
