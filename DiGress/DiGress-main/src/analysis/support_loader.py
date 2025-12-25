# src/analysis/support_loader.py
import os
from src.datasets.epinions_dataset import _read_signed_digraph

# NOTE: 调整为你的真实路径
ROOT = "data/epinions"

def load_epinions_graph():
    txt_path = os.path.join(ROOT, "soc-sign-epinions.txt")
    G = _read_signed_digraph(txt_path)
    return G
