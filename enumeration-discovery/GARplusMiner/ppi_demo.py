from __future__ import annotations

import os
import time
from pathlib import Path

from garplus_demo_runner import GarplusRunConfig, run_demo
from garplus_ml_predicates import MLPredicateConfig
from ppi_loader import build_ppi_seed_pattern, load_ppi_csv
from sampled_pt_loader import build_sampled_seed_pattern, load_sampled_pt_graph


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_SUBDIR = "\u53bb\u75c5\u56fe\u6570\u636e"
DATA_DIR = Path(os.environ.get("GARPLUS_DATA_DIR", str(BASE_DIR / DEFAULT_DATA_SUBDIR)))
PROCESSED_DIR = Path(os.environ.get("GARPLUS_PROCESSED_DIR", str(BASE_DIR / "processed")))


CONFIG = GarplusRunConfig(
    dataset_name="PPI",
    interaction_csv_path=str(DATA_DIR / "protein_protein_signed.csv"),
    node_csv_path=str(DATA_DIR / "protein.csv"),
    node_csv_label="protein_csv",
    sampled_pt_path=str(PROCESSED_DIR / "ppi" / "ppi_selected.pt"),
    sampled_graph_loader=load_sampled_pt_graph,
    csv_graph_loader=load_ppi_csv,
    seed_builder=build_sampled_seed_pattern,
    fallback_interaction_name="protein_protein_signed.csv",
    fallback_node_name="protein.csv",
    force_edge_label="candidate_interaction",
    edge_label_column="Experimental System",
    pattern_bn_cache_path=str(PROCESSED_DIR / "ppi" / "pattern_bn.pkl"),
    predicate_bn_cache_path=str(PROCESSED_DIR / "ppi" / "predicate_bn_negative.pkl"),
    ml_predicates=MLPredicateConfig(
        enabled=True,
        equivalence_threshold=0.80,
        similarity_threshold=0.85,
    ),
)


def main() -> None:
    run_demo(CONFIG)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("running cost:", time.time() - start_time)
