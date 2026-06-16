from __future__ import annotations

import os
import time
from functools import partial
from pathlib import Path

from garplus_demo_runner import GarplusRunConfig, run_demo
from garplus_ml_predicates import MLPredicateConfig
from relation_sampled_loader import RelationGraphConfig, build_source_seed_pattern, load_relation_sampled_pt_graph


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_SUBDIR = "\u53bb\u75c5\u56fe\u6570\u636e"
DATA_DIR = Path(os.environ.get("GARPLUS_DATA_DIR", str(BASE_DIR / DEFAULT_DATA_SUBDIR)))
PROCESSED_DIR = Path(os.environ.get("GARPLUS_PROCESSED_DIR", str(BASE_DIR / "processed")))


RELATION = RelationGraphConfig(
    relation_name="TI",
    source_label="Gene",
    target_label="Disease",
    source_index_column="gene_index",
    target_index_column="disease_index",
    default_edge_label="gene_disease",
    edge_csv_path=str(DATA_DIR / "gene_disease_signed.csv"),
    source_node_csv_path=str(DATA_DIR / "gene.csv"),
    target_node_csv_path=str(DATA_DIR / "disease.csv"),
    load_node_attributes=False,
)


CONFIG = GarplusRunConfig(
    dataset_name="TI",
    interaction_csv_path=RELATION.edge_csv_path,
    node_csv_path=None,
    node_csv_label="node_csvs",
    sampled_pt_path=str(PROCESSED_DIR / "ti" / "ti_selected.pt"),
    sampled_graph_loader=partial(load_relation_sampled_pt_graph, RELATION),
    seed_builder=partial(build_source_seed_pattern, source_label="Gene"),
    fallback_interaction_name="gene_disease_signed.csv",
    fallback_node_name="gene.csv",
    force_edge_label="gene_disease",
    edge_label_column="EdgeLabel",
    pattern_bn_cache_path=str(PROCESSED_DIR / "ti" / "pattern_bn.pkl"),
    predicate_bn_cache_path=str(PROCESSED_DIR / "ti" / "predicate_bn_negative.pkl"),
    ml_predicates=MLPredicateConfig(
        enabled=True,
        equivalence_threshold=0.95,
        similarity_threshold=0.80,
    ),
)


def main() -> None:
    run_demo(CONFIG)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("running cost:", time.time() - start_time)
