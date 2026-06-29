from __future__ import annotations

import os
import time
from functools import partial
from pathlib import Path

from garplus_demo_runner import GarplusRunConfig, run_demo
from garplus_ml_predicates import MLPredicateConfig
from predicate_enrichment import PredicateEnrichmentConfig
from relation_sampled_loader import RelationGraphConfig, build_source_seed_pattern, load_relation_csv_graph, load_relation_sampled_pt_graph


BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_SUBDIR = "\u53bb\u75c5\u56fe\u6570\u636e"
DATA_DIR = Path(os.environ.get("GARPLUS_DATA_DIR", str(BASE_DIR / DEFAULT_DATA_SUBDIR)))
PROCESSED_DIR = Path(os.environ.get("GARPLUS_PROCESSED_DIR", str(BASE_DIR / "processed")))
PATTERN_DEBUG = os.environ.get("GARPLUS_PATTERN_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
PATTERN_DEBUG_LIMIT = int(os.environ.get("GARPLUS_PATTERN_DEBUG_LIMIT", "500"))


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
    source_node_index_column="index",
    target_node_index_column="index",
    load_node_attributes=True,
    source_edge_attr_columns=("geneid", "genesymbol"),
    target_edge_attr_columns=("diseaseid", "diseasename"),
    excluded_edge_attr_columns=("gene_index", "disease_index", "node_1", "node_2"),
)


CONFIG = GarplusRunConfig(
    dataset_name="TI",
    mode="fp-growth",
    # decision_tree_max_depth=4,
    fp_growth_max_itemset_size=4,
    interaction_csv_path=RELATION.edge_csv_path,
    node_csv_path=None,
    node_csv_label="node_csvs",
    sampled_pt_path=str(PROCESSED_DIR / "ti" / "ti_selected.pt"),
    sampled_graph_loader=partial(load_relation_sampled_pt_graph, RELATION),
    verification_graph_loader=partial(load_relation_csv_graph, RELATION),
    seed_builder=partial(build_source_seed_pattern, source_label="Gene"),
    fallback_interaction_name="gene_disease_signed.csv",
    fallback_node_name="gene.csv",
    force_edge_label="gene_disease",
    edge_label_column="EdgeLabel",
    pattern_bn_cache_path=str(PROCESSED_DIR / "ti" / "pattern_bn.pkl"),
    predicate_bn_focus_targets=("negative", "positive"),
    predicate_bn_cache_path=str(PROCESSED_DIR / "ti" / "predicate_bn_negative.pkl"),
    deduped_rules_output_path=str(PROCESSED_DIR / "ti" / "deduped_rules.txt"),
    include_ml_predicate_targets=False,
    include_edge_existing_target=False,
    undirected=False,
    undirected_pattern=False,
    topology_only_pattern_dedup=True,
    topology_dedupe_respect_direction=True,
    global_rematch_max_instances=None,
    global_match_scope="sampled",
    max_radius = 3,
    max_add_edge = 2,
    enable_rule_payload_generation=False,
    pattern_extension_only=PATTERN_DEBUG,
    pattern_extension_debug=PATTERN_DEBUG,
    pattern_extension_debug_limit=PATTERN_DEBUG_LIMIT,
    inject_sampled_frequent_patterns=not PATTERN_DEBUG,
    filter_degree_predicates=True,
    ignored_predicate_key_tokens=("interaction_label", "omimids","altgeneids","edge_existing","degree", "high_degree", "sampled_", "augmented_negative", "direction_role", "edgelabel", "ml_equivalence"),
    ignored_target_values=("unknown", "neutral"),
    drop_ignored_target_edges=True,
    predicate_enrichment=PredicateEnrichmentConfig(
        inference_edge_predicates=True,
        inference_presence_key="inferencegenesymbol",
    ),
    ml_predicates=MLPredicateConfig(
        enabled=True,
        equivalence_threshold=0.95,
        similarity_threshold=0.80,
        precomputed_edge_csv_path="/home/yyyy/codework/GARplus/enumeration-discovery/GARplusMiner/GARplus-ml-predicate/gene_disease_signed.csv",
        offline_csv_path=str(PROCESSED_DIR / "ti" / "ml_predicates.csv"),
    ),
)


def main() -> None:
    run_demo(CONFIG)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("running cost:", time.time() - start_time)
