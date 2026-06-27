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
DEBUG_MATCH_EXPANSION = os.environ.get("GARPLUS_DEBUG_MATCH_EXPANSION", "1").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_TRANSACTION_COST = os.environ.get("GARPLUS_DEBUG_TRANSACTION_COST", "1").strip().lower() in {"1", "true", "yes", "on"}
DEBUG_SAMPLE_MATCHES = int(os.environ.get("GARPLUS_DEBUG_SAMPLE_MATCHES", "3"))


RELATION = RelationGraphConfig(
    relation_name="DDA",
    source_label="Drug",
    target_label="Disease",
    source_index_column="chemical_index",
    target_index_column="disease_index",
    default_edge_label="drug_disease",
    edge_csv_path=str(DATA_DIR / "drug_disease_signed.csv"),
    source_node_csv_path=str(DATA_DIR / "drug.csv"),
    target_node_csv_path=str(DATA_DIR / "disease.csv"),
    source_node_index_column="index",
    target_node_index_column="index",
    load_node_attributes=True,
    excluded_node_attr_columns=(
        "original_index",
        "source_node_id",
        "chemicalid",
        "chemicalname",
        "casrn",
        "synonyms",
        "description",
        "drug_interactions",
        "external_identifiers",
        "external_links",
        "general_references",
        "references",
        "patents",
    ),
    # These relation-table columns describe the Drug endpoint, not one interaction.
    source_edge_attr_columns=(),
    # These relation-table columns describe the Disease endpoint, not one interaction.
    target_edge_attr_columns=("diseasename", "diseaseid"),
    excluded_edge_attr_columns=("chemical_index", "disease_index", "node_1", "node_2"),
)


CONFIG = GarplusRunConfig(
    dataset_name="DDA",
    mode="pattern-only" if PATTERN_DEBUG else "decision-tree",
    interaction_csv_path=RELATION.edge_csv_path,
    node_csv_path=None,
    node_csv_label="node_csvs",
    sampled_pt_path=str(PROCESSED_DIR / "dda" / "dda_selected.pt"),
    sampled_graph_loader=partial(load_relation_sampled_pt_graph, RELATION),
    verification_graph_loader=partial(load_relation_csv_graph, RELATION),
    seed_builder=partial(build_source_seed_pattern, source_label="Drug"),
    fallback_interaction_name="drug_disease_signed.csv",
    fallback_node_name="drug.csv",
    force_edge_label="drug_disease",
    edge_label_column="EdgeLabel",
    pattern_bn_cache_path=str(PROCESSED_DIR / "dda" / "pattern_bn.pkl"),
    predicate_bn_focus_targets=("negative", "positive"),
    predicate_bn_cache_path=str(PROCESSED_DIR / "dda" / "predicate_bn_negative.pkl"),
    deduped_rules_output_path=str(PROCESSED_DIR / "dda" / "deduped_rules.txt"),
    enable_target_recall=False,
    enable_rule_payload_generation=False,
    include_ml_predicate_targets=False,
    include_edge_existing_target=True,
    undirected=False,
    undirected_pattern=False,
    max_radius = 2,
    max_add_edge = 2,
    topology_only_pattern_dedup=True,
    topology_dedupe_respect_direction=True,
    global_rematch_max_instances=None,
    pattern_extension_only=PATTERN_DEBUG,
    pattern_extension_debug=PATTERN_DEBUG,
    pattern_extension_debug_limit=PATTERN_DEBUG_LIMIT,
    debug_match_expansion=DEBUG_MATCH_EXPANSION,
    debug_transaction_cost=DEBUG_TRANSACTION_COST,
    debug_sample_matches=DEBUG_SAMPLE_MATCHES,
    inject_sampled_frequent_patterns=not PATTERN_DEBUG,
    filter_degree_predicates=True,
    ignored_predicate_key_tokens=(
        "interaction_label",
        "degree",
        "high_degree",
        "sampled_",
        "augmented_negative",
        "direction_role",
        "edgelabel",
        "ml_equivalence",
        "source_row_id",
        "pubmedids",
        "original_index",
        "source_node_id",
        "chemicalid",
        "chemicalname",
        "casrn",
        "synonyms",
        "description",
        "drug_interactions",
        "external_",
        "reference",
        "patents",
        "groups",
        "categories"
    ),
    drop_target_entity_features=True,
    ignored_target_values=("unknown", "neutral"),
    drop_ignored_target_edges=True,
    predicate_enrichment=PredicateEnrichmentConfig(
        inference_edge_predicates=True,
        inference_presence_key="inferencechemicalname",
    ),
    ml_predicates=MLPredicateConfig(
        enabled=True,
        equivalence_threshold=0.95,
        similarity_threshold=0.80,
        precomputed_edge_csv_path="/home/yyyy/codework/GARplus/enumeration-discovery/GARplusMiner/GARplus-ml-predicate/drug_disease_signed.csv",
        offline_csv_path=str(PROCESSED_DIR / "dda" / "ml_predicates.csv"),
    ),
)


def main() -> None:
    run_demo(CONFIG)


if __name__ == "__main__":
    start_time = time.time()
    main()
    print("running cost:", time.time() - start_time)
