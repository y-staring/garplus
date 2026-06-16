"""TI/gene-disease-specific field configuration and equivalence helpers."""

from __future__ import annotations

from typing import Mapping

from util.dda import DISEASE_FIELDS
from util.utils import clean_text, jaccard, values_for


GENE_FIELDS = (
    "GeneSymbol",
    "GeneName",
    "Synonyms",
    "GeneID",
    "PharmGKBIDs",
    "UniProtIDs",
    "source",
)

GENE_ID_FIELDS = ("Node_1", "GeneID")
DISEASE_ID_FIELDS = ("Node_2", "DiseaseID")
GENE_NAME_FIELDS = ("GeneSymbol", "GeneName", "Synonyms")
DISEASE_NAME_FIELDS = ("DiseaseName", "Synonyms")


def equivalence_score(gene: Mapping[str, object], disease: Mapping[str, object]) -> float:
    gene_ids = {clean_text(gene.get(field), lower=True) for field in GENE_ID_FIELDS}
    disease_ids = {clean_text(disease.get(field), lower=True) for field in DISEASE_ID_FIELDS}
    gene_ids.discard("")
    disease_ids.discard("")
    if gene_ids & disease_ids:
        return 1.0
    return jaccard(values_for(gene, GENE_NAME_FIELDS), values_for(disease, DISEASE_NAME_FIELDS))


def is_equivalent(gene: Mapping[str, object], disease: Mapping[str, object], *, threshold: float = 0.95) -> bool:
    return equivalence_score(gene, disease) >= threshold
