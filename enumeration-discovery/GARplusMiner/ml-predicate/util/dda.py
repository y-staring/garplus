"""DDA-specific field configuration and equivalence helpers."""

from __future__ import annotations

from typing import Mapping

from util.utils import clean_text, jaccard, values_for


DRUG_FIELDS = (
    "ChemicalName",
    "Synonyms",
    "name",
    "description",
    "indication",
    "mechanism-of-action",
    "categories",
    "atc-codes",
    "targets",
    "enzymes",
    "pathways",
)

DISEASE_FIELDS = (
    "DiseaseName",
    "Synonyms",
    "Definition",
    "Categories",
    "SlimMappings",
    "ParentIDs",
    "TreeNumbers",
    "AltDiseaseIDs",
)

DRUG_ID_FIELDS = ("Node_1", "ChemicalID")
DISEASE_ID_FIELDS = ("Node_2", "DiseaseID")
DRUG_NAME_FIELDS = ("ChemicalName", "Synonyms", "name")
DISEASE_NAME_FIELDS = ("DiseaseName", "Synonyms")


def equivalence_score(drug: Mapping[str, object], disease: Mapping[str, object]) -> float:
    drug_ids = {clean_text(drug.get(field), lower=True) for field in DRUG_ID_FIELDS}
    disease_ids = {clean_text(disease.get(field), lower=True) for field in DISEASE_ID_FIELDS}
    drug_ids.discard("")
    disease_ids.discard("")
    if drug_ids & disease_ids:
        return 1.0
    return jaccard(values_for(drug, DRUG_NAME_FIELDS), values_for(disease, DISEASE_NAME_FIELDS))


def is_equivalent(drug: Mapping[str, object], disease: Mapping[str, object], *, threshold: float = 0.95) -> bool:
    return equivalence_score(drug, disease) >= threshold
