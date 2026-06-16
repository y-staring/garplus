"""PPI-specific field configuration and equivalence helpers."""

from __future__ import annotations

from typing import Mapping

from util.utils import any_exact_match, any_overlap, jaccard, values_for


PROTEIN_FIELDS = (
    "official_symbol",
    "synonyms",
    "protein_names",
    "Protein names",
    "Gene Names",
    "Gene Names (synonym)",
    "Gene Ontology IDs",
    "Keywords",
    "Protein families",
    "Function [CC]",
    "Involvement in disease",
)

PPI_EXACT_ID_PAIRS = (
    ("index", "index"),
    ("Entrez Gene Interactor", "Entrez Gene Interactor"),
    ("BioGRID ID Interactor", "BioGRID ID Interactor"),
    ("Official Symbol Interactor", "Official Symbol Interactor"),
    ("Systematic Name Interactor", "Systematic Name Interactor"),
    ("Entrez Gene ID", "Entrez Gene ID"),
    ("BioGRID ID", "BioGRID ID"),
    ("official_symbol", "official_symbol"),
)

PPI_ACCESSION_PAIRS = (
    ("SWISS-PROT Accessions", "SWISS-PROT Accessions"),
    ("TREMBL Accessions", "TREMBL Accessions"),
    ("REFSEQ Accessions", "REFSEQ Accessions"),
    ("UniProtIDs", "UniProtIDs"),
    ("Entry", "Entry"),
)

PPI_ALIAS_FIELDS = (
    "Official Symbol Interactor",
    "official_symbol",
    "Synonyms Interactor",
    "synonyms",
    "protein_names",
    "Protein names",
    "Gene Names",
    "Gene Names (synonym)",
)


def equivalence_score(left: Mapping[str, object], right: Mapping[str, object]) -> float:
    if any_exact_match(left, right, PPI_EXACT_ID_PAIRS) or any_overlap(left, right, PPI_ACCESSION_PAIRS):
        return 1.0
    return jaccard(values_for(left, PPI_ALIAS_FIELDS), values_for(right, PPI_ALIAS_FIELDS))


def is_equivalent(left: Mapping[str, object], right: Mapping[str, object], *, threshold: float = 0.80) -> bool:
    return equivalence_score(left, right) >= threshold
