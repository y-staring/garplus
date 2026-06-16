"""Entity equivalence labels for GAR+."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping, Sequence

from util import dda, ppi, ti
from util.utils import ObjectPair, validate_pairs


def ppi_equivalence_scores(pairs: Sequence[ObjectPair]) -> list[float]:
    validate_pairs(pairs)
    return [ppi.equivalence_score(left, right) for left, right in pairs]


def ppi_equivalence_labels(pairs: Sequence[ObjectPair], *, threshold: float = 0.80) -> list[int]:
    return [int(score >= threshold) for score in ppi_equivalence_scores(pairs)]


def dda_equivalence_scores(pairs: Sequence[ObjectPair]) -> list[float]:
    validate_pairs(pairs)
    return [dda.equivalence_score(drug, disease) for drug, disease in pairs]


def dda_equivalence_labels(pairs: Sequence[ObjectPair], *, threshold: float = 0.95) -> list[int]:
    return [int(score >= threshold) for score in dda_equivalence_scores(pairs)]


def ti_equivalence_scores(pairs: Sequence[ObjectPair]) -> list[float]:
    validate_pairs(pairs)
    return [ti.equivalence_score(gene, disease) for gene, disease in pairs]


def ti_equivalence_labels(pairs: Sequence[ObjectPair], *, threshold: float = 0.95) -> list[int]:
    return [int(score >= threshold) for score in ti_equivalence_scores(pairs)]


def equivalence_labels(
    tables: Mapping[str, Sequence[ObjectPair]],
    *,
    ppi_threshold: float = 0.80,
    dda_threshold: float = 0.95,
    ti_threshold: float = 0.95,
) -> dict[str, list[int]]:
    return {
        "ppi": ppi_equivalence_labels(tables.get("ppi", []), threshold=ppi_threshold),
        "dda": dda_equivalence_labels(tables.get("dda", []), threshold=dda_threshold),
        "ti": ti_equivalence_labels(tables.get("ti", []), threshold=ti_threshold),
    }


def _ppi_pair(row: Mapping[str, object]) -> ObjectPair:
    return (
        {
            "index": row.get("index_A", ""),
            "Entrez Gene Interactor": row.get("Entrez Gene Interactor A", ""),
            "BioGRID ID Interactor": row.get("BioGRID ID Interactor A", ""),
            "Official Symbol Interactor": row.get("Official Symbol Interactor A", ""),
            "Systematic Name Interactor": row.get("Systematic Name Interactor A", ""),
            "SWISS-PROT Accessions": row.get("SWISS-PROT Accessions Interactor A", ""),
            "TREMBL Accessions": row.get("TREMBL Accessions Interactor A", ""),
            "REFSEQ Accessions": row.get("REFSEQ Accessions Interactor A", ""),
            "Synonyms Interactor": row.get("Synonyms Interactor A", ""),
        },
        {
            "index": row.get("index_B", ""),
            "Entrez Gene Interactor": row.get("Entrez Gene Interactor B", ""),
            "BioGRID ID Interactor": row.get("BioGRID ID Interactor B", ""),
            "Official Symbol Interactor": row.get("Official Symbol Interactor B", ""),
            "Systematic Name Interactor": row.get("Systematic Name Interactor B", ""),
            "SWISS-PROT Accessions": row.get("SWISS-PROT Accessions Interactor B", ""),
            "TREMBL Accessions": row.get("TREMBL Accessions Interactor B", ""),
            "REFSEQ Accessions": row.get("REFSEQ Accessions Interactor B", ""),
            "Synonyms Interactor": row.get("Synonyms Interactor B", ""),
        },
    )


def _write_csv_labels(
    dataset: str,
    input_path: Path,
    output_path: Path,
    threshold: float,
    score_func,
    pair_func,
) -> dict[str, object]:
    rows = ones = zeros = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as in_file, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_file:
        reader = csv.DictReader(in_file)
        fieldnames = [name for name in reader.fieldnames or [] if name not in {"equivalence_score", "equivalence_pred"}]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames + ["equivalence_score", "equivalence_pred"])
        writer.writeheader()
        for row in reader:
            score = score_func(*pair_func(row))
            label = int(score >= threshold)
            row = {name: row.get(name, "") for name in fieldnames}
            row["equivalence_score"] = f"{score:.6f}"
            row["equivalence_pred"] = label
            writer.writerow(row)
            rows += 1
            ones += label
            zeros += 1 - label
    return {
        "dataset": dataset,
        "input": str(input_path),
        "output": str(output_path),
        "rows": rows,
        "equivalence_1": ones,
        "equivalence_0": zeros,
        "threshold": threshold,
    }


def run_csv(
    input_dir: Path,
    output_dir: Path,
    *,
    ppi_threshold: float = 0.80,
    dda_threshold: float = 0.95,
    ti_threshold: float = 0.95,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = [
        _write_csv_labels(
            "ppi",
            input_dir / "protein_protein_signed.csv",
            output_dir / "protein_protein_signed.csv",
            ppi_threshold,
            ppi.equivalence_score,
            _ppi_pair,
        ),
        _write_csv_labels(
            "dda",
            input_dir / "drug_disease_signed.csv",
            output_dir / "drug_disease_signed.csv",
            dda_threshold,
            dda.equivalence_score,
            lambda row: (row, row),
        ),
        _write_csv_labels(
            "ti",
            input_dir / "gene_disease_signed.csv",
            output_dir / "gene_disease_signed.csv",
            ti_threshold,
            ti.equivalence_score,
            lambda row: (row, row),
        ),
    ]
    with (output_dir / "equivalence_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=root / "rudik" / "GAR+数据")
    parser.add_argument("--output-dir", type=Path, default=root / "rudik" / "GARplus_data_equivalence")
    parser.add_argument("--ppi-threshold", type=float, default=0.80)
    parser.add_argument("--dda-threshold", type=float, default=0.95)
    parser.add_argument("--ti-threshold", type=float, default=0.95)
    args = parser.parse_args()
    summary = run_csv(
        args.input_dir,
        args.output_dir,
        ppi_threshold=args.ppi_threshold,
        dda_threshold=args.dda_threshold,
        ti_threshold=args.ti_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
