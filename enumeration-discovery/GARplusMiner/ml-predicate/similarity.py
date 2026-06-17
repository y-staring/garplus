"""Semantic similarity/relatedness labels for GAR+."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from util import dda, ppi, ti
from util.utils import (
    ObjectPair,
    embedding_scores,
    labels_from_scores,
    object_text,
    validate_pairs,
)


DEFAULT_EMBEDDING_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"


def _model_labels(
    pairs: Sequence[ObjectPair],
    left_fields: Sequence[str],
    right_fields: Sequence[str],
    *,
    threshold: float,
    model_name: str,
    batch_size: int,
    max_chars: int,
) -> list[int]:
    validate_pairs(pairs)
    text_pairs = [
        (object_text(left, left_fields, max_chars=max_chars), object_text(right, right_fields, max_chars=max_chars))
        for left, right in pairs
    ]
    scores = embedding_scores(text_pairs, model_name=model_name, batch_size=batch_size)
    return labels_from_scores(scores, threshold)


def ppi_similarity_labels(
    pairs: Sequence[ObjectPair],
    *,
    threshold: float = 0.94,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    max_chars: int = 900,
) -> list[int]:
    return _model_labels(
        pairs,
        ppi.PROTEIN_FIELDS,
        ppi.PROTEIN_FIELDS,
        threshold=threshold,
        model_name=model_name,
        batch_size=batch_size,
        max_chars=max_chars,
    )


def dda_relatedness_labels(
    pairs: Sequence[ObjectPair],
    *,
    threshold: float = 0.90,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    max_chars: int = 900,
) -> list[int]:
    return _model_labels(
        pairs,
        dda.DRUG_FIELDS,
        dda.DISEASE_FIELDS,
        threshold=threshold,
        model_name=model_name,
        batch_size=batch_size,
        max_chars=max_chars,
    )


def ti_relatedness_labels(
    pairs: Sequence[ObjectPair],
    *,
    threshold: float = 1.00,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    max_chars: int = 900,
) -> list[int]:
    return _model_labels(
        pairs,
        ti.GENE_FIELDS,
        ti.DISEASE_FIELDS,
        threshold=threshold,
        model_name=model_name,
        batch_size=batch_size,
        max_chars=max_chars,
    )


dda_similarity_labels = dda_relatedness_labels
ti_similarity_labels = ti_relatedness_labels


def similarity_labels(
    tables: Mapping[str, Sequence[ObjectPair]],
    *,
    ppi_threshold: float = 0.94,
    dda_threshold: float = 0.90,
    ti_threshold: float = 1.00,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    max_chars: int = 900,
) -> dict[str, list[int]]:
    return {
        "ppi": ppi_similarity_labels(
            tables.get("ppi", []),
            threshold=ppi_threshold,
            model_name=model_name,
            batch_size=batch_size,
            max_chars=max_chars,
        ),
        "dda": dda_relatedness_labels(
            tables.get("dda", []),
            threshold=dda_threshold,
            model_name=model_name,
            batch_size=batch_size,
            max_chars=max_chars,
        ),
        "ti": ti_relatedness_labels(
            tables.get("ti", []),
            threshold=ti_threshold,
            model_name=model_name,
            batch_size=batch_size,
            max_chars=max_chars,
        ),
    }


def _ppi_pair(row: Mapping[str, object]) -> ObjectPair:
    return (
        {
            "official_symbol": row.get("Official Symbol Interactor A", ""),
            "synonyms": row.get("Synonyms Interactor A", ""),
            "SWISS-PROT Accessions": row.get("SWISS-PROT Accessions Interactor A", ""),
            "TREMBL Accessions": row.get("TREMBL Accessions Interactor A", ""),
            "REFSEQ Accessions": row.get("REFSEQ Accessions Interactor A", ""),
        },
        {
            "official_symbol": row.get("Official Symbol Interactor B", ""),
            "synonyms": row.get("Synonyms Interactor B", ""),
            "SWISS-PROT Accessions": row.get("SWISS-PROT Accessions Interactor B", ""),
            "TREMBL Accessions": row.get("TREMBL Accessions Interactor B", ""),
            "REFSEQ Accessions": row.get("REFSEQ Accessions Interactor B", ""),
        },
    )


def _write_csv_labels(
    dataset: str,
    input_path: Path,
    output_path: Path,
    left_fields: Sequence[str],
    right_fields: Sequence[str],
    threshold: float,
    pair_func,
    *,
    model_name: str,
    batch_size: int,
    chunk_size: int,
    max_chars: int,
) -> dict[str, object]:
    started = time.time()
    rows = 0
    total = 0.0
    score_min = None
    score_max = None
    pred_counts = Counter()
    interaction_counts = Counter()
    batch_rows: list[dict[str, object]] = []
    batch_pairs: list[tuple[str, str]] = []

    def flush(writer: csv.DictWriter) -> None:
        nonlocal total, score_min, score_max
        if not batch_rows:
            return
        scores = embedding_scores(batch_pairs, model_name=model_name, batch_size=batch_size)
        labels = labels_from_scores(scores, threshold)
        for row, score, label in zip(batch_rows, scores, labels):
            row["similarity_score"] = f"{score:.6f}"
            row["similarity_pred"] = label
            writer.writerow(row)
            pred_counts[label] += 1
            total += score
            score_min = score if score_min is None else min(score_min, score)
            score_max = score if score_max is None else max(score_max, score)
        batch_rows.clear()
        batch_pairs.clear()

    with input_path.open("r", encoding="utf-8-sig", newline="") as in_file, output_path.open(
        "w", encoding="utf-8", newline=""
    ) as out_file:
        reader = csv.DictReader(in_file)
        fieldnames = [name for name in reader.fieldnames or [] if name not in {"similarity_score", "similarity_pred"}]
        writer = csv.DictWriter(out_file, fieldnames=fieldnames + ["similarity_score", "similarity_pred"])
        writer.writeheader()
        for row in reader:
            rows += 1
            interaction_counts[row.get("interaction_label", "")] += 1
            left, right = pair_func(row)
            batch_rows.append({name: row.get(name, "") for name in fieldnames})
            batch_pairs.append(
                (
                    object_text(left, left_fields, max_chars=max_chars),
                    object_text(right, right_fields, max_chars=max_chars),
                )
            )
            if len(batch_rows) >= chunk_size:
                flush(writer)
        flush(writer)

    return {
        "dataset": dataset,
        "model": model_name,
        "input": str(input_path),
        "output": str(output_path),
        "rows": rows,
        "similarity_1": pred_counts[1],
        "similarity_0": pred_counts[0],
        "threshold": threshold,
        "score_min": round(score_min or 0.0, 6),
        "score_max": round(score_max or 0.0, 6),
        "score_mean": round(total / rows, 6) if rows else 0.0,
        "interaction_labels": dict(interaction_counts),
        "seconds": round(time.time() - started, 2),
    }


def run_csv(
    input_dir: Path,
    output_dir: Path,
    *,
    ppi_threshold: float = 0.94,
    dda_threshold: float = 0.90,
    ti_threshold: float = 1.00,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    chunk_size: int = 2048,
    max_chars: int = 900,
) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = (
        (
            "ppi",
            "protein_protein_signed.csv",
            ppi.PROTEIN_FIELDS,
            ppi.PROTEIN_FIELDS,
            ppi_threshold,
            _ppi_pair,
        ),
        ("dda", "drug_disease_signed.csv", dda.DRUG_FIELDS, dda.DISEASE_FIELDS, dda_threshold, lambda row: (row, row)),
        ("ti", "gene_disease_signed.csv", ti.GENE_FIELDS, ti.DISEASE_FIELDS, ti_threshold, lambda row: (row, row)),
    )
    summary = [
        _write_csv_labels(
            dataset,
            input_dir / filename,
            output_dir / filename,
            left_fields,
            right_fields,
            threshold,
            pair_func,
            model_name=model_name,
            batch_size=batch_size,
            chunk_size=chunk_size,
            max_chars=max_chars,
        )
        for dataset, filename, left_fields, right_fields, threshold, pair_func in jobs
    ]
    with (output_dir / "similarity_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=root / "rudik" / "GAR+\u6570\u636e")
    parser.add_argument("--output-dir", type=Path, default=root / "rudik" / "GARplus_data_similarity")
    parser.add_argument("--ppi-threshold", type=float, default=0.94)
    parser.add_argument("--dda-threshold", type=float, default=0.90)
    parser.add_argument("--ti-threshold", type=float, default=1.00)
    parser.add_argument("--model-name", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=2048)
    parser.add_argument("--max-chars", type=int, default=900)
    args = parser.parse_args()
    summary = run_csv(
        args.input_dir,
        args.output_dir,
        ppi_threshold=args.ppi_threshold,
        dda_threshold=args.dda_threshold,
        ti_threshold=args.ti_threshold,
        model_name=args.model_name,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        max_chars=args.max_chars,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
