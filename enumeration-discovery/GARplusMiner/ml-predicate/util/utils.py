"""Utilities for GAR+ ML predicate functions."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Mapping, Sequence


Object = Mapping[str, object]
ObjectPair = Sequence[Object]
MISSING_VALUES = {"", "-", "null", "none", "nan", "na", "n/a"}


def clean_text(value: object, *, lower: bool = False) -> str:
    text = str(value or "").strip()
    if text.lower() in MISSING_VALUES:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.lower() if lower else text


def split_values(value: object, *, lower: bool = False) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    text = text.replace("{", "").replace("}", "").replace("[", "").replace("]", "").replace('"', "")
    output: list[str] = []
    seen: set[str] = set()
    for piece in re.split(r"\||;|,", text):
        token = clean_text(piece, lower=lower)
        key = token.lower()
        if token and key not in seen:
            output.append(token)
            seen.add(key)
    return output


def values_for(obj: Object, fields: Iterable[str], *, lower: bool = True) -> set[str]:
    values: set[str] = set()
    for field in fields:
        raw = obj.get(field, "")
        single = clean_text(raw, lower=lower)
        if single:
            values.add(single)
        values.update(split_values(raw, lower=lower))
    values.discard("")
    return values


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def any_exact_match(left: Object, right: Object, field_pairs: Iterable[tuple[str, str]]) -> bool:
    for left_field, right_field in field_pairs:
        left_value = clean_text(left.get(left_field), lower=True)
        right_value = clean_text(right.get(right_field), lower=True)
        if left_value and left_value == right_value:
            return True
    return False


def any_overlap(left: Object, right: Object, field_pairs: Iterable[tuple[str, str]]) -> bool:
    for left_field, right_field in field_pairs:
        if values_for(left, (left_field,)) & values_for(right, (right_field,)):
            return True
    return False


def object_text(obj: Object, fields: Sequence[str], *, max_chars: int = 900) -> str:
    parts: list[str] = []
    for field in fields:
        value = clean_text(obj.get(field))
        if not value:
            continue
        if "|" in value or ";" in value or "," in value:
            value = "; ".join(split_values(value))
        parts.append(f"{field}: {value}")
    return ". ".join(parts)[:max_chars]


def validate_pairs(pairs: Sequence[ObjectPair]) -> None:
    for index, pair in enumerate(pairs):
        if len(pair) != 2:
            raise ValueError(f"Pair at row {index} must contain exactly two objects")

@lru_cache(maxsize=4)
def load_sentence_transformer(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install model dependencies with: py -3.12 -m pip install numpy sentence-transformers") from exc
    try:
        import torch  # type: ignore
    except ImportError:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_name, device=device)


def embedding_scores(text_pairs: Sequence[tuple[str, str]], model_name: str, batch_size: int = 64) -> list[float]:
    if not text_pairs:
        return []
    model = load_sentence_transformer(model_name)
    texts: list[str] = []
    for left, right in text_pairs:
        texts.append(left)
        texts.append(right)
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)

    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install numpy with: py -3.12 -m pip install numpy") from exc

    scores: list[float] = []
    for index in range(0, len(embeddings), 2):
        cosine = float(np.dot(embeddings[index], embeddings[index + 1]))
        scores.append(abs(max(-1.0, min(1.0, cosine))))
    return scores


def labels_from_scores(scores: Sequence[float], threshold: float) -> list[int]:
    return [int(score >= threshold) for score in scores]
