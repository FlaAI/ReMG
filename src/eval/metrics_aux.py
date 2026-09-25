"""Auxiliary 2x2 intervention metrics for P/R eval reports.

"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean
from typing import Any


def _normalized_prediction(row: dict[str, Any]) -> str:
    """Return a comparable prediction string for consistency checks."""

    value = row.get("normalized_prediction") or row.get("raw_prediction") or ""
    return str(value).strip().lower()


def content_soft_consistency_rate(scored_rows: list[dict[str, Any]]) -> float:
    """Fraction of fixed-(quad,s) content pairs with identical predictions across z."""

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in scored_rows:
        if row["query_type"] != "content":
            continue
        key = (row["quadruplet_id"], str(row["content_id"]))
        groups[key].append(_normalized_prediction(row))

    if not groups:
        return 0.0
    consistent = [len(set(predictions)) == 1 for predictions in groups.values()]
    return safe_mean([float(value) for value in consistent])


def content_worst_z_accuracy(scored_rows: list[dict[str, Any]]) -> float:
    """Mean over fixed-(quad,s) of min correctness across z variants."""

    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in scored_rows:
        if row["query_type"] != "content":
            continue
        key = (row["quadruplet_id"], str(row["content_id"]))
        groups[key].append(bool(row["is_correct"]))

    if not groups:
        return 0.0
    return safe_mean([float(min(values)) for values in groups.values()])


def paralinguistic_soft_consistency_rate(scored_rows: list[dict[str, Any]]) -> float:
    """Fraction of fixed-(quad,z) para pairs with identical predictions across s."""

    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in scored_rows:
        if row["query_type"] != "paralinguistic":
            continue
        key = (row["quadruplet_id"], str(row["paralinguistic_label"]))
        groups[key].append(_normalized_prediction(row))

    if not groups:
        return 0.0
    consistent = [len(set(predictions)) == 1 for predictions in groups.values()]
    return safe_mean([float(value) for value in consistent])


def paralinguistic_worst_s_accuracy(scored_rows: list[dict[str, Any]]) -> float:
    """Mean over fixed-(quad,z) of min correctness across the two content stems."""

    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in scored_rows:
        if row["query_type"] != "paralinguistic":
            continue
        key = (row["quadruplet_id"], str(row["paralinguistic_label"]))
        groups[key].append(bool(row["is_correct"]))

    if not groups:
        return 0.0
    return safe_mean([float(min(values)) for values in groups.values()])


def safe_mean(values: list[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty list."""

    if not values:
        return 0.0
    return float(mean(values))


def compute_auxiliary_metric_bundle(scored_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute auxiliary intervention-matrix metrics from scored rows."""

    return {
        "content_soft_consistency_rate": content_soft_consistency_rate(scored_rows),
        "content_worst_z_accuracy": content_worst_z_accuracy(scored_rows),
        "paralinguistic_soft_consistency_rate": paralinguistic_soft_consistency_rate(
            scored_rows
        ),
        "paralinguistic_worst_s_accuracy": paralinguistic_worst_s_accuracy(scored_rows),
    }


def build_r_modality_gap_metrics(
    *,
    content_text_accuracy: float,
    content_audio_accuracy: float,
) -> dict[str, float]:
    """Derive R-layer text-vs-audio gap metrics for one Base run."""

    absolute_gap = content_text_accuracy - content_audio_accuracy
    text_retention = (
        content_audio_accuracy / content_text_accuracy
        if content_text_accuracy > 0
        else 0.0
    )
    gap_closure = text_retention
    return {
        "content_text_accuracy": content_text_accuracy,
        "content_audio_accuracy": content_audio_accuracy,
        "absolute_modality_gap": absolute_gap,
        "text_retention": text_retention,
        "gap_closure": gap_closure,
    }
