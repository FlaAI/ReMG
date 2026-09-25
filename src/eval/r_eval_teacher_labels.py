"""Teacher-text labels and stratified reporting for frozen R eval unique s.

"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_report
from realmg.data.r_text_teacher import build_teacher_example
from realmg.eval.p_harness import compute_metric_bundle, safe_mean
from realmg.eval.r_eval_text import build_r_eval_text_examples, r_eval_text_examples_path
from realmg.eval.r_harness import cluster_bootstrap_r_metric_bundle, write_jsonl
from realmg.eval.r_verifiers import GeneralVerifierScorer, verify_teacher_prediction


R_EVAL_TEACHER_TEXT_EXAMPLES_PATH = "Data/manifests/r_eval_teacher_text_examples.jsonl"
R_EVAL_TEACHER_LABELS_PATH = "Data/manifests/r_eval_teacher_text_labels.jsonl"
R_EVAL_TEACHER_LABELS_REGISTRY = "Data/cards/r_eval_teacher_text_examples.json"


def r_eval_teacher_text_examples_file() -> Path:
    """Return the eval Teacher-text examples manifest."""

    return resolve_from_repo_root(R_EVAL_TEACHER_TEXT_EXAMPLES_PATH)


def r_eval_teacher_labels_file(predictions_file: str | None = None) -> Path:
    """Return the scored Teacher-text labels path.

    If predictions_file is given, write/read a sibling labels file so models
    never share one labels JSONL. Legacy qwen2.5 file remains at the unslugged path.
    """

    if predictions_file is None:
        return resolve_from_repo_root(R_EVAL_TEACHER_LABELS_PATH)
    predictions_path = resolve_from_repo_root(predictions_file)
    stem = predictions_path.stem
    if stem.startswith("r_eval_text_predictions_"):
        slug = stem[len("r_eval_text_predictions_") :]
        return predictions_path.with_name(f"r_eval_teacher_text_labels_{slug}.jsonl")
    if stem.startswith("r_eval_teacher_text_predictions_"):
        slug = stem[len("r_eval_teacher_text_predictions_") :]
        return predictions_path.with_name(f"r_eval_teacher_text_labels_{slug}.jsonl")
    return predictions_path.with_name(stem.replace("predictions", "labels") + ".jsonl")


def prepare_r_eval_teacher_text_examples() -> None:
    """Export one Teacher-text example per unique eval content_id."""

    examples, registry = build_r_eval_text_examples()
    teacher_examples = [
        build_teacher_example(
            {
                "content_id": example["content_id"],
                "source_name": example["source_name"],
                "prompt_text": example["prompt_text"],
                "answer_text": example["reference_answer"],
                "metadata": example.get("metadata", {}),
            }
        )
        for example in examples
    ]
    write_jsonl(teacher_examples, r_eval_teacher_text_examples_file())
    write_report(
        {
            **registry,
            "stage": "r_eval_teacher_text_prepare",
            "output_manifest": R_EVAL_TEACHER_TEXT_EXAMPLES_PATH,
        },
        resolve_from_repo_root(R_EVAL_TEACHER_LABELS_REGISTRY),
    )
    print(
        f"[prepare-r-eval-teacher-text] unique_s={len(teacher_examples)}",
        flush=True,
    )


def build_teacher_label_rows(
    examples: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Verify Teacher-text predictions for eval unique s."""

    example_by_id = {row["example_id"]: row for row in examples}
    scorer = GeneralVerifierScorer()
    labels: list[dict[str, Any]] = []
    for prediction_row in predictions:
        example = example_by_id[prediction_row["example_id"]]
        prediction_text = str(
            prediction_row.get("prediction_text")
            or prediction_row.get("prediction")
            or ""
        ).strip()
        result = verify_teacher_prediction(
            source_name=example["source_name"],
            prompt_text=example["prompt_text"],
            reference_answer=example["reference_answer"],
            prediction_text=prediction_text,
            general_verifier=scorer,
        )
        labels.append(
            {
                "content_id": example["content_id"],
                "source_name": example["source_name"],
                "teacher_text_correct": bool(result.is_correct),
                "teacher_text_verifier": result.verifier_method,
                "teacher_text_prediction": prediction_text,
                "normalized_prediction": result.normalized_prediction,
            }
        )
    return labels


def filter_r_eval_teacher_text_predictions(predictions_file: str) -> None:
    """Score Teacher-text predictions on eval unique s and write label manifest."""

    predictions_path = resolve_from_repo_root(predictions_file)
    examples = read_jsonl(r_eval_teacher_text_examples_file())
    predictions = read_jsonl(predictions_path)
    labels = build_teacher_label_rows(examples, predictions)
    labels_path = r_eval_teacher_labels_file(str(predictions_path))
    write_jsonl(labels, labels_path)
    correct_count = sum(1 for row in labels if row["teacher_text_correct"])
    card_stem = labels_path.stem
    write_report(
        {
            "stage": "r_eval_teacher_text_filter",
            "predictions_manifest": str(predictions_path),
            "labels_manifest": str(labels_path),
            "unique_s": len(labels),
            "teacher_text_correct_count": correct_count,
            "teacher_text_correct_rate": correct_count / len(labels) if labels else 0.0,
        },
        resolve_from_repo_root(f"Data/cards/{card_stem}_report.json"),
    )
    print(
        f"[filter-r-eval-teacher-text] unique_s={len(labels)} "
        f"correct={correct_count}",
        flush=True,
    )


def load_teacher_label_index() -> dict[str, dict[str, Any]]:
    """Load eval Teacher-text labels keyed by content_id."""

    return {
        row["content_id"]: row for row in read_jsonl(r_eval_teacher_labels_file())
    }


def build_r_eval_teacher_stratified_report(
    scored_audio_rows: list[dict[str, Any]],
    *,
    audio_predictions_path: Path,
    labels_path: Path | None = None,
) -> dict[str, Any]:
    """Split scored R eval audio metrics by Teacher-text correctness."""

    label_index = {
        row["content_id"]: row
        for row in read_jsonl(labels_path or r_eval_teacher_labels_file())
    }
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_labels: list[str] = []
    for row in scored_audio_rows:
        content_id = str(row["content_id"])
        label_row = label_index.get(content_id)
        if label_row is None:
            missing_labels.append(content_id)
            continue
        bucket_name = (
            "teacher_text_correct"
            if label_row["teacher_text_correct"]
            else "teacher_text_incorrect"
        )
        buckets[bucket_name].append(row)

    if missing_labels:
        preview = sorted(set(missing_labels))[:5]
        raise ValueError(
            "Missing Teacher-text labels for content_ids: "
            f"{preview} (+{len(set(missing_labels)) - len(preview)} more)"
        )

    def summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
        metrics = compute_metric_bundle(rows)
        intervals = cluster_bootstrap_r_metric_bundle(rows)
        return {
            "n": {
                "queries": len(rows),
                "unique_s": len({str(row["content_id"]) for row in rows}),
                "quadruplets": len({row["quadruplet_id"] for row in rows}),
            },
            "metrics": {
                metric_name: {
                    "point_estimate": metric_value,
                    "ci95": intervals[metric_name],
                }
                for metric_name, metric_value in metrics.items()
            },
        }

    return {
        "layer": "R",
        "split": "eval",
        "purpose": "teacher_text_stratified_audio_eval",
        "prediction_source": str(audio_predictions_path),
        "labels_manifest": str(labels_path or r_eval_teacher_labels_file()),
        "strata": {
            bucket_name: summarize_bucket(rows)
            for bucket_name, rows in sorted(buckets.items())
        },
        "overall_teacher_text_correct_rate_on_unique_s": safe_mean(
            [
                float(label_index[content_id]["teacher_text_correct"])
                for content_id in sorted(
                    {str(row["content_id"]) for row in scored_audio_rows}
                )
            ]
        ),
    }


def score_r_eval_teacher_stratified(audio_scored_file: str) -> None:
    """Write a Teacher-text stratified report from one scored audio eval file."""

    scored_path = resolve_from_repo_root(audio_scored_file)
    scored_rows = read_jsonl(scored_path)
    labels_path = None
    name = scored_path.name
    if name.startswith("r_eval_predictions_") and name.endswith("_scored.jsonl"):
        slug = name[len("r_eval_predictions_") : -len("_scored.jsonl")]
        candidate = scored_path.with_name(f"r_eval_teacher_text_labels_{slug}.jsonl")
        if candidate.exists():
            labels_path = candidate
        elif slug == "qwen25_omni_7b_bailian_base":
            labels_path = r_eval_teacher_labels_file()
    report = build_r_eval_teacher_stratified_report(
        scored_rows,
        audio_predictions_path=scored_path.with_name(
            scored_path.name.replace("_scored.jsonl", ".jsonl")
        ),
        labels_path=labels_path,
    )
    output_path = scored_path.with_name(scored_path.stem + "_teacher_strata_report.json")
    write_report(report, output_path)
    print(
        "[score-r-eval-teacher-strata] "
        f"correct_queries={report['strata'].get('teacher_text_correct', {}).get('n', {}).get('queries', 0)} "
        f"incorrect_queries={report['strata'].get('teacher_text_incorrect', {}).get('n', {}).get('queries', 0)}",
        flush=True,
    )
