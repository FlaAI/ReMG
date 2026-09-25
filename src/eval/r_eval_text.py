"""R-layer text baseline eval examples and modality-gap reporting.

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_report
from realmg.eval.metrics_aux import build_r_modality_gap_metrics
from realmg.eval.p_harness import safe_mean
from realmg.eval.r_harness import (
    attach_r_scored_predictions,
    eval_examples_path,
    load_r_eval_examples,
    write_jsonl,
)
from realmg.eval.r_verifiers import assign_verifier_method


R_EVAL_TEXT_EXAMPLES_PATH = "Data/manifests/r_eval_text_examples.jsonl"
R_EVAL_TEXT_REGISTRY_PATH = "Data/cards/r_eval_text_examples.json"


def r_eval_text_examples_path() -> Path:
    """Return the frozen R eval text-baseline examples manifest."""

    return resolve_from_repo_root(R_EVAL_TEXT_EXAMPLES_PATH)


def build_r_eval_text_examples() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Export one text-only example per unique eval content_id."""

    examples_by_content: dict[str, dict[str, Any]] = {}
    for example in read_jsonl(eval_examples_path()):
        if example["query_type"] != "content":
            continue
        content_id = str(example["metadata"]["content_id"])
        if content_id in examples_by_content:
            continue
        prompt_text = str(example["metadata"]["prompt_text"])
        answer_text = str(example["target"]["answer_text"])
        source_name = str(example["metadata"]["source_name"])
        examples_by_content[content_id] = {
            "example_id": content_id,
            "layer": "R",
            "split": "eval",
            "stage": "eval_text_baseline",
            "content_id": content_id,
            "source_name": source_name,
            "prompt_text": prompt_text,
            "reference_answer": answer_text,
            "verifier_method": assign_verifier_method(
                source_name,
                prompt_text,
                answer_text,
            ),
            "metadata": {
                "representative_quadruplet_id": example["quadruplet_id"],
                "representative_utterance_id": example["utterance_id"],
            },
        }

    examples = sorted(
        examples_by_content.values(),
        key=lambda row: row["content_id"],
    )
    by_source: dict[str, int] = {}
    by_verifier: dict[str, int] = {}
    for example in examples:
        by_source[example["source_name"]] = by_source.get(example["source_name"], 0) + 1
        by_verifier[example["verifier_method"]] = (
            by_verifier.get(example["verifier_method"], 0) + 1
        )

    registry = {
        "layer": "R",
        "split": "eval",
        "purpose": "text_baseline_for_modality_gap",
        "example_count": len(examples),
        "by_source": dict(sorted(by_source.items())),
        "by_verifier_method": dict(sorted(by_verifier.items())),
        "notes": [
            "One row per unique eval content_id (unique s).",
            "Score with score-r-eval-text; then build gap via score-r-modality-gap.",
        ],
    }
    return examples, registry


def prepare_r_eval_text_examples() -> None:
    """Freeze text-only R eval examples for Acc_text measurement."""

    examples, registry = build_r_eval_text_examples()
    write_jsonl(examples, r_eval_text_examples_path())
    write_report(registry, resolve_from_repo_root(R_EVAL_TEXT_REGISTRY_PATH))
    print(
        f"[prepare-r-eval-text] unique_s={registry['example_count']}",
        flush=True,
    )


def load_r_eval_text_examples() -> dict[str, dict[str, Any]]:
    """Load frozen text-baseline examples keyed by content_id."""

    return {
        record["example_id"]: record
        for record in read_jsonl(r_eval_text_examples_path())
    }


def score_r_eval_text_predictions(predictions_file: str) -> None:
    """Score text-only R eval predictions with the routed Teacher-text verifiers."""

    predictions_path = resolve_from_repo_root(predictions_file)
    text_examples = load_r_eval_text_examples()
    audio_examples = load_r_eval_examples()

    # Reuse the audio scorer by projecting text rows onto one representative
    # content audio example per unique s.
    projected_examples: dict[str, dict[str, Any]] = {}
    for content_id, text_example in text_examples.items():
        audio_match = next(
            (
                example
                for example_id, example in audio_examples.items()
                if example["query_type"] == "content"
                and str(example["metadata"]["content_id"]) == content_id
            ),
            None,
        )
        if audio_match is None:
            raise KeyError(f"No audio eval example found for content_id={content_id}")
        projected_examples[content_id] = {
            **audio_match,
            "example_id": content_id,
            "metadata": {
                **audio_match["metadata"],
                "prompt_text": text_example["prompt_text"],
            },
            "target": {
                **audio_match["target"],
                "answer_text": text_example["reference_answer"],
            },
        }

    predictions = read_jsonl(predictions_path)
    scored_rows = attach_r_scored_predictions(projected_examples, predictions)

    scored_output_path = predictions_path.with_name(
        predictions_path.stem + "_scored.jsonl"
    )
    report_output_path = predictions_path.with_name(
        predictions_path.stem + "_report.json"
    )
    write_jsonl(scored_rows, scored_output_path)
    write_report(
        build_r_eval_text_score_report(scored_rows, predictions_path),
        report_output_path,
    )
    print(
        f"[score-r-eval-text] unique_s={len(scored_rows)} "
        f"accuracy={safe_mean([float(row['is_correct']) for row in scored_rows]):.4f}",
        flush=True,
    )
    from realmg.eval.score_locked_eval import (
        maybe_write_r_locked_report_for_slug,
        slug_from_r_text_path,
    )

    maybe_write_r_locked_report_for_slug(slug_from_r_text_path(predictions_path))


def build_r_eval_text_score_report(
    scored_rows: list[dict[str, Any]],
    predictions_path: Path,
) -> dict[str, Any]:
    """Build a compact report for the text-only R eval baseline."""

    accuracy = safe_mean([float(row["is_correct"]) for row in scored_rows])
    return {
        "layer": "R",
        "split": "eval",
        "purpose": "text_baseline_for_modality_gap",
        "prediction_source": str(predictions_path),
        "n": {
            "unique_s": len(scored_rows),
            "queries": len(scored_rows),
        },
        "metrics": {
            "content_text_accuracy": {
                "point_estimate": accuracy,
            },
            "parse_rate": {
                "point_estimate": safe_mean(
                    [float(row["is_parseable"]) for row in scored_rows]
                ),
            },
        },
    }


def build_r_modality_gap_report(
    *,
    audio_report_path: str,
    text_report_path: str,
) -> dict[str, Any]:
    """Combine scored audio/text reports into gap metrics for one Base run."""

    audio_report = json.loads(
        resolve_from_repo_root(audio_report_path).read_text(encoding="utf-8")
    )
    text_report = json.loads(
        resolve_from_repo_root(text_report_path).read_text(encoding="utf-8")
    )
    content_audio_accuracy = audio_report["metrics"]["content_query_accuracy"][
        "point_estimate"
    ]
    content_text_accuracy = text_report["metrics"]["content_text_accuracy"][
        "point_estimate"
    ]
    gap_metrics = build_r_modality_gap_metrics(
        content_text_accuracy=content_text_accuracy,
        content_audio_accuracy=content_audio_accuracy,
    )
    return {
        "layer": "R",
        "split": "eval",
        "purpose": "modality_gap_base",
        "audio_report": audio_report_path,
        "text_report": text_report_path,
        "n": {
            "audio_content_queries": audio_report["n"]["content_queries"],
            "text_unique_s": text_report["n"]["unique_s"],
        },
        "metrics": {
            metric_name: {"point_estimate": metric_value}
            for metric_name, metric_value in gap_metrics.items()
        },
    }


def score_r_modality_gap(
    *,
    audio_report_path: str,
    text_report_path: str,
    output_path: str | None = None,
) -> None:
    """Write the R-layer text-vs-audio gap report for one Base run."""

    report = build_r_modality_gap_report(
        audio_report_path=audio_report_path,
        text_report_path=text_report_path,
    )
    if output_path is None:
        audio_stem = Path(audio_report_path).stem.replace("_report", "")
        output_path = str(
            resolve_from_repo_root(
                f"Data/manifests/{audio_stem}_modality_gap_report.json"
            )
        )
    write_report(report, resolve_from_repo_root(output_path))
    print(
        "[score-r-modality-gap] "
        f"text={report['metrics']['content_text_accuracy']['point_estimate']:.4f} "
        f"audio={report['metrics']['content_audio_accuracy']['point_estimate']:.4f} "
        f"gap={report['metrics']['absolute_modality_gap']['point_estimate']:.4f}",
        flush=True,
    )
