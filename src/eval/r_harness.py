"""R-layer eval harness: content = audio-is-question; para = MMSU 2-class.

"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_report
from realmg.data.r_quadruplets import R_QUADRUPLETS_PATH, R_UTTERANCES_PATH
from realmg.eval.p_harness import compute_metric_bundle, percentile, safe_mean
from realmg.eval.q_para_prompts import R_EMOTION_LABELS, eval_q_para_prompt
from realmg.eval.r_verifiers import GeneralVerifierScorer, verify_teacher_prediction


R_EVAL_EXAMPLES_PATH = "Data/manifests/r_eval_examples.jsonl"
R_PARA_PROMPT = eval_q_para_prompt(R_EMOTION_LABELS)
R_BOOTSTRAP_SEED = 42
R_BOOTSTRAP_SAMPLES = 1000


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write JSONL records with stable UTF-8 formatting."""

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_r_eval_quadruplets() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load eval-split R quadruplets and utterance lookup."""

    quads = [
        row
        for row in read_jsonl(resolve_from_repo_root(R_QUADRUPLETS_PATH))
        if row["split"] == "eval"
    ]
    utterances = {
        row["utterance_id"]: row
        for row in read_jsonl(resolve_from_repo_root(R_UTTERANCES_PATH))
    }
    return quads, utterances


def build_r_content_example(
    quadruplet: dict[str, Any],
    utterance: dict[str, Any],
) -> dict[str, Any]:
    """Build one R content query: audio is the question; no extra text stem."""

    return {
        "example_id": (
            f"{quadruplet['quadruplet_id']}::{utterance['utterance_id']}::content"
        ),
        "layer": "R",
        "split": "eval",
        "quadruplet_id": quadruplet["quadruplet_id"],
        "utterance_id": utterance["utterance_id"],
        "query_type": "content",
        "audio_path": utterance["audio_path"],
        "prompt": "",
        "target": {
            "answer_text": utterance["metadata"]["answer_text"],
            "content_id": utterance["content_id"],
        },
        "metadata": {
            "speaker_id": utterance["speaker_id"],
            "engine": utterance["tts_engine"],
            "paralinguistic_label": utterance["paralinguistic_label"],
            "content_id": utterance["content_id"],
            "prompt_text": utterance["metadata"]["prompt_text"],
            "source_name": utterance["source_name"],
        },
    }


def build_r_paralinguistic_example(
    quadruplet: dict[str, Any],
    utterance: dict[str, Any],
) -> dict[str, Any]:
    """Build one R paralinguistic query with the frozen MMSU surface."""

    return {
        "example_id": (
            f"{quadruplet['quadruplet_id']}::{utterance['utterance_id']}::para"
        ),
        "layer": "R",
        "split": "eval",
        "quadruplet_id": quadruplet["quadruplet_id"],
        "utterance_id": utterance["utterance_id"],
        "query_type": "paralinguistic",
        "audio_path": utterance["audio_path"],
        "prompt": R_PARA_PROMPT,
        "choices": list(R_EMOTION_LABELS),
        "target": {
            "label": utterance["paralinguistic_label"],
        },
        "metadata": {
            "speaker_id": utterance["speaker_id"],
            "engine": utterance["tts_engine"],
            "paralinguistic_label": utterance["paralinguistic_label"],
            "content_id": utterance["content_id"],
        },
    }


def build_r_eval_examples() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand eval quadruplets into content + para queries per audio."""

    quadruplets, utterances = load_r_eval_quadruplets()
    examples: list[dict[str, Any]] = []
    speakers: set[str] = set()
    engines: set[str] = set()
    for quadruplet in quadruplets:
        for utterance_id in quadruplet["utterance_ids"]:
            utterance = utterances[utterance_id]
            speakers.add(str(utterance["speaker_id"]))
            engines.add(str(utterance["tts_engine"]))
            examples.append(build_r_content_example(quadruplet, utterance))
            examples.append(build_r_paralinguistic_example(quadruplet, utterance))

    registry = {
        "layer": "R",
        "split": "eval",
        "quadruplet_count": len(quadruplets),
        "audio_count": len(quadruplets) * 4,
        "query_count": len(examples),
        "content_query_count": len(quadruplets) * 4,
        "paralinguistic_query_count": len(quadruplets) * 4,
        "speaker_count": len(speakers),
        "engine_count": len(engines),
        "paired_metric_n": len(quadruplets),
        "content_prompt_style": "audio_is_the_question_empty_text_stem",
        "paralinguistic_prompt_style": (
            "MMSU Emotion Recognition wording; two-way closed labels"
        ),
        "paralinguistic_prompt_source": "mmsu_emotion_recognition",
        "emotion_labels": list(R_EMOTION_LABELS),
        "note": (
            "Paired 'both-z / both-s correct' metrics are defined over "
            f"n={len(quadruplets)} eval quadruplets."
        ),
    }
    return examples, registry


def prepare_r_eval_examples() -> None:
    """Freeze R eval query examples from carved quadruplets."""

    examples, registry = build_r_eval_examples()
    examples_path = resolve_from_repo_root("Data/manifests/r_eval_examples.jsonl")
    registry_path = resolve_from_repo_root("Data/cards/r_eval_examples.json")
    write_jsonl(examples, examples_path)
    write_report(registry, registry_path)
    print(
        f"[prepare-r-eval] quads={registry['quadruplet_count']} "
        f"queries={registry['query_count']} "
        f"paired_metric_n={registry['paired_metric_n']}",
        flush=True,
    )


def eval_examples_path() -> Path:
    """Return the frozen R eval examples manifest."""

    return resolve_from_repo_root(R_EVAL_EXAMPLES_PATH)


def load_r_eval_examples() -> dict[str, dict[str, Any]]:
    """Load serialized R eval examples as an id-indexed lookup."""

    return {
        record["example_id"]: record for record in read_jsonl(eval_examples_path())
    }


def extract_prediction_text(record: dict[str, Any]) -> str:
    """Pick the first supported prediction field from one model output row."""

    for field_name in ("prediction", "prediction_text", "response", "text", "output"):
        if field_name in record:
            return str(record[field_name]).strip()
    raise KeyError(
        "Prediction row must include one of: prediction, prediction_text, "
        "response, text, output."
    )


def score_r_paralinguistic_prediction(prediction_text: str) -> str | None:
    """Map a raw paralinguistic prediction to neutral or happy."""

    normalized = prediction_text.strip().lower()
    for label in R_EMOTION_LABELS:
        if normalized == label:
            return label
    return None


def score_r_content_prediction(
    prediction_text: str,
    example: dict[str, Any],
    *,
    general_verifier: GeneralVerifierScorer | None = None,
) -> tuple[bool, str | None, str]:
    """Score one R content answer via the routed Teacher-text verifier."""

    metadata = example["metadata"]
    result = verify_teacher_prediction(
        source_name=metadata["source_name"],
        prompt_text=metadata["prompt_text"],
        reference_answer=example["target"]["answer_text"],
        prediction_text=prediction_text,
        general_verifier=general_verifier,
    )
    return (
        bool(result.is_correct),
        result.normalized_prediction,
        result.verifier_method,
    )


def attach_r_scored_predictions(
    examples: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    general_verifier: GeneralVerifierScorer | None = None,
    allow_incomplete: bool = False,
) -> list[dict[str, Any]]:
    """Attach verifier outcomes and paired-metric fields to frozen R examples."""

    scorer = general_verifier or GeneralVerifierScorer()
    scored_rows: list[dict[str, Any]] = []
    seen_example_ids: set[str] = set()
    total = len(predictions)

    for idx, prediction_row in enumerate(predictions):
        if idx > 0 and idx % 25 == 0:
            print(
                f"[attach_r_scored_predictions] {idx}/{total} scored, "
                f"current_correct={sum(1 for r in scored_rows if r['is_correct'])}/{idx}",
                flush=True,
            )
        example_id = prediction_row["example_id"]
        if example_id not in examples:
            raise KeyError(f"Unknown example_id in predictions: {example_id}")
        if example_id in seen_example_ids:
            raise ValueError(f"Duplicate prediction for example_id: {example_id}")

        example = examples[example_id]
        prediction_text = extract_prediction_text(prediction_row)
        if example["query_type"] == "content":
            is_correct, normalized_answer, verifier_method = score_r_content_prediction(
                prediction_text,
                example,
                general_verifier=scorer,
            )
            is_parseable = normalized_answer is not None and bool(prediction_text)
        else:
            normalized_answer = score_r_paralinguistic_prediction(prediction_text)
            is_correct = normalized_answer == example["target"]["label"]
            is_parseable = normalized_answer is not None
            verifier_method = "label_match"

        scored_rows.append(
            {
                "example_id": example_id,
                "layer": "R",
                "quadruplet_id": example["quadruplet_id"],
                "utterance_id": example["utterance_id"],
                "query_type": example["query_type"],
                "speaker_id": example["metadata"]["speaker_id"],
                "content_id": example["metadata"]["content_id"],
                "paralinguistic_label": example["metadata"]["paralinguistic_label"],
                "engine": example["metadata"].get("engine"),
                "source_name": example["metadata"].get("source_name"),
                "target": example["target"],
                "raw_prediction": prediction_text,
                "normalized_prediction": normalized_answer,
                "verifier_method": verifier_method,
                "is_correct": is_correct,
                "is_parseable": is_parseable,
            }
        )
        seen_example_ids.add(example_id)

    missing_example_ids = set(examples) - seen_example_ids
    if missing_example_ids and not allow_incomplete:
        preview = sorted(missing_example_ids)[:5]
        raise ValueError(
            "Predictions are incomplete. Missing example_ids: "
            f"{preview} (+{len(missing_example_ids) - len(preview)} more)"
        )
    for example_id in sorted(missing_example_ids):
        example = examples[example_id]
        scored_rows.append(
            {
                "example_id": example_id,
                "layer": "R",
                "quadruplet_id": example["quadruplet_id"],
                "utterance_id": example["utterance_id"],
                "query_type": example["query_type"],
                "speaker_id": example["metadata"]["speaker_id"],
                "content_id": example["metadata"]["content_id"],
                "paralinguistic_label": example["metadata"]["paralinguistic_label"],
                "engine": example["metadata"].get("engine"),
                "source_name": example["metadata"].get("source_name"),
                "target": example["target"],
                "raw_prediction": "",
                "normalized_prediction": None,
                "verifier_method": "missing_prediction",
                "is_correct": False,
                "is_parseable": False,
            }
        )
    return scored_rows


def cluster_bootstrap_r_metric_bundle(
    scored_rows: list[dict[str, Any]],
    *,
    seed: int = R_BOOTSTRAP_SEED,
    samples: int = R_BOOTSTRAP_SAMPLES,
) -> dict[str, dict[str, float]]:
    """Estimate 95% CIs with content_id (unique s) cluster bootstrap."""

    cluster_ids = sorted({str(row["content_id"]) for row in scored_rows})
    rows_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        rows_by_cluster[str(row["content_id"])].append(row)

    metric_samples: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)
    for _ in range(samples):
        bootstrap_rows: list[dict[str, Any]] = []
        for sampled_cluster in (rng.choice(cluster_ids) for _ in cluster_ids):
            bootstrap_rows.extend(rows_by_cluster[sampled_cluster])
        bundle = compute_metric_bundle(bootstrap_rows)
        for metric_name, metric_value in bundle.items():
            metric_samples[metric_name].append(metric_value)

    intervals: dict[str, dict[str, float]] = {}
    for metric_name, values in metric_samples.items():
        ordered = sorted(values)
        intervals[metric_name] = {
            "low": percentile(ordered, 0.025),
            "high": percentile(ordered, 0.975),
        }
    return intervals


def build_r_score_report(
    scored_rows: list[dict[str, Any]],
    predictions_path: Path,
) -> dict[str, Any]:
    """Build the paper-facing report for one scored R eval run."""

    metrics = compute_metric_bundle(scored_rows)
    intervals = cluster_bootstrap_r_metric_bundle(scored_rows)
    speaker_count = len({str(row["speaker_id"]) for row in scored_rows})
    content_count = len({str(row["content_id"]) for row in scored_rows})
    quadruplet_count = len({row["quadruplet_id"] for row in scored_rows})

    return {
        "layer": "R",
        "split": "eval",
        "prediction_source": str(predictions_path),
        "bootstrap": {
            "cluster_unit": "content_id",
            "samples": R_BOOTSTRAP_SAMPLES,
            "seed": R_BOOTSTRAP_SEED,
        },
        "n": {
            "unique_s": content_count,
            "speakers": speaker_count,
            "quadruplets": quadruplet_count,
            "queries": len(scored_rows),
            "content_queries": sum(
                1 for row in scored_rows if row["query_type"] == "content"
            ),
            "paralinguistic_queries": sum(
                1 for row in scored_rows if row["query_type"] == "paralinguistic"
            ),
        },
        "metrics": {
            metric_name: {
                "point_estimate": metric_value,
                "ci95": intervals[metric_name],
            }
            for metric_name, metric_value in metrics.items()
        },
    }


def score_r_eval_predictions(
    predictions_file: str,
    *,
    allow_incomplete: bool = False,
) -> None:
    """Score one model prediction file against the frozen R eval set."""

    predictions_path = resolve_from_repo_root(predictions_file)
    examples = load_r_eval_examples()
    predictions = read_jsonl(predictions_path)
    scored_rows = attach_r_scored_predictions(
        examples,
        predictions,
        allow_incomplete=allow_incomplete,
    )

    scored_output_path = predictions_path.with_name(
        predictions_path.stem + "_scored.jsonl"
    )
    report_output_path = predictions_path.with_name(
        predictions_path.stem + "_report.json"
    )
    write_jsonl(scored_rows, scored_output_path)
    report = build_r_score_report(scored_rows, predictions_path)
    write_report(report, report_output_path)
    print(
        f"[score-r-eval] scored={len(scored_rows)} "
        f"quads={report['n']['quadruplets']}",
        flush=True,
    )
    from realmg.eval.score_locked_eval import (
        maybe_write_r_locked_report_for_slug,
        slug_from_r_audio_path,
    )

    maybe_write_r_locked_report_for_slug(slug_from_r_audio_path(predictions_path))
