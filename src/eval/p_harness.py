"""P-layer evaluation harness for frozen content and paralinguistic queries."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_report
from realmg.eval.metrics_aux import compute_auxiliary_metric_bundle
from realmg.eval.q_para_prompts import P_EMOTION_LABELS, eval_q_para_prompt


def safe_mean(values: list[float]) -> float:
    """Return the arithmetic mean, or 0.0 for an empty list."""

    if not values:
        return 0.0
    return float(mean(values))


P_EVAL_PROTOCOLS = ("unseen_speaker", "unseen_content")
P_PAPER_PROTOCOL = "paper"
P_SCORABLE_PROTOCOLS = (*P_EVAL_PROTOCOLS, P_PAPER_PROTOCOL)
P_CONTENT_PROMPT = (
    "Which transcript matches this audio?\n"
    "A. {option_a}\n"
    "B. {option_b}\n"
    "Answer with A or B only."
)
# Frozen eval / Teacher-ceiling surface: MMSU Emotion Recognition wording.
P_PARA_PROMPT = eval_q_para_prompt(P_EMOTION_LABELS)
P_BOOTSTRAP_SEED = 42
P_BOOTSTRAP_SAMPLES = 1000


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def utterance_manifest_path(protocol: str) -> Path:
    """Return the final frozen utterance manifest for one P protocol."""

    if protocol == P_PAPER_PROTOCOL:
        return resolve_from_repo_root(
            "Data/manifests/p_utterances_unseen_speaker_lex_filtered.jsonl"
        )
    return resolve_from_repo_root(
        f"Data/manifests/p_utterances_{protocol}_lex_filtered.jsonl"
    )


def quadruplet_manifest_path(protocol: str) -> Path:
    """Return the final frozen quadruplet manifest for one P protocol."""

    if protocol == P_PAPER_PROTOCOL:
        return resolve_from_repo_root("Data/manifests/p_quadruplets_paper_eval.jsonl")
    return resolve_from_repo_root(
        f"Data/manifests/p_quadruplets_{protocol}_lex_filtered.jsonl"
    )


def eval_examples_path(protocol: str) -> Path:
    """Return the output path for serialized eval examples."""

    return resolve_from_repo_root(f"Data/manifests/p_eval_examples_{protocol}.jsonl")


def eval_registry_path(protocol: str) -> Path:
    """Return the output path for the eval registry summary."""

    return resolve_from_repo_root(f"Data/cards/p_eval_examples_{protocol}.json")


def teacher_ceiling_examples_path(protocol: str) -> Path:
    """Return the output path for teacher q_para ceiling examples."""

    return resolve_from_repo_root(
        f"Data/manifests/p_teacher_ceiling_examples_{protocol}.jsonl"
    )


def teacher_ceiling_registry_path(protocol: str) -> Path:
    """Return the output path for the teacher q_para ceiling registry."""

    return resolve_from_repo_root(
        f"Data/cards/p_teacher_ceiling_examples_{protocol}.json"
    )


def load_eval_protocol_data(
    protocol: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load frozen eval quadruplets and utterances for one P protocol."""

    quadruplet_records = read_jsonl(quadruplet_manifest_path(protocol))
    if protocol == P_PAPER_PROTOCOL:
        quadruplets = list(quadruplet_records)
    else:
        quadruplets = [
            record for record in quadruplet_records if record["split"] == "eval"
        ]
    utterances = {
        record["utterance_id"]: record
        for record in read_jsonl(utterance_manifest_path(protocol))
    }
    return quadruplets, utterances


def build_content_example(
    protocol: str,
    quadruplet: dict[str, Any],
    utterance: dict[str, Any],
    option_map: dict[str, str],
) -> dict[str, Any]:
    """Build one frozen content-identification eval example."""

    canonical_transcripts = quadruplet["metadata"]["canonical_transcripts"]
    content_id = utterance["content_id"]
    target_option = option_map[content_id]

    return {
        "example_id": (
            f"{quadruplet['quadruplet_id']}::{utterance['utterance_id']}::content"
        ),
        "layer": "P",
        "protocol": protocol,
        "split": "eval",
        "quadruplet_id": quadruplet["quadruplet_id"],
        "utterance_id": utterance["utterance_id"],
        "query_type": "content",
        "audio_path": utterance["audio_path"],
        "prompt": P_CONTENT_PROMPT.format(
            option_a=canonical_transcripts[quadruplet["content_pair"][0]],
            option_b=canonical_transcripts[quadruplet["content_pair"][1]],
        ),
        "choices": {
            "A": canonical_transcripts[quadruplet["content_pair"][0]],
            "B": canonical_transcripts[quadruplet["content_pair"][1]],
        },
        "target": {
            "choice": target_option,
            "content_id": content_id,
            "transcript": utterance["transcript"],
        },
        "metadata": {
            "speaker_id": utterance["speaker_id"],
            "paralinguistic_label": utterance["paralinguistic_label"],
            "content_id": content_id,
        },
    }


def build_paralinguistic_example(
    protocol: str,
    quadruplet: dict[str, Any],
    utterance: dict[str, Any],
) -> dict[str, Any]:
    """Build one frozen paralinguistic eval example."""

    return {
        "example_id": (
            f"{quadruplet['quadruplet_id']}::{utterance['utterance_id']}::para"
        ),
        "layer": "P",
        "protocol": protocol,
        "split": "eval",
        "quadruplet_id": quadruplet["quadruplet_id"],
        "utterance_id": utterance["utterance_id"],
        "query_type": "paralinguistic",
        "audio_path": utterance["audio_path"],
        "prompt": P_PARA_PROMPT,
        "choices": list(P_EMOTION_LABELS),
        "target": {
            "label": utterance["paralinguistic_label"],
        },
        "metadata": {
            "speaker_id": utterance["speaker_id"],
            "paralinguistic_label": utterance["paralinguistic_label"],
            "content_id": utterance["content_id"],
        },
    }


def build_eval_examples_for_protocol(protocol: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand frozen eval quadruplets into one content and one para query per audio."""

    quadruplets, utterances = load_eval_protocol_data(protocol)
    examples: list[dict[str, Any]] = []
    unique_speakers: set[str] = set()
    unique_contents: set[str] = set()
    unique_slots: set[int] = set()

    for quadruplet in quadruplets:
        option_map = {
            quadruplet["content_pair"][0]: "A",
            quadruplet["content_pair"][1]: "B",
        }
        for utterance_id in quadruplet["utterance_ids"]:
            utterance = utterances[utterance_id]
            unique_speakers.add(str(utterance["speaker_id"]))
            unique_contents.add(str(utterance["content_id"]))
            unique_slots.add(int(utterance["metadata"]["utt_slot"]))
            examples.append(
                build_content_example(protocol, quadruplet, utterance, option_map)
            )
            examples.append(
                build_paralinguistic_example(protocol, quadruplet, utterance)
            )

    registry = {
        "protocol": protocol,
        "split": "eval",
        "quadruplet_count": len(quadruplets),
        "audio_count": len(quadruplets) * 4,
        "query_count": len(examples),
        "content_query_count": len(quadruplets) * 4,
        "paralinguistic_query_count": len(quadruplets) * 4,
        "speaker_count": len(unique_speakers),
        "speaker_content_id_count": len(unique_contents),
        "utterance_slot_count": len(unique_slots),
        "emotion_labels": list(P_EMOTION_LABELS),
        "content_prompt_style": "two-way identification within the quadruplet",
        "paralinguistic_prompt_style": (
            "MMSU Emotion Recognition wording; five-way closed labels"
        ),
        "paralinguistic_prompt_source": "mmsu_emotion_recognition",
    }
    return examples, registry


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write JSONL records with stable UTF-8 formatting."""

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_p_eval_examples(protocol: str) -> None:
    """Create frozen P-layer eval examples for one or all construction protocols."""

    if protocol == P_PAPER_PROTOCOL:
        raise ValueError(
            "Paper examples must preserve construction example_id values. "
            "Use realmg prepare-p-paper-eval."
        )
    protocols = P_EVAL_PROTOCOLS if protocol == "all" else (protocol,)
    for protocol_name in protocols:
        examples, registry = build_eval_examples_for_protocol(protocol_name)
        write_jsonl(examples, eval_examples_path(protocol_name))
        write_report(registry, eval_registry_path(protocol_name))


def build_teacher_ceiling_examples(
    protocol: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Export only the P q_para examples used for the Teacher ceiling."""

    examples = [
        example
        for example in read_jsonl(eval_examples_path(protocol))
        if example["query_type"] == "paralinguistic"
    ]
    unique_speakers = {
        str(example["metadata"]["speaker_id"]) for example in examples
    }
    unique_slots = {
        int(example["metadata"]["content_id"].split("_")[-1]) for example in examples
    }
    registry = {
        "layer": "P",
        "protocol": protocol,
        "split": "eval",
        "purpose": "teacher_q_para_ceiling",
        "query_type": "paralinguistic",
        "query_count": len(examples),
        "audio_count": len(examples),
        "quadruplet_count": len({example["quadruplet_id"] for example in examples}),
        "speaker_count": len(unique_speakers),
        "utterance_slot_count": len(unique_slots),
        "prompt_style": (
            "MMSU Emotion Recognition wording; same surface as loss-2 eval; "
            "no extra teacher prompt prior"
        ),
        "paralinguistic_prompt_source": "mmsu_emotion_recognition",
        "emotion_labels": list(P_EMOTION_LABELS),
    }
    return examples, registry


def prepare_p_teacher_ceiling_examples(protocol: str) -> None:
    """Create the P q_para-only examples for Teacher ceiling measurement."""

    protocols = P_EVAL_PROTOCOLS if protocol == "all" else (protocol,)
    for protocol_name in protocols:
        examples, registry = build_teacher_ceiling_examples(protocol_name)
        write_jsonl(examples, teacher_ceiling_examples_path(protocol_name))
        write_report(registry, teacher_ceiling_registry_path(protocol_name))


def normalize_prediction_text(value: Any) -> str:
    """Normalize one raw prediction string for rule-based scoring."""

    if value is None:
        return ""
    return str(value).strip().lower()


def extract_prediction_payload(record: dict[str, Any]) -> str:
    """Pick the first supported prediction field from one model output row."""

    for field_name in ("prediction", "prediction_text", "response", "text", "output"):
        if field_name in record:
            return normalize_prediction_text(record[field_name])
    raise KeyError(
        "Prediction row must include one of: prediction, prediction_text, response, text, output."
    )


def score_content_prediction(prediction_text: str, example: dict[str, Any]) -> str | None:
    """Map a raw content prediction to A/B when possible."""

    target_choices = example["choices"]
    if prediction_text in {"a", "a.", "option a"}:
        return "A"
    if prediction_text in {"b", "b.", "option b"}:
        return "B"

    if prediction_text == normalize_prediction_text(target_choices["A"]):
        return "A"
    if prediction_text == normalize_prediction_text(target_choices["B"]):
        return "B"
    return None


def score_paralinguistic_prediction(
    prediction_text: str,
) -> str | None:
    """Map a raw paralinguistic prediction to one of the frozen labels."""

    for label in P_EMOTION_LABELS:
        if prediction_text == label:
            return label
    return None


def load_eval_examples(protocol: str) -> dict[str, dict[str, Any]]:
    """Load serialized eval examples as an id-indexed lookup."""

    return {
        record["example_id"]: record
        for record in read_jsonl(eval_examples_path(protocol))
    }


def load_teacher_ceiling_examples(protocol: str) -> dict[str, dict[str, Any]]:
    """Load serialized teacher q_para ceiling examples as an id lookup."""

    return {
        record["example_id"]: record
        for record in read_jsonl(teacher_ceiling_examples_path(protocol))
    }


def load_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    """Load model prediction rows from JSONL."""

    return read_jsonl(predictions_path)


def attach_scored_predictions(
    examples: dict[str, dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach parsed predictions and correctness flags to frozen examples."""

    scored_rows: list[dict[str, Any]] = []
    seen_example_ids: set[str] = set()

    for prediction_row in predictions:
        example_id = prediction_row["example_id"]
        if example_id not in examples:
            raise KeyError(f"Unknown example_id in predictions: {example_id}")
        if example_id in seen_example_ids:
            raise ValueError(f"Duplicate prediction for example_id: {example_id}")

        example = examples[example_id]
        prediction_text = extract_prediction_payload(prediction_row)
        normalized_answer: str | None
        is_correct = False

        if example["query_type"] == "content":
            normalized_answer = score_content_prediction(prediction_text, example)
            is_correct = normalized_answer == example["target"]["choice"]
        else:
            normalized_answer = score_paralinguistic_prediction(prediction_text)
            is_correct = normalized_answer == example["target"]["label"]

        scored_rows.append(
            {
                "example_id": example_id,
                "protocol": example["protocol"],
                "quadruplet_id": example["quadruplet_id"],
                "utterance_id": example["utterance_id"],
                "query_type": example["query_type"],
                "speaker_id": example["metadata"]["speaker_id"],
                "content_id": example["metadata"]["content_id"],
                "paralinguistic_label": example["metadata"]["paralinguistic_label"],
                "target": example["target"],
                "raw_prediction": prediction_text,
                "normalized_prediction": normalized_answer,
                "is_correct": is_correct,
                "is_parseable": normalized_answer is not None,
            }
        )
        seen_example_ids.add(example_id)

    missing_example_ids = set(examples) - seen_example_ids
    if missing_example_ids:
        raise ValueError(
            "Predictions are incomplete. Missing example_ids: "
            f"{sorted(missing_example_ids)[:5]}"
        )

    return scored_rows


def compute_metric_bundle(scored_rows: list[dict[str, Any]]) -> dict[str, float]:
    """Compute query-level and paired 2x2 metrics from scored predictions."""

    content_rows = [row for row in scored_rows if row["query_type"] == "content"]
    para_rows = [
        row for row in scored_rows if row["query_type"] == "paralinguistic"
    ]

    content_accuracy = safe_mean([float(row["is_correct"]) for row in content_rows])
    para_accuracy = safe_mean([float(row["is_correct"]) for row in para_rows])
    parse_rate = safe_mean([float(row["is_parseable"]) for row in scored_rows])

    content_pairs: dict[tuple[str, str], list[bool]] = defaultdict(list)
    para_pairs: dict[tuple[str, str], list[bool]] = defaultdict(list)
    content_quadruplets: dict[str, list[bool]] = defaultdict(list)
    para_quadruplets: dict[str, list[bool]] = defaultdict(list)
    para_by_label: dict[str, list[bool]] = defaultdict(list)

    for row in content_rows:
        content_pairs[(row["quadruplet_id"], row["content_id"])].append(row["is_correct"])
        content_quadruplets[row["quadruplet_id"]].append(row["is_correct"])

    for row in para_rows:
        para_pairs[(row["quadruplet_id"], row["paralinguistic_label"])].append(
            row["is_correct"]
        )
        para_quadruplets[row["quadruplet_id"]].append(row["is_correct"])
        para_by_label[row["paralinguistic_label"]].append(row["is_correct"])

    content_all_z_correct = safe_mean(
        [float(all(pair_values)) for pair_values in content_pairs.values()]
    )
    para_all_s_correct = safe_mean(
        [float(all(pair_values)) for pair_values in para_pairs.values()]
    )
    content_quadruplet_all_correct = safe_mean(
        [float(all(values)) for values in content_quadruplets.values()]
    )
    para_quadruplet_all_correct = safe_mean(
        [float(all(values)) for values in para_quadruplets.values()]
    )
    para_macro_class_accuracy = safe_mean(
        [safe_mean([float(value) for value in values]) for values in para_by_label.values()]
    )

    return {
        "parse_rate": parse_rate,
        "content_query_accuracy": content_accuracy,
        "content_all_z_correct_rate": content_all_z_correct,
        "content_quadruplet_all_correct_rate": content_quadruplet_all_correct,
        "paralinguistic_query_accuracy": para_accuracy,
        "paralinguistic_all_s_correct_rate": para_all_s_correct,
        "paralinguistic_quadruplet_all_correct_rate": para_quadruplet_all_correct,
        "paralinguistic_macro_class_accuracy": para_macro_class_accuracy,
        **compute_auxiliary_metric_bundle(scored_rows),
    }


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Return a simple linear-interpolated percentile from sorted samples."""

    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = fraction * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + weight * (upper_value - lower_value)


def cluster_bootstrap_metric_bundle(
    scored_rows: list[dict[str, Any]],
    seed: int = P_BOOTSTRAP_SEED,
    samples: int = P_BOOTSTRAP_SAMPLES,
) -> dict[str, dict[str, float]]:
    """Estimate 95% CIs with speaker-cluster bootstrap."""

    speakers = sorted({str(row["speaker_id"]) for row in scored_rows})
    rows_by_speaker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        rows_by_speaker[str(row["speaker_id"])].append(row)

    metric_samples: dict[str, list[float]] = defaultdict(list)
    rng = random.Random(seed)

    for _ in range(samples):
        bootstrap_rows: list[dict[str, Any]] = []
        for sampled_speaker in (rng.choice(speakers) for _ in speakers):
            bootstrap_rows.extend(rows_by_speaker[sampled_speaker])
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


def build_score_report(
    protocol: str,
    scored_rows: list[dict[str, Any]],
    predictions_path: Path,
) -> dict[str, Any]:
    """Build the paper-facing report for one scored P eval run."""

    metrics = compute_metric_bundle(scored_rows)
    intervals = cluster_bootstrap_metric_bundle(scored_rows)
    speaker_count = len({str(row["speaker_id"]) for row in scored_rows})
    quadruplet_count = len({row["quadruplet_id"] for row in scored_rows})

    return {
        "layer": "P",
        "protocol": protocol,
        "split": "eval",
        "prediction_source": str(predictions_path),
        "bootstrap": {
            "cluster_unit": "speaker_id",
            "samples": P_BOOTSTRAP_SAMPLES,
            "seed": P_BOOTSTRAP_SEED,
        },
        "n": {
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


def parse_protocol_from_predictions_path(predictions_path: Path) -> str:
    """Infer the P protocol name from the predictions filename."""

    filename = predictions_path.name.lower()
    # `_paper_` and must not fall through.
    if f"_{P_PAPER_PROTOCOL}_" in filename:
        return P_PAPER_PROTOCOL
    path_text = str(predictions_path).lower()
    for protocol in P_EVAL_PROTOCOLS:
        if protocol in path_text:
            return protocol
    raise ValueError(
        "Could not infer protocol from predictions path. Include paper, "
        "unseen_speaker, or unseen_content in the filename."
    )


def score_p_eval_predictions(predictions_file: str) -> None:
    """Score one model prediction file against the frozen P eval set."""

    predictions_path = resolve_from_repo_root(predictions_file)
    protocol = parse_protocol_from_predictions_path(predictions_path)
    examples = load_eval_examples(protocol)
    predictions = load_predictions(predictions_path)
    scored_rows = attach_scored_predictions(examples, predictions)

    scored_output_path = predictions_path.with_name(
        predictions_path.stem + "_scored.jsonl"
    )
    report_output_path = predictions_path.with_name(
        predictions_path.stem + "_report.json"
    )
    write_jsonl(scored_rows, scored_output_path)
    write_report(
        build_score_report(protocol, scored_rows, predictions_path),
        report_output_path,
    )
    if protocol == P_PAPER_PROTOCOL:
        from realmg.eval.score_locked_eval import (
            slug_from_p_paper_path,
            write_p_locked_report_from_scored,
        )

        write_p_locked_report_from_scored(
            slug=slug_from_p_paper_path(predictions_path),
            scored_rows=scored_rows,
            scored_path=scored_output_path,
        )


def build_teacher_ceiling_report(
    protocol: str,
    scored_rows: list[dict[str, Any]],
    predictions_path: Path,
) -> dict[str, Any]:
    """Build a compact report for the Teacher q_para ceiling measurement."""

    full_report = build_score_report(protocol, scored_rows, predictions_path)
    metric_names = (
        "parse_rate",
        "paralinguistic_query_accuracy",
        "paralinguistic_all_s_correct_rate",
        "paralinguistic_quadruplet_all_correct_rate",
        "paralinguistic_macro_class_accuracy",
    )
    return {
        **full_report,
        "purpose": "teacher_q_para_ceiling",
        "query_type": "paralinguistic",
        "metrics": {
            metric_name: full_report["metrics"][metric_name]
            for metric_name in metric_names
        },
    }


def score_p_teacher_ceiling_predictions(predictions_file: str) -> None:
    """Score Teacher q_para ceiling predictions on the frozen P eval subset."""

    predictions_path = resolve_from_repo_root(predictions_file)
    protocol = parse_protocol_from_predictions_path(predictions_path)
    examples = load_teacher_ceiling_examples(protocol)
    predictions = load_predictions(predictions_path)
    scored_rows = attach_scored_predictions(examples, predictions)

    scored_output_path = predictions_path.with_name(
        predictions_path.stem + "_scored.jsonl"
    )
    report_output_path = predictions_path.with_name(
        predictions_path.stem + "_report.json"
    )
    write_jsonl(scored_rows, scored_output_path)
    write_report(
        build_teacher_ceiling_report(protocol, scored_rows, predictions_path),
        report_output_path,
    )


def iter_p_api_query_protocols(protocol: str) -> tuple[str, ...]:
    """Return P protocols an API runner should query.

    remain available by explicit name and are not paper columns.
    """

    if protocol == "all":
        return (P_PAPER_PROTOCOL,)
    if protocol not in P_SCORABLE_PROTOCOLS:
        raise ValueError(
            f"Unsupported protocol {protocol!r}. Expected one of {P_SCORABLE_PROTOCOLS}."
        )
    return (protocol,)


def add_eval_parser_args(
    parser: argparse.ArgumentParser,
    *,
    default: str = "all",
) -> None:
    """Attach the shared P eval protocol argument to a subparser."""

    if default == P_PAPER_PROTOCOL:
        help_text = (
            "P eval set to query. Default paper is the locked table "
            "(220 quads / 880 audios). all is an alias for paper. "
            "Construction logs: unseen_speaker or unseen_content."
        )
    else:
        help_text = (
            "Which frozen P protocol to export. all = construction protocols "
            "only. Paper examples: realmg prepare-p-paper-eval."
        )
    parser.add_argument(
        "--protocol",
        choices=(*P_SCORABLE_PROTOCOLS, "all"),
        default=default,
        help=help_text,
    )
