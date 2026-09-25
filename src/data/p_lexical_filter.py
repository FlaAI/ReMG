"""P-layer lexical leakage filter with a frozen external text emotion classifier."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report


TEXT_EMOTION_MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
NON_NEUTRAL_CONFIDENCE_MIN = 0.9
LEAK_LABELS = frozenset({"anger", "joy", "sadness", "surprise"})
P_PROTOCOLS = ("unseen_speaker", "unseen_content")
LEXICAL_SCORE_CACHE_PATH = "Data/cards/p_lexical_scores.jsonl"


@lru_cache(maxsize=1)
def load_text_emotion_classifier() -> Any:
    """Load the frozen DistilRoBERTa emotion classifier once."""

    import torch
    from transformers import pipeline

    device = 0 if torch.cuda.is_available() else -1
    return pipeline(
        "text-classification",
        model=TEXT_EMOTION_MODEL_NAME,
        top_k=None,
        device=device,
    )


def classify_transcript_emotion(transcript: str) -> dict[str, Any]:
    """Return label scores and the strongest non-neutral emotion for one transcript."""

    classifier = load_text_emotion_classifier()
    label_scores = {
        item["label"]: float(item["score"]) for item in classifier(transcript)[0]
    }
    predicted_label = max(label_scores, key=label_scores.get)
    leak_label = max(LEAK_LABELS, key=lambda label: label_scores[label])
    leak_score = label_scores[leak_label]
    return {
        "predicted_label": predicted_label,
        "predicted_score": label_scores[predicted_label],
        "non_neutral_label": leak_label,
        "non_neutral_score": leak_score,
        "label_scores": label_scores,
    }


def transcript_has_lexical_leak(score_row: dict[str, Any]) -> bool:
    """Drop a script if text alone confidently predicts a non-neutral emotion."""

    return score_row["non_neutral_score"] >= NON_NEUTRAL_CONFIDENCE_MIN


def collect_unique_scripts(
    utterance_records: list[dict[str, Any]],
) -> dict[int, str]:
    """Collect one official transcript per ESD utterance slot."""

    scripts: dict[int, str] = {}
    for record in utterance_records:
        utt_slot = record["metadata"]["utt_slot"]
        if utt_slot not in scripts:
            scripts[utt_slot] = record["transcript"]
    return scripts


def score_unique_scripts(scripts: dict[int, str]) -> dict[int, dict[str, Any]]:
    """Score each unique ESD script with the frozen text classifier."""

    scored_scripts: dict[int, dict[str, Any]] = {}
    for utt_slot, transcript in sorted(scripts.items()):
        score_row = classify_transcript_emotion(transcript)
        score_row["utt_slot"] = utt_slot
        score_row["transcript"] = transcript
        score_row["lexical_leak"] = transcript_has_lexical_leak(score_row)
        score_row["text_emotion_model"] = TEXT_EMOTION_MODEL_NAME
        scored_scripts[utt_slot] = score_row
    return scored_scripts


def filter_utterance_records(
    utterance_records: list[dict[str, Any]],
    scored_scripts: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep utterances whose script does not leak emotion from text alone."""

    kept_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    for record in utterance_records:
        score_row = scored_scripts[record["metadata"]["utt_slot"]]
        updated_record = dict(record)
        updated_record["metadata"] = dict(record["metadata"])
        updated_record["metadata"]["text_emotion_label"] = score_row["predicted_label"]
        updated_record["metadata"]["text_emotion_non_neutral_score"] = score_row[
            "non_neutral_score"
        ]
        if score_row["lexical_leak"]:
            updated_record["metadata"]["lexical_rejection_reason"] = "text_emotion"
            rejected_records.append(updated_record)
        else:
            kept_records.append(updated_record)
    return kept_records, rejected_records


def filter_quadruplet_records(
    quadruplet_records: list[dict[str, Any]],
    scored_scripts: dict[int, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop a quadruplet if either content script leaks emotion from text."""

    kept_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    for record in quadruplet_records:
        leaked_slots = [
            utt_slot
            for utt_slot in record["metadata"]["utt_slots"]
            if scored_scripts[utt_slot]["lexical_leak"]
        ]
        updated_record = dict(record)
        updated_record["metadata"] = dict(record["metadata"])
        if leaked_slots:
            updated_record["metadata"]["lexical_rejection_reason"] = "text_emotion"
            updated_record["metadata"]["leaked_utt_slots"] = leaked_slots
            rejected_records.append(updated_record)
        else:
            kept_records.append(updated_record)
    return kept_records, rejected_records


def build_filter_report(
    protocol: str,
    utterance_records: list[dict[str, Any]],
    kept_utterances: list[dict[str, Any]],
    rejected_utterances: list[dict[str, Any]],
    quadruplet_records: list[dict[str, Any]],
    kept_quadruplets: list[dict[str, Any]],
    rejected_quadruplets: list[dict[str, Any]],
    leaked_slot_count: int,
    scored_script_count: int,
) -> dict[str, Any]:
    """Summarize lexical filtering results for one protocol."""

    return {
        "protocol": protocol,
        "text_emotion_model": TEXT_EMOTION_MODEL_NAME,
        "leak_labels": sorted(LEAK_LABELS),
        "non_neutral_confidence_min": NON_NEUTRAL_CONFIDENCE_MIN,
        "unique_script_count": scored_script_count,
        "leaked_script_count": leaked_slot_count,
        "utterance_count_before": len(utterance_records),
        "utterance_count_after": len(kept_utterances),
        "utterance_rejected_count": len(rejected_utterances),
        "quadruplet_count_before": len(quadruplet_records),
        "quadruplet_count_after": len(kept_quadruplets),
        "quadruplet_rejected_count": len(rejected_quadruplets),
    }


def filter_protocol_bundle(
    protocol: str,
    scored_scripts: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """Apply lexical filtering to one WER-filtered P-layer protocol bundle."""

    utterance_path = resolve_from_repo_root(
        f"Data/manifests/p_utterances_{protocol}_filtered.jsonl"
    )
    quadruplet_path = resolve_from_repo_root(
        f"Data/manifests/p_quadruplets_{protocol}_filtered.jsonl"
    )
    utterance_records = read_jsonl(utterance_path)
    quadruplet_records = read_jsonl(quadruplet_path)

    kept_utterances, rejected_utterances = filter_utterance_records(
        utterance_records, scored_scripts
    )
    kept_quadruplets, rejected_quadruplets = filter_quadruplet_records(
        quadruplet_records, scored_scripts
    )

    write_jsonl(
        kept_utterances,
        utterance_path.with_name(f"p_utterances_{protocol}_lex_filtered.jsonl"),
    )
    write_jsonl(
        rejected_utterances,
        utterance_path.with_name(f"p_utterances_{protocol}_lex_rejected.jsonl"),
    )
    write_jsonl(
        kept_quadruplets,
        quadruplet_path.with_name(f"p_quadruplets_{protocol}_lex_filtered.jsonl"),
    )
    write_jsonl(
        rejected_quadruplets,
        quadruplet_path.with_name(f"p_quadruplets_{protocol}_lex_rejected.jsonl"),
    )

    leaked_slot_count = sum(
        1 for score_row in scored_scripts.values() if score_row["lexical_leak"]
    )
    report = build_filter_report(
        protocol,
        utterance_records,
        kept_utterances,
        rejected_utterances,
        quadruplet_records,
        kept_quadruplets,
        rejected_quadruplets,
        leaked_slot_count,
        len(scored_scripts),
    )
    write_report(
        report,
        resolve_from_repo_root(f"Data/cards/p_lexical_filter_{protocol}.json"),
    )
    return report


def write_lexical_score_cache(scored_scripts: dict[int, dict[str, Any]]) -> None:
    """Write unique-script emotion scores for the data card."""

    rows = [scored_scripts[utt_slot] for utt_slot in sorted(scored_scripts)]
    write_jsonl(rows, resolve_from_repo_root(LEXICAL_SCORE_CACHE_PATH))


def apply_lexical_leak_rule(score_row: dict[str, Any]) -> dict[str, Any]:
    """Recompute leak flags from cached label scores under the frozen rule."""

    label_scores = score_row["label_scores"]
    leak_label = max(LEAK_LABELS, key=lambda label: label_scores[label])
    leak_score = float(label_scores[leak_label])
    updated_row = dict(score_row)
    updated_row["non_neutral_label"] = leak_label
    updated_row["non_neutral_score"] = leak_score
    updated_row["lexical_leak"] = leak_score >= NON_NEUTRAL_CONFIDENCE_MIN
    updated_row["leak_labels"] = sorted(LEAK_LABELS)
    updated_row["text_emotion_model"] = TEXT_EMOTION_MODEL_NAME
    return updated_row


def load_scored_scripts_from_cache() -> dict[int, dict[str, Any]]:
    """Load cached classifier scores if they already contain label scores."""

    cache_path = resolve_from_repo_root(LEXICAL_SCORE_CACHE_PATH)
    if not cache_path.exists():
        return {}

    scored_scripts: dict[int, dict[str, Any]] = {}
    for row in read_jsonl(cache_path):
        if "label_scores" not in row:
            return {}
        scored_scripts[row["utt_slot"]] = apply_lexical_leak_rule(row)
    return scored_scripts


def filter_p_layer_lexical() -> None:
    """Filter WER-kept P-layer items whose canonical text leaks emotion."""

    scored_scripts = load_scored_scripts_from_cache()
    if not scored_scripts:
        scripts = collect_unique_scripts(
            read_jsonl(resolve_from_repo_root("Data/manifests/p_utterances_draft.jsonl"))
        )
        scored_scripts = {
            utt_slot: apply_lexical_leak_rule(score_row)
            for utt_slot, score_row in score_unique_scripts(scripts).items()
        }
    write_lexical_score_cache(scored_scripts)

    reports = {
        protocol: filter_protocol_bundle(protocol, scored_scripts)
        for protocol in P_PROTOCOLS
    }
    write_report(
        {
            "text_emotion_model": TEXT_EMOTION_MODEL_NAME,
            "leak_labels": sorted(LEAK_LABELS),
            "non_neutral_confidence_min": NON_NEUTRAL_CONFIDENCE_MIN,
            **reports,
        },
        resolve_from_repo_root("Data/cards/p_lexical_filter_summary.json"),
    )
