"""P-layer WER filtering with Whisper-medium."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report
from realmg.data.wer import transcribe_audio_file, wer_percent


WHISPER_MODEL_NAME = "medium"
SINGLE_WER_MAX_PERCENT = 10.0
DELTA_WER_MAX_TRAIN_DEV_PERCENT = 5.0
DELTA_WER_MAX_EVAL_PERCENT = 3.0
PRIMARY_PARALINGUISTIC_PAIR = ("neutral", "happy")


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def load_wer_score_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load cached WER scores keyed by utterance id."""

    if not cache_path.exists():
        return {}

    cached_scores: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            cached_scores[row["utterance_id"]] = row
    return cached_scores


def append_wer_score_cache(cache_path: Path, score_row: dict[str, Any]) -> None:
    """Append one WER score row to the cache file."""

    ensure_parent(cache_path)
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(score_row, ensure_ascii=False) + "\n")


def score_utterances_with_whisper(
    utterance_records: list[dict[str, Any]],
    cache_path: Path,
) -> dict[str, dict[str, Any]]:
    """Transcribe utterances and compute WER against their reference transcripts."""

    cached_scores = load_wer_score_cache(cache_path)
    unique_records = {record["utterance_id"]: record for record in utterance_records}

    total_count = len(unique_records)
    scored_count = 0

    for index, utterance_id in enumerate(sorted(unique_records), start=1):
        if utterance_id in cached_scores:
            continue

        record = unique_records[utterance_id]
        audio_path = resolve_from_repo_root(record["audio_path"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio file: {audio_path}")

        hypothesis = transcribe_audio_file(audio_path, WHISPER_MODEL_NAME)
        reference = record["transcript"]
        score_row = {
            "utterance_id": utterance_id,
            "audio_path": record["audio_path"],
            "reference_transcript": reference,
            "hypothesis_transcript": hypothesis,
            "wer_percent": wer_percent(reference, hypothesis),
            "whisper_model": WHISPER_MODEL_NAME,
        }
        cached_scores[utterance_id] = score_row
        append_wer_score_cache(cache_path, score_row)
        scored_count += 1

        if scored_count % 50 == 0 or index == total_count:
            print(
                f"Scored {index}/{total_count} utterances ({scored_count} new)",
                flush=True,
            )

    return cached_scores


def delta_wer_threshold(split_name: str) -> float:
    """Return the delta-WER threshold for one split."""

    if split_name == "eval":
        return DELTA_WER_MAX_EVAL_PERCENT
    return DELTA_WER_MAX_TRAIN_DEV_PERCENT


def utterance_passes_single_wer(score_row: dict[str, Any]) -> bool:
    """Check whether one utterance passes the single-WER threshold."""

    return score_row["wer_percent"] <= SINGLE_WER_MAX_PERCENT


def content_pair_passes_delta_wer(
    left_score: float,
    right_score: float,
    split_name: str,
) -> bool:
    """Check whether two emotions for the same content pass delta-WER."""

    return abs(left_score - right_score) <= delta_wer_threshold(split_name)


def build_content_delta_lookup(
    utterance_records: list[dict[str, Any]],
    wer_scores: dict[str, dict[str, Any]],
) -> dict[tuple[str, str, str], bool]:
    """Mark whether each content id passes neutral-vs-happy delta WER."""

    scores_by_key: dict[tuple[str, str, str], float] = {}
    split_by_content: dict[tuple[str, str], str] = {}

    for record in utterance_records:
        if record["paralinguistic_label"] not in PRIMARY_PARALINGUISTIC_PAIR:
            continue
        key = (record["speaker_id"], record["content_id"], record["paralinguistic_label"])
        scores_by_key[key] = wer_scores[record["utterance_id"]]["wer_percent"]
        split_by_content[(record["speaker_id"], record["content_id"])] = record["split"]

    delta_lookup: dict[tuple[str, str, str], bool] = {}
    for (speaker_id, content_id), split_name in split_by_content.items():
        neutral_key = (speaker_id, content_id, "neutral")
        happy_key = (speaker_id, content_id, "happy")
        if neutral_key not in scores_by_key or happy_key not in scores_by_key:
            delta_lookup[(speaker_id, content_id, split_name)] = False
            continue
        delta_lookup[(speaker_id, content_id, split_name)] = content_pair_passes_delta_wer(
            scores_by_key[neutral_key],
            scores_by_key[happy_key],
            split_name,
        )
    return delta_lookup


def filter_utterance_records(
    utterance_records: list[dict[str, Any]],
    wer_scores: dict[str, dict[str, Any]],
    delta_lookup: dict[tuple[str, str, str], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split utterances into kept and rejected sets with reasons."""

    kept_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []

    for record in utterance_records:
        score_row = wer_scores[record["utterance_id"]]
        rejection_reasons: list[str] = []

        if not utterance_passes_single_wer(score_row):
            rejection_reasons.append("single_wer")

        content_key = (
            record["speaker_id"],
            record["content_id"],
            record["split"],
        )
        if record["paralinguistic_label"] in PRIMARY_PARALINGUISTIC_PAIR:
            if not delta_lookup.get(content_key, False):
                rejection_reasons.append("delta_wer")

        updated_record = dict(record)
        updated_record["metadata"] = dict(record["metadata"])
        updated_record["metadata"]["wer_percent"] = score_row["wer_percent"]
        updated_record["metadata"]["hypothesis_transcript"] = score_row["hypothesis_transcript"]

        if rejection_reasons:
            updated_record["metadata"]["wer_rejection_reasons"] = rejection_reasons
            rejected_records.append(updated_record)
        else:
            kept_records.append(updated_record)

    return kept_records, rejected_records


def filter_quadruplet_records(
    quadruplet_records: list[dict[str, Any]],
    kept_utterance_ids: set[str],
    delta_lookup: dict[tuple[str, str, str], bool],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter quadruplets using kept utterances and content delta-WER."""

    kept_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []

    for record in quadruplet_records:
        rejection_reasons: list[str] = []
        if not all(utterance_id in kept_utterance_ids for utterance_id in record["utterance_ids"]):
            rejection_reasons.append("utterance_wer")

        speaker_id = record["metadata"]["speaker_id"]
        split_name = record["split"]
        for content_id in record["content_pair"]:
            if not delta_lookup.get((speaker_id, content_id, split_name), False):
                rejection_reasons.append("content_delta_wer")
                break

        updated_record = dict(record)
        updated_record["metadata"] = dict(record["metadata"])
        if rejection_reasons:
            updated_record["metadata"]["wer_rejection_reasons"] = rejection_reasons
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
) -> dict[str, Any]:
    """Summarize WER filtering results for one protocol."""

    return {
        "protocol": protocol,
        "whisper_model": WHISPER_MODEL_NAME,
        "single_wer_max_percent": SINGLE_WER_MAX_PERCENT,
        "delta_wer_max_train_dev_percent": DELTA_WER_MAX_TRAIN_DEV_PERCENT,
        "delta_wer_max_eval_percent": DELTA_WER_MAX_EVAL_PERCENT,
        "utterance_count_before": len(utterance_records),
        "utterance_count_after": len(kept_utterances),
        "utterance_rejected_count": len(rejected_utterances),
        "quadruplet_count_before": len(quadruplet_records),
        "quadruplet_count_after": len(kept_quadruplets),
        "quadruplet_rejected_count": len(rejected_quadruplets),
    }


def filter_protocol_bundle(
    protocol: str,
    utterance_manifest_path: Path,
    quadruplet_manifest_path: Path,
    wer_scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply WER filtering to one P-layer protocol bundle."""

    utterance_records = read_jsonl(utterance_manifest_path)
    quadruplet_records = read_jsonl(quadruplet_manifest_path)
    delta_lookup = build_content_delta_lookup(utterance_records, wer_scores)

    kept_utterances, rejected_utterances = filter_utterance_records(
        utterance_records, wer_scores, delta_lookup
    )
    kept_utterance_ids = {record["utterance_id"] for record in kept_utterances}
    kept_quadruplets, rejected_quadruplets = filter_quadruplet_records(
        quadruplet_records, kept_utterance_ids, delta_lookup
    )

    filtered_utterance_path = utterance_manifest_path.with_name(
        utterance_manifest_path.name.replace(".jsonl", "_filtered.jsonl")
    )
    rejected_utterance_path = utterance_manifest_path.with_name(
        utterance_manifest_path.name.replace(".jsonl", "_wer_rejected.jsonl")
    )
    filtered_quadruplet_path = quadruplet_manifest_path.with_name(
        quadruplet_manifest_path.name.replace(".jsonl", "_filtered.jsonl")
    )
    rejected_quadruplet_path = quadruplet_manifest_path.with_name(
        quadruplet_manifest_path.name.replace(".jsonl", "_wer_rejected.jsonl")
    )
    report_path = resolve_from_repo_root(
        f"Data/cards/p_wer_filter_{protocol}.json"
    )

    write_jsonl(kept_utterances, filtered_utterance_path)
    write_jsonl(rejected_utterances, rejected_utterance_path)
    write_jsonl(kept_quadruplets, filtered_quadruplet_path)
    write_jsonl(rejected_quadruplets, rejected_quadruplet_path)

    report = build_filter_report(
        protocol,
        utterance_records,
        kept_utterances,
        rejected_utterances,
        quadruplet_records,
        kept_quadruplets,
        rejected_quadruplets,
    )
    write_report(report, report_path)
    return report


def filter_p_layer_wer() -> None:
    """Score P-layer utterances with Whisper-medium and apply WER filters."""

    draft_manifest_path = resolve_from_repo_root("Data/manifests/p_utterances_draft.jsonl")
    wer_cache_path = resolve_from_repo_root("Data/cards/p_wer_scores.jsonl")

    draft_records = read_jsonl(draft_manifest_path)
    wer_scores = score_utterances_with_whisper(draft_records, wer_cache_path)

    speaker_report = filter_protocol_bundle(
        "unseen_speaker",
        resolve_from_repo_root("Data/manifests/p_utterances_unseen_speaker.jsonl"),
        resolve_from_repo_root("Data/manifests/p_quadruplets_unseen_speaker.jsonl"),
        wer_scores,
    )
    content_report = filter_protocol_bundle(
        "unseen_content",
        resolve_from_repo_root("Data/manifests/p_utterances_unseen_content.jsonl"),
        resolve_from_repo_root("Data/manifests/p_quadruplets_unseen_content.jsonl"),
        wer_scores,
    )

    summary_path = resolve_from_repo_root("Data/cards/p_wer_filter_summary.json")
    write_report(
        {
            "whisper_model": WHISPER_MODEL_NAME,
            "unseen_speaker": speaker_report,
            "unseen_content": content_report,
        },
        summary_path,
    )
