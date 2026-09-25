"""P-layer canonical content and quadruplet construction for ESD."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report


P_PRIMARY_PARALINGUISTIC_PAIR = ("neutral", "happy")
P_PAIRING_SEED = 42
CANONICAL_EMOTION_RAW = "neutral"


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def token_set(text: str) -> set[str]:
    """Tokenize text for overlap checks."""

    normalized = re.sub(r"[^\w\s]", " ", text.lower())
    return {token for token in normalized.split() if token}


def word_count(text: str) -> int:
    """Count words in a transcript."""

    return len(token_set(text))


def jaccard_similarity(left: str, right: str) -> float:
    """Compute word-level Jaccard similarity between two transcripts."""

    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return intersection / union


def content_pair_is_valid(left_transcript: str, right_transcript: str) -> bool:
    """Check whether two canonical transcripts satisfy P-layer pairing rules."""

    left_count = word_count(left_transcript)
    right_count = word_count(right_transcript)
    if left_count == 0 or right_count == 0:
        return False

    max_count = max(left_count, right_count)
    min_count = min(left_count, right_count)
    if max_count - min_count > 3:
        return False
    if (max_count - min_count) / max_count > 0.3:
        return False
    if jaccard_similarity(left_transcript, right_transcript) > 0.5:
        return False
    return True


def build_canonical_content_records(
    utterance_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build canonical content entries from neutral ESD transcripts."""

    canonical_records: list[dict[str, Any]] = []
    for record in utterance_records:
        if record["metadata"]["emotion_raw"] != CANONICAL_EMOTION_RAW:
            continue

        canonical_records.append(
            {
                "content_id": record["content_id"],
                "speaker_id": record["speaker_id"],
                "utt_slot": record["metadata"]["utt_slot"],
                "canonical_transcript": record["transcript"].strip(),
                "source_utterance_id": record["utterance_id"],
            }
        )

    canonical_records.sort(
        key=lambda item: (item["speaker_id"], item["utt_slot"])
    )
    return canonical_records


def pair_canonical_content_items(
    items: list[dict[str, Any]], seed: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair canonical content items within one speaker using greedy matching."""

    pool = list(items)
    rng = random.Random(seed)
    rng.shuffle(pool)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    used: set[str] = set()

    for index, left_item in enumerate(pool):
        if left_item["content_id"] in used:
            continue

        for right_item in pool[index + 1:]:
            if right_item["content_id"] in used:
                continue
            if not content_pair_is_valid(
                left_item["canonical_transcript"],
                right_item["canonical_transcript"],
            ):
                continue

            pairs.append((left_item, right_item))
            used.add(left_item["content_id"])
            used.add(right_item["content_id"])
            break

    return pairs


def utterance_lookup(
    utterance_records: list[dict[str, Any]],
) -> dict[tuple[str, str, str], str]:
    """Map speaker, content, and plan label to manifest utterance ids."""

    lookup: dict[tuple[str, str, str], str] = {}
    for record in utterance_records:
        key = (
            record["speaker_id"],
            record["content_id"],
            record["paralinguistic_label"],
        )
        lookup[key] = record["utterance_id"]
    return lookup


def build_quadruplet_record(
    protocol: str,
    split_name: str,
    speaker_id: str,
    left_item: dict[str, Any],
    right_item: dict[str, Any],
    paralinguistic_pair: tuple[str, str],
    lookup: dict[tuple[str, str, str], str],
) -> dict[str, Any]:
    """Build one P-layer quadruplet record."""

    content_pair = [left_item["content_id"], right_item["content_id"]]
    utterance_ids: list[str] = []

    for content_id in content_pair:
        for plan_label in paralinguistic_pair:
            key = (speaker_id, content_id, plan_label)
            if key not in lookup:
                raise KeyError(f"Missing utterance for {key}")
            utterance_ids.append(lookup[key])

    quadruplet_id = (
        f"p-{protocol}-{split_name}-{speaker_id}-"
        f"{left_item['utt_slot']:06d}-{right_item['utt_slot']:06d}-"
        f"{paralinguistic_pair[0]}-{paralinguistic_pair[1]}"
    )

    return {
        "quadruplet_id": quadruplet_id,
        "layer": "P",
        "split": split_name,
        "content_pair": content_pair,
        "paralinguistic_pair": list(paralinguistic_pair),
        "utterance_ids": utterance_ids,
        "metadata": {
            "protocol": protocol,
            "speaker_id": speaker_id,
            "canonical_transcripts": {
                left_item["content_id"]: left_item["canonical_transcript"],
                right_item["content_id"]: right_item["canonical_transcript"],
            },
            "utt_slots": [left_item["utt_slot"], right_item["utt_slot"]],
        },
    }


def build_quadruplets_for_protocol(
    protocol: str,
    utterance_records: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build canonical content and quadruplets for one P-layer protocol."""

    canonical_records = build_canonical_content_records(utterance_records)
    lookup = utterance_lookup(utterance_records)
    quadruplet_records: list[dict[str, Any]] = []

    for split_name in ("train", "construction-dev", "eval"):
        split_records = [
            record for record in utterance_records if record["split"] == split_name
        ]
        split_canonical = [
            item
            for item in canonical_records
            if any(
                record["content_id"] == item["content_id"]
                and record["speaker_id"] == item["speaker_id"]
                for record in split_records
            )
        ]

        by_speaker: dict[str, list[dict[str, Any]]] = {}
        for item in split_canonical:
            by_speaker.setdefault(item["speaker_id"], []).append(item)

        for speaker_id, items in sorted(by_speaker.items()):
            pairs = pair_canonical_content_items(items, seed + int(speaker_id))
            for left_item, right_item in pairs:
                quadruplet_records.append(
                    build_quadruplet_record(
                        protocol,
                        split_name,
                        speaker_id,
                        left_item,
                        right_item,
                        P_PRIMARY_PARALINGUISTIC_PAIR,
                        lookup,
                    )
                )

    return canonical_records, quadruplet_records


def build_quadruplet_registry(
    protocol: str,
    canonical_records: list[dict[str, Any]],
    quadruplet_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize quadruplet construction for auditing."""

    counts_by_split: dict[str, int] = {}
    for record in quadruplet_records:
        counts_by_split[record["split"]] = counts_by_split.get(record["split"], 0) + 1

    return {
        "protocol": protocol,
        "canonical_source_emotion": CANONICAL_EMOTION_RAW,
        "paralinguistic_pair": list(P_PRIMARY_PARALINGUISTIC_PAIR),
        "pairing_seed": P_PAIRING_SEED,
        "canonical_content_count": len(canonical_records),
        "quadruplet_counts_by_split": counts_by_split,
        "quadruplet_count": len(quadruplet_records),
    }


def prepare_p_quadruplets() -> None:
    """Create canonical content and quadruplet files for both P main protocols."""

    unseen_speaker_manifest = resolve_from_repo_root(
        "Data/manifests/p_utterances_unseen_speaker.jsonl"
    )
    unseen_content_manifest = resolve_from_repo_root(
        "Data/manifests/p_utterances_unseen_content.jsonl"
    )

    unseen_speaker_records = read_jsonl(unseen_speaker_manifest)
    unseen_content_records = read_jsonl(unseen_content_manifest)

    speaker_canonical, speaker_quadruplets = build_quadruplets_for_protocol(
        "unseen_speaker",
        unseen_speaker_records,
        P_PAIRING_SEED,
    )
    content_canonical, content_quadruplets = build_quadruplets_for_protocol(
        "unseen_content",
        unseen_content_records,
        P_PAIRING_SEED,
    )

    write_jsonl(
        speaker_canonical,
        resolve_from_repo_root(
            "Data/manifests/p_canonical_content_unseen_speaker.jsonl"
        ),
    )
    write_jsonl(
        speaker_quadruplets,
        resolve_from_repo_root(
            "Data/manifests/p_quadruplets_unseen_speaker.jsonl"
        ),
    )
    write_report(
        build_quadruplet_registry(
            "unseen_speaker", speaker_canonical, speaker_quadruplets
        ),
        resolve_from_repo_root("Data/manifests/p_quadruplets_unseen_speaker.json"),
    )

    write_jsonl(
        content_canonical,
        resolve_from_repo_root(
            "Data/manifests/p_canonical_content_unseen_content.jsonl"
        ),
    )
    write_jsonl(
        content_quadruplets,
        resolve_from_repo_root(
            "Data/manifests/p_quadruplets_unseen_content.jsonl"
        ),
    )
    write_report(
        build_quadruplet_registry(
            "unseen_content", content_canonical, content_quadruplets
        ),
        resolve_from_repo_root("Data/manifests/p_quadruplets_unseen_content.json"),
    )
