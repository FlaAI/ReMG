"""Download and verify the Hugging Face ESD mirror used by RealMG."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

from datasets import Audio, Dataset, DatasetDict, load_dataset, load_from_disk

from realmg.data.config import load_local_paths, resolve_from_repo_root


HF_ESD_REPO = "jspaulsen/esd"
EXPECTED_ENGLISH_SPEAKERS = 10
EXPECTED_UTTERANCES_PER_ENGLISH_SPEAKER = 1750
EXPECTED_UTTERANCE_SLOTS_PER_SPEAKER = 350
EXPECTED_EMOTIONS = {"neutral", "happiness", "anger", "sadness", "surprise"}
EMOTION_TO_PLAN_LABEL = {
    "anger": "angry",
    "happiness": "happy",
    "neutral": "neutral",
    "sadness": "sad",
    "surprise": "surprise",
}
P_UNSEEN_SPEAKER_SPLIT = {
    "eval": ["0011", "0015"],
    "construction-dev": ["0012"],
}
P_UNSEEN_CONTENT_SPLIT_SEED = 42
P_UNSEEN_CONTENT_EVAL_SLOTS = 50
P_UNSEEN_CONTENT_DEV_SLOTS = 35


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def official_utterance_id(speaker_id: str, utt_slot: int) -> str:
    """Format the official ESD utterance id from speaker and slot index."""

    return f"{speaker_id}_{utt_slot:06d}"


def map_esd_emotion_to_plan_label(emotion: str) -> str:
    """Map ESD emotion labels to the plan's paralinguistic label set."""

    return EMOTION_TO_PLAN_LABEL[emotion]


def relative_audio_path_for_row(row: dict[str, Any]) -> str:
    """Return the repository-relative wav path for one English ESD row."""

    speaker_id = row["speaker_id"]
    emotion = row["emotion"]
    official_id = row["official_utterance_id"]
    return (
        f"Data/raw/esd/wav/en/{speaker_id}/{emotion}/{official_id}.wav"
    )


def iter_english_rows_in_dataset_order(dataset_dict: DatasetDict) -> Iterator[dict[str, Any]]:
    """Yield English rows in the original HF dataset order."""

    train_split: Dataset = dataset_dict["train"]
    for row in train_split:
        if row["language"] == "en":
            yield row


def annotate_english_utterance_slots(rows: Iterator[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assign utterance slots from block position within each speaker-emotion group."""

    slot_counters: dict[tuple[str, str], int] = defaultdict(int)
    annotated_rows: list[dict[str, Any]] = []

    for row in rows:
        speaker_id = row["speaker_id"]
        emotion = row["emotion"]
        slot_counters[(speaker_id, emotion)] += 1
        utt_slot = slot_counters[(speaker_id, emotion)]

        annotated_row = dict(row)
        annotated_row["utt_slot"] = utt_slot
        annotated_row["official_utterance_id"] = official_utterance_id(speaker_id, utt_slot)
        annotated_rows.append(annotated_row)

    return annotated_rows


def build_integrity_report(rows: list[dict[str, Any]]) -> dict:
    """Compute structural checks using official utterance slots, not raw transcript."""

    speaker_counts = Counter(row["speaker_id"] for row in rows)
    emotion_counts = Counter(row["emotion"] for row in rows)
    emotions_by_speaker_slot: dict[tuple[str, int], set[str]] = defaultdict(set)
    slot_counts_per_speaker: dict[str, set[int]] = defaultdict(set)

    for row in rows:
        speaker_id = row["speaker_id"]
        utt_slot = row["utt_slot"]
        emotions_by_speaker_slot[(speaker_id, utt_slot)].add(row["emotion"])
        slot_counts_per_speaker[speaker_id].add(utt_slot)

    incomplete_parallel_groups = {
        f"{speaker_id}_{utt_slot:06d}": sorted(emotions)
        for (speaker_id, utt_slot), emotions in emotions_by_speaker_slot.items()
        if len(emotions) != 5
    }

    return {
        "hf_repo": HF_ESD_REPO,
        "grouping_key": "official_utterance_id_from_block_position",
        "english_row_count": len(rows),
        "speaker_count": len(speaker_counts),
        "speaker_ids": sorted(speaker_counts),
        "speaker_utterance_counts": dict(sorted(speaker_counts.items())),
        "emotion_counts": dict(sorted(emotion_counts.items())),
        "utterance_slot_count_per_speaker": {
            speaker_id: len(slots)
            for speaker_id, slots in sorted(slot_counts_per_speaker.items())
        },
        "incomplete_parallel_groups": incomplete_parallel_groups,
        "checks": {
            "speaker_count_is_10": len(speaker_counts) == EXPECTED_ENGLISH_SPEAKERS,
            "all_speakers_have_1750_rows": all(
                count == EXPECTED_UTTERANCES_PER_ENGLISH_SPEAKER
                for count in speaker_counts.values()
            ),
            "all_speakers_have_350_utterance_slots": all(
                len(slots) == EXPECTED_UTTERANCE_SLOTS_PER_SPEAKER
                for slots in slot_counts_per_speaker.values()
            ),
            "emotion_label_set_matches_plan": set(emotion_counts) == EXPECTED_EMOTIONS,
            "all_speaker_slot_groups_cover_5_emotions": not incomplete_parallel_groups,
        },
    }


def build_p_utterance_record(row: dict[str, Any]) -> dict[str, Any]:
    """Build one P-layer utterance manifest record from an annotated ESD row."""

    speaker_id = row["speaker_id"]
    utt_slot = row["utt_slot"]
    emotion = row["emotion"]
    official_id = row["official_utterance_id"]
    plan_label = map_esd_emotion_to_plan_label(emotion)
    manifest_utterance_id = f"p-esd-{speaker_id}-{utt_slot:06d}-{plan_label}"
    audio_path = relative_audio_path_for_row(row)

    return {
        "utterance_id": manifest_utterance_id,
        "layer": "P",
        "source_name": "esd",
        "split": "train",
        "content_id": official_id,
        "paralinguistic_label": plan_label,
        "audio_path": audio_path,
        "transcript": row["transcript"].strip(),
        "speaker_id": speaker_id,
        "reference_speaker_id": None,
        "tts_engine": None,
        "metadata": {
            "hf_repo": HF_ESD_REPO,
            "official_utterance_id": official_id,
            "emotion_raw": emotion,
            "utt_slot": utt_slot,
            "gender": row["gender"],
            "audio_export_pending": False,
        },
    }


def write_report(report: dict[str, Any], report_path: Path) -> None:
    """Write the integrity report in JSON format."""

    ensure_parent(report_path)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> None:
    """Write manifest records as JSONL."""

    ensure_parent(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(input_path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries."""

    records: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def export_english_wavs(rows: list[dict[str, Any]]) -> None:
    """Export English ESD audio bytes to stable local wav files."""

    for row in rows:
        audio = row["audio"]
        audio_bytes = audio["bytes"]
        if audio_bytes is None:
            raise ValueError("Expected HF audio bytes for ESD export.")

        output_path = resolve_from_repo_root(relative_audio_path_for_row(row))
        ensure_parent(output_path)
        if not output_path.exists():
            output_path.write_bytes(audio_bytes)


def split_for_unseen_speaker(speaker_id: str) -> str:
    """Return the split name for the unseen-speaker P-layer protocol."""

    if speaker_id in P_UNSEEN_SPEAKER_SPLIT["eval"]:
        return "eval"
    if speaker_id in P_UNSEEN_SPEAKER_SPLIT["construction-dev"]:
        return "construction-dev"
    return "train"


def build_unseen_content_slot_split() -> dict[str, set[int]]:
    """Create the shared utterance-slot split for the main unseen-content protocol."""

    all_slots = list(range(1, EXPECTED_UTTERANCE_SLOTS_PER_SPEAKER + 1))
    rng = random.Random(P_UNSEEN_CONTENT_SPLIT_SEED)
    rng.shuffle(all_slots)

    eval_slots = set(all_slots[:P_UNSEEN_CONTENT_EVAL_SLOTS])
    construction_dev_slots = set(
        all_slots[
            P_UNSEEN_CONTENT_EVAL_SLOTS:
            P_UNSEEN_CONTENT_EVAL_SLOTS + P_UNSEEN_CONTENT_DEV_SLOTS
        ]
    )
    train_slots = set(
        all_slots[P_UNSEEN_CONTENT_EVAL_SLOTS + P_UNSEEN_CONTENT_DEV_SLOTS:]
    )
    return {
        "eval": eval_slots,
        "construction-dev": construction_dev_slots,
        "train": train_slots,
    }


def split_for_unseen_content(utt_slot: int, slot_split: dict[str, set[int]]) -> str:
    """Return the split name for the unseen-content P-layer protocol."""

    if utt_slot in slot_split["eval"]:
        return "eval"
    if utt_slot in slot_split["construction-dev"]:
        return "construction-dev"
    return "train"


def build_unseen_speaker_registry(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the unseen-speaker split assignment for auditing."""

    counts_by_split = Counter(record["split"] for record in records)
    speaker_to_split = {
        speaker_id: split_for_unseen_speaker(speaker_id)
        for speaker_id in sorted({record["speaker_id"] for record in records})
    }
    return {
        "protocol": "p_unseen_speaker_main",
        "eval_speakers": P_UNSEEN_SPEAKER_SPLIT["eval"],
        "construction_dev_speakers": P_UNSEEN_SPEAKER_SPLIT["construction-dev"],
        "train_speakers": [
            speaker_id
            for speaker_id, split in speaker_to_split.items()
            if split == "train"
        ],
        "utterance_counts_by_split": dict(sorted(counts_by_split.items())),
        "speaker_to_split": speaker_to_split,
    }


def build_unseen_content_registry(
    records: list[dict[str, Any]], slot_split: dict[str, set[int]]
) -> dict[str, Any]:
    """Summarize the unseen-content split assignment for auditing."""

    counts_by_split = Counter(record["split"] for record in records)
    return {
        "protocol": "p_unseen_content_main",
        "seed": P_UNSEEN_CONTENT_SPLIT_SEED,
        "eval_slots": sorted(slot_split["eval"]),
        "construction_dev_slots": sorted(slot_split["construction-dev"]),
        "train_slot_count": len(slot_split["train"]),
        "utterance_counts_by_split": dict(sorted(counts_by_split.items())),
    }


def prepare_p_unseen_speaker_split() -> None:
    """Create the main P-layer unseen-speaker split files."""

    draft_manifest_path = resolve_from_repo_root("Data/manifests/p_utterances_draft.jsonl")
    split_manifest_path = resolve_from_repo_root(
        "Data/manifests/p_utterances_unseen_speaker.jsonl"
    )
    split_registry_path = resolve_from_repo_root(
        "Data/manifests/p_unseen_speaker_split.json"
    )

    draft_records = read_jsonl(draft_manifest_path)
    split_records: list[dict[str, Any]] = []

    for record in draft_records:
        updated_record = dict(record)
        updated_record["split"] = split_for_unseen_speaker(record["speaker_id"])
        split_records.append(updated_record)

    split_registry = build_unseen_speaker_registry(split_records)
    write_jsonl(split_records, split_manifest_path)
    write_report(split_registry, split_registry_path)


def prepare_p_unseen_content_split() -> None:
    """Create the main P-layer unseen-content split files."""

    draft_manifest_path = resolve_from_repo_root("Data/manifests/p_utterances_draft.jsonl")
    split_manifest_path = resolve_from_repo_root(
        "Data/manifests/p_utterances_unseen_content.jsonl"
    )
    split_registry_path = resolve_from_repo_root(
        "Data/manifests/p_unseen_content_split.json"
    )

    slot_split = build_unseen_content_slot_split()
    draft_records = read_jsonl(draft_manifest_path)
    split_records: list[dict[str, Any]] = []

    for record in draft_records:
        updated_record = dict(record)
        utt_slot = updated_record["metadata"]["utt_slot"]
        updated_record["split"] = split_for_unseen_content(utt_slot, slot_split)
        split_records.append(updated_record)

    split_registry = build_unseen_content_registry(split_records, slot_split)
    write_jsonl(split_records, split_manifest_path)
    write_report(split_registry, split_registry_path)


def write_local_paths_if_missing() -> Path:
    """Create the local path config with the agreed default roots."""

    config_path = resolve_from_repo_root("Data/configs/local_paths.toml")
    if config_path.exists():
        return config_path

    contents = """[storage]
raw_root = "Data/raw"
audio_root = "Data/audio"

[sources]
esd_root = "Data/raw/esd"
vctk_root = "Data/raw/vctk"
tulu3_root = "Data/raw/tulu3"
natural_reasoning_root = "Data/raw/natural_reasoning"

[models]
indextts2_root = ""
cosyvoice2_root = ""
asr_root = ""
ser_root = "Data/models/emotion2vec_plus_large"
speechbrain_ser_root = "Data/models/speechbrain_emotion_iemocap"
"""
    ensure_parent(config_path)
    config_path.write_text(contents, encoding="utf-8")
    return config_path


def load_or_download_esd(esd_root: Path, cache_dir: Path, dataset_dir: Path) -> DatasetDict:
    """Load a saved ESD snapshot or download it from Hugging Face."""

    if dataset_dir.exists():
        dataset = load_from_disk(str(dataset_dir))
    else:
        esd_root.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(HF_ESD_REPO, cache_dir=str(cache_dir))
        dataset = dataset.cast_column("audio", Audio(decode=False))
        dataset.save_to_disk(str(dataset_dir))

    if isinstance(dataset, DatasetDict):
        dataset["train"] = dataset["train"].cast_column("audio", Audio(decode=False))
        return dataset

    raise TypeError("Expected ESD snapshot to be a DatasetDict.")


def download_and_verify_esd() -> None:
    """Download the ESD mirror and emit integrity plus P-layer manifest drafts."""

    write_local_paths_if_missing()
    local_paths = load_local_paths()
    esd_root = resolve_from_repo_root(local_paths["sources"]["esd_root"])
    cache_dir = esd_root / "hf_cache"
    dataset_dir = esd_root / "hf_snapshot"
    report_path = resolve_from_repo_root("Data/cards/esd_english_integrity_report.json")
    manifest_path = resolve_from_repo_root("Data/manifests/p_utterances_draft.jsonl")

    dataset = load_or_download_esd(esd_root, cache_dir, dataset_dir)
    annotated_rows = annotate_english_utterance_slots(
        iter_english_rows_in_dataset_order(dataset)
    )
    export_english_wavs(annotated_rows)
    report = build_integrity_report(annotated_rows)
    manifest_records = [build_p_utterance_record(row) for row in annotated_rows]

    write_report(report, report_path)
    write_jsonl(manifest_records, manifest_path)
