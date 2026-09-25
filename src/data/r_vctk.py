"""VCTK reference-speaker pool for the R layer."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from realmg.data.config import load_local_paths, resolve_from_repo_root
from realmg.data.esd import write_report


VCTK_OFFICIAL_ZIP_URL = (
    "https://datashare.is.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
)
R_REFERENCE_SPLIT_SEED = 42
R_EVAL_SPEAKERS_PER_GENDER = 8
R_CONSTRUCTION_DEV_FRACTION = 0.10
# p315 has no text files in VCTK 0.92.
VCTK_SPEAKERS_WITHOUT_TEXT = frozenset({"p315"})


def normalize_vctk_gender(raw_gender: str) -> str:
    """Map VCTK gender strings to female/male."""

    lowered = raw_gender.strip().lower()
    if lowered in {"f", "female"}:
        return "female"
    if lowered in {"m", "male"}:
        return "male"
    raise ValueError(f"Unrecognized VCTK gender: {raw_gender}")


def find_vctk_speaker_info_file(vctk_root: Path) -> Path:
    """Locate speaker-info.txt in a local VCTK 0.92 extract."""

    candidates = [
        vctk_root / "speaker-info.txt",
        vctk_root / "VCTK-Corpus-0.92" / "speaker-info.txt",
        vctk_root / "VCTK-Corpus" / "speaker-info.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "VCTK speaker-info.txt was not found. Hugging Face "
        "CSTR-Edinburgh/vctk can no longer be loaded (dataset scripts "
        f"are unsupported). Place VCTK 0.92 under {vctk_root} "
        f"(official zip: {VCTK_OFFICIAL_ZIP_URL})."
    )


def parse_vctk_speaker_info(info_path: Path) -> dict[str, dict[str, Any]]:
    """Parse official VCTK speaker metadata."""

    speakers: dict[str, dict[str, Any]] = {}
    for raw_line in info_path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if not fields or fields[0] == "ID":
            continue
        speaker_id = fields[0]
        if speaker_id in VCTK_SPEAKERS_WITHOUT_TEXT:
            continue
        speakers[speaker_id] = {
            "speaker_id": speaker_id,
            "age": fields[1],
            "gender": normalize_vctk_gender(fields[2]),
            "accent": fields[3] if len(fields) > 3 else "",
            "region": " ".join(fields[4:]) if len(fields) > 4 else "",
        }
    return speakers


def assign_vctk_reference_splits(
    speakers: dict[str, dict[str, Any]],
    seed: int,
) -> dict[str, str]:
    """Assign eval / construction-dev / train using a gender-balanced eval pool."""

    rng = random.Random(seed)
    by_gender: dict[str, list[str]] = {"female": [], "male": []}
    for speaker_id, record in speakers.items():
        by_gender[record["gender"]].append(speaker_id)

    eval_ids: list[str] = []
    remaining_ids: list[str] = []
    for gender in ("female", "male"):
        pool = list(by_gender[gender])
        rng.shuffle(pool)
        if len(pool) < R_EVAL_SPEAKERS_PER_GENDER:
            raise ValueError(f"Not enough {gender} VCTK speakers for the eval pool.")
        eval_ids.extend(pool[:R_EVAL_SPEAKERS_PER_GENDER])
        remaining_ids.extend(pool[R_EVAL_SPEAKERS_PER_GENDER:])

    rng.shuffle(remaining_ids)
    construction_dev_count = max(1, round(len(remaining_ids) * R_CONSTRUCTION_DEV_FRACTION))
    construction_dev_ids = remaining_ids[:construction_dev_count]
    train_ids = remaining_ids[construction_dev_count:]

    speaker_to_split = {speaker_id: "eval" for speaker_id in eval_ids}
    speaker_to_split.update(
        {speaker_id: "construction-dev" for speaker_id in construction_dev_ids}
    )
    speaker_to_split.update({speaker_id: "train" for speaker_id in train_ids})
    return speaker_to_split


def build_vctk_split_report(
    speakers: dict[str, dict[str, Any]],
    speaker_to_split: dict[str, str],
    info_path: Path,
) -> dict[str, Any]:
    """Summarize the frozen R-layer reference-speaker split."""

    split_ids: dict[str, list[str]] = {
        "eval": [],
        "construction-dev": [],
        "train": [],
    }
    for speaker_id, split_name in sorted(speaker_to_split.items()):
        split_ids[split_name].append(speaker_id)

    eval_gender_counts = {"female": 0, "male": 0}
    for speaker_id in split_ids["eval"]:
        eval_gender_counts[speakers[speaker_id]["gender"]] += 1

    return {
        "source": "VCTK-Corpus-0.92",
        "speaker_info_path": str(info_path),
        "seed": R_REFERENCE_SPLIT_SEED,
        "eval_speakers_per_gender": R_EVAL_SPEAKERS_PER_GENDER,
        "construction_dev_fraction": R_CONSTRUCTION_DEV_FRACTION,
        "excluded_speakers": sorted(VCTK_SPEAKERS_WITHOUT_TEXT),
        "speaker_count": len(speakers),
        "eval_gender_counts": eval_gender_counts,
        "split_counts": {name: len(ids) for name, ids in split_ids.items()},
        "eval_speakers": split_ids["eval"],
        "construction_dev_speakers": split_ids["construction-dev"],
        "train_speakers": split_ids["train"],
        "speakers": {
            speaker_id: {**record, "split": speaker_to_split[speaker_id]}
            for speaker_id, record in sorted(speakers.items())
        },
    }


def prepare_r_vctk_reference_speakers() -> None:
    """Freeze the R-layer reference-speaker split from a local VCTK extract."""

    local_paths = load_local_paths()
    vctk_root = resolve_from_repo_root(local_paths["sources"]["vctk_root"])
    info_path = find_vctk_speaker_info_file(vctk_root)
    speakers = parse_vctk_speaker_info(info_path)
    speaker_to_split = assign_vctk_reference_splits(speakers, R_REFERENCE_SPLIT_SEED)
    report = build_vctk_split_report(speakers, speaker_to_split, info_path)
    write_report(
        report,
        resolve_from_repo_root("Data/cards/r_vctk_reference_split.json"),
    )
