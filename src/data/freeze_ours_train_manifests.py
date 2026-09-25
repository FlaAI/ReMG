"""Freeze Ours train / construction-dev manifests for Qwen2.5-Omni-7B.

"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report
from realmg.data.r_quadruplets import R_QUADRUPLETS_PATH, R_UTTERANCES_PATH
from realmg.eval.p_harness import (
    build_content_example,
    build_paralinguistic_example,
    utterance_manifest_path,
)
from realmg.eval.p_paper_eval import (
    select_paper_quadruplets,
    utterance_ids_in_quadruplets,
)
from realmg.eval.r_harness import (
    build_r_content_example,
    build_r_paralinguistic_example,
)


P_SPEAKER_SPLIT_PATH = "Data/manifests/p_unseen_speaker_split.json"
P_CONTENT_SPLIT_PATH = "Data/manifests/p_unseen_content_split.json"
P_LEX_QUAD_PATHS = (
    "Data/manifests/p_quadruplets_unseen_speaker_lex_filtered.jsonl",
    "Data/manifests/p_quadruplets_unseen_content_lex_filtered.jsonl",
)
P_UTTERANCE_LOOKUP_PATH = (
    "Data/manifests/p_utterances_unseen_speaker_lex_filtered.jsonl"
)

P_INTERSECTION_TRAIN_PATH = (
    "Data/manifests/p_intersection_train_utterances.jsonl"
)
P_CONSTRUCTION_DEV_UTT_PATH = (
    "Data/manifests/p_construction_dev_utterances.jsonl"
)
P_CONSTRUCTION_DEV_EXAMPLES_PATH = (
    "Data/manifests/p_construction_dev_examples.jsonl"
)
R_CONSTRUCTION_DEV_EXAMPLES_PATH = (
    "Data/manifests/r_construction_dev_examples.jsonl"
)
FREEZE_CARD_PATH = "Data/cards/ours_train_construction_dev_freeze.json"

EXPECTED_P_TRAIN = 2164
EXPECTED_P_DEV = 670


def load_json(path: Path) -> dict[str, Any]:
    """Load one UTF-8 JSON object."""

    return json.loads(path.read_text(encoding="utf-8"))


def parse_p_utterance_id(utterance_id: str) -> tuple[str, int, str]:
    """Parse speaker, slot, and emotion from a P utterance id."""

    parts = utterance_id.split("-")
    return parts[2], int(parts[3]), parts[4]


def load_p_membership() -> tuple[set[str], set[str], set[int], set[int]]:
    """Return train/dev speakers and train/eval/dev slots from frozen splits."""

    speaker_split = load_json(resolve_from_repo_root(P_SPEAKER_SPLIT_PATH))
    content_split = load_json(resolve_from_repo_root(P_CONTENT_SPLIT_PATH))
    train_speakers = set(speaker_split["train_speakers"])
    dev_speakers = set(speaker_split["construction_dev_speakers"])
    eval_slots = set(content_split["eval_slots"])
    dev_slots = set(content_split["construction_dev_slots"])
    train_slots = set(range(1, 351)) - eval_slots - dev_slots
    return train_speakers, dev_speakers, train_slots, eval_slots | dev_slots


def collect_lex_quadruplet_utterance_ids() -> set[str]:
    """Union utterance ids covered by both lex-filtered P protocols."""

    utterance_ids: set[str] = set()
    for relative_path in P_LEX_QUAD_PATHS:
        for row in read_jsonl(resolve_from_repo_root(relative_path)):
            utterance_ids.update(row["utterance_ids"])
    return utterance_ids


def classify_p_utterance_id(
    utterance_id: str,
    *,
    train_speakers: set[str],
    train_slots: set[int],
    eval_slots: set[int],
    eval_speakers: set[str],
) -> str:
    """Map one P utterance to train / construction-dev / eval / other."""

    speaker, slot, _emotion = parse_p_utterance_id(utterance_id)
    if speaker in eval_speakers or slot in eval_slots:
        return "eval"
    if speaker in train_speakers and slot in train_slots:
        return "train"
    return "construction-dev"


def freeze_p_train_and_dev_utterances() -> dict[str, Any]:
    """Write frozen P intersection train and construction-dev utterance lists."""

    train_speakers, _dev_speakers, train_slots, held_out_slots = load_p_membership()
    speaker_split = load_json(resolve_from_repo_root(P_SPEAKER_SPLIT_PATH))
    eval_speakers = set(speaker_split["eval_speakers"])
    eval_slots = set(
        load_json(resolve_from_repo_root(P_CONTENT_SPLIT_PATH))["eval_slots"]
    )
    utterance_lookup = {
        row["utterance_id"]: row
        for row in read_jsonl(resolve_from_repo_root(P_UTTERANCE_LOOKUP_PATH))
    }
    paper_ids = {
        row["utterance_id"]
        for row in read_jsonl(
            resolve_from_repo_root("Data/manifests/p_eval_examples_paper.jsonl")
        )
    }

    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for utterance_id in sorted(collect_lex_quadruplet_utterance_ids()):
        bucket = classify_p_utterance_id(
            utterance_id,
            train_speakers=train_speakers,
            train_slots=train_slots,
            eval_slots=eval_slots,
            eval_speakers=eval_speakers,
        )
        if bucket not in {"train", "construction-dev"}:
            continue
        if utterance_id not in utterance_lookup:
            raise KeyError(f"Missing utterance row for {utterance_id}")
        row = dict(utterance_lookup[utterance_id])
        row["split"] = bucket
        row["metadata"] = {
            **row.get("metadata", {}),
            "membership": "intersection_train"
            if bucket == "train"
            else "construction_dev_residual",
            "source_pool": "lex_filtered_quadruplet_union",
        }
        if bucket == "train":
            train_rows.append(row)
        else:
            dev_rows.append(row)

    if len(train_rows) != EXPECTED_P_TRAIN:
        raise ValueError(
            f"Expected {EXPECTED_P_TRAIN} P train utterances, got {len(train_rows)}"
        )
    if len(dev_rows) != EXPECTED_P_DEV:
        raise ValueError(
            f"Expected {EXPECTED_P_DEV} P construction-dev utterances, got {len(dev_rows)}"
        )

    train_ids = {row["utterance_id"] for row in train_rows}
    dev_ids = {row["utterance_id"] for row in dev_rows}
    if train_ids & paper_ids:
        raise ValueError("P train overlaps paper eval examples.")
    if dev_ids & paper_ids:
        raise ValueError("P construction-dev overlaps paper eval examples.")
    if train_ids & dev_ids:
        raise ValueError("P train overlaps construction-dev.")

    root = resolve_from_repo_root(".")
    for row in train_rows + dev_rows:
        if not (root / row["audio_path"]).exists():
            raise FileNotFoundError(row["audio_path"])

    write_jsonl(train_rows, resolve_from_repo_root(P_INTERSECTION_TRAIN_PATH))
    write_jsonl(dev_rows, resolve_from_repo_root(P_CONSTRUCTION_DEV_UTT_PATH))
    return {
        "p_train_count": len(train_rows),
        "p_construction_dev_count": len(dev_rows),
        "p_train_emotion_counts": dict(
            Counter(row["paralinguistic_label"] for row in train_rows)
        ),
        "p_construction_dev_emotion_counts": dict(
            Counter(row["paralinguistic_label"] for row in dev_rows)
        ),
        "p_paper_overlap_train": 0,
        "p_paper_overlap_construction_dev": 0,
        "held_out_slot_count_for_membership": len(held_out_slots),
    }


def build_r_construction_dev_examples() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand R mass-carved construction-dev quadruplets into scoreable queries."""

    quadruplets = [
        row
        for row in read_jsonl(resolve_from_repo_root(R_QUADRUPLETS_PATH))
        if row["split"] == "construction-dev"
    ]
    utterances = {
        row["utterance_id"]: row
        for row in read_jsonl(resolve_from_repo_root(R_UTTERANCES_PATH))
    }
    examples: list[dict[str, Any]] = []
    speakers: set[str] = set()
    for quadruplet in quadruplets:
        for utterance_id in quadruplet["utterance_ids"]:
            utterance = utterances[utterance_id]
            speakers.add(str(utterance["speaker_id"]))
            content = build_r_content_example(quadruplet, utterance)
            para = build_r_paralinguistic_example(quadruplet, utterance)
            content["split"] = "construction-dev"
            para["split"] = "construction-dev"
            examples.append(content)
            examples.append(para)

    root = resolve_from_repo_root(".")
    for example in examples:
        if not (root / example["audio_path"]).exists():
            raise FileNotFoundError(example["audio_path"])

    registry = {
        "layer": "R",
        "split": "construction-dev",
        "quadruplet_count": len(quadruplets),
        "audio_count": len(quadruplets) * 4,
        "query_count": len(examples),
        "content_query_count": sum(
            1 for example in examples if example["query_type"] == "content"
        ),
        "paralinguistic_query_count": sum(
            1 for example in examples if example["query_type"] == "paralinguistic"
        ),
        "speaker_count": len(speakers),
        "purpose": "ours_checkpoint_selection_primary_r_acc_audio",
    }
    return examples, registry


def load_p_construction_dev_quadruplets(
    protocol: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load construction-dev lex-filtered quadruplets for one protocol."""

    path = resolve_from_repo_root(
        f"Data/manifests/p_quadruplets_{protocol}_lex_filtered.jsonl"
    )
    quadruplets = [row for row in read_jsonl(path) if row["split"] == "construction-dev"]
    utterances = {
        row["utterance_id"]: row
        for row in read_jsonl(utterance_manifest_path(protocol))
    }
    return quadruplets, utterances


def build_p_construction_dev_examples() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Expand P construction-dev quads with speaker-first conflict drop."""

    speaker_quadruplets, speaker_utterances = load_p_construction_dev_quadruplets(
        "unseen_speaker"
    )
    content_quadruplets, content_utterances = load_p_construction_dev_quadruplets(
        "unseen_content"
    )
    kept, dropped_content = select_paper_quadruplets(
        speaker_quadruplets, content_quadruplets
    )
    utterances = {**speaker_utterances, **content_utterances}
    examples: list[dict[str, Any]] = []
    for quadruplet in kept:
        protocol = str(quadruplet["metadata"]["protocol"])
        option_map = {
            quadruplet["content_pair"][0]: "A",
            quadruplet["content_pair"][1]: "B",
        }
        for utterance_id in quadruplet["utterance_ids"]:
            utterance = utterances[utterance_id]
            content = build_content_example(
                protocol, quadruplet, utterance, option_map
            )
            para = build_paralinguistic_example(protocol, quadruplet, utterance)
            content["split"] = "construction-dev"
            para["split"] = "construction-dev"
            content["protocol"] = "construction_dev"
            para["protocol"] = "construction_dev"
            examples.append(content)
            examples.append(para)

    root = resolve_from_repo_root(".")
    for example in examples:
        if not (root / example["audio_path"]).exists():
            raise FileNotFoundError(example["audio_path"])

    registry = {
        "layer": "P",
        "split": "construction-dev",
        "conflict_rule": "keep_unseen_speaker_then_nonoverlapping_unseen_content",
        "source_construction_dev_quadruplets": {
            "unseen_speaker": len(speaker_quadruplets),
            "unseen_content": len(content_quadruplets),
        },
        "dropped_unseen_content_quadruplet_count": len(dropped_content),
        "quadruplet_count": len(kept),
        "audio_count": len(utterance_ids_in_quadruplets(kept)),
        "query_count": len(examples),
        "content_query_count": sum(
            1 for example in examples if example["query_type"] == "content"
        ),
        "paralinguistic_query_count": sum(
            1 for example in examples if example["query_type"] == "paralinguistic"
        ),
        "purpose": "ours_checkpoint_selection_secondary_p_para_score",
    }
    return examples, registry


def prepare_ours_train_construction_dev_manifests() -> dict[str, Any]:
    """Freeze P train/dev utterances and R/P construction-dev score examples."""

    p_utt_summary = freeze_p_train_and_dev_utterances()
    r_examples, r_registry = build_r_construction_dev_examples()
    p_examples, p_registry = build_p_construction_dev_examples()
    write_jsonl(r_examples, resolve_from_repo_root(R_CONSTRUCTION_DEV_EXAMPLES_PATH))
    write_jsonl(p_examples, resolve_from_repo_root(P_CONSTRUCTION_DEV_EXAMPLES_PATH))
    card = {
        "stage": "ours_train_construction_dev_freeze",
        "backbone_note": "shared across backbones; first use Qwen2.5-Omni-7B",
        **p_utt_summary,
        "r_construction_dev_examples": r_registry,
        "p_construction_dev_examples": p_registry,
        "paths": {
            "p_intersection_train": P_INTERSECTION_TRAIN_PATH,
            "p_construction_dev_utterances": P_CONSTRUCTION_DEV_UTT_PATH,
            "r_construction_dev_examples": R_CONSTRUCTION_DEV_EXAMPLES_PATH,
            "p_construction_dev_examples": P_CONSTRUCTION_DEV_EXAMPLES_PATH,
        },
    }
    write_report(card, resolve_from_repo_root(FREEZE_CARD_PATH))
    return card
