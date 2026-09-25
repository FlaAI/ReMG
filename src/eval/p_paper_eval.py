"""Paper P eval table: conflict-free union of construction eval quadruplets.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report
from realmg.eval.p_harness import (
    P_PAPER_PROTOCOL,
    eval_examples_path,
    eval_registry_path,
    load_eval_protocol_data,
    score_p_eval_predictions,
    score_p_teacher_ceiling_predictions,
    teacher_ceiling_examples_path,
    teacher_ceiling_registry_path,
)


P_PAPER_EXPECTED_QUADRUPLETS = 220
P_PAPER_EXPECTED_AUDIOS = 880
P_PAPER_CONFLICT_RULE = "keep_unseen_speaker_then_nonoverlapping_unseen_content"

P_PAPER_PREDICTION_SLUGS = (
    "qwen25_omni_7b_bailian_base",
    "qwen3_omni_flash_bailian_base",
    "qwen35_omni_flash_bailian_base",
    "gemini_3_7_flash_poloapi_base",
    "gemini_3_1_pro_preview_poloapi_base",
    "gpt_audio_2025_08_28_ohmygpt_base",
)
P_PAPER_TEACHER_CEILING_SLUGS = ("qwen25_omni_7b_bailian_base",)


def paper_quadruplet_manifest_path() -> Path:
    """Return the frozen paper-eval quadruplet JSONL path."""

    return resolve_from_repo_root("Data/manifests/p_quadruplets_paper_eval.jsonl")


def paper_eval_card_path() -> Path:
    """Return the construction card for the paper P table."""

    return resolve_from_repo_root("Data/cards/p_eval_paper.json")


def load_construction_eval_quadruplets(protocol: str) -> list[dict[str, Any]]:
    """Load eval-split quadruplets for one construction protocol."""

    quadruplets, _utterances = load_eval_protocol_data(protocol)
    return quadruplets


def utterance_ids_in_quadruplets(quadruplets: list[dict[str, Any]]) -> set[str]:
    """Collect utterance ids covered by a quadruplet list."""

    utterance_ids: set[str] = set()
    for quadruplet in quadruplets:
        utterance_ids.update(quadruplet["utterance_ids"])
    return utterance_ids


def select_paper_quadruplets(
    speaker_quadruplets: list[dict[str, Any]],
    content_quadruplets: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep all speaker eval quads, then non-overlapping content eval quads."""

    speaker_utterance_ids = utterance_ids_in_quadruplets(speaker_quadruplets)
    kept = list(speaker_quadruplets)
    dropped_content: list[dict[str, Any]] = []
    for quadruplet in content_quadruplets:
        if set(quadruplet["utterance_ids"]) & speaker_utterance_ids:
            dropped_content.append(quadruplet)
        else:
            kept.append(quadruplet)
    return kept, dropped_content


def validate_paper_quadruplets(kept: list[dict[str, Any]]) -> set[str]:
    """Fail if the freeze counts or unique-audio constraint do not hold."""

    utterance_ids = utterance_ids_in_quadruplets(kept)
    if len(kept) != P_PAPER_EXPECTED_QUADRUPLETS:
        raise ValueError(
            f"Expected {P_PAPER_EXPECTED_QUADRUPLETS} paper quadruplets, got {len(kept)}"
        )
    if len(utterance_ids) != P_PAPER_EXPECTED_AUDIOS:
        raise ValueError(
            f"Expected {P_PAPER_EXPECTED_AUDIOS} unique audios, got {len(utterance_ids)}"
        )
    if len(utterance_ids) != 4 * len(kept):
        raise ValueError("Paper quadruplets are not audio-disjoint.")
    return utterance_ids


def select_examples_for_quadruplets(
    examples: list[dict[str, Any]],
    kept_quadruplet_ids: set[str],
) -> list[dict[str, Any]]:
    """Keep construction examples whose quadruplet survived conflict drop."""

    return [
        example
        for example in examples
        if example["quadruplet_id"] in kept_quadruplet_ids
    ]


def build_paper_eval_registry(
    kept_quadruplets: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    dropped_content: list[dict[str, Any]],
    speaker_eval_count: int,
    content_eval_count: int,
) -> dict[str, Any]:
    """Summarize the paper P table for the data card and eval registry."""

    unique_speakers = {str(quad["metadata"]["speaker_id"]) for quad in kept_quadruplets}
    unique_slots: set[int] = set()
    for quadruplet in kept_quadruplets:
        unique_slots.update(int(slot) for slot in quadruplet["metadata"]["utt_slots"])
    kept_content_count = sum(
        1
        for quadruplet in kept_quadruplets
        if quadruplet["metadata"]["protocol"] == "unseen_content"
    )
    kept_utterance_ids = utterance_ids_in_quadruplets(kept_quadruplets)
    orphan_ids = sorted(
        utterance_ids_in_quadruplets(dropped_content) - kept_utterance_ids
    )
    return {
        "protocol": P_PAPER_PROTOCOL,
        "split": "eval",
        "purpose": "paper_p_main_table",
        "conflict_rule": P_PAPER_CONFLICT_RULE,
        "source_eval_quadruplets": {
            "unseen_speaker": speaker_eval_count,
            "unseen_content": content_eval_count,
        },
        "dropped_unseen_content_quadruplet_count": len(dropped_content),
        "dropped_unseen_content_quadruplet_ids": [
            quadruplet["quadruplet_id"] for quadruplet in dropped_content
        ],
        "orphan_audio_count": len(orphan_ids),
        "orphan_utterance_ids": orphan_ids,
        "quadruplet_count": len(kept_quadruplets),
        "audio_count": len(kept_utterance_ids),
        "query_count": len(examples),
        "content_query_count": sum(
            1 for example in examples if example["query_type"] == "content"
        ),
        "paralinguistic_query_count": sum(
            1 for example in examples if example["query_type"] == "paralinguistic"
        ),
        "speaker_count": len(unique_speakers),
        "utterance_slot_count": len(unique_slots),
        "kept_from_unseen_speaker": speaker_eval_count,
        "kept_from_unseen_content": kept_content_count,
        "example_ids_preserved": True,
        "naive_utterance_union": 894,
    }


def write_paper_eval_table() -> dict[str, Any]:
    """Freeze paper quadruplets, eval examples, and teacher-ceiling examples."""

    speaker_quadruplets = load_construction_eval_quadruplets("unseen_speaker")
    content_quadruplets = load_construction_eval_quadruplets("unseen_content")
    kept, dropped_content = select_paper_quadruplets(
        speaker_quadruplets, content_quadruplets
    )
    validate_paper_quadruplets(kept)
    kept_quadruplet_ids = {quadruplet["quadruplet_id"] for quadruplet in kept}

    speaker_examples = read_jsonl(eval_examples_path("unseen_speaker"))
    content_examples = read_jsonl(eval_examples_path("unseen_content"))
    paper_examples = select_examples_for_quadruplets(
        speaker_examples, kept_quadruplet_ids
    ) + select_examples_for_quadruplets(content_examples, kept_quadruplet_ids)
    expected_queries = P_PAPER_EXPECTED_AUDIOS * 2
    if len(paper_examples) != expected_queries:
        raise ValueError(
            f"Expected {expected_queries} paper queries, got {len(paper_examples)}"
        )

    registry = build_paper_eval_registry(
        kept,
        paper_examples,
        dropped_content,
        speaker_eval_count=len(speaker_quadruplets),
        content_eval_count=len(content_quadruplets),
    )

    write_jsonl(kept, paper_quadruplet_manifest_path())
    write_jsonl(paper_examples, eval_examples_path(P_PAPER_PROTOCOL))
    write_report(registry, eval_registry_path(P_PAPER_PROTOCOL))
    write_report(registry, paper_eval_card_path())

    ceiling_examples = [
        example
        for example in paper_examples
        if example["query_type"] == "paralinguistic"
    ]
    ceiling_registry = {
        "layer": "P",
        "protocol": P_PAPER_PROTOCOL,
        "split": "eval",
        "purpose": "teacher_q_para_ceiling",
        "query_type": "paralinguistic",
        "query_count": len(ceiling_examples),
        "audio_count": len(ceiling_examples),
        "quadruplet_count": len(kept),
        "speaker_count": registry["speaker_count"],
        "utterance_slot_count": registry["utterance_slot_count"],
        "example_ids_preserved": True,
        "conflict_rule": P_PAPER_CONFLICT_RULE,
    }
    write_jsonl(ceiling_examples, teacher_ceiling_examples_path(P_PAPER_PROTOCOL))
    write_report(ceiling_registry, teacher_ceiling_registry_path(P_PAPER_PROTOCOL))
    return registry


def construction_prediction_path(kind: str, protocol: str, slug: str) -> Path:
    """Return one construction-protocol prediction JSONL path."""

    filename = f"{kind}_{protocol}_{slug}.jsonl"
    return resolve_from_repo_root(f"Data/manifests/{filename}")


def paper_prediction_path(kind: str, slug: str) -> Path:
    """Return the paper-protocol prediction JSONL path for one slug."""

    filename = f"{kind}_{P_PAPER_PROTOCOL}_{slug}.jsonl"
    return resolve_from_repo_root(f"Data/manifests/{filename}")


def merge_prediction_files(
    kind: str,
    slug: str,
    kept_example_ids: set[str],
) -> Path:
    """Copy kept construction prediction rows into the paper prediction file."""

    speaker_path = construction_prediction_path(kind, "unseen_speaker", slug)
    content_path = construction_prediction_path(kind, "unseen_content", slug)
    if not speaker_path.is_file() or not content_path.is_file():
        raise FileNotFoundError(
            f"Missing construction predictions for {kind} slug={slug}: "
            f"{speaker_path} / {content_path}"
        )
    merged = [
        row
        for row in read_jsonl(speaker_path) + read_jsonl(content_path)
        if row["example_id"] in kept_example_ids
    ]
    merged_ids = [row["example_id"] for row in merged]
    if len(merged_ids) != len(set(merged_ids)):
        raise ValueError(f"Duplicate prediction example_id after merge for {slug}")
    missing = kept_example_ids - set(merged_ids)
    if missing:
        raise ValueError(
            f"{kind} slug={slug} missing {len(missing)} paper example_ids; "
            f"examples={sorted(missing)[:5]}"
        )
    output_path = paper_prediction_path(kind, slug)
    write_jsonl(merged, output_path)
    return output_path


def merge_paper_predictions() -> list[Path]:
    """Subset all frozen Base P predictions onto the paper example set."""

    paper_example_ids = {
        example["example_id"] for example in read_jsonl(eval_examples_path(P_PAPER_PROTOCOL))
    }
    ceiling_example_ids = {
        example["example_id"]
        for example in read_jsonl(teacher_ceiling_examples_path(P_PAPER_PROTOCOL))
    }
    written: list[Path] = []
    for slug in P_PAPER_PREDICTION_SLUGS:
        written.append(
            merge_prediction_files("p_eval_predictions", slug, paper_example_ids)
        )
    for slug in P_PAPER_TEACHER_CEILING_SLUGS:
        written.append(
            merge_prediction_files(
                "p_teacher_ceiling_predictions", slug, ceiling_example_ids
            )
        )
    return written


def score_merged_paper_predictions(prediction_paths: list[Path]) -> None:
    """Score paper prediction files against the frozen paper example set."""

    for path in prediction_paths:
        relative = str(path)
        if "teacher_ceiling" in path.name:
            score_p_teacher_ceiling_predictions(relative)
            print(f"[prepare-p-paper-eval] scored teacher ceiling {path.name}")
        else:
            score_p_eval_predictions(relative)
            print(f"[prepare-p-paper-eval] scored {path.name}")


def prepare_p_paper_eval(*, merge_predictions: bool = True) -> dict[str, Any]:
    """Build the paper P table and optionally copy existing predictions onto it."""

    registry = write_paper_eval_table()
    if merge_predictions:
        written = merge_paper_predictions()
        registry["merged_prediction_files"] = [str(path) for path in written]
        write_report(registry, paper_eval_card_path())
        write_report(registry, eval_registry_path(P_PAPER_PROTOCOL))
        score_merged_paper_predictions(written)
    print(
        "[prepare-p-paper-eval] "
        f"quads={registry['quadruplet_count']} audios={registry['audio_count']} "
        f"queries={registry['query_count']} orphans={registry['orphan_audio_count']}"
    )
    return registry
