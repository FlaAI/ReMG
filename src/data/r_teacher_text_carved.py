"""Teacher-text screening prep for carved train (per-backbone).

"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report
from realmg.data.r_quadruplets import R_UTTERANCES_PATH
from realmg.data.r_tts_mass_carve import CARVE_UNITS_PATH
from realmg.data.r_text_teacher import (
    build_teacher_example,
    filter_r_teacher_text_predictions,
    is_successful_prediction_row,
    load_prediction_index,
    run_r_teacher_text_api,
    teacher_predictions_path,
)


BACKBONE_SLUG = "qwen25_omni_7b"
BACKBONE_LABEL = "Qwen2.5-Omni-7B"
TEACHER_BACKEND = "bailian_openai_compatible"

CARVED_TRAIN_EXAMPLES_PATH = (
    "Data/manifests/r_teacher_text_carved_train_examples.jsonl"
)
CARVED_TRAIN_PREDICTIONS_PATH = (
    f"Data/manifests/r_teacher_text_carved_train_predictions_{BACKBONE_SLUG}.jsonl"
)
CARVED_TRAIN_FAILURES_PATH = (
    f"Data/manifests/r_teacher_text_carved_train_failures_{BACKBONE_SLUG}.jsonl"
)
CARVED_TRAIN_CORRECT_PATH = (
    f"Data/manifests/r_carved_train_teacher_correct_{BACKBONE_SLUG}.jsonl"
)
CARVED_TRAIN_UNITS_CORRECT_PATH = (
    f"Data/manifests/r_carved_train_units_teacher_correct_{BACKBONE_SLUG}.jsonl"
)
CARVED_TRAIN_UTTERANCES_CORRECT_PATH = (
    f"Data/manifests/r_carved_train_utterances_teacher_correct_{BACKBONE_SLUG}.jsonl"
)
CARVED_TRAIN_REPORT_PATH = (
    f"Data/cards/r_teacher_text_carved_train_{BACKBONE_SLUG}_report.json"
)
CARVED_TRAIN_RUN_REPORT_PATH = (
    f"Data/cards/r_teacher_text_carved_train_{BACKBONE_SLUG}_run_report.json"
)

BACKBONE_SPECS = {
    "qwen25_omni_7b": {
        "label": "Qwen2.5-Omni-7B",
        "teacher_backend": "bailian_openai_compatible",
    },
    "phi4_mm": {
        "label": "Phi-4-MM",
        "teacher_backend": "local_phi4_mm",
    },
    "qwen2_audio_7b": {
        "label": "Qwen2-Audio-7B-Instruct",
        "teacher_backend": "local_qwen2_audio",
    },
}


def _refresh_backbone_paths() -> None:
    """Rebuild namespaced carved-train paths after a backbone switch."""

    global CARVED_TRAIN_PREDICTIONS_PATH, CARVED_TRAIN_FAILURES_PATH
    global CARVED_TRAIN_CORRECT_PATH, CARVED_TRAIN_UNITS_CORRECT_PATH
    global CARVED_TRAIN_UTTERANCES_CORRECT_PATH, CARVED_TRAIN_REPORT_PATH
    global CARVED_TRAIN_RUN_REPORT_PATH
    CARVED_TRAIN_PREDICTIONS_PATH = (
        f"Data/manifests/r_teacher_text_carved_train_predictions_{BACKBONE_SLUG}.jsonl"
    )
    CARVED_TRAIN_FAILURES_PATH = (
        f"Data/manifests/r_teacher_text_carved_train_failures_{BACKBONE_SLUG}.jsonl"
    )
    CARVED_TRAIN_CORRECT_PATH = (
        f"Data/manifests/r_carved_train_teacher_correct_{BACKBONE_SLUG}.jsonl"
    )
    CARVED_TRAIN_UNITS_CORRECT_PATH = (
        f"Data/manifests/r_carved_train_units_teacher_correct_{BACKBONE_SLUG}.jsonl"
    )
    CARVED_TRAIN_UTTERANCES_CORRECT_PATH = (
        f"Data/manifests/r_carved_train_utterances_teacher_correct_{BACKBONE_SLUG}.jsonl"
    )
    CARVED_TRAIN_REPORT_PATH = (
        f"Data/cards/r_teacher_text_carved_train_{BACKBONE_SLUG}_report.json"
    )
    CARVED_TRAIN_RUN_REPORT_PATH = (
        f"Data/cards/r_teacher_text_carved_train_{BACKBONE_SLUG}_run_report.json"
    )


def set_carved_teacher_backbone(slug: str) -> None:
    """Switch Teacher-text carved-train artifacts to one student backbone."""

    global BACKBONE_SLUG, BACKBONE_LABEL, TEACHER_BACKEND
    if slug not in BACKBONE_SPECS:
        raise ValueError(
            f"Unknown carved-train backbone {slug}. "
            f"Known: {sorted(BACKBONE_SPECS)}"
        )
    spec = BACKBONE_SPECS[slug]
    BACKBONE_SLUG = slug
    BACKBONE_LABEL = spec["label"]
    TEACHER_BACKEND = spec["teacher_backend"]
    _refresh_backbone_paths()


def carved_train_examples_path() -> Path:
    """Return the carved-train Teacher-text export path."""

    return resolve_from_repo_root(CARVED_TRAIN_EXAMPLES_PATH)


def carved_train_predictions_path() -> Path:
    """Return the backbone-specific carved-train predictions path."""

    return resolve_from_repo_root(CARVED_TRAIN_PREDICTIONS_PATH)


def carved_train_failures_path() -> Path:
    """Return the backbone-specific carved-train API failures path."""

    return resolve_from_repo_root(CARVED_TRAIN_FAILURES_PATH)


def carved_train_correct_path() -> Path:
    """Return the backbone-specific Teacher-correct content_id manifest."""

    return resolve_from_repo_root(CARVED_TRAIN_CORRECT_PATH)


def load_carved_train_units() -> list[dict[str, Any]]:
    """Load carved units restricted to the train split."""

    return [
        unit
        for unit in read_jsonl(resolve_from_repo_root(CARVE_UNITS_PATH))
        if unit["split"] == "train"
    ]


def build_carved_train_text_record(unit: dict[str, Any]) -> dict[str, Any]:
    """Build a filtered-pool-shaped row from one carved train unit."""

    return {
        "content_id": unit["content_id"],
        "source_name": unit["source_name"],
        "prompt_text": unit["prompt_text"],
        "answer_text": unit["answer_text"],
        "metadata": {
            "mass_text_id": unit["mass_text_id"],
            "unit_id": unit["unit_id"],
            "engine": unit["engine"],
            "speaker_id": unit["speaker_id"],
            "split": unit["split"],
        },
    }


def prepare_r_teacher_text_carved_train() -> None:
    """Export Teacher-text examples for every unique carved-train content_id."""

    train_units = load_carved_train_units()
    by_content: dict[str, dict[str, Any]] = {}
    for unit in train_units:
        by_content.setdefault(unit["content_id"], unit)

    examples = [
        {
            **build_teacher_example(build_carved_train_text_record(unit)),
            "backbone_slug": BACKBONE_SLUG,
            "backbone_label": BACKBONE_LABEL,
            "teacher_backend": TEACHER_BACKEND,
            "scope": "carved_train",
        }
        for unit in sorted(by_content.values(), key=lambda row: row["content_id"])
    ]
    write_jsonl(examples, carved_train_examples_path())

    by_source = Counter(example["source_name"] for example in examples)
    by_verifier = Counter(example["verifier_method"] for example in examples)
    write_report(
        {
            "stage": "teacher_text_carved_train_prepare",
            "backbone_slug": BACKBONE_SLUG,
            "backbone_label": BACKBONE_LABEL,
            "scope": "carved_train_only",
            "train_unit_count": len(train_units),
            "unique_content_count": len(examples),
            "by_source": dict(sorted(by_source.items())),
            "by_verifier_method": dict(sorted(by_verifier.items())),
            "outputs": {
                "examples": CARVED_TRAIN_EXAMPLES_PATH,
                "predictions": CARVED_TRAIN_PREDICTIONS_PATH,
                "correct": CARVED_TRAIN_CORRECT_PATH,
            },
            "notes": [
                "Teacher-text correct is per-backbone; do not reuse for other students.",
            ],
        },
        resolve_from_repo_root(CARVED_TRAIN_REPORT_PATH),
    )
    print(
        f"[prepare-r-teacher-text-carved-train] unique_s={len(examples)} "
        f"backbone={BACKBONE_SLUG}",
        flush=True,
    )


def import_global_predictions_for_carved_train() -> int:
    """Copy successful global prediction rows that hit carved-train content_ids."""

    examples = read_jsonl(carved_train_examples_path())
    target_ids = {example["example_id"] for example in examples}
    global_path = teacher_predictions_path()
    if not global_path.exists():
        return 0

    carved_index = load_prediction_index(carved_train_predictions_path())
    imported = 0
    out_path = carved_train_predictions_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for row in read_jsonl(global_path):
            if not is_successful_prediction_row(row):
                continue
            example_id = row["example_id"]
            if example_id not in target_ids or example_id in carved_index:
                continue
            append_row = {
                **row,
                "imported_from": str(global_path),
                "backbone_slug": BACKBONE_SLUG,
            }
            handle.write(json.dumps(append_row, ensure_ascii=False) + "\n")
            carved_index[example_id] = append_row
            imported += 1
    if imported:
        print(
            f"[import-global-predictions] imported={imported} into {out_path.name}",
            flush=True,
        )
    return imported


def run_r_teacher_text_carved_train(
    *,
    limit: int | None = None,
    resume: bool = True,
    request_interval_seconds: float = 0.2,
    max_retries: int = 3,
    shard_index: int = 0,
    num_shards: int = 1,
) -> None:
    """Query the backbone Teacher-text backend for unqueried carved-train s."""

    if TEACHER_BACKEND == "local_phi4_mm":
        from realmg.train.phi4_teacher_text import run_phi4_teacher_text_carved_train

        run_phi4_teacher_text_carved_train(
            limit=limit,
            resume=resume,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        return
    if TEACHER_BACKEND == "local_qwen2_audio":
        from realmg.train.qwen2_audio_teacher_text import (
            run_qwen2_audio_teacher_text_carved_train,
        )

        if num_shards != 1 or shard_index != 0:
            raise ValueError(
                "Qwen2-Audio Teacher-text sharding is not implemented yet."
            )
        run_qwen2_audio_teacher_text_carved_train(limit=limit, resume=resume)
        return
    import_global_predictions_for_carved_train()
    run_r_teacher_text_api(
        limit=limit,
        resume=resume,
        request_interval_seconds=request_interval_seconds,
        max_retries=max_retries,
        examples_path=carved_train_examples_path(),
        predictions_path=carved_train_predictions_path(),
        failures_path=carved_train_failures_path(),
        run_report_path=resolve_from_repo_root(CARVED_TRAIN_RUN_REPORT_PATH),
        update_sample_plan=False,
    )


def merge_carved_train_teacher_correct(
    kept_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Join Teacher-correct content_ids with carved train units and utterances."""

    kept_ids = {record["content_id"] for record in kept_records}
    train_units = load_carved_train_units()
    kept_units = [unit for unit in train_units if unit["content_id"] in kept_ids]
    kept_unit_ids = {unit["unit_id"] for unit in kept_units}

    utterances = [
        row
        for row in read_jsonl(resolve_from_repo_root(R_UTTERANCES_PATH))
        if row["split"] == "train" and row["metadata"]["unit_id"] in kept_unit_ids
    ]

    for record in kept_records:
        record.setdefault("metadata", {})
        record["metadata"].update(
            {
                "backbone_slug": BACKBONE_SLUG,
                "backbone_label": BACKBONE_LABEL,
                "scope": "carved_train",
            }
        )

    write_jsonl(kept_records, carved_train_correct_path())
    write_jsonl(kept_units, resolve_from_repo_root(CARVED_TRAIN_UNITS_CORRECT_PATH))
    write_jsonl(
        utterances,
        resolve_from_repo_root(CARVED_TRAIN_UTTERANCES_CORRECT_PATH),
    )

    return {
        "unique_s_queried": len(
            load_prediction_index(carved_train_predictions_path())
        ),
        "unique_s_correct": len(kept_records),
        "train_units_before": len(train_units),
        "train_units_after": len(kept_units),
        "train_utterances_after": len(utterances),
        "retention_rate_units": len(kept_units) / len(train_units) if train_units else 0.0,
        "engine_counts_after": dict(
            sorted(Counter(unit["engine"] for unit in kept_units).items())
        ),
        "source_counts_after": dict(
            sorted(Counter(record["source_name"] for record in kept_records).items())
        ),
    }


def filter_r_teacher_text_carved_train(
    *,
    predictions_file: str | None = None,
) -> None:
    """Verify carved-train predictions and write backbone-specific train masks."""

    predictions_path = (
        resolve_from_repo_root(predictions_file)
        if predictions_file
        else carved_train_predictions_path()
    )
    kept_records = filter_r_teacher_text_predictions(
        predictions_file=str(predictions_path),
        examples_path=carved_train_examples_path(),
        output_correct_path=carved_train_correct_path(),
        report_path=resolve_from_repo_root(CARVED_TRAIN_REPORT_PATH),
        merge_existing_correct=False,
    )
    merge_stats = merge_carved_train_teacher_correct(kept_records)

    report_path = resolve_from_repo_root(CARVED_TRAIN_REPORT_PATH)
    base_report = json.loads(report_path.read_text(encoding="utf-8"))
    write_report(
        {
            **base_report,
            "stage": "teacher_text_carved_train_filter",
            "merge": merge_stats,
            "outputs": {
                "teacher_correct": CARVED_TRAIN_CORRECT_PATH,
                "train_units": CARVED_TRAIN_UNITS_CORRECT_PATH,
                "train_utterances": CARVED_TRAIN_UTTERANCES_CORRECT_PATH,
                "predictions": CARVED_TRAIN_PREDICTIONS_PATH,
            },
            "notes": [
                "Do not reuse this kept set for a different student backbone.",
                "Eval/dev carved pool is unchanged.",
            ],
        },
        report_path,
    )
    print(
        f"[filter-r-teacher-text-carved-train] kept_s={merge_stats['unique_s_correct']} "
        f"train_units={merge_stats['train_units_after']}/"
        f"{merge_stats['train_units_before']}",
        flush=True,
    )
