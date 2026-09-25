"""Score a finished API eval run when every frozen example has a prediction.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from realmg.data.esd import read_jsonl
from realmg.data.r_text_teacher import load_prediction_index


def scored_predictions_path(predictions_path: Path) -> Path:
    """Return the *_scored.jsonl sibling of a predictions file."""

    return predictions_path.with_name(predictions_path.stem + "_scored.jsonl")


def scored_file_covers_examples(
    scored_path: Path,
    examples: list[dict[str, Any]],
) -> bool:
    """True when a scored JSONL already has every frozen example_id."""

    if not scored_path.is_file():
        return False
    scored_ids = {str(row["example_id"]) for row in read_jsonl(scored_path)}
    return all(str(example["example_id"]) in scored_ids for example in examples)


def missing_prediction_ids(
    examples: list[dict[str, Any]],
    predictions_path: Path,
) -> list[str]:
    """Return example_ids that do not yet have a prediction row."""

    if not predictions_path.is_file():
        return [str(example["example_id"]) for example in examples]
    finished = load_prediction_index(predictions_path)
    return [
        str(example["example_id"])
        for example in examples
        if str(example["example_id"]) not in finished
    ]


def _skip_or_missing(
    *,
    examples: list[dict[str, Any]],
    predictions_path: Path,
    aborted: bool,
    log_prefix: str,
) -> list[str] | None:
    """Return missing ids, or None when scoring should proceed."""

    if aborted:
        print(f"[{log_prefix}] skip scoring: run aborted", flush=True)
        return []
    missing = missing_prediction_ids(examples, predictions_path)
    if missing:
        print(
            f"[{log_prefix}] skip scoring: {len(missing)} predictions missing",
            flush=True,
        )
        return missing
    return None


def score_p_eval_if_complete(
    *,
    examples: list[dict[str, Any]],
    predictions_path: Path,
    aborted: bool,
    log_prefix: str,
) -> None:
    """Score P predictions (paper locked catalog included) when the run is full."""

    if _skip_or_missing(
        examples=examples,
        predictions_path=predictions_path,
        aborted=aborted,
        log_prefix=log_prefix,
    ) is not None:
        return
    from realmg.eval.p_harness import score_p_eval_predictions

    score_p_eval_predictions(str(predictions_path))


def score_p_teacher_ceiling_if_complete(
    *,
    examples: list[dict[str, Any]],
    predictions_path: Path,
    aborted: bool,
    log_prefix: str,
) -> None:
    """Score Teacher q_para ceiling predictions when the run is full."""

    if _skip_or_missing(
        examples=examples,
        predictions_path=predictions_path,
        aborted=aborted,
        log_prefix=log_prefix,
    ) is not None:
        return
    from realmg.eval.p_harness import score_p_teacher_ceiling_predictions

    score_p_teacher_ceiling_predictions(str(predictions_path))


def score_r_eval_if_complete(
    *,
    examples: list[dict[str, Any]],
    predictions_path: Path,
    aborted: bool,
    log_prefix: str,
) -> None:
    """Score R audio predictions and write locked R metrics if text scores exist."""

    if _skip_or_missing(
        examples=examples,
        predictions_path=predictions_path,
        aborted=aborted,
        log_prefix=log_prefix,
    ) is not None:
        return
    scored_path = scored_predictions_path(predictions_path)
    if scored_file_covers_examples(scored_path, examples):
        from realmg.eval.score_locked_eval import (
            maybe_write_r_locked_report_for_slug,
            slug_from_r_audio_path,
        )

        print(f"[{log_prefix}] reuse existing scored file", flush=True)
        maybe_write_r_locked_report_for_slug(slug_from_r_audio_path(predictions_path))
        return
    from realmg.eval.r_harness import score_r_eval_predictions

    score_r_eval_predictions(str(predictions_path))


def score_r_eval_text_if_complete(
    *,
    examples: list[dict[str, Any]],
    predictions_path: Path,
    aborted: bool,
    log_prefix: str,
) -> None:
    """Score R text predictions and write locked R metrics if audio scores exist."""

    if _skip_or_missing(
        examples=examples,
        predictions_path=predictions_path,
        aborted=aborted,
        log_prefix=log_prefix,
    ) is not None:
        return
    scored_path = scored_predictions_path(predictions_path)
    if scored_file_covers_examples(scored_path, examples):
        from realmg.eval.score_locked_eval import (
            maybe_write_r_locked_report_for_slug,
            slug_from_r_text_path,
        )

        print(f"[{log_prefix}] reuse existing scored file", flush=True)
        maybe_write_r_locked_report_for_slug(slug_from_r_text_path(predictions_path))
        return
    from realmg.eval.r_eval_text import score_r_eval_text_predictions

    score_r_eval_text_predictions(str(predictions_path))
