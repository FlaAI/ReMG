"""Teacher-text screening for the R-layer text pool.

"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI

from realmg.data.bailian_api import (
    build_bailian_client,
    bailian_model_name,
    default_bailian_request_options,
)
from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report
from realmg.data.r_text_filter import FILTERED_MANIFEST_PATH
# Do not import r_verifiers at module top: it pulls torch/transformers and breaks
# prediction-only local-server clients. Import inside verifier-using functions.


FILTERED_INPUT_PATH = FILTERED_MANIFEST_PATH
TEACHER_EXAMPLES_PATH = "Data/manifests/r_teacher_text_examples.jsonl"
TEACHER_PREDICTIONS_PATH = "Data/manifests/r_teacher_text_predictions.jsonl"
TEACHER_FAILURES_PATH = "Data/manifests/r_teacher_text_failures.jsonl"
TEACHER_CORRECT_PATH = "Data/manifests/r_text_candidates_teacher_correct.jsonl"
TEACHER_SCREEN_REPORT_PATH = "Data/cards/r_teacher_text_filter_report.json"
TEACHER_EXAMPLES_CARD_PATH = "Data/cards/r_teacher_text_examples.json"
TEACHER_SAMPLE_PLAN_PATH = "Data/cards/r_teacher_text_sample_plan.json"

TEACHER_TARGET_COUNT = 5000
TEACHER_INITIAL_SAMPLE = 5000
TEACHER_MAX_ROUNDS = 3
TEACHER_SAMPLE_SEED = 42
TEACHER_RETENTION_BUFFER = 1.15

TEACHER_ABORT_CONSECUTIVE_FAILURES = 10
TEACHER_ABORT_WINDOW_SIZE = 20
TEACHER_ABORT_WINDOW_FAILURE_RATE = 0.50
TEACHER_PROGRESS_LOG_EVERY = 50


def teacher_examples_path() -> Path:
    """Return the export path for Teacher-text inference examples."""

    return resolve_from_repo_root(TEACHER_EXAMPLES_PATH)


def teacher_predictions_path() -> Path:
    """Return the default Teacher-text prediction manifest path."""

    return resolve_from_repo_root(TEACHER_PREDICTIONS_PATH)


def teacher_failures_path() -> Path:
    """Return the exhausted Teacher-text API failure manifest path."""

    return resolve_from_repo_root(TEACHER_FAILURES_PATH)


def teacher_correct_path() -> Path:
    """Return the kept Teacher-correct R-text manifest path."""

    return resolve_from_repo_root(TEACHER_CORRECT_PATH)


def build_teacher_example(record: dict[str, Any]) -> dict[str, Any]:
    """Convert one filtered R-text row into a Teacher-text example."""

    from realmg.eval.r_verifiers import assign_verifier_method

    prompt_text = record["prompt_text"]
    reference_answer = record["answer_text"]
    source_name = record["source_name"]
    verifier_method = assign_verifier_method(
        source_name,
        prompt_text,
        reference_answer,
    )
    return {
        "example_id": record["content_id"],
        "layer": "R",
        "stage": "teacher_text_screen",
        "content_id": record["content_id"],
        "source_name": source_name,
        "prompt_text": prompt_text,
        "reference_answer": reference_answer,
        "verifier_method": verifier_method,
        "metadata": dict(record.get("metadata", {})),
    }


def build_teacher_examples_report(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize exported Teacher-text examples."""

    by_source = Counter(example["source_name"] for example in examples)
    by_verifier = Counter(example["verifier_method"] for example in examples)
    return {
        "input_manifest": FILTERED_INPUT_PATH,
        "output_manifest": TEACHER_EXAMPLES_PATH,
        "example_count": len(examples),
        "by_source": dict(sorted(by_source.items())),
        "by_verifier_method": dict(sorted(by_verifier.items())),
        "teacher_backend": "bailian_openai_compatible",
        "notes": [
            "Teacher sees text-only prompt_text; audio is the question at train/eval time.",
        ],
    }


def prepare_r_teacher_text_examples() -> None:
    """Export Teacher-text examples from the filtered R-text pool."""

    filtered_records = read_jsonl(resolve_from_repo_root(FILTERED_INPUT_PATH))
    examples = [build_teacher_example(record) for record in filtered_records]
    write_jsonl(examples, teacher_examples_path())
    write_report(build_teacher_examples_report(examples), resolve_from_repo_root(TEACHER_EXAMPLES_CARD_PATH))


def is_successful_prediction_row(record: dict[str, Any]) -> bool:
    """Return whether one prediction row is a usable Teacher success."""

    if record.get("error") or record.get("status") == "exhausted_retries":
        return False
    return bool(str(record.get("prediction_text", "")).strip())


def load_prediction_index(path: Path) -> dict[str, dict[str, Any]]:
    """Load successful prediction rows keyed by example_id."""

    if not path.exists():
        return {}
    return {
        record["example_id"]: record
        for record in read_jsonl(path)
        if is_successful_prediction_row(record)
    }


def rewrite_successful_predictions(path: Path) -> int:
    """Drop error rows from the predictions file and keep successes only."""

    if not path.exists():
        return 0
    kept = [
        record
        for record in read_jsonl(path)
        if is_successful_prediction_row(record)
    ]
    write_jsonl(kept, path)
    return len(kept)


def should_abort_teacher_api_run(
    *,
    consecutive_failures: int,
    recent_outcomes: list[bool],
    abort_consecutive_failures: int,
    abort_window_size: int,
    abort_window_failure_rate: float,
) -> str | None:
    """Return an abort reason when Bailian failures become systemic."""

    if consecutive_failures >= abort_consecutive_failures:
        return (
            f"consecutive_failures>={abort_consecutive_failures} "
            f"(got {consecutive_failures})"
        )

    if len(recent_outcomes) >= abort_window_size:
        window = recent_outcomes[-abort_window_size:]
        failure_rate = sum(1 for ok in window if not ok) / len(window)
        if failure_rate >= abort_window_failure_rate:
            return (
                f"window_failure_rate>={abort_window_failure_rate:.0%} "
                f"over last {abort_window_size} attempts "
                f"(got {failure_rate:.0%})"
            )
    return None


def list_unqueried_examples(
    examples: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return examples that do not yet have a prediction row."""

    return [
        example
        for example in examples
        if example["example_id"] not in predictions
    ]


def sample_unqueried_examples(
    examples: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Randomly sample unqueried Teacher-text examples."""

    pool = list_unqueried_examples(examples, predictions)
    if sample_size >= len(pool):
        return pool
    rng = random.Random(seed)
    return rng.sample(pool, sample_size)


def load_teacher_correct_records() -> list[dict[str, Any]]:
    """Load the accumulated Teacher-correct R-text pool if it exists."""

    path = teacher_correct_path()
    if not path.exists():
        return []
    return read_jsonl(path)


def merge_teacher_correct_records(
    existing_records: list[dict[str, Any]],
    new_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge Teacher-correct rows without duplicate content_id."""

    merged: dict[str, dict[str, Any]] = {
        record["content_id"]: record for record in existing_records
    }
    for record in new_records:
        merged[record["content_id"]] = record
    return list(merged.values())


def count_scorable_predictions(
    examples: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> int:
    """Count queried examples that can be scored by a verifier."""

    scorable = 0
    for example in examples:
        row = predictions.get(example["example_id"])
        if row is None or row.get("error"):
            continue
        scorable += 1
    return scorable


def build_teacher_sample_plan(
    *,
    target_count: int = TEACHER_TARGET_COUNT,
    retention_buffer: float = TEACHER_RETENTION_BUFFER,
    initial_sample: int = TEACHER_INITIAL_SAMPLE,
    seed: int = TEACHER_SAMPLE_SEED,
) -> dict[str, Any]:
    """Recommend the next Teacher-text API sample size."""

    examples = read_jsonl(teacher_examples_path())
    predictions = load_prediction_index(teacher_predictions_path())
    kept_records = load_teacher_correct_records()
    unqueried = list_unqueried_examples(examples, predictions)
    scorable = count_scorable_predictions(examples, predictions)
    kept_count = len(kept_records)
    remaining_needed = max(0, target_count - kept_count)

    retention_rate = kept_count / scorable if scorable else None
    if remaining_needed == 0:
        recommended_sample = 0
        status = "target_reached"
    elif retention_rate is None or scorable == 0:
        recommended_sample = min(initial_sample, len(unqueried))
        status = "initial_probe"
    elif retention_rate == 0:
        recommended_sample = min(initial_sample, len(unqueried))
        status = "zero_retention_retry"
    else:
        raw_need = math.ceil(remaining_needed / retention_rate * retention_buffer)
        recommended_sample = min(raw_need, len(unqueried))
        status = "top_up"

    return {
        "target_count": target_count,
        "kept_count": kept_count,
        "remaining_needed": remaining_needed,
        "queried_count": len(predictions),
        "scorable_count": scorable,
        "unqueried_count": len(unqueried),
        "retention_rate": retention_rate,
        "retention_buffer": retention_buffer,
        "recommended_sample_size": recommended_sample,
        "sample_seed": seed,
        "status": status,
        "notes": [
            "Use recommended_sample_size with `run-r-teacher-text --sample-size`.",
        ],
    }


def plan_r_teacher_text_sample(
    *,
    target_count: int = TEACHER_TARGET_COUNT,
    retention_buffer: float = TEACHER_RETENTION_BUFFER,
    initial_sample: int = TEACHER_INITIAL_SAMPLE,
    seed: int = TEACHER_SAMPLE_SEED,
) -> None:
    """Write the next Teacher-text sampling recommendation."""

    write_report(
        build_teacher_sample_plan(
            target_count=target_count,
            retention_buffer=retention_buffer,
            initial_sample=initial_sample,
            seed=seed,
        ),
        resolve_from_repo_root(TEACHER_SAMPLE_PLAN_PATH),
    )


def append_prediction_row(path: Path, record: dict[str, Any]) -> None:
    """Append one prediction row and fsync so an OOM kill cannot lose it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def upsert_prediction_rows(
    path: Path,
    updates: dict[str, dict[str, Any]],
) -> int:
    """Replace or append prediction rows keyed by example_id."""

    rows = read_jsonl(path) if path.exists() else []
    index = {row["example_id"]: position for position, row in enumerate(rows)}
    replaced = 0
    for example_id, record in updates.items():
        if example_id in index:
            rows[index[example_id]] = record
            replaced += 1
        else:
            rows.append(record)
    write_jsonl(rows, path)
    return replaced


def extract_completion_text(completion: Any) -> str:
    """Extract plain text from one OpenAI-compatible chat completion."""

    message = completion.choices[0].message
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return ""


def extract_omni_stream_text(stream: Any) -> str:
    """Accumulate plain text from a Bailian Omni streaming completion."""

    parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            parts.append(content)
            continue
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif isinstance(item, str):
                    parts.append(item)
    return "".join(parts).strip()


def omni_eval_create_extras(model_name: str) -> dict[str, Any]:
    """Return compatible-mode extras required by Bailian Omni eval calls.

    off; pin enable_thinking=False so eval never silently enters hybrid think.
    """

    extras: dict[str, Any] = {}
    name = model_name.strip().lower()
    if "omni" in name:
        extras["stream"] = True
        extras["stream_options"] = {"include_usage": True}
    if "qwen3" in name and "omni" in name:
        extras["extra_body"] = {"enable_thinking": False}
    return extras


def call_teacher_text_api(
    client: OpenAI,
    *,
    prompt_text: str,
    request_options: dict[str, Any],
    model: str | None = None,
) -> str:
    """Call Bailian Qwen-Omni once with text-only input."""

    model_id = model or bailian_model_name()
    create_kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": request_options["temperature"],
        "max_tokens": request_options["max_tokens"],
    }
    modalities = request_options.get("modalities")
    if modalities:
        create_kwargs["modalities"] = modalities
    create_kwargs.update(omni_eval_create_extras(model_id))
    completion = client.chat.completions.create(**create_kwargs)
    if create_kwargs.get("stream"):
        return extract_omni_stream_text(completion)
    return extract_completion_text(completion)


def run_r_teacher_text_api(
    *,
    sample_size: int | None = None,
    limit: int | None = None,
    seed: int = TEACHER_SAMPLE_SEED,
    resume: bool = True,
    request_interval_seconds: float = 0.2,
    max_retries: int = 3,
    abort_consecutive_failures: int = TEACHER_ABORT_CONSECUTIVE_FAILURES,
    abort_window_size: int = TEACHER_ABORT_WINDOW_SIZE,
    abort_window_failure_rate: float = TEACHER_ABORT_WINDOW_FAILURE_RATE,
    examples_path: Path | None = None,
    predictions_path: Path | None = None,
    failures_path: Path | None = None,
    run_report_path: Path | None = None,
    update_sample_plan: bool = True,
) -> None:
    """Fill Teacher-text predictions through the Bailian OpenAI-compatible API."""

    examples_file = examples_path or teacher_examples_path()
    output_path = predictions_path or teacher_predictions_path()
    failures_file = failures_path or teacher_failures_path()
    report_path = run_report_path or resolve_from_repo_root(
        "Data/cards/r_teacher_text_run_report.json"
    )

    all_examples = read_jsonl(examples_file)
    if resume:
        rewrite_successful_predictions(output_path)
    finished = load_prediction_index(output_path) if resume else {}

    if sample_size is not None:
        examples = sample_unqueried_examples(
            all_examples,
            finished,
            sample_size,
            seed,
        )
    elif limit is not None:
        examples = list_unqueried_examples(all_examples, finished)[:limit]
    else:
        examples = list_unqueried_examples(all_examples, finished)

    client = build_bailian_client()
    request_options = default_bailian_request_options()

    processed = 0
    skipped = 0
    failed = 0
    consecutive_failures = 0
    recent_outcomes: list[bool] = []
    abort_reason: str | None = None

    for example in examples:
        example_id = example["example_id"]
        if example_id in finished:
            skipped += 1
            continue

        last_error: Exception | None = None
        prediction_text = ""
        for attempt in range(max_retries):
            try:
                prediction_text = call_teacher_text_api(
                    client,
                    prompt_text=example["prompt_text"],
                    request_options=request_options,
                )
                if not prediction_text.strip():
                    raise RuntimeError("empty_prediction_text")
                append_prediction_row(
                    output_path,
                    {
                        "example_id": example_id,
                        "content_id": example["content_id"],
                        "source_name": example["source_name"],
                        "prediction_text": prediction_text,
                        "teacher_backend": "bailian_openai_compatible",
                        "teacher_model": bailian_model_name(),
                    },
                )
                finished[example_id] = {"example_id": example_id}
                processed += 1
                consecutive_failures = 0
                recent_outcomes.append(True)
                last_error = None
                if processed % TEACHER_PROGRESS_LOG_EVERY == 0:
                    print(
                        f"[run-r-teacher-text] successes={processed} "
                        f"exhausted_failures={failed} remaining_in_batch="
                        f"{len(examples) - processed - failed - skipped}",
                        flush=True,
                    )
                break
            except Exception as exc:  # noqa: BLE001 - retry wrapper
                last_error = exc
                time.sleep(min(2.0 ** attempt, 8.0))

        if last_error is not None:
            failed += 1
            consecutive_failures += 1
            recent_outcomes.append(False)
            append_prediction_row(
                failures_file,
                {
                    "example_id": example_id,
                    "content_id": example["content_id"],
                    "source_name": example["source_name"],
                    "status": "exhausted_retries",
                    "retry_count": max_retries,
                    "error": str(last_error),
                    "teacher_backend": "bailian_openai_compatible",
                    "teacher_model": bailian_model_name(),
                },
            )
            print(
                f"[run-r-teacher-text] exhausted_retries "
                f"example_id={example_id} error={last_error}",
                flush=True,
            )

            abort_reason = should_abort_teacher_api_run(
                consecutive_failures=consecutive_failures,
                recent_outcomes=recent_outcomes,
                abort_consecutive_failures=abort_consecutive_failures,
                abort_window_size=abort_window_size,
                abort_window_failure_rate=abort_window_failure_rate,
            )
            if abort_reason is not None:
                print(
                    f"[run-r-teacher-text] ABORT: {abort_reason}",
                    flush=True,
                )
                break

        if request_interval_seconds > 0:
            time.sleep(request_interval_seconds)

    write_report(
        {
            "examples_manifest": str(examples_file),
            "predictions_manifest": str(output_path),
            "failures_manifest": str(failures_file),
            "processed_count": processed,
            "skipped_existing_count": skipped,
            "exhausted_failure_count": failed,
            "aborted": abort_reason is not None,
            "abort_reason": abort_reason,
            "abort_policy": {
                "consecutive_failures": abort_consecutive_failures,
                "window_size": abort_window_size,
                "window_failure_rate": abort_window_failure_rate,
                "max_retries": max_retries,
            },
            "requested_sample_size": sample_size,
            "requested_limit": limit,
            "sample_seed": seed if sample_size is not None else None,
            "resume": resume,
            "request_interval_seconds": request_interval_seconds,
            "notes": [
                "Only successful answers are written to predictions.",
                "Exhausted retries are flagged in the failures manifest.",
            ],
        },
        report_path,
    )
    if update_sample_plan:
        plan_r_teacher_text_sample(seed=seed)
    if abort_reason is not None:
        raise RuntimeError(f"Teacher-text API aborted: {abort_reason}")


def build_teacher_filter_report(
    *,
    queried_count: int,
    kept_records: list[dict[str, Any]],
    new_kept_count: int,
    rejection_counts: Counter[str],
    verifier_counts: Counter[str],
    merged_kept_count: int,
) -> dict[str, Any]:
    """Summarize Teacher-text screening."""

    kept_by_source = Counter(record["source_name"] for record in kept_records)
    return {
        "input_manifest": TEACHER_EXAMPLES_PATH,
        "predictions_manifest": TEACHER_PREDICTIONS_PATH,
        "output_manifest": TEACHER_CORRECT_PATH,
        "queried_count": queried_count,
        "new_kept_count": new_kept_count,
        "merged_kept_count": merged_kept_count,
        "retention_rate": new_kept_count / queried_count if queried_count else None,
        "kept_by_source": dict(sorted(kept_by_source.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "verifier_counts": dict(sorted(verifier_counts.items())),
        "frozen_verifiers": {
            "natural_reasoning": "TIGER-Lab/general-verifier",
            "tulu3_open_qa": "qa_metrics.pedant.PEDANT (VoiceBench SD-QA path)",
            "tulu3_mcq": "VoiceBench mcq option extraction",
        },
        "notes": [
            "Losses still do not use reference answers; this stage only filters trainable s.",
        ],
    }


def filter_r_teacher_text_predictions(
    *,
    predictions_file: str | None = None,
    verifier_batch_log_every: int = 200,
    examples_path: Path | None = None,
    output_correct_path: Path | None = None,
    report_path: Path | None = None,
    merge_existing_correct: bool = True,
) -> list[dict[str, Any]]:
    """Keep queried R-text rows whose Teacher-text answers pass the routed verifier."""

    from realmg.eval.r_verifiers import (
        GeneralVerifierScorer,
        verify_teacher_prediction,
    )

    examples_file = examples_path or teacher_examples_path()
    examples = read_jsonl(examples_file)
    example_index = {example["example_id"]: example for example in examples}
    predictions_path = (
        resolve_from_repo_root(predictions_file)
        if predictions_file
        else teacher_predictions_path()
    )
    predictions = load_prediction_index(predictions_path)
    queried_examples = [
        example_index[example_id]
        for example_id in predictions
        if example_id in example_index
    ]

    general_verifier = GeneralVerifierScorer()
    new_kept_records: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    verifier_counts: Counter[str] = Counter()

    for index, example in enumerate(queried_examples, start=1):
        example_id = example["example_id"]
        prediction_row = predictions[example_id]
        if prediction_row.get("error"):
            rejection_counts["teacher_api_error"] += 1
            continue

        result = verify_teacher_prediction(
            source_name=example["source_name"],
            prompt_text=example["prompt_text"],
            reference_answer=example["reference_answer"],
            prediction_text=str(prediction_row.get("prediction_text", "")),
            general_verifier=general_verifier,
        )
        verifier_counts[result.verifier_method] += 1
        if result.is_correct:
            new_kept_records.append(
                {
                    "content_id": example["content_id"],
                    "source_name": example["source_name"],
                    "prompt_text": example["prompt_text"],
                    "answer_text": example["reference_answer"],
                    "metadata": {
                        **example.get("metadata", {}),
                        "teacher_text_verifier": result.verifier_method,
                        "teacher_text_prediction": prediction_row.get(
                            "prediction_text", ""
                        ),
                    },
                }
            )
        else:
            rejection_counts["teacher_text_incorrect"] += 1

        if (
            result.verifier_method == "general_verifier"
            and verifier_batch_log_every > 0
            and index % verifier_batch_log_every == 0
        ):
            write_report(
                {
                    "progress_queried_examples": index,
                    "new_kept_so_far": len(new_kept_records),
                    "rejected_so_far": sum(rejection_counts.values()),
                },
                resolve_from_repo_root("Data/cards/r_teacher_text_filter_progress.json"),
            )

    merged_kept_records = (
        merge_teacher_correct_records(
            load_teacher_correct_records(),
            new_kept_records,
        )
        if merge_existing_correct
        else new_kept_records
    )
    correct_output = output_correct_path or teacher_correct_path()
    screen_report = report_path or resolve_from_repo_root(TEACHER_SCREEN_REPORT_PATH)
    write_jsonl(merged_kept_records, correct_output)
    write_report(
        build_teacher_filter_report(
            queried_count=len(queried_examples),
            kept_records=new_kept_records,
            new_kept_count=len(new_kept_records),
            rejection_counts=rejection_counts,
            verifier_counts=verifier_counts,
            merged_kept_count=len(merged_kept_records),
        ),
        screen_report,
    )
    if merge_existing_correct:
        plan_r_teacher_text_sample()
    return new_kept_records


def run_r_teacher_text_rounds(
    *,
    target_count: int = TEACHER_TARGET_COUNT,
    initial_sample: int = TEACHER_INITIAL_SAMPLE,
    max_rounds: int = TEACHER_MAX_ROUNDS,
    retention_buffer: float = TEACHER_RETENTION_BUFFER,
    seed: int = TEACHER_SAMPLE_SEED,
    request_interval_seconds: float = 0.2,
    max_retries: int = 3,
) -> None:
    """Run up to a few Teacher-text API rounds until the kept pool reaches target."""

    round_reports: list[dict[str, Any]] = []
    for round_index in range(1, max_rounds + 1):
        plan = build_teacher_sample_plan(
            target_count=target_count,
            retention_buffer=retention_buffer,
            initial_sample=initial_sample,
            seed=seed + round_index - 1,
        )
        sample_size = plan["recommended_sample_size"]
        if sample_size <= 0:
            round_reports.append(
                {
                    "round": round_index,
                    "status": plan["status"],
                    "sample_size": 0,
                    "kept_count": plan["kept_count"],
                }
            )
            break

        run_r_teacher_text_api(
            sample_size=sample_size,
            seed=seed + round_index - 1,
            resume=True,
            request_interval_seconds=request_interval_seconds,
            max_retries=max_retries,
        )
        filter_r_teacher_text_predictions()
        plan = build_teacher_sample_plan(
            target_count=target_count,
            retention_buffer=retention_buffer,
            initial_sample=initial_sample,
            seed=seed + round_index - 1,
        )
        round_reports.append(
            {
                "round": round_index,
                "sample_size": sample_size,
                "kept_count": plan["kept_count"],
                "retention_rate": plan["retention_rate"],
                "remaining_needed": plan["remaining_needed"],
                "status": plan["status"],
            }
        )
        if plan["remaining_needed"] <= 0:
            break

    write_report(
        {
            "target_count": target_count,
            "max_rounds": max_rounds,
            "rounds": round_reports,
            "final_plan": build_teacher_sample_plan(
                target_count=target_count,
                retention_buffer=retention_buffer,
                initial_sample=initial_sample,
                seed=seed,
            ),
        },
        resolve_from_repo_root("Data/cards/r_teacher_text_rounds_report.json"),
    )
