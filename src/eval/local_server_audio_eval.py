"""Local OpenAI-compatible server eval helpers (vLLM on GPU box).

"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_report
from realmg.data.local_server_api import (
    build_local_server_client,
    default_local_server_request_options,
    local_server_audio_content_format,
    local_server_model_name,
)
from realmg.data.r_text_teacher import (
    TEACHER_ABORT_CONSECUTIVE_FAILURES,
    TEACHER_ABORT_WINDOW_FAILURE_RATE,
    TEACHER_ABORT_WINDOW_SIZE,
    TEACHER_PROGRESS_LOG_EVERY,
    append_prediction_row,
    load_prediction_index,
    rewrite_successful_predictions,
    should_abort_teacher_api_run,
)
from realmg.eval.eval_finish import (
    score_p_eval_if_complete,
    score_r_eval_if_complete,
    score_r_eval_text_if_complete,
)
from realmg.eval.p_harness import (
    P_SCORABLE_PROTOCOLS,
    eval_examples_path,
    iter_p_api_query_protocols,
)

BACKEND_SLUG = "local_server_base"
BACKEND_NAME = "local_server_openai_compatible"
R_CONTENT_MAX_TOKENS = 4096
R_PARA_MAX_TOKENS = 64
R_EVAL_EXAMPLES_PATH = "Data/manifests/r_eval_examples.jsonl"
R_EVAL_TEXT_EXAMPLES_PATH = "Data/manifests/r_eval_text_examples.jsonl"


def r_eval_examples_path() -> Path:
    """Return the frozen R eval examples manifest."""

    return resolve_from_repo_root(R_EVAL_EXAMPLES_PATH)


def r_eval_text_examples_path() -> Path:
    """Return the frozen R eval text-baseline examples manifest."""

    return resolve_from_repo_root(R_EVAL_TEXT_EXAMPLES_PATH)


def model_slug(model_name: str) -> str:
    """Filesystem-safe slug for one local-server model id."""

    slug = model_name.strip()
    for old, new in (("/", "_"), (":", "_"), (".", "_"), ("-", "_")):
        slug = slug.replace(old, new)
    return slug


def condition_slug_for_model(model_name: str) -> str:
    """Return the per-model eval filename slug."""

    return f"{model_slug(model_name)}_{BACKEND_SLUG}"


def resolve_eval_model(model_name: str | None) -> str:
    """Use an explicit model id, or the local-server toml default."""

    if model_name is None or not str(model_name).strip():
        return local_server_model_name()
    return str(model_name).strip()


def encode_local_audio_base64(audio_path: Path) -> tuple[str, str]:
    """Encode a local audio file as raw base64 for input_audio."""

    if not audio_path.exists():
        raise FileNotFoundError(f"Missing eval audio: {audio_path}")
    suffix = audio_path.suffix.lower().lstrip(".")
    if not suffix:
        raise ValueError(f"Audio path has no format suffix: {audio_path}")
    payload = base64.b64encode(audio_path.read_bytes()).decode("utf-8")
    return payload, suffix


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
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts).strip()
    return ""


def build_local_server_audio_content_part(
    audio_path: Path,
    *,
    audio_content_format: str | None = None,
) -> dict[str, Any]:
    """Build one multimodal audio content part for the local server."""

    audio_data, audio_format = encode_local_audio_base64(audio_path)
    fmt = audio_content_format or local_server_audio_content_format()
    if fmt == "audio_url":
        mime = "audio/wav" if audio_format == "wav" else f"audio/{audio_format}"
        return {
            "type": "audio_url",
            "audio_url": {"url": f"data:{mime};base64,{audio_data}"},
        }
    return {
        "type": "input_audio",
        "input_audio": {
            "data": audio_data,
            "format": audio_format,
        },
    }


def call_local_server_audio_text_api(
    client: OpenAI,
    *,
    audio_path: Path,
    prompt_text: str,
    request_options: dict[str, Any],
    model: str | None = None,
) -> str:
    """Call the local server once with audio and optional text prompt."""

    content: list[dict[str, Any]] = []
    if prompt_text.strip():
        content.append({"type": "text", "text": prompt_text})
    content.append(build_local_server_audio_content_part(audio_path))

    completion = client.chat.completions.create(
        model=resolve_eval_model(model),
        messages=[{"role": "user", "content": content}],
        temperature=request_options["temperature"],
        max_tokens=request_options["max_tokens"],
    )
    return extract_completion_text(completion)


def call_local_server_text_api(
    client: OpenAI,
    *,
    prompt_text: str,
    request_options: dict[str, Any],
    model: str | None = None,
) -> str:
    """Call the local server once with text-only input."""

    completion = client.chat.completions.create(
        model=resolve_eval_model(model),
        messages=[{"role": "user", "content": prompt_text}],
        temperature=request_options["temperature"],
        max_tokens=request_options["max_tokens"],
    )
    return extract_completion_text(completion)


def list_pending_eval_examples(
    examples: list[dict[str, Any]],
    finished: dict[str, dict[str, Any]],
    *,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Return unqueried eval examples in file order, optionally capped."""

    pending = [
        example
        for example in examples
        if example["example_id"] not in finished
    ]
    if limit is not None:
        return pending[:limit]
    return pending


def p_eval_predictions_path(protocol: str, model_name: str | None = None) -> Path:
    """Return the prediction JSONL path for one P protocol local-server run."""

    slug = condition_slug_for_model(resolve_eval_model(model_name))
    return resolve_from_repo_root(
        f"Data/manifests/p_eval_predictions_{protocol}_{slug}.jsonl"
    )


def p_eval_failures_path(protocol: str, model_name: str | None = None) -> Path:
    """Return the exhausted-failure JSONL path for one P protocol local-server run."""

    slug = condition_slug_for_model(resolve_eval_model(model_name))
    return resolve_from_repo_root(
        f"Data/manifests/p_eval_failures_{protocol}_{slug}.jsonl"
    )


def p_eval_run_report_path(protocol: str, model_name: str | None = None) -> Path:
    """Return the run report path for one P protocol local-server run."""

    slug = condition_slug_for_model(resolve_eval_model(model_name))
    return resolve_from_repo_root(f"Data/cards/p_eval_run_{protocol}_{slug}.json")


def r_eval_predictions_path(model_name: str | None = None) -> Path:
    """Return the prediction JSONL path for the R local-server run."""

    slug = condition_slug_for_model(resolve_eval_model(model_name))
    return resolve_from_repo_root(f"Data/manifests/r_eval_predictions_{slug}.jsonl")


def r_eval_failures_path(model_name: str | None = None) -> Path:
    """Return the exhausted-failure JSONL path for the R local-server run."""

    slug = condition_slug_for_model(resolve_eval_model(model_name))
    return resolve_from_repo_root(f"Data/manifests/r_eval_failures_{slug}.jsonl")


def r_eval_run_report_path(model_name: str | None = None) -> Path:
    """Return the run report path for the R local-server run."""

    slug = condition_slug_for_model(resolve_eval_model(model_name))
    return resolve_from_repo_root(f"Data/cards/r_eval_run_{slug}.json")


def r_eval_text_predictions_path(model_name: str | None = None) -> Path:
    """Return the prediction JSONL path for the R text baseline local-server run."""

    slug = condition_slug_for_model(resolve_eval_model(model_name))
    return resolve_from_repo_root(
        f"Data/manifests/r_eval_text_predictions_{slug}.jsonl"
    )


def r_eval_text_failures_path(model_name: str | None = None) -> Path:
    """Return the failure JSONL path for the R text baseline local-server run."""

    slug = condition_slug_for_model(resolve_eval_model(model_name))
    return resolve_from_repo_root(
        f"Data/manifests/r_eval_text_failures_{slug}.jsonl"
    )


def r_eval_text_run_report_path(model_name: str | None = None) -> Path:
    """Return the run report path for the R text baseline local-server run."""

    slug = condition_slug_for_model(resolve_eval_model(model_name))
    return resolve_from_repo_root(f"Data/cards/r_eval_text_run_{slug}.json")


def request_options_for_r_example(
    example: dict[str, Any],
    base_options: dict[str, Any],
) -> dict[str, Any]:
    """Choose generation limits for one R eval query type.

    R_CONTENT_MAX_TOKENS (e.g. TARS on 24GB with max_model_len=4096), honor
    that ceiling so completion cannot exceed the engine context window.
    """

    configured = int(base_options["max_tokens"])
    if example["query_type"] == "content":
        if configured < R_CONTENT_MAX_TOKENS:
            max_tokens = configured
        else:
            max_tokens = R_CONTENT_MAX_TOKENS
    else:
        max_tokens = R_PARA_MAX_TOKENS
    return {**base_options, "max_tokens": max_tokens}


def run_p_eval_local_server(
    *,
    protocol: str,
    limit: int | None = None,
    resume: bool = True,
    request_interval_seconds: float = 0.0,
    max_retries: int = 3,
    model: str | None = None,
    score_on_complete: bool = False,
) -> None:
    """Fill local-server predictions for one or all frozen P eval protocols."""

    model_id = resolve_eval_model(model)
    if protocol == "all":
        for one_protocol in iter_p_api_query_protocols("all"):
            run_p_eval_local_server(
                protocol=one_protocol,
                limit=limit,
                resume=resume,
                request_interval_seconds=request_interval_seconds,
                max_retries=max_retries,
                model=model_id,
                score_on_complete=score_on_complete,
            )
        return

    if protocol not in P_SCORABLE_PROTOCOLS:
        raise ValueError(
            f"Unsupported protocol {protocol!r}. Expected one of {P_SCORABLE_PROTOCOLS}."
        )

    examples_file = eval_examples_path(protocol)
    output_path = p_eval_predictions_path(protocol, model_id)
    failures_file = p_eval_failures_path(protocol, model_id)
    report_path = p_eval_run_report_path(protocol, model_id)

    examples = read_jsonl(examples_file)
    if resume:
        rewrite_successful_predictions(output_path)
    finished = load_prediction_index(output_path) if resume else {}
    already_done_count = len(finished)
    pending = list_pending_eval_examples(examples, finished, limit=limit)

    client = build_local_server_client()
    request_options = default_local_server_request_options()

    processed = 0
    failed = 0
    consecutive_failures = 0
    recent_outcomes: list[bool] = []
    abort_reason: str | None = None

    print(
        f"[run-p-eval-local-server] protocol={protocol} pending={len(pending)} "
        f"already_done={already_done_count} model={model_id}",
        flush=True,
    )

    for example in pending:
        example_id = example["example_id"]
        audio_path = resolve_from_repo_root(example["audio_path"])
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                prediction_text = call_local_server_audio_text_api(
                    client,
                    audio_path=audio_path,
                    prompt_text=example["prompt"],
                    request_options=request_options,
                    model=model_id,
                )
                if not prediction_text.strip():
                    raise RuntimeError("empty_prediction_text")
                append_prediction_row(
                    output_path,
                    {
                        "example_id": example_id,
                        "prediction_text": prediction_text,
                        "model": model_id,
                        "backend": BACKEND_NAME,
                        "condition": "base",
                        "protocol": protocol,
                        "layer": "P",
                        "query_type": example["query_type"],
                        "audio_path": example["audio_path"],
                    },
                )
                finished[example_id] = {"example_id": example_id}
                processed += 1
                consecutive_failures = 0
                recent_outcomes.append(True)
                last_error = None
                if processed % TEACHER_PROGRESS_LOG_EVERY == 0:
                    print(
                        f"[run-p-eval-local-server] successes={processed} "
                        f"exhausted_failures={failed} "
                        f"remaining_in_batch={len(pending) - processed - failed}",
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
                    "status": "exhausted_retries",
                    "retry_count": max_retries,
                    "error": str(last_error),
                    "model": model_id,
                    "backend": BACKEND_NAME,
                    "protocol": protocol,
                    "audio_path": example["audio_path"],
                },
            )
            abort_reason = should_abort_teacher_api_run(
                consecutive_failures=consecutive_failures,
                recent_outcomes=recent_outcomes,
                abort_consecutive_failures=TEACHER_ABORT_CONSECUTIVE_FAILURES,
                abort_window_size=TEACHER_ABORT_WINDOW_SIZE,
                abort_window_failure_rate=TEACHER_ABORT_WINDOW_FAILURE_RATE,
            )
            if abort_reason is not None:
                print(f"[run-p-eval-local-server] ABORT: {abort_reason}", flush=True)
                break

        if request_interval_seconds > 0:
            time.sleep(request_interval_seconds)

    write_report(
        {
            "stage": "p_eval_local_server_base",
            "protocol": protocol,
            "condition": "base",
            "backbone_slug": model_slug(model_id),
            "backend": BACKEND_NAME,
            "model": model_id,
            "examples_manifest": str(examples_file),
            "predictions_manifest": str(output_path),
            "failures_manifest": str(failures_file),
            "processed_count": processed,
            "already_done_count": already_done_count,
            "exhausted_failure_count": failed,
            "aborted": abort_reason is not None,
            "abort_reason": abort_reason,
            "requested_limit": limit,
            "resume": resume,
            "request_interval_seconds": request_interval_seconds,
            "score_on_complete": score_on_complete,
            "notes": [
                f"Local-server predictions use slug {condition_slug_for_model(model_id)}.",
                "Default: copy predictions JSONL to dev machine and run score-p-eval locally.",
            ],
        },
        report_path,
    )
    print(
        f"[run-p-eval-local-server] done protocol={protocol} "
        f"new_successes={processed} failures={failed} "
        f"total_predictions={len(load_prediction_index(output_path))}",
        flush=True,
    )
    if score_on_complete:
        score_p_eval_if_complete(
            examples=examples,
            predictions_path=output_path,
            aborted=abort_reason is not None,
            log_prefix="run-p-eval-local-server",
        )


def run_r_eval_local_server(
    *,
    limit: int | None = None,
    resume: bool = True,
    request_interval_seconds: float = 0.0,
    max_retries: int = 3,
    model: str | None = None,
    score_on_complete: bool = False,
) -> None:
    """Fill local-server predictions for the frozen R eval set."""

    model_id = resolve_eval_model(model)
    examples_file = r_eval_examples_path()
    output_path = r_eval_predictions_path(model_id)
    failures_file = r_eval_failures_path(model_id)
    report_path = r_eval_run_report_path(model_id)

    examples = read_jsonl(examples_file)
    if resume:
        rewrite_successful_predictions(output_path)
    finished = load_prediction_index(output_path) if resume else {}
    already_done_count = len(finished)
    pending = list_pending_eval_examples(examples, finished, limit=limit)

    client = build_local_server_client()
    base_options = default_local_server_request_options()

    processed = 0
    failed = 0
    consecutive_failures = 0
    recent_outcomes: list[bool] = []
    abort_reason: str | None = None

    print(
        f"[run-r-eval-local-server] pending={len(pending)} "
        f"already_done={already_done_count} model={model_id}",
        flush=True,
    )

    for example in pending:
        example_id = example["example_id"]
        audio_path = resolve_from_repo_root(example["audio_path"])
        request_options = request_options_for_r_example(example, base_options)
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                prediction_text = call_local_server_audio_text_api(
                    client,
                    audio_path=audio_path,
                    prompt_text=str(example.get("prompt") or ""),
                    request_options=request_options,
                    model=model_id,
                )
                if not prediction_text.strip():
                    raise RuntimeError("empty_prediction_text")
                append_prediction_row(
                    output_path,
                    {
                        "example_id": example_id,
                        "prediction_text": prediction_text,
                        "model": model_id,
                        "backend": BACKEND_NAME,
                        "condition": "base",
                        "layer": "R",
                        "query_type": example["query_type"],
                        "audio_path": example["audio_path"],
                    },
                )
                finished[example_id] = {"example_id": example_id}
                processed += 1
                consecutive_failures = 0
                recent_outcomes.append(True)
                last_error = None
                if processed % TEACHER_PROGRESS_LOG_EVERY == 0:
                    print(
                        f"[run-r-eval-local-server] successes={processed} "
                        f"exhausted_failures={failed} "
                        f"remaining_in_batch={len(pending) - processed - failed}",
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
                    "status": "exhausted_retries",
                    "retry_count": max_retries,
                    "error": str(last_error),
                    "model": model_id,
                    "backend": BACKEND_NAME,
                    "layer": "R",
                    "audio_path": example["audio_path"],
                },
            )
            abort_reason = should_abort_teacher_api_run(
                consecutive_failures=consecutive_failures,
                recent_outcomes=recent_outcomes,
                abort_consecutive_failures=TEACHER_ABORT_CONSECUTIVE_FAILURES,
                abort_window_size=TEACHER_ABORT_WINDOW_SIZE,
                abort_window_failure_rate=TEACHER_ABORT_WINDOW_FAILURE_RATE,
            )
            if abort_reason is not None:
                print(f"[run-r-eval-local-server] ABORT: {abort_reason}", flush=True)
                break

        if request_interval_seconds > 0:
            time.sleep(request_interval_seconds)

    write_report(
        {
            "stage": "r_eval_local_server_base",
            "condition": "base",
            "backbone_slug": model_slug(model_id),
            "backend": BACKEND_NAME,
            "model": model_id,
            "examples_manifest": str(examples_file),
            "predictions_manifest": str(output_path),
            "failures_manifest": str(failures_file),
            "processed_count": processed,
            "already_done_count": already_done_count,
            "exhausted_failure_count": failed,
            "aborted": abort_reason is not None,
            "abort_reason": abort_reason,
            "requested_limit": limit,
            "resume": resume,
            "request_interval_seconds": request_interval_seconds,
            "content_max_tokens": R_CONTENT_MAX_TOKENS,
            "score_on_complete": score_on_complete,
            "notes": [
                f"Local-server predictions use slug {condition_slug_for_model(model_id)}.",
                "Default: score-r-eval on dev machine after copying JSONL.",
            ],
        },
        report_path,
    )
    print(
        f"[run-r-eval-local-server] done new_successes={processed} failures={failed} "
        f"total_predictions={len(load_prediction_index(output_path))}",
        flush=True,
    )
    if score_on_complete:
        score_r_eval_if_complete(
            examples=examples,
            predictions_path=output_path,
            aborted=abort_reason is not None,
            log_prefix="run-r-eval-local-server",
        )


def run_r_eval_text_local_server(
    *,
    limit: int | None = None,
    resume: bool = True,
    request_interval_seconds: float = 0.0,
    max_retries: int = 3,
    model: str | None = None,
    score_on_complete: bool = False,
) -> None:
    """Fill local-server text-only predictions for unique eval content_id rows."""

    model_id = resolve_eval_model(model)
    examples_file = r_eval_text_examples_path()
    output_path = r_eval_text_predictions_path(model_id)
    failures_file = r_eval_text_failures_path(model_id)
    report_path = r_eval_text_run_report_path(model_id)
    request_options = default_local_server_request_options()
    configured = int(request_options["max_tokens"])
    request_options = {
        **request_options,
        "max_tokens": (
            configured
            if configured < R_CONTENT_MAX_TOKENS
            else R_CONTENT_MAX_TOKENS
        ),
    }

    examples = read_jsonl(examples_file)
    if resume:
        rewrite_successful_predictions(output_path)
    finished = load_prediction_index(output_path) if resume else {}
    already_done_count = len(finished)
    pending = list_pending_eval_examples(examples, finished, limit=limit)

    client = build_local_server_client()
    processed = 0
    failed = 0
    consecutive_failures = 0
    recent_outcomes: list[bool] = []
    abort_reason: str | None = None

    print(
        f"[run-r-eval-text-local-server] pending={len(pending)} "
        f"already_done={already_done_count} model={model_id}",
        flush=True,
    )

    for example in pending:
        example_id = example["example_id"]
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                prediction_text = call_local_server_text_api(
                    client,
                    prompt_text=example["prompt_text"],
                    request_options=request_options,
                    model=model_id,
                )
                if not prediction_text.strip():
                    raise RuntimeError("empty_prediction_text")
                append_prediction_row(
                    output_path,
                    {
                        "example_id": example_id,
                        "content_id": example["content_id"],
                        "prediction_text": prediction_text,
                        "model": model_id,
                        "backend": BACKEND_NAME,
                        "condition": "base",
                        "layer": "R",
                        "stage": "eval_text_baseline",
                        "source_name": example["source_name"],
                    },
                )
                finished[example_id] = {"example_id": example_id}
                processed += 1
                consecutive_failures = 0
                recent_outcomes.append(True)
                last_error = None
                if processed % TEACHER_PROGRESS_LOG_EVERY == 0:
                    print(
                        f"[run-r-eval-text-local-server] successes={processed} "
                        f"exhausted_failures={failed} "
                        f"remaining_in_batch={len(pending) - processed - failed}",
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
                    "status": "exhausted_retries",
                    "retry_count": max_retries,
                    "error": str(last_error),
                    "model": model_id,
                    "backend": BACKEND_NAME,
                },
            )
            abort_reason = should_abort_teacher_api_run(
                consecutive_failures=consecutive_failures,
                recent_outcomes=recent_outcomes,
                abort_consecutive_failures=TEACHER_ABORT_CONSECUTIVE_FAILURES,
                abort_window_size=TEACHER_ABORT_WINDOW_SIZE,
                abort_window_failure_rate=TEACHER_ABORT_WINDOW_FAILURE_RATE,
            )
            if abort_reason is not None:
                print(
                    f"[run-r-eval-text-local-server] ABORT: {abort_reason}",
                    flush=True,
                )
                break

        if request_interval_seconds > 0:
            time.sleep(request_interval_seconds)

    write_report(
        {
            "stage": "r_eval_text_local_server_base",
            "condition": "base",
            "backbone_slug": model_slug(model_id),
            "backend": BACKEND_NAME,
            "model": model_id,
            "examples_manifest": str(examples_file),
            "predictions_manifest": str(output_path),
            "failures_manifest": str(failures_file),
            "processed_count": processed,
            "already_done_count": already_done_count,
            "exhausted_failure_count": failed,
            "aborted": abort_reason is not None,
            "abort_reason": abort_reason,
            "requested_limit": limit,
            "resume": resume,
            "request_interval_seconds": request_interval_seconds,
            "score_on_complete": score_on_complete,
            "notes": [
                "Default: score-r-eval-text on dev machine after copying JSONL.",
            ],
        },
        report_path,
    )
    print(
        f"[run-r-eval-text-local-server] done new_successes={processed} "
        f"failures={failed} "
        f"total_predictions={len(load_prediction_index(output_path))}",
        flush=True,
    )
    if score_on_complete:
        score_r_eval_text_if_complete(
            examples=examples,
            predictions_path=output_path,
            aborted=abort_reason is not None,
            log_prefix="run-r-eval-text-local-server",
        )


def transfer_predictions_path(benchmark: str, model_name: str | None = None) -> Path:
    """Return the prediction JSONL path for one transfer local-server run."""

    slug = condition_slug_for_model(resolve_eval_model(model_name))
    return resolve_from_repo_root(
        f"Data/manifests/transfer_{benchmark}_predictions_{slug}.jsonl"
    )


def transfer_failures_path(benchmark: str, model_name: str | None = None) -> Path:
    """Return the failure JSONL path for one transfer local-server run."""

    slug = condition_slug_for_model(resolve_eval_model(model_name))
    return resolve_from_repo_root(
        f"Data/manifests/transfer_{benchmark}_failures_{slug}.jsonl"
    )


def transfer_run_report_path(benchmark: str, model_name: str | None = None) -> Path:
    """Return the run report path for one transfer local-server run."""

    slug = condition_slug_for_model(resolve_eval_model(model_name))
    return resolve_from_repo_root(
        f"Data/cards/transfer_{benchmark}_run_{slug}.json"
    )


def request_options_for_transfer_example(
    example: dict[str, Any],
    base_options: dict[str, Any],
) -> dict[str, Any]:
    """Choose generation limits for one Transfer example."""

    configured = int(base_options["max_tokens"])
    benchmark = str(example.get("benchmark") or "")
    if benchmark == "voicebench_sdqa":
        max_tokens = configured if configured < R_CONTENT_MAX_TOKENS else R_CONTENT_MAX_TOKENS
    elif benchmark.startswith("voicebench_"):
        max_tokens = min(configured, 256)
    else:
        max_tokens = min(configured, R_PARA_MAX_TOKENS)
    return {**base_options, "max_tokens": max_tokens}


def run_transfer_eval_local_server(
    *,
    benchmark: str,
    limit: int | None = None,
    resume: bool = True,
    request_interval_seconds: float = 0.0,
    max_retries: int = 3,
    model: str | None = None,
) -> None:
    """Fill local-server predictions for one frozen Transfer benchmark."""

    from realmg.eval.transfer_eval import TRANSFER_BENCHMARKS, transfer_examples_path

    if benchmark == "all":
        for one_benchmark in TRANSFER_BENCHMARKS:
            run_transfer_eval_local_server(
                benchmark=one_benchmark,
                limit=limit,
                resume=resume,
                request_interval_seconds=request_interval_seconds,
                max_retries=max_retries,
                model=model,
            )
        return

    if benchmark not in TRANSFER_BENCHMARKS:
        raise ValueError(
            f"Unsupported benchmark {benchmark!r}. Expected one of {TRANSFER_BENCHMARKS}."
        )

    model_id = resolve_eval_model(model)
    examples_file = transfer_examples_path(benchmark)
    output_path = transfer_predictions_path(benchmark, model_id)
    failures_file = transfer_failures_path(benchmark, model_id)
    report_path = transfer_run_report_path(benchmark, model_id)

    examples = read_jsonl(examples_file)
    if resume:
        rewrite_successful_predictions(output_path)
    finished = load_prediction_index(output_path) if resume else {}
    already_done_count = len(finished)
    pending = list_pending_eval_examples(examples, finished, limit=limit)

    client = build_local_server_client()
    base_options = default_local_server_request_options()

    processed = 0
    failed = 0
    consecutive_failures = 0
    recent_outcomes: list[bool] = []
    abort_reason: str | None = None

    print(
        f"[run-transfer-eval-local-server] benchmark={benchmark} "
        f"pending={len(pending)} already_done={already_done_count} model={model_id}",
        flush=True,
    )

    for example in pending:
        example_id = example["example_id"]
        audio_path = resolve_from_repo_root(example["audio_path"])
        request_options = request_options_for_transfer_example(example, base_options)
        last_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                prediction_text = call_local_server_audio_text_api(
                    client,
                    audio_path=audio_path,
                    prompt_text=str(example.get("prompt") or ""),
                    request_options=request_options,
                    model=model_id,
                )
                append_prediction_row(
                    output_path,
                    {
                        "example_id": example_id,
                        "prediction_text": prediction_text,
                        "model": model_id,
                        "backend": BACKEND_NAME,
                        "condition": "base",
                        "layer": "transfer",
                        "benchmark": benchmark,
                        "audio_path": example["audio_path"],
                    },
                )
                finished[example_id] = {"example_id": example_id}
                processed += 1
                consecutive_failures = 0
                recent_outcomes.append(True)
                last_error = None
                if processed % TEACHER_PROGRESS_LOG_EVERY == 0:
                    print(
                        f"[run-transfer-eval-local-server] benchmark={benchmark} "
                        f"successes={processed} exhausted_failures={failed} "
                        f"remaining_in_batch={len(pending) - processed - failed}",
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
                    "status": "exhausted_retries",
                    "retry_count": max_retries,
                    "error": str(last_error),
                    "model": model_id,
                    "backend": BACKEND_NAME,
                    "benchmark": benchmark,
                    "audio_path": example["audio_path"],
                },
            )
            abort_reason = should_abort_teacher_api_run(
                consecutive_failures=consecutive_failures,
                recent_outcomes=recent_outcomes,
                abort_consecutive_failures=TEACHER_ABORT_CONSECUTIVE_FAILURES,
                abort_window_size=TEACHER_ABORT_WINDOW_SIZE,
                abort_window_failure_rate=TEACHER_ABORT_WINDOW_FAILURE_RATE,
            )
            if abort_reason is not None:
                print(
                    f"[run-transfer-eval-local-server] ABORT: {abort_reason}",
                    flush=True,
                )
                break

        if request_interval_seconds > 0:
            time.sleep(request_interval_seconds)

    write_report(
        {
            "stage": "transfer_eval_local_server_base",
            "benchmark": benchmark,
            "condition": "base",
            "backbone_slug": model_slug(model_id),
            "backend": BACKEND_NAME,
            "model": model_id,
            "examples_manifest": str(examples_file),
            "predictions_manifest": str(output_path),
            "failures_manifest": str(failures_file),
            "processed_count": processed,
            "already_done_count": already_done_count,
            "exhausted_failure_count": failed,
            "aborted": abort_reason is not None,
            "abort_reason": abort_reason,
            "requested_limit": limit,
            "resume": resume,
            "request_interval_seconds": request_interval_seconds,
            "notes": [
                "Score with: realmg score-transfer-eval --predictions-file <jsonl>",
            ],
        },
        report_path,
    )
    print(
        f"[run-transfer-eval-local-server] done benchmark={benchmark} "
        f"new_successes={processed} failures={failed} "
        f"total_predictions={len(load_prediction_index(output_path))}",
        flush=True,
    )


def default_smoke_example() -> dict[str, Any]:
    """Return one frozen P paper eval content example for connectivity smoke tests."""

    examples = read_jsonl(eval_examples_path("paper"))
    for example in examples:
        if example.get("query_type") == "content":
            return example
    raise RuntimeError("No P paper content example found for smoke test.")


def smoke_local_server_audio(*, model: str | None = None) -> dict[str, Any]:
    """Run one frozen paper P eval audio query against the local inference server."""

    model_id = resolve_eval_model(model)
    example = default_smoke_example()
    audio_path = resolve_from_repo_root(example["audio_path"])
    client = build_local_server_client()
    request_options = default_local_server_request_options()

    prediction_text = call_local_server_audio_text_api(
        client,
        audio_path=audio_path,
        prompt_text=str(example.get("prompt", "")),
        request_options=request_options,
        model=model_id,
    )

    report = {
        "stage": "local_server_audio_smoke",
        "backend": BACKEND_SLUG,
        "model": model_id,
        "example_id": example["example_id"],
        "audio_path": example["audio_path"],
        "query_type": example.get("query_type"),
        "gold_choice": example.get("target", {}).get("choice"),
        "prediction_text": prediction_text,
        "request_options": request_options,
    }
    output_path = resolve_from_repo_root(
        f"Data/cards/local_server_smoke_{model_slug(model_id)}.json"
    )
    write_report(report, output_path)
    print(
        "[smoke-local-server-audio] "
        f"model={model_id} example_id={example['example_id']} "
        f"prediction={prediction_text!r} report={output_path}",
        flush=True,
    )
    return report
