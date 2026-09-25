"""Phi-4-MM local Teacher-text generate for carved-train screening.

"""

from __future__ import annotations

import time
from typing import Any

import torch

from realmg.data.esd import read_jsonl, write_report
from realmg.data.r_teacher_text_carved import (
    BACKBONE_LABEL,
    BACKBONE_SLUG,
    CARVED_TRAIN_RUN_REPORT_PATH,
    carved_train_examples_path,
    carved_train_failures_path,
    carved_train_predictions_path,
)
from realmg.data.r_text_teacher import (
    TEACHER_PROGRESS_LOG_EVERY,
    append_prediction_row,
    load_prediction_index,
    rewrite_successful_predictions,
)
from realmg.data.config import resolve_from_repo_root
from realmg.train.ours_model import load_phi4_processor, load_phi4_with_speech_merged
from realmg.train.phi4_io import INPUT_MODE_LANGUAGE, phi4_user_prompt


DEFAULT_MODEL_DIR = "microsoft/Phi-4-multimodal-instruct"
MAX_NEW_TOKENS = 256


def _generate_one_text(
    model: Any,
    processor: Any,
    prompt_text: str,
    *,
    max_new_tokens: int,
    pad_token_id: int,
) -> str:
    """Greedy-decode one text-only Teacher-text answer."""

    prompt = phi4_user_prompt(user_text=prompt_text, has_audio=False)
    encoded = processor(text=prompt, return_tensors="pt")
    model_device = next(model.parameters()).device
    encoded = {
        key: value.to(model_device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }
    encoded["input_mode"] = torch.tensor(
        [INPUT_MODE_LANGUAGE],
        dtype=torch.long,
        device=model_device,
    )
    prompt_len = int(encoded["input_ids"].shape[1])
    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            use_cache=False,
        )
    response_ids = output[0, prompt_len:]
    text = processor.tokenizer.decode(
        response_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return text.strip()


def run_phi4_teacher_text_carved_train(
    *,
    model_dir: str = DEFAULT_MODEL_DIR,
    limit: int | None = None,
    resume: bool = True,
    shard_index: int = 0,
    num_shards: int = 1,
) -> None:
    """Fill Phi-4 Teacher-text predictions for carved-train unique s.

    position so two GPU processes can share one predictions JSONL without
    overlapping ``example_id``. When ``num_shards > 1``, the caller must rewrite
    successful predictions once before launching shards (avoids dual-writer races).
    """

    if not torch.cuda.is_available():
        raise RuntimeError("Phi-4 Teacher-text generate requires CUDA.")
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(
            f"shard_index must be in [0, {num_shards}), got {shard_index}"
        )
    examples = read_jsonl(carved_train_examples_path())
    predictions_path = carved_train_predictions_path()
    failures_path = carved_train_failures_path()
    if resume and num_shards == 1:
        rewrite_successful_predictions(predictions_path)
    finished = load_prediction_index(predictions_path) if resume else {}
    pending = [row for row in examples if row["example_id"] not in finished]
    if limit is not None:
        pending = pending[:limit]
    if num_shards > 1:
        pending = [
            row
            for index, row in enumerate(pending)
            if index % num_shards == shard_index
        ]
    print(
        f"[phi4-teacher-text] pending={len(pending)} already={len(finished)} "
        f"backbone={BACKBONE_SLUG} shard={shard_index}/{num_shards}",
        flush=True,
    )
    if not pending:
        return

    device = torch.device("cuda")
    processor = load_phi4_processor(model_dir)
    model = load_phi4_with_speech_merged(model_dir, dtype=torch.bfloat16)
    model.to(device)
    model.eval()
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_token_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    t0 = time.time()
    done = 0
    failed = 0
    for example in pending:
        example_id = example["example_id"]
        try:
            prediction_text = _generate_one_text(
                model,
                processor,
                example["prompt_text"],
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=pad_token_id,
            )
            if not prediction_text:
                raise RuntimeError("empty_prediction_text")
            append_prediction_row(
                predictions_path,
                {
                    "example_id": example_id,
                    "content_id": example["content_id"],
                    "source_name": example["source_name"],
                    "prediction_text": prediction_text,
                    "teacher_backend": "local_phi4_mm",
                    "teacher_model": BACKBONE_LABEL,
                },
            )
            done += 1
        except Exception as exc:  # noqa: BLE001 - keep the run going
            if failed == 0:
                import traceback

                traceback.print_exc()
            failed += 1
            append_prediction_row(
                failures_path,
                {
                    "example_id": example_id,
                    "content_id": example["content_id"],
                    "source_name": example["source_name"],
                    "status": "exhausted_retries",
                    "error": str(exc),
                    "teacher_backend": "local_phi4_mm",
                },
            )
        if (done + failed) % TEACHER_PROGRESS_LOG_EVERY == 0:
            print(
                f"[phi4-teacher-text] shard={shard_index}/{num_shards} "
                f"done={done} failed={failed} elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
    report_path = resolve_from_repo_root(CARVED_TRAIN_RUN_REPORT_PATH)
    if num_shards > 1:
        report_path = report_path.with_name(
            f"{report_path.stem}_shard{shard_index}{report_path.suffix}"
        )
    write_report(
        {
            "stage": "teacher_text_carved_train_phi4",
            "backbone_slug": BACKBONE_SLUG,
            "shard_index": shard_index,
            "num_shards": num_shards,
            "successes": done,
            "failures": failed,
            "elapsed_sec": round(time.time() - t0, 1),
            "predictions": str(predictions_path),
        },
        report_path,
    )
    print(
        f"[phi4-teacher-text] finished shard={shard_index}/{num_shards} "
        f"successes={done} failures={failed}",
        flush=True,
    )
