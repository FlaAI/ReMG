"""
Ours training loop: Audio→Text reverse KL + paralinguistic FKL anchoring.

"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from realmg.data.esd import write_report
from realmg.eval.q_para_prompts import (
    EVAL_STEM,
    MMSU_EVAL_STEM,
    ONE_LABEL_INSTRUCTION,
    P_EMOTION_LABELS,
    R_EMOTION_LABELS,
    R_PARA_TEACHER_TASK_PRIVILEGE,
    TRAIN_STEMS,
    assert_task_privilege_has_no_closed_set_label,
)
from realmg.train.omni_io import (
    build_audio_user_conversation,
    build_text_user_conversation,
    encode_conversations,
    generate_response_ids,
    logprobs_from_generate_scores,
    pad_token_id_from_processor,
    response_lengths,
    response_mask_from_ids,
    response_step_logits,
    response_token_logprobs,
    response_topk_teacher_student_fkl,
    stack_generate_logits,
)
from realmg.train.ours_data import (
    OursTrainSample,
    SqrtMixSampler,
    load_p_train_rows,
    load_r_train_rows,
    r_train_utterances_rel,
    summarize_train_pools,
)
from realmg.train.ours_model import BACKBONE_PHI4_MM, BACKBONE_QWEN25_OMNI
from realmg.train.ours_losses import (
    clipped_reverse_kl_ppo_loss,
    reverse_kl_warmup_scale,
    three_arm_loss_weights,
)
from realmg.train.ours_prefetch import OneAheadPrefetcher
from realmg.train.r_cnt_teacher_privilege import R_CNT_TEACHER_TASK_PRIVILEGE
from realmg.train.para_targets import (
    truncate_para_response_ids_to_first_line_closed_label,
)


DEFAULT_MODEL_DIR = "Qwen/Qwen2.5-Omni-7B"
DEFAULT_OUTPUT_DIR = "Data/ckpts/ours_qwen25_omni_7b_v5_brief"
DEFAULT_VANILLA_OPD_OUTPUT_DIR = (
    "Data/ckpts/ours_qwen25_omni_7b_vanilla_opd"
)
VANILLA_OPD_OUTPUT_DIR_NAME = "ours_qwen25_omni_7b_vanilla_opd"
DEFAULT_PHI4_MODEL_DIR = "microsoft/Phi-4-multimodal-instruct"
DEFAULT_PHI4_OUTPUT_DIR = "Data/ckpts/ours_phi4_mm_v5_brief"
DEFAULT_PHI4_R_TRAIN = (
    "Data/manifests/r_carved_train_utterances_teacher_correct_phi4_mm.jsonl"
)
LOCKED_MICRO_BATCH_SIZE = 4
LOCKED_GRAD_ACCUM = 32
FROZEN_OUTPUT_DIR_NAMES = {
    "ours_qwen25_omni_7b_v2_maxmicro": "",
    "ours_qwen25_omni_7b_v3_qc": "",
    "ours_qwen25_omni_7b_v4_priv": "",
    "ours_qwen25_omni_7b_v5_brief": "",
    "ours_qwen25_omni_7b_v6_textkd": "",
    "ckd_kc_qwen25_omni_7b": "",
    "cord_kc_qwen25_omni_7b": "",
    "xopd_kc_qwen25_omni_7b": "",
    "tars_kc_qwen25_omni_7b": "",
}


@dataclass
class PreparedMicrobatch:
    """CPU/GPU-ready micro-batch. R fills cnt+para encodings. P fills para.

    prompt = student public para. teacher_para = privileged teacher encoding
    on R. None on P means the teacher uses the same encoding as prompt.
    """

    layer: str
    samples: list[OursTrainSample]
    student_prompt: dict[str, Any] | None = None
    teacher_prompt: dict[str, Any] | None = None
    prompt: dict[str, Any] | None = None
    teacher_para: dict[str, Any] | None = None


def tokenizer_from_processor(processor: Any) -> Any:
    """Return the tokenizer attached to an Omni processor."""

    return getattr(processor, "tokenizer", processor)


def para_decode_stats(texts: list[str], labels: Sequence[str]) -> dict[str, int]:
    """Coarse para-format probes: exact label, happy mention, long non-label."""

    allowed = {label.lower() for label in labels}
    exact = 0
    happy = 0
    long_nonlabel = 0
    for text in texts:
        lowered = text.lower()
        first = ""
        if text.strip():
            first = text.strip().splitlines()[0].strip().strip(".,!").lower()
        if first in allowed:
            exact += 1
        if "happy" in lowered:
            happy += 1
        if first not in allowed and len(text.split()) >= 8:
            long_nonlabel += 1
    return {
        "n": len(texts),
        "exact_one_label": exact,
        "mentions_happy": happy,
        "long_nonlabel": long_nonlabel,
    }


def compute_loss1_batch(
    student: Any,
    prepared: PreparedMicrobatch,
    *,
    max_new_tokens: int,
    clip_eps: float,
    pad_token_id: int,
) -> tuple[Tensor, list[int]]:
    """On-policy Audio→Text reverse KL for a homogeneous R micro-batch.

    Reuses generate scores as logprob_old (skips one teacher-forcing forward).
    """

    if prepared.student_prompt is None or prepared.teacher_prompt is None:
        raise ValueError("R micro-batch missing encoded cnt prompts.")
    student_prompt = prepared.student_prompt
    teacher_prompt = prepared.teacher_prompt

    was_training = student.training
    student.eval()
    with torch.no_grad():
        response_ids, scores = generate_response_ids(
            student,
            student_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=pad_token_id,
            return_scores=True,
        )
        assert isinstance(response_ids, Tensor)
        response_mask = response_mask_from_ids(response_ids, pad_token_id=pad_token_id)
        lengths = response_lengths(response_mask)
        logprob_old = logprobs_from_generate_scores(scores, response_ids)
        with student.disable_adapter():
            logprob_teacher = response_token_logprobs(
                student, teacher_prompt, response_ids, response_mask=response_mask
            )
    if was_training:
        student.train()
    logprob_new = response_token_logprobs(
        student, student_prompt, response_ids, response_mask=response_mask
    )
    loss = clipped_reverse_kl_ppo_loss(
        logprob_new,
        logprob_old,
        logprob_teacher,
        clip_eps=clip_eps,
        response_mask=response_mask,
    )
    return loss, lengths


def compute_loss2_batch(
    student: Any,
    prepared: PreparedMicrobatch,
    *,
    max_new_tokens: int,
    top_k: int,
    pad_token_id: int,
    tokenizer: Any,
) -> tuple[Tensor, list[int], list[str]]:
    """Offline forward KL on teacher para, forced onto the student public prompt.

    Teacher generate uses teacher_para (R: privileged, P: same as student).
    Trajectories are cut to the first-line closed-set label. Rows with no
    first-line label are dropped from the FKL denominator.
    """

    if prepared.prompt is None:
        raise ValueError("Micro-batch missing encoded student para prompt.")
    student_prompt = prepared.prompt
    teacher_prompt = prepared.teacher_para
    if teacher_prompt is None:
        teacher_prompt = student_prompt
    labels = R_EMOTION_LABELS if prepared.layer == "R" else P_EMOTION_LABELS

    was_training = student.training
    student.eval()
    with torch.no_grad():
        with student.disable_adapter():
            response_ids, scores = generate_response_ids(
                student,
                teacher_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
                return_scores=True,
            )
            assert isinstance(response_ids, Tensor)
            if response_ids.shape[1] == 0:
                teacher_logits = student_prompt["input_ids"].new_zeros(
                    (response_ids.shape[0], 0, 1)
                )
            else:
                teacher_logits = stack_generate_logits(scores)
            (
                response_ids,
                teacher_logits,
                response_mask,
                lengths,
                kept_flags,
                texts,
            ) = truncate_para_response_ids_to_first_line_closed_label(
                tokenizer,
                response_ids,
                teacher_logits,
                labels=labels,
                pad_token_id=pad_token_id,
            )
    if was_training:
        student.train()
    n_kept = sum(1 for kept in kept_flags if kept)
    if n_kept == 0:
        zero = torch.zeros(
            (),
            device=student_prompt["input_ids"].device,
            dtype=torch.float32,
        )
        return zero, lengths, texts
    student_logits = response_step_logits(student, student_prompt, response_ids)
    loss = response_topk_teacher_student_fkl(
        teacher_logits.detach(),
        student_logits,
        top_k=top_k,
        response_mask=response_mask,
    )
    return loss, lengths, texts


def _move_batch_to_device(
    batch: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Move encoded tensors to the training device/dtype."""

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            tensor = value.to(device, non_blocking=True)
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            moved[key] = tensor
        else:
            moved[key] = value
    return moved


def prepare_microbatch_on_cpu(
    samples: list[OursTrainSample],
    processor: Any,
) -> PreparedMicrobatch:
    """Encode a homogeneous micro-batch on CPU for prefetch."""

    if not samples:
        raise ValueError("Empty micro-batch.")
    layer = samples[0].layer
    if any(sample.layer != layer for sample in samples):
        raise ValueError("Micro-batch must be same-layer for padded batching.")
    cpu = torch.device("cpu")
    student_para_convs = [
        build_audio_user_conversation(sample.audio_path, sample.para_prompt)
        for sample in samples
    ]
    prompt = encode_conversations(processor, student_para_convs, device=cpu)
    teacher_para = None
    if any(
        sample.teacher_para_prompt != sample.para_prompt for sample in samples
    ):
        teacher_para_convs = [
            build_audio_user_conversation(
                sample.audio_path, sample.teacher_para_prompt
            )
            for sample in samples
        ]
        teacher_para = encode_conversations(
            processor, teacher_para_convs, device=cpu
        )
    if layer != "R":
        return PreparedMicrobatch(
            layer=layer,
            samples=samples,
            prompt=prompt,
            teacher_para=teacher_para,
        )
    student_convs = [
        build_audio_user_conversation(sample.audio_path, "") for sample in samples
    ]
    teacher_convs = [
        build_text_user_conversation(sample.teacher_text_prompt)
        for sample in samples
    ]
    return PreparedMicrobatch(
        layer="R",
        samples=samples,
        student_prompt=encode_conversations(processor, student_convs, device=cpu),
        teacher_prompt=encode_conversations(processor, teacher_convs, device=cpu),
        prompt=prompt,
        teacher_para=teacher_para,
    )


def materialize_microbatch_on_device(
    prepared: PreparedMicrobatch,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> PreparedMicrobatch:
    """Copy prefetched CPU tensors onto CUDA."""

    return PreparedMicrobatch(
        layer=prepared.layer,
        samples=prepared.samples,
        student_prompt=(
            None
            if prepared.student_prompt is None
            else _move_batch_to_device(
                prepared.student_prompt, device=device, dtype=dtype
            )
        ),
        teacher_prompt=(
            None
            if prepared.teacher_prompt is None
            else _move_batch_to_device(
                prepared.teacher_prompt, device=device, dtype=dtype
            )
        ),
        prompt=(
            None
            if prepared.prompt is None
            else _move_batch_to_device(prepared.prompt, device=device, dtype=dtype)
        ),
        teacher_para=(
            None
            if prepared.teacher_para is None
            else _move_batch_to_device(
                prepared.teacher_para, device=device, dtype=dtype
            )
        ),
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON object as a line."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _refuse_frozen_checkpoint_overwrite(output_dir: Path) -> None:
    """Keep completed runs as frozen artifacts."""

    if output_dir.name in FROZEN_OUTPUT_DIR_NAMES:
        raise ValueError(
            f"Refusing to write into frozen run {output_dir.name}."
        )


def _refuse_ablation_overwrite(output_dir: Path) -> None:
    _refuse_frozen_checkpoint_overwrite(output_dir)


def _refuse_vanilla_opd_output_collision(
    output_dir: Path, *, vanilla_opd: bool
) -> None:
    """Isolate vanilla_opd writes from v5 / frozen dirs and the reverse."""

    name = output_dir.name
    if vanilla_opd:
        if name != VANILLA_OPD_OUTPUT_DIR_NAME:
            raise ValueError(
                f"--vanilla-opd requires output dir named "
                f"{VANILLA_OPD_OUTPUT_DIR_NAME}, got {name}."
            )
        return
    if name == VANILLA_OPD_OUTPUT_DIR_NAME:
        raise ValueError(
            "Refusing to write ours_qwen25_omni_7b_vanilla_opd without "
            "--vanilla-opd."
        )


def _fmt_opt(value: float | None, *, digits: int) -> str:
    """Format an optional float for the step log line."""

    if value is None:
        return "na"
    return f"{value:.{digits}f}"


def _log_micro_rows(
    *,
    log_path: Path,
    length_stats_path: Path,
    epoch: int,
    optimizer_step: int,
    prepared: PreparedMicrobatch,
    slot_loss: float,
    loss1_value: float | None,
    loss2_value: float,
    lengths_cnt: list[int],
    lengths_para: list[int],
    para_texts: list[str],
    micro_batch_size: int,
    max_new_tokens_r: int,
    max_new_tokens_p: int,
) -> None:
    """Write per-utterance train log and response-length rows."""

    for index, sample in enumerate(prepared.samples):
        _append_jsonl(
            log_path,
            {
                "epoch": epoch,
                "optimizer_step": optimizer_step,
                "layer": sample.layer,
                "utterance_id": sample.utterance_id,
                "slot_loss": slot_loss,
                "loss1": loss1_value,
                "loss2": loss2_value,
                "response_tokens_cnt": (
                    lengths_cnt[index] if prepared.layer == "R" else None
                ),
                "response_tokens_para": lengths_para[index],
                "para_text": para_texts[index],
                "para_skipped": lengths_para[index] == 0,
                "micro_batch_size": micro_batch_size,
            },
        )
        if prepared.layer == "R":
            _append_jsonl(
                length_stats_path,
                {
                    "layer": "R",
                    "arm": "r_cnt",
                    "response_tokens": lengths_cnt[index],
                    "max_new_tokens": max_new_tokens_r,
                    "optimizer_step": optimizer_step,
                },
            )
        _append_jsonl(
            length_stats_path,
            {
                "layer": sample.layer,
                "arm": "r_para" if sample.layer == "R" else "p_para",
                "response_tokens": lengths_para[index],
                "max_new_tokens": max_new_tokens_p,
                "optimizer_step": optimizer_step,
            },
        )


def run_ours_training(
    *,
    model_dir: Path,
    output_dir: Path,
    epochs: int = 1,
    micro_batch_size: int = LOCKED_MICRO_BATCH_SIZE,
    grad_accum: int = LOCKED_GRAD_ACCUM,
    learning_rate: float = 1e-4,
    seed: int = 42,
    max_new_tokens_r: int = 512,
    max_new_tokens_p: int = 64,
    clip_eps: float = 0.2,
    fkl_top_k: int = 16,
    max_optimizer_steps: int | None = None,
    prefetch: bool = True,
    backbone: str = BACKBONE_QWEN25_OMNI,
    vanilla_opd: bool = False,
) -> dict[str, Any]:
    """Run Ours loop and save epoch-end LoRA adapters."""

    if vanilla_opd and backbone != BACKBONE_QWEN25_OMNI:
        raise ValueError("vanilla_opd ablation is Omni-only.")
    _refuse_vanilla_opd_output_collision(output_dir, vanilla_opd=vanilla_opd)
    _refuse_ablation_overwrite(output_dir)
    if max_new_tokens_r != 512:
        raise ValueError("v5_brief R-cnt cap is locked to 512 to match frozen eval.")
    if micro_batch_size < 1:
        raise ValueError("micro_batch_size must be >= 1.")
    if grad_accum < 1:
        raise ValueError("grad_accum must be >= 1.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Ours training requires CUDA.")

    from realmg.train.ours_model import build_ours_student, save_lora_checkpoint

    r_rows = load_r_train_rows()
    p_rows = load_p_train_rows()
    sampler = SqrtMixSampler(r_rows, p_rows, seed=seed)
    arm_weights = three_arm_loss_weights(sampler.prob_r)
    weight_r_cnt = arm_weights["r_cnt"]
    weight_r_para = arm_weights["r_para"]
    weight_p_para = arm_weights["p_para"]
    samples_per_opt_step = micro_batch_size * grad_accum
    steps_per_epoch = math.ceil((len(r_rows) + len(p_rows)) / samples_per_opt_step)
    total_optimizer_steps = steps_per_epoch * epochs
    if max_optimizer_steps is not None:
        total_optimizer_steps = min(total_optimizer_steps, max_optimizer_steps)

    processor, student = build_ours_student(
        model_dir, device=device, backbone=backbone
    )
    tokenizer = tokenizer_from_processor(processor)
    dtype = next(student.parameters()).dtype
    pad_token_id = pad_token_id_from_processor(processor)
    optimizer = AdamW(
        (p for p in student.parameters() if p.requires_grad),
        lr=learning_rate,
        weight_decay=0.0,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.03 * total_optimizer_steps)),
        num_training_steps=total_optimizer_steps,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    length_stats_path = output_dir / "response_length_stats.jsonl"
    metrics: list[dict[str, Any]] = []
    optimizer_step = 0
    t0 = time.time()

    def _prepare_next() -> PreparedMicrobatch:
        samples = sampler.sample_homogeneous_microbatch(micro_batch_size)
        # Vanilla: still consume sampler RNG on P (incl. build_*_sample), but
        # skip expensive encode — trajectory of later R draws stays aligned.
        if vanilla_opd and samples[0].layer == "P":
            return PreparedMicrobatch(layer="P", samples=samples)
        return prepare_microbatch_on_cpu(samples, processor)

    prefetcher: OneAheadPrefetcher[PreparedMicrobatch] | None = None
    if prefetch:
        prefetcher = OneAheadPrefetcher(_prepare_next)

    student.train()
    optimizer.zero_grad(set_to_none=True)
    if vanilla_opd:
        print(
            "[ours] vanilla_opd: L1 only (concise privilege kept); "
            "P slots advance sampler then skip loss; "
            f"weight_r_cnt={weight_r_cnt:.4f} prob_r={sampler.prob_r:.4f} "
            f"seed={seed}",
            flush=True,
        )
    else:
        print(
            "[ours] 3-arm weights "
            f"r_cnt={weight_r_cnt:.4f} r_para={weight_r_para:.4f} "
            f"p_para={weight_p_para:.4f} "
            f"E[mass]={arm_weights['expected_mass_r_cnt']:.3f}:"
            f"{arm_weights['expected_mass_r_para']:.3f}:"
            f"{arm_weights['expected_mass_p_para']:.3f} "
            f"(cnt:para={arm_weights['expected_cnt_para_ratio']:.3f}) "
            f"prob_r={sampler.prob_r:.4f}",
            flush=True,
        )

    try:
        for epoch in range(epochs):
            epoch_loss_sum = 0.0
            for _ in range(steps_per_epoch):
                if optimizer_step >= total_optimizer_steps:
                    break
                slot_losses: list[float] = []
                step_loss1: list[float] = []
                step_loss2: list[float] = []
                step_len_cnt: list[int] = []
                step_len_para: list[int] = []
                step_hit_cap = 0
                step_para_n = 0
                step_exact = 0
                step_happy = 0
                step_leak = 0
                step_para_skip = 0
                n_r_slots = 0
                n_p_slots = 0
                for _acc in range(grad_accum):
                    prepared_cpu = (
                        prefetcher.next() if prefetcher is not None else _prepare_next()
                    )
                    if vanilla_opd and prepared_cpu.layer == "P":
                        n_p_slots += 1
                        slot_losses.append(0.0)
                        continue
                    prepared = materialize_microbatch_on_device(
                        prepared_cpu, device=device, dtype=dtype
                    )
                    lengths_cnt: list[int] = []
                    loss1_value: float | None = None
                    if prepared.layer == "R":
                        n_r_slots += 1
                        loss1, lengths_cnt = compute_loss1_batch(
                            student,
                            prepared,
                            max_new_tokens=max_new_tokens_r,
                            clip_eps=clip_eps,
                            pad_token_id=pad_token_id,
                        )
                        loss1 = loss1 * reverse_kl_warmup_scale(
                            optimizer_step, total_optimizer_steps
                        )
                        loss1_value = float(loss1.detach().item())
                        step_loss1.append(loss1_value)
                        step_len_cnt.extend(lengths_cnt)
                        ((loss1 * weight_r_cnt) / grad_accum).backward()
                        del loss1
                    else:
                        n_p_slots += 1

                    if vanilla_opd:
                        # R-only L1; no L2 / no R-para privilege path.
                        if loss1_value is None:
                            raise RuntimeError("vanilla_opd R slot missing loss1.")
                        slot_loss = weight_r_cnt * loss1_value
                        slot_losses.append(slot_loss)
                        empty_para = [0] * len(prepared.samples)
                        empty_texts = [""] * len(prepared.samples)
                        _log_micro_rows(
                            log_path=log_path,
                            length_stats_path=length_stats_path,
                            epoch=epoch,
                            optimizer_step=optimizer_step,
                            prepared=prepared,
                            slot_loss=slot_loss,
                            loss1_value=loss1_value,
                            loss2_value=0.0,
                            lengths_cnt=lengths_cnt,
                            lengths_para=empty_para,
                            para_texts=empty_texts,
                            micro_batch_size=micro_batch_size,
                            max_new_tokens_r=max_new_tokens_r,
                            max_new_tokens_p=max_new_tokens_p,
                        )
                        continue

                    loss2, lengths_para, para_texts = compute_loss2_batch(
                        student,
                        prepared,
                        max_new_tokens=max_new_tokens_p,
                        top_k=fkl_top_k,
                        pad_token_id=pad_token_id,
                        tokenizer=tokenizer,
                    )
                    loss2_value = float(loss2.detach().item())
                    n_kept = sum(1 for length in lengths_para if length > 0)
                    step_para_skip += len(lengths_para) - n_kept
                    step_loss2.append(loss2_value)
                    step_len_para.extend(lengths_para)
                    labels = (
                        R_EMOTION_LABELS if prepared.layer == "R" else P_EMOTION_LABELS
                    )
                    decode_stats = para_decode_stats(para_texts, labels)
                    step_para_n += decode_stats["n"]
                    step_exact += decode_stats["exact_one_label"]
                    step_happy += decode_stats["mentions_happy"]
                    step_leak += decode_stats["long_nonlabel"]
                    step_hit_cap += sum(
                        1 for length in lengths_para if length >= max_new_tokens_p
                    )
                    if prepared.layer == "R":
                        if loss1_value is None:
                            raise RuntimeError("R slot missing loss1.")
                        if n_kept > 0:
                            ((loss2 * weight_r_para) / grad_accum).backward()
                        slot_loss = (
                            weight_r_cnt * loss1_value
                            + (weight_r_para * loss2_value if n_kept > 0 else 0.0)
                        )
                    else:
                        if n_kept > 0:
                            ((loss2 * weight_p_para) / grad_accum).backward()
                        slot_loss = (
                            weight_p_para * loss2_value if n_kept > 0 else 0.0
                        )
                    del loss2
                    slot_losses.append(slot_loss)
                    _log_micro_rows(
                        log_path=log_path,
                        length_stats_path=length_stats_path,
                        epoch=epoch,
                        optimizer_step=optimizer_step,
                        prepared=prepared,
                        slot_loss=slot_loss,
                        loss1_value=loss1_value,
                        loss2_value=loss2_value,
                        lengths_cnt=lengths_cnt,
                        lengths_para=lengths_para,
                        para_texts=para_texts,
                        micro_batch_size=micro_batch_size,
                        max_new_tokens_r=max_new_tokens_r,
                        max_new_tokens_p=max_new_tokens_p,
                    )

                torch.nn.utils.clip_grad_norm_(
                    (p for p in student.parameters() if p.requires_grad),
                    max_norm=1.0,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                mean_loss = sum(slot_losses) / max(1, len(slot_losses))
                epoch_loss_sum += mean_loss
                mean_loss1 = (
                    sum(step_loss1) / len(step_loss1) if step_loss1 else None
                )
                mean_loss2 = (
                    sum(step_loss2) / len(step_loss2) if step_loss2 else None
                )
                mean_len_cnt = (
                    sum(step_len_cnt) / len(step_len_cnt) if step_len_cnt else None
                )
                mean_len_para = (
                    sum(step_len_para) / len(step_len_para) if step_len_para else None
                )
                hit_cap_rate = (
                    step_hit_cap / max(1, len(step_len_para))
                )
                exact_rate = step_exact / max(1, step_para_n)
                mass_r_cnt = n_r_slots * weight_r_cnt
                mass_r_para = n_r_slots * weight_r_para
                mass_p_para = n_p_slots * weight_p_para
                print(
                    f"[ours] epoch={epoch} step={optimizer_step}/"
                    f"{total_optimizer_steps} loss={mean_loss:.4f} "
                    f"loss1={_fmt_opt(mean_loss1, digits=4)} "
                    f"loss2={_fmt_opt(mean_loss2, digits=4)} "
                    f"len_cnt={_fmt_opt(mean_len_cnt, digits=1)} "
                    f"len_para={_fmt_opt(mean_len_para, digits=1)} "
                    f"para_hit_cap={hit_cap_rate:.2f} "
                    f"para_exact={exact_rate:.2f} "
                    f"happy={step_happy}/{max(1, step_para_n)} "
                    f"leakish={step_leak}/{max(1, step_para_n)} "
                    f"para_skip={step_para_skip}/{max(1, step_para_n)} "
                    f"slots_R={n_r_slots} slots_P={n_p_slots} "
                    f"mass={mass_r_cnt:.1f}:{mass_r_para:.1f}:{mass_p_para:.1f} "
                    f"micro={micro_batch_size} accum={grad_accum}",
                    flush=True,
                )

            ckpt_dir = output_dir / f"epoch_{epoch}"
            save_lora_checkpoint(student, ckpt_dir)
            metrics.append(
                {
                    "epoch": epoch,
                    "optimizer_steps": optimizer_step,
                    "mean_loss": epoch_loss_sum / max(1, steps_per_epoch),
                    "checkpoint": str(ckpt_dir),
                }
            )
            if optimizer_step >= total_optimizer_steps:
                break
    finally:
        if prefetcher is not None:
            prefetcher.close()

    summary = {
        "status": "completed",
        "device": str(device),
        "model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "recipe": "vanilla_opd" if vanilla_opd else "v5_brief_cnt_teacher",
        "vanilla_opd": vanilla_opd,
        "practical_arms": ["R-cnt"] if vanilla_opd else ["R-cnt", "R-para", "P-para"],
        "skipped_arm": "P-cnt+all-L2" if vanilla_opd else "P-cnt",
        "arm_loss_target_ratio": "L1-only" if vanilla_opd else "2:1:1",
        "arm_loss_weights": arm_weights,
        "epochs": epochs,
        "micro_batch_size": micro_batch_size,
        "grad_accum": grad_accum,
        "effective_batch_utterances": samples_per_opt_step,
        "max_new_tokens_r": max_new_tokens_r,
        "max_new_tokens_p": max_new_tokens_p,
        "para_stop_at_first_line_label": True,
        "para_one_label_instruction": ONE_LABEL_INSTRUCTION,
        "train_para_stem_count": len(TRAIN_STEMS),
        "train_includes_mmsu_eval_stem": MMSU_EVAL_STEM in TRAIN_STEMS
        or EVAL_STEM in TRAIN_STEMS,
        "r_para_teacher_privilege": (
            None if vanilla_opd else R_PARA_TEACHER_TASK_PRIVILEGE
        ),
        "r_cnt_teacher_privilege": R_CNT_TEACHER_TASK_PRIVILEGE,
        "ema_teacher": False,
        "sdr_kl_clip": False,
        "on_policy_opsd": False,
        "unfreeze_audio_tower_proj": False,
        "prefetch": prefetch,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": optimizer_step,
        "elapsed_sec": round(time.time() - t0, 1),
        "epoch_metrics": metrics,
        "backbone": backbone,
        "r_train_utterances": r_train_utterances_rel(),
        "prob_r": sampler.prob_r,
        "r_train_count": len(r_rows),
        "p_train_count": len(p_rows),
        "seed": seed,
        "response_length_stats_path": str(length_stats_path),
    }
    write_report(summary, output_dir / "train_summary.json")
    return summary


def _assert_eval_aligned_conversation(sample: OursTrainSample) -> None:
    """Fail dry-run if training chats drift from frozen local eval."""

    audio_only = build_audio_user_conversation(sample.audio_path, "")
    if audio_only[0]["role"] != "user":
        raise ValueError("Audio chat must be a user turn.")
    if any(turn.get("role") == "system" for turn in audio_only):
        raise ValueError("Training chat must not include a system turn.")
    audio_types = [part["type"] for part in audio_only[0]["content"]]
    if audio_types != ["audio"]:
        raise ValueError(f"R-cnt chat must be audio-only, got {audio_types}")
    para_chat = build_audio_user_conversation(sample.audio_path, sample.para_prompt)
    para_types = [part["type"] for part in para_chat[0]["content"]]
    if para_types != ["text", "audio"]:
        raise ValueError(f"Para chat must be text then audio, got {para_types}")
    if ONE_LABEL_INSTRUCTION not in sample.para_prompt:
        raise ValueError("Train para prompt missing one-label instruction.")
    if R_PARA_TEACHER_TASK_PRIVILEGE in sample.para_prompt:
        raise ValueError("Student public para prompt must not include teacher privilege.")
    teacher_para_chat = build_audio_user_conversation(
        sample.audio_path, sample.teacher_para_prompt
    )
    teacher_para_types = [part["type"] for part in teacher_para_chat[0]["content"]]
    if teacher_para_types != ["text", "audio"]:
        raise ValueError(
            f"Teacher para chat must be text then audio, got {teacher_para_types}"
        )
    if sample.layer == "R":
        if not sample.teacher_para_prompt.startswith(R_PARA_TEACHER_TASK_PRIVILEGE):
            raise ValueError("R teacher para prompt must start with the task privilege.")
        if sample.teacher_para_prompt == sample.para_prompt:
            raise ValueError("R teacher para prompt must differ from the public prompt.")
        if not sample.teacher_text_prompt.startswith(R_CNT_TEACHER_TASK_PRIVILEGE):
            raise ValueError("R teacher cnt text must start with the concise privilege.")
        if sample.teacher_text_prompt == sample.text_prompt:
            raise ValueError("R teacher cnt text must differ from the raw question.")
        if sample.text_prompt.strip() not in sample.teacher_text_prompt:
            raise ValueError("R teacher cnt text must still contain the question.")
        if R_CNT_TEACHER_TASK_PRIVILEGE in sample.text_prompt:
            raise ValueError("Raw R question must not already include the cnt privilege.")
        if R_CNT_TEACHER_TASK_PRIVILEGE in sample.para_prompt:
            raise ValueError("Student para prompt must not include the cnt privilege.")
        gold = str((sample.metadata or {}).get("answer_text") or "")
        if gold.strip() and gold.strip() in R_CNT_TEACHER_TASK_PRIVILEGE:
            raise ValueError("Cnt teacher privilege must not contain the gold answer.")
    else:
        if sample.teacher_para_prompt != sample.para_prompt:
            raise ValueError("P teacher para prompt must match the public prompt.")
        if R_PARA_TEACHER_TASK_PRIVILEGE in sample.teacher_para_prompt:
            raise ValueError("P teacher para must not include the R privilege.")
        if sample.teacher_text_prompt != sample.text_prompt:
            raise ValueError("P teacher text must match the transcript field.")
        if R_CNT_TEACHER_TASK_PRIVILEGE in sample.teacher_text_prompt:
            raise ValueError("P teacher text must not include the R-cnt privilege.")
    teacher = build_text_user_conversation(sample.teacher_text_prompt)
    if teacher[0]["role"] != "user":
        raise ValueError("Teacher chat must be a user turn.")


def dry_run_ours_data(n_samples: int = 8, seed: int = 42) -> dict[str, Any]:
    """Local no-GPU check of train pools and a few mixed samples."""

    assert_task_privilege_has_no_closed_set_label(R_PARA_TEACHER_TASK_PRIVILEGE)
    assert_task_privilege_has_no_closed_set_label(R_CNT_TEACHER_TASK_PRIVILEGE)
    if MMSU_EVAL_STEM not in TRAIN_STEMS and EVAL_STEM not in TRAIN_STEMS:
        raise ValueError("Train stem pool must include the frozen MMSU eval stem.")
    summary = summarize_train_pools()
    sampler = SqrtMixSampler(load_r_train_rows(), load_p_train_rows(), seed=seed)
    preview = []
    for sample in sampler.iter_microbatches(n_samples):
        _assert_eval_aligned_conversation(sample)
        preview.append(
            {
                "layer": sample.layer,
                "utterance_id": sample.utterance_id,
                "audio_exists": sample.audio_path.exists(),
                "text_prompt_chars": len(sample.text_prompt),
                "teacher_text_prompt_chars": len(sample.teacher_text_prompt),
                "teacher_cnt_privileged": (
                    sample.teacher_text_prompt != sample.text_prompt
                ),
                "para_prompt_chars": len(sample.para_prompt),
                "teacher_para_prompt_chars": len(sample.teacher_para_prompt),
                "has_para_prompt": True,
                "has_one_label_instruction": ONE_LABEL_INSTRUCTION in sample.para_prompt,
                "student_has_mmsu_stem": "How does the speaker feel in the recording?"
                in sample.para_prompt,
                "teacher_para_privileged": (
                    sample.teacher_para_prompt != sample.para_prompt
                ),
            }
        )
    summary["preview"] = preview
    summary["status"] = "DRY_RUN_DATA_OK"
    return summary


def summarize_response_length_file(path: Path) -> dict[str, Any]:
    """Aggregate response_length_stats.jsonl into percentiles by arm/layer."""

    buckets: dict[str, list[int]] = {}
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = str(row.get("arm") or row.get("layer"))
            buckets.setdefault(key, []).append(int(row["response_tokens"]))

    def _pct(values: list[int], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(q * (len(ordered) - 1))))
        return float(ordered[idx])

    out: dict[str, Any] = {"status": "ok", "path": str(path)}
    for key, values in buckets.items():
        out[key] = {
            "n": len(values),
            "mean": (sum(values) / len(values)) if values else None,
            "p50": _pct(values, 0.50),
            "p90": _pct(values, 0.90),
            "p95": _pct(values, 0.95),
            "p99": _pct(values, 0.99),
            "max": max(values) if values else None,
        }
    return out


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI for Ours train / data dry-run / length summary."""

    parser = argparse.ArgumentParser(description="Train Ours on Qwen2.5-Omni-7B")
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=LOCKED_MICRO_BATCH_SIZE,
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=LOCKED_GRAD_ACCUM,
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens-r", type=int, default=512)
    parser.add_argument("--max-new-tokens-p", type=int, default=64)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--fkl-top-k", type=int, default=16)
    parser.add_argument("--max-optimizer-steps", type=int, default=None)
    parser.add_argument(
        "--backbone",
        default=BACKBONE_QWEN25_OMNI,
        choices=(BACKBONE_QWEN25_OMNI, BACKBONE_PHI4_MM),
        help="Student backbone. Qwen defaults stay unchanged.",
    )
    parser.add_argument(
        "--r-train-utterances",
        default=None,
        help="Override R teacher-correct utterance jsonl (Phi-4 uses phi4_mm).",
    )
    parser.add_argument(
        "--no-prefetch",
        action="store_true",
        help="Disable one-ahead CPU encode prefetch.",
    )
    parser.add_argument(
        "--vanilla-opd",
        action="store_true",
    )
    parser.add_argument(
        "--dry-run-data",
        action="store_true",
        help="Only inventory train pools (no model / no GPU).",
    )
    parser.add_argument(
        "--summarize-response-lengths",
        type=Path,
        default=None,
        help="Summarize a response_length_stats.jsonl and exit.",
    )
    return parser


def _apply_backbone_cli_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill Phi-4 model/output/R-train paths when the caller left Qwen defaults."""

    if args.r_train_utterances:
        os.environ["REALMG_R_TRAIN_UTTERANCES"] = str(args.r_train_utterances)
    if args.backbone != BACKBONE_PHI4_MM:
        return args
    if str(args.model_dir) == DEFAULT_MODEL_DIR:
        args.model_dir = Path(DEFAULT_PHI4_MODEL_DIR)
    if str(args.output_dir) == DEFAULT_OUTPUT_DIR:
        args.output_dir = Path(DEFAULT_PHI4_OUTPUT_DIR)
    if not args.r_train_utterances:
        os.environ["REALMG_R_TRAIN_UTTERANCES"] = DEFAULT_PHI4_R_TRAIN
    return args


def main(argv: list[str] | None = None) -> int:
    """Entry point used by `python -m realmg.train.ours_train`."""

    args = _apply_backbone_cli_defaults(build_arg_parser().parse_args(argv))
    if getattr(args, "vanilla_opd", False):
        if str(args.output_dir) == DEFAULT_OUTPUT_DIR:
            args.output_dir = Path(DEFAULT_VANILLA_OPD_OUTPUT_DIR)
        if args.backbone != BACKBONE_QWEN25_OMNI:
            raise SystemExit("vanilla_opd is Omni-only.")
    if args.summarize_response_lengths is not None:
        summary = summarize_response_length_file(args.summarize_response_lengths)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.dry_run_data:
        summary = dry_run_ours_data(seed=args.seed)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    run_ours_training(
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        micro_batch_size=args.micro_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        seed=args.seed,
        max_new_tokens_r=args.max_new_tokens_r,
        max_new_tokens_p=args.max_new_tokens_p,
        clip_eps=args.clip_eps,
        fkl_top_k=args.fkl_top_k,
        max_optimizer_steps=args.max_optimizer_steps,
        prefetch=not args.no_prefetch,
        backbone=args.backbone,
        vanilla_opd=bool(getattr(args, "vanilla_opd", False)),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
