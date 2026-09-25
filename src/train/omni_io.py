"""Qwen2.5-Omni conversation encode / generate / token logprob helpers.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor


def build_audio_user_conversation(audio_path: Path, user_text: str) -> list[dict[str, Any]]:
    """Build one user turn aligned to frozen local-server audio eval."""

    content: list[dict[str, Any]] = []
    if user_text.strip():
        content.append({"type": "text", "text": user_text})
    content.append({"type": "audio", "audio": str(audio_path)})
    return [{"role": "user", "content": content}]


def build_text_user_conversation(user_text: str) -> list[dict[str, Any]]:
    """Build a text-only user turn for the loss-1 teacher (text(s))."""

    return [{"role": "user", "content": [{"type": "text", "text": user_text}]}]


def _ensure_left_padding(processor: Any) -> None:
    """Prefer left padding for decoder-only batched generate."""

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token


def encode_conversations(
    processor: Any,
    conversations: list[list[dict[str, Any]]],
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Tokenize one or more student conversations into a padded batch on device."""

    from realmg.train.phi4_io import encode_phi4_conversations, is_phi4_processor

    if is_phi4_processor(processor):
        return encode_phi4_conversations(
            processor, conversations, device=device, dtype=dtype
        )

    from qwen_omni_utils import process_mm_info

    if not conversations:
        raise ValueError("encode_conversations requires at least one conversation.")
    _ensure_left_padding(processor)

    texts: list[str] = []
    audio_items: list[Any] = []
    has_audio = False
    for conversation in conversations:
        text = processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        audios, images, videos = process_mm_info(
            conversation, use_audio_in_video=False
        )
        if images or videos:
            raise ValueError("Ours training expects audio/text only.")
        texts.append(text)
        if audios:
            has_audio = True
            audio_items.append(audios[0] if len(audios) == 1 else audios)
        else:
            audio_items.append(None)

    processor_kwargs: dict[str, Any] = {
        "text": texts,
        "return_tensors": "pt",
        "padding": True,
        "use_audio_in_video": False,
    }
    if has_audio:
        processor_kwargs["audio"] = audio_items
        processor_kwargs["images"] = None
        processor_kwargs["videos"] = None
    inputs = processor(**processor_kwargs)
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, Tensor):
            tensor = value.to(device)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            moved[key] = tensor
        else:
            moved[key] = value
    return moved


def encode_conversation(
    processor: Any,
    conversation: list[dict[str, Any]],
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Tokenize one Omni conversation into a single-item batch on device."""

    return encode_conversations(
        processor, [conversation], device=device, dtype=dtype
    )


def clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    """Shallow-clone a batch, cloning tensors."""

    out: dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.clone() if isinstance(value, Tensor) else value
    return out


def append_token_ids(batch: dict[str, Any], token_ids: Tensor) -> dict[str, Any]:
    """Append generated token ids to input_ids / attention_mask."""

    if token_ids.dim() == 1:
        token_ids = token_ids.unsqueeze(0)
    out = clone_batch(batch)
    out["input_ids"] = torch.cat(
        [out["input_ids"], token_ids.to(out["input_ids"].device)], dim=1
    )
    ones = torch.ones(
        token_ids.shape,
        dtype=out["attention_mask"].dtype,
        device=out["attention_mask"].device,
    )
    out["attention_mask"] = torch.cat([out["attention_mask"], ones], dim=1)
    return out


def pad_token_id_from_processor(processor: Any) -> int:
    """Resolve pad token id from processor/tokenizer."""

    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", 0)
    return int(pad_id)


def response_mask_from_ids(response_ids: Tensor, *, pad_token_id: int) -> Tensor:
    """Boolean mask over non-pad response positions, shape [B, T]."""

    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)
    return response_ids.ne(pad_token_id)


def response_lengths(response_mask: Tensor) -> list[int]:
    """Per-row valid response token counts."""

    return [int(x) for x in response_mask.sum(dim=-1).detach().cpu().tolist()]


def decode_response_ids(
    tokenizer: Any,
    response_ids: Tensor,
    *,
    pad_token_id: int,
) -> list[str]:
    """Decode generated response rows, dropping pad ids."""

    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)
    texts: list[str] = []
    for row in response_ids.detach().cpu().tolist():
        kept = [int(token) for token in row if int(token) != pad_token_id]
        texts.append(tokenizer.decode(kept, skip_special_tokens=True))
    return texts


def logprobs_from_generate_scores(
    scores: tuple[Tensor, ...] | list[Tensor],
    response_ids: Tensor,
) -> Tensor:
    """Gather per-token log-probs from HF generate scores.

    Args:
        scores: length-T tuple of [B, V] next-token logits.
        response_ids: [B, T] generated ids (pad allowed on the right).
    Returns:
        [B, T] log-probs aligned with response_ids.
    """

    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)
    if len(scores) != response_ids.shape[1]:
        raise ValueError(
            f"scores length {len(scores)} != response width {response_ids.shape[1]}"
        )
    rows: list[Tensor] = []
    for step, step_logits in enumerate(scores):
        log_probs = torch.log_softmax(step_logits, dim=-1)
        token = response_ids[:, step].unsqueeze(-1)
        rows.append(log_probs.gather(1, token).squeeze(-1))
    return torch.stack(rows, dim=1)


def stack_generate_logits(
    scores: tuple[Tensor, ...] | list[Tensor],
) -> Tensor:
    """Stack generate step logits to [B, T, V]."""

    return torch.stack(list(scores), dim=1)


@torch.no_grad()
def generate_response_ids(
    model: Any,
    prompt_batch: dict[str, Any],
    *,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 1.0,
    top_p: float = 1.0,
    pad_token_id: int | None = None,
    return_scores: bool = False,
) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
    """Generate continuation token ids (response only, no prompt).

    When return_scores=True, also returns the per-step logits tuple from HF.
    """

    generate_kwargs: dict[str, Any] = {
        **prompt_batch,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    if pad_token_id is not None:
        generate_kwargs["pad_token_id"] = pad_token_id
    if return_scores:
        generate_kwargs["output_scores"] = True
        generate_kwargs["return_dict_in_generate"] = True

    # Thinker rejects return_audio; full Omni accepts it.
    try:
        output = model.generate(**generate_kwargs, return_audio=False)
    except (TypeError, ValueError):
        output = model.generate(**generate_kwargs)

    prompt_len = int(prompt_batch["input_ids"].shape[1])
    if return_scores:
        sequences = output.sequences
        response_ids = sequences[:, prompt_len:]
        scores = output.scores
        if scores is None:
            raise RuntimeError("generate returned no scores despite return_scores=True.")
        return response_ids, scores
    return output[:, prompt_len:]


def response_token_logprobs(
    model: Any,
    prompt_batch: dict[str, Any],
    response_ids: Tensor,
    *,
    response_mask: Tensor | None = None,
) -> Tensor:
    """Return per-token log-probs, shape [B, T] (B=1 squeezes not applied)."""

    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)
    batch_size, resp_len = response_ids.shape
    if resp_len == 0:
        return response_ids.new_zeros((batch_size, 0), dtype=torch.float32)

    full = append_token_ids(prompt_batch, response_ids)
    outputs = model(**full, use_cache=False)
    prompt_len = int(prompt_batch["input_ids"].shape[1])
    step_logits = outputs.logits[:, prompt_len - 1 : -1, :]
    log_probs = torch.log_softmax(step_logits, dim=-1)
    # Pad positions are ignored later via response_mask in the loss.
    _ = response_mask
    return log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)


def response_step_logits(
    model: Any,
    prompt_batch: dict[str, Any],
    response_ids: Tensor,
) -> Tensor:
    """Return raw logits at each response step, shape [B, T, V]."""

    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)
    full = append_token_ids(prompt_batch, response_ids)
    outputs = model(**full, use_cache=False)
    prompt_len = int(prompt_batch["input_ids"].shape[1])
    return outputs.logits[:, prompt_len - 1 : -1, :]


def response_topk_teacher_student_fkl(
    teacher_logits: Tensor,
    student_logits: Tensor,
    *,
    top_k: int = 16,
    response_mask: Tensor | None = None,
) -> Tensor:
    """Mean forward KL over teacher top-k support.

    Accepts [T, V] or [B, T, V]. Optional response_mask [B, T] (or [T]).
    """

    if teacher_logits.shape != student_logits.shape:
        raise ValueError("Teacher/student logit shapes must match.")
    if teacher_logits.numel() == 0:
        return student_logits.new_zeros(())

    if teacher_logits.dim() == 2:
        teacher_logits = teacher_logits.unsqueeze(0)
        student_logits = student_logits.unsqueeze(0)
    if response_mask is None:
        response_mask = teacher_logits.new_ones(
            teacher_logits.shape[:2], dtype=torch.bool
        )
    elif response_mask.dim() == 1:
        response_mask = response_mask.unsqueeze(0)

    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    top_k_eff = min(top_k, teacher_log_probs.shape[-1])
    top_values, top_indices = torch.topk(teacher_log_probs, k=top_k_eff, dim=-1)
    teacher_top = top_values.exp()
    teacher_top = teacher_top / teacher_top.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    student_top = student_log_probs.gather(-1, top_indices)
    fkl = (teacher_top * (teacher_top.clamp_min(1e-12).log() - student_top)).sum(dim=-1)
    masked = fkl * response_mask.to(dtype=fkl.dtype)
    denom = response_mask.to(dtype=fkl.dtype).sum().clamp_min(1.0)
    return masked.sum() / denom
