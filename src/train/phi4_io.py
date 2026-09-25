"""Phi-4-MM conversation encode for Ours training.

"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor


USER_PROMPT = "<|user|>"
ASSISTANT_PROMPT = "<|assistant|>"
PROMPT_SUFFIX = "<|end|>"
AUDIO_PLACEHOLDER = "<|audio_1|>"
INPUT_MODE_LANGUAGE = 0
INPUT_MODE_SPEECH = 2


def is_phi4_processor(processor: Any) -> bool:
    """Return whether this processor is the Phi-4-MM local-code processor."""

    name = type(processor).__name__
    return "Phi4" in name or "Phi4MM" in name


def phi4_user_prompt(*, user_text: str, has_audio: bool) -> str:
    """Build one Phi-4 user string plus the generation suffix."""

    text = user_text.strip()
    if has_audio and text:
        body = f"{text}{AUDIO_PLACEHOLDER}"
    elif has_audio:
        body = AUDIO_PLACEHOLDER
    else:
        body = text
    return f"{USER_PROMPT}{body}{PROMPT_SUFFIX}{ASSISTANT_PROMPT}"


def _ensure_left_padding(processor: Any) -> None:
    """Prefer left padding for decoder-only batched generate."""

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token = tokenizer.eos_token


def _conversation_text_and_audio(
    conversation: list[dict[str, Any]],
) -> tuple[str, Path | None]:
    """Extract concatenated user text and the first audio path."""

    texts: list[str] = []
    audio_path: Path | None = None
    for turn in conversation:
        for part in turn.get("content") or []:
            part_type = part.get("type")
            if part_type == "text":
                texts.append(str(part.get("text") or ""))
            elif part_type == "audio":
                if audio_path is None:
                    audio_path = Path(str(part["audio"]))
    return " ".join(t for t in texts if t.strip()).strip(), audio_path


def _read_audio_pair(audio_path: Path) -> tuple[Any, int]:
    """Load one wav as (array, sample_rate) for the Phi-4 processor."""

    import soundfile

    audio, sample_rate = soundfile.read(str(audio_path))
    return audio, int(sample_rate)


def _pad_left(sequences: list[Tensor], padding_value: int) -> Tensor:
    """Left-pad 1-D token rows to a shared length."""

    max_len = max(int(row.shape[0]) for row in sequences)
    out = sequences[0].new_full((len(sequences), max_len), padding_value)
    for index, row in enumerate(sequences):
        out[index, max_len - int(row.shape[0]) :] = row
    return out


def _cat_with_pad(tensors: list[Tensor], dim: int, padding_value: float = 0.0) -> Tensor:
    """Concatenate along dim, padding every other dim to the max size."""

    ndim = tensors[0].dim()
    out_size = [max(item.shape[axis] for item in tensors) for axis in range(ndim)]
    out_size[dim] = sum(item.shape[dim] for item in tensors)
    output = tensors[0].new_full(out_size, padding_value)
    cursor = 0
    for item in tensors:
        slices = [slice(0, item.shape[axis]) for axis in range(ndim)]
        slices[dim] = slice(cursor, cursor + item.shape[dim])
        output[tuple(slices)] = item
        cursor += item.shape[dim]
    return output


def _move_batch(
    batch: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict[str, Any]:
    """Move tensor values to device/dtype; leave the rest unchanged."""

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, Tensor):
            tensor = value.to(device)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            moved[key] = tensor
        else:
            moved[key] = value
    return moved


def encode_one_phi4_conversation(
    processor: Any,
    conversation: list[dict[str, Any]],
) -> dict[str, Any]:
    """Tokenize one Phi-4 conversation on CPU."""

    user_text, audio_path = _conversation_text_and_audio(conversation)
    has_audio = audio_path is not None
    prompt = phi4_user_prompt(user_text=user_text, has_audio=has_audio)
    processor_kwargs: dict[str, Any] = {"text": prompt, "return_tensors": "pt"}
    if has_audio:
        assert audio_path is not None
        processor_kwargs["audios"] = [_read_audio_pair(audio_path)]
    encoded = processor(**processor_kwargs)
    out: dict[str, Any] = {}
    for key, value in encoded.items():
        out[key] = value
    out["input_mode"] = torch.tensor(
        [INPUT_MODE_SPEECH if has_audio else INPUT_MODE_LANGUAGE],
        dtype=torch.long,
    )
    return out


def encode_phi4_conversations(
    processor: Any,
    conversations: list[list[dict[str, Any]]],
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """Pad one homogeneous Phi-4 micro-batch (all audio or all text)."""

    if not conversations:
        raise ValueError("encode_phi4_conversations requires at least one conversation.")
    _ensure_left_padding(processor)
    rows = [
        encode_one_phi4_conversation(processor, conversation)
        for conversation in conversations
    ]
    modes = [int(row["input_mode"].reshape(-1)[0].item()) for row in rows]
    if any(mode != modes[0] for mode in modes):
        raise ValueError("Phi-4 micro-batch mixed LANGUAGE and SPEECH input_mode.")
    tokenizer = getattr(processor, "tokenizer", processor)
    pad_id = int(getattr(tokenizer, "pad_token_id", 0) or 0)
    input_ids = _pad_left([row["input_ids"].reshape(-1) for row in rows], pad_id)
    attention_mask = (input_ids != pad_id).long()
    batch: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_mode": torch.tensor(modes, dtype=torch.long),
    }
    if modes[0] == INPUT_MODE_SPEECH:
        audio_embeds = [row["input_audio_embeds"] for row in rows]
        batch["input_audio_embeds"] = _cat_with_pad(audio_embeds, dim=0)
        batch["audio_embed_sizes"] = torch.cat(
            [row["audio_embed_sizes"].reshape(-1) for row in rows]
        )
        audio_masks = [
            row["input_audio_embeds"].new_ones(
                (row["input_audio_embeds"].shape[1],),
                dtype=torch.bool,
            )
            for row in rows
        ]
        max_audio = max(int(mask.shape[0]) for mask in audio_masks)
        padded_mask = audio_masks[0].new_zeros((len(audio_masks), max_audio), dtype=torch.bool)
        for index, mask in enumerate(audio_masks):
            padded_mask[index, : int(mask.shape[0])] = mask
        batch["audio_attention_mask"] = padded_mask
    else:
        # LANGUAGE: Phi-4-MM forward asserts at least one of audio/image
        # embeddings is not None, but it must not receive image_embeds (empty
        # placeholders still trigger the vision path). Pass empty audio_embeds
        # and audio sizes, and leave image keys absent so the model defaults
        # them to None.
        if "input_audio_embeds" in rows[0]:
            batch["input_audio_embeds"] = torch.stack(
                [row["input_audio_embeds"] for row in rows]
            )
        if "audio_embed_sizes" in rows[0]:
            batch["audio_embed_sizes"] = torch.stack(
                [row["audio_embed_sizes"] for row in rows]
            )
    return _move_batch(batch, device=device, dtype=dtype)
