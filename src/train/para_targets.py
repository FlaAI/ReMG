"""Cut para teacher trajectories to the first-line closed-set label.

"""

from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import Tensor

from realmg.train.omni_io import decode_response_ids, response_mask_from_ids


FIRST_LINE_PUNCT = ".,!?;:\"'` "


def normalize_first_line_as_label(text: str) -> str:
    """Return the first line with wrapping punctuation stripped, lowercased."""

    stripped = text.strip()
    if not stripped:
        return ""
    first = stripped.splitlines()[0].strip()
    return first.strip(FIRST_LINE_PUNCT).lower()


def first_line_is_closed_set_label(text: str, labels: Sequence[str]) -> bool:
    """True iff the first line is exactly one closed-set label."""

    allowed = {label.lower() for label in labels}
    return normalize_first_line_as_label(text) in allowed


def count_tokens_through_first_line_label(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    labels: Sequence[str],
    pad_token_id: int,
) -> int | None:
    """Smallest prefix length whose decode first-line is a closed-set label."""

    kept = [int(token) for token in token_ids if int(token) != int(pad_token_id)]
    if not kept:
        return None
    for width in range(1, len(kept) + 1):
        decoded = tokenizer.decode(kept[:width], skip_special_tokens=True)
        if first_line_is_closed_set_label(decoded, labels):
            return width
    return None


def truncate_para_response_ids_to_first_line_closed_label(
    tokenizer: Any,
    response_ids: Tensor,
    teacher_logits: Tensor,
    *,
    labels: Sequence[str],
    pad_token_id: int,
) -> tuple[Tensor, Tensor, Tensor, list[int], list[bool], list[str]]:
    """Slice teacher ids/logits to the first-line label, or mask the row out.

    Returns truncated ids, truncated logits, response mask, kept lengths,
    kept flags, and decoded texts (distilled prefix if kept, else raw).
    """

    if response_ids.dim() == 1:
        response_ids = response_ids.unsqueeze(0)
    if teacher_logits.dim() == 2:
        teacher_logits = teacher_logits.unsqueeze(0)
    batch_size, raw_width = response_ids.shape
    raw_texts = decode_response_ids(
        tokenizer, response_ids, pad_token_id=pad_token_id
    )
    if raw_width == 0:
        empty_ids = response_ids[:, :0]
        empty_logits = teacher_logits[:, :0, :]
        empty_mask = empty_ids.new_zeros((batch_size, 0), dtype=torch.bool)
        return (
            empty_ids,
            empty_logits,
            empty_mask,
            [0] * batch_size,
            [False] * batch_size,
            raw_texts,
        )

    keep_widths: list[int] = []
    kept_flags: list[bool] = []
    for row in response_ids.detach().cpu().tolist():
        width = count_tokens_through_first_line_label(
            tokenizer,
            row,
            labels=labels,
            pad_token_id=pad_token_id,
        )
        if width is None:
            keep_widths.append(0)
            kept_flags.append(False)
        else:
            keep_widths.append(width)
            kept_flags.append(True)

    max_keep = max(keep_widths, default=0)
    if max_keep == 0:
        empty_ids = response_ids[:, :0]
        empty_logits = teacher_logits[:, :0, :]
        empty_mask = empty_ids.new_zeros((batch_size, 0), dtype=torch.bool)
        return (
            empty_ids,
            empty_logits,
            empty_mask,
            keep_widths,
            kept_flags,
            raw_texts,
        )

    truncated = response_ids.new_full((batch_size, max_keep), int(pad_token_id))
    for index, width in enumerate(keep_widths):
        if width > 0:
            truncated[index, :width] = response_ids[index, :width]
    truncated_logits = teacher_logits[:, :max_keep, :]
    truncated_mask = response_mask_from_ids(truncated, pad_token_id=pad_token_id)
    for index, kept in enumerate(kept_flags):
        if not kept:
            truncated_mask[index] = False
    distilled_texts = decode_response_ids(
        tokenizer, truncated, pad_token_id=pad_token_id
    )
    texts = [
        distilled_texts[index] if kept else raw_texts[index]
        for index, kept in enumerate(kept_flags)
    ]
    return (
        truncated,
        truncated_logits,
        truncated_mask,
        keep_widths,
        kept_flags,
        texts,
    )
