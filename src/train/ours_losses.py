"""Ours loss helpers: sampled-token reverse KL and top-k forward KL."""

from __future__ import annotations

import torch
from torch import Tensor


def clipped_reverse_kl_ppo_loss(
    logprob_new: Tensor,
    logprob_old: Tensor,
    logprob_teacher: Tensor,
    *,
    clip_eps: float = 0.2,
    response_mask: Tensor | None = None,
) -> Tensor:
    """PPO-style clipped surrogate for on-policy reverse KL (EOPD-style).

    Advantage A = log π_T - log π_old (detached). Minimize -min(rA, clip(r)A).
    Tensors may be [T] or [B, T]. Optional response_mask matches logprob shape.
    """

    if logprob_new.numel() == 0:
        return logprob_new.new_zeros(())
    if logprob_new.dim() == 1:
        logprob_new = logprob_new.unsqueeze(0)
        logprob_old = logprob_old.unsqueeze(0)
        logprob_teacher = logprob_teacher.unsqueeze(0)
    if response_mask is None:
        response_mask = logprob_new.new_ones(logprob_new.shape, dtype=torch.bool)
    elif response_mask.dim() == 1:
        response_mask = response_mask.unsqueeze(0)

    # Inactive positions must be zeroed BEFORE exp/mul. Batched generate pads
    # finished rows with pad_token_id; when pad_id == eos_id those scores are
    # often -inf. Computing ratio on them yields inf, then inf*0 == NaN even
    # after multiplying by the boolean mask.
    inactive = ~response_mask
    logprob_new = logprob_new.masked_fill(inactive, 0.0)
    logprob_old = logprob_old.masked_fill(inactive, 0.0)
    logprob_teacher = logprob_teacher.masked_fill(inactive, 0.0)

    advantage = (logprob_teacher - logprob_old).detach()
    ratio = torch.exp(logprob_new - logprob_old.detach())
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
    # EOPD minimizes max(-rA, -clip(r)A) == -min(rA, clip(r)A).
    per_token = -torch.min(unclipped, clipped)
    masked = per_token * response_mask.to(dtype=per_token.dtype)
    denom = response_mask.to(dtype=per_token.dtype).sum().clamp_min(1.0)
    return masked.sum() / denom


def reverse_kl_warmup_scale(step_index: int, total_optimizer_steps: int) -> float:
    """Linear 0→1 warmup over the first 10% of optimizer steps."""

    if total_optimizer_steps <= 0:
        return 1.0
    warmup = max(1, int(0.1 * total_optimizer_steps))
    return float(min(1.0, (step_index + 1) / warmup))


def three_arm_loss_weights(prob_r: float) -> dict[str, float]:
    """Slot coefficients for expected R-cnt:R-para:P-para = 2:1:1.

    Sqrt mix still chooses the layer. P-cnt is skipped, so all content mass
    sits on R-cnt and must equal the two para arms combined. Coefficients
    also keep E[slot weight]=1, matching the old per-utterance scale.

    For p_R=0.5 this is (1.0, 0.5, 0.5) on (R-cnt, R-para, P-para).
    """

    prob_r = min(max(float(prob_r), 1e-6), 1.0 - 1e-6)
    prob_p = 1.0 - prob_r
    weight_r_cnt = 1.0 / (2.0 * prob_r)
    weight_r_para = 1.0 / (4.0 * prob_r)
    weight_p_para = 1.0 / (4.0 * prob_p)
    mass_r_cnt = prob_r * weight_r_cnt
    mass_r_para = prob_r * weight_r_para
    mass_p_para = prob_p * weight_p_para
    return {
        "r_cnt": weight_r_cnt,
        "r_para": weight_r_para,
        "p_para": weight_p_para,
        "expected_mass_r_cnt": mass_r_cnt,
        "expected_mass_r_para": mass_r_para,
        "expected_mass_p_para": mass_p_para,
        "expected_cnt_para_ratio": mass_r_cnt / (mass_r_para + mass_p_para),
    }
