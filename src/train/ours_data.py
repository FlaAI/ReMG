"""R/P train pools and sqrt-size layer sampler for Ours.

"""

from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl
from realmg.eval.q_para_prompts import (
    EVAL_STEM,
    MMSU_EVAL_STEM,
    ONE_LABEL_INSTRUCTION,
    P_EMOTION_LABELS,
    R_EMOTION_LABELS,
    R_PARA_TEACHER_TASK_PRIVILEGE,
    TRAIN_STEMS,
    attach_r_para_teacher_privilege,
    sample_train_q_para_prompt,
)
from realmg.train.ours_losses import three_arm_loss_weights
from realmg.train.r_cnt_teacher_privilege import (
    R_CNT_TEACHER_TASK_PRIVILEGE,
    attach_r_cnt_teacher_privilege,
)


R_TRAIN_UTTERANCES = (
    "Data/manifests/r_carved_train_utterances_teacher_correct_qwen25_omni_7b.jsonl"
)
P_TRAIN_UTTERANCES = "Data/manifests/p_intersection_train_utterances.jsonl"
R_TRAIN_UTTERANCES_ENV = "REALMG_R_TRAIN_UTTERANCES"

LayerName = Literal["R", "P"]


@dataclass(frozen=True)
class OursTrainSample:
    """One utterance. R runs cnt+para. P runs para only.

    para_prompt is the student public stem. teacher_para_prompt equals
    para_prompt on P, and adds the task-level privilege on R.
    teacher_text_prompt is the loss-1 teacher text(s) turn: privilege plus
    the question on R, unused on P.
    """

    layer: LayerName
    utterance_id: str
    audio_path: Path
    text_prompt: str
    teacher_text_prompt: str
    para_prompt: str
    teacher_para_prompt: str
    metadata: dict[str, Any]


def resolve_audio_path(relative_or_absolute: str) -> Path:
    """Resolve a repo-relative audio path to an absolute Path."""

    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return resolve_from_repo_root(relative_or_absolute)


def r_train_utterances_rel() -> str:
    """Return the R train utterance manifest, overridable per backbone."""

    override = os.environ.get(R_TRAIN_UTTERANCES_ENV, "").strip()
    if override:
        return override.replace("\\", "/")
    return R_TRAIN_UTTERANCES


def load_r_train_rows() -> list[dict[str, Any]]:
    """Load teacher-correct R train utterances."""

    rel = r_train_utterances_rel()
    path = Path(rel)
    if path.is_absolute():
        return read_jsonl(path)
    return read_jsonl(resolve_from_repo_root(rel))


def load_p_train_rows() -> list[dict[str, Any]]:
    """Load frozen P intersection train utterances."""

    return read_jsonl(resolve_from_repo_root(P_TRAIN_UTTERANCES))


def build_r_sample(row: dict[str, Any], *, rng: random.Random) -> OursTrainSample:
    """Build one R sample: audio-is-question cnt plus 2-class para."""

    meta = row.get("metadata") or {}
    text_prompt = str(meta.get("prompt_text") or row.get("transcript") or "")
    if not text_prompt.strip():
        raise ValueError(f"R row missing prompt_text: {row['utterance_id']}")
    para_prompt = sample_train_q_para_prompt(R_EMOTION_LABELS, rng=rng)
    return OursTrainSample(
        layer="R",
        utterance_id=str(row["utterance_id"]),
        audio_path=resolve_audio_path(str(row["audio_path"])),
        text_prompt=text_prompt,
        teacher_text_prompt=attach_r_cnt_teacher_privilege(text_prompt),
        para_prompt=para_prompt,
        teacher_para_prompt=attach_r_para_teacher_privilege(para_prompt),
        metadata={
            "content_id": row.get("content_id"),
            "paralinguistic_label": row.get("paralinguistic_label"),
            "answer_text": meta.get("answer_text"),
        },
    )


def build_p_sample(row: dict[str, Any], *, rng: random.Random) -> OursTrainSample:
    """Build one P sample with a 5-class train-stem paralinguistic prompt."""

    para_prompt = sample_train_q_para_prompt(P_EMOTION_LABELS, rng=rng)
    return OursTrainSample(
        layer="P",
        utterance_id=str(row["utterance_id"]),
        audio_path=resolve_audio_path(str(row["audio_path"])),
        text_prompt=str(row.get("transcript") or ""),
        teacher_text_prompt=str(row.get("transcript") or ""),
        para_prompt=para_prompt,
        teacher_para_prompt=para_prompt,
        metadata={
            "content_id": row.get("content_id"),
            "paralinguistic_label": row.get("paralinguistic_label"),
        },
    )


class SqrtMixSampler:
    """Alternate R/P steps with P(layer)=sqrt(n_layer)/Z."""

    def __init__(
        self,
        r_rows: list[dict[str, Any]],
        p_rows: list[dict[str, Any]],
        *,
        seed: int = 42,
    ) -> None:
        if not r_rows or not p_rows:
            raise ValueError("Both R and P train pools must be non-empty.")
        self.r_rows = r_rows
        self.p_rows = p_rows
        self.rng = random.Random(seed)
        weight_r = math.sqrt(len(r_rows))
        weight_p = math.sqrt(len(p_rows))
        self.prob_r = weight_r / (weight_r + weight_p)
        self._r_i = 0
        self._p_i = 0
        self.rng.shuffle(self.r_rows)
        self.rng.shuffle(self.p_rows)

    def _next_row(self, layer: LayerName) -> dict[str, Any]:
        """Take the next row from one pool, reshuffling at wrap-around."""

        if layer == "R":
            if self._r_i >= len(self.r_rows):
                self.rng.shuffle(self.r_rows)
                self._r_i = 0
            row = self.r_rows[self._r_i]
            self._r_i += 1
            return row
        if self._p_i >= len(self.p_rows):
            self.rng.shuffle(self.p_rows)
            self._p_i = 0
        row = self.p_rows[self._p_i]
        self._p_i += 1
        return row

    def sample(self) -> OursTrainSample:
        """Draw one training sample with sqrt-size layer mix."""

        layer: LayerName = "R" if self.rng.random() < self.prob_r else "P"
        return self.sample_from_layer(layer)

    def sample_from_layer(self, layer: LayerName) -> OursTrainSample:
        """Draw one sample from a fixed layer."""

        row = self._next_row(layer)
        if layer == "R":
            return build_r_sample(row, rng=self.rng)
        return build_p_sample(row, rng=self.rng)

    def sample_homogeneous_microbatch(self, micro_batch_size: int) -> list[OursTrainSample]:
        """Draw a same-layer micro-batch (required for true padded batching).

        Layer is chosen once with the sqrt mix, then ``micro_batch_size`` rows
        are taken from that layer only.
        """

        if micro_batch_size < 1:
            raise ValueError("micro_batch_size must be >= 1.")
        layer: LayerName = "R" if self.rng.random() < self.prob_r else "P"
        return [self.sample_from_layer(layer) for _ in range(micro_batch_size)]

    def iter_microbatches(self, n: int) -> Iterator[OursTrainSample]:
        """Yield n independent single samples (debug / dry-run)."""

        for _ in range(n):
            yield self.sample()


def summarize_train_pools(*, effective_batch: int = 128) -> dict[str, Any]:
    """Return counts and paths for dry-run inventory."""

    r_rows = load_r_train_rows()
    p_rows = load_p_train_rows()
    missing_r = sum(
        1 for row in r_rows if not resolve_audio_path(str(row["audio_path"])).exists()
    )
    missing_p = sum(
        1 for row in p_rows if not resolve_audio_path(str(row["audio_path"])).exists()
    )
    weight_r = math.sqrt(len(r_rows))
    weight_p = math.sqrt(len(p_rows))
    prob_r = weight_r / (weight_r + weight_p)
    return {
        "r_train_count": len(r_rows),
        "p_train_count": len(p_rows),
        "r_audio_missing": missing_r,
        "p_audio_missing": missing_p,
        "prob_r": prob_r,
        "prob_p": 1.0 - prob_r,
        "practical_arms": ["R-cnt", "R-para", "P-para"],
        "skipped_arm": "P-cnt",
        "arm_loss_weights": three_arm_loss_weights(prob_r),
        "effective_batch_lock": effective_batch,
        "steps_per_epoch_estimate": math.ceil(
            (len(r_rows) + len(p_rows)) / max(1, effective_batch)
        ),
        "one_label_instruction": ONE_LABEL_INSTRUCTION,
        "train_para_stem_count": len(TRAIN_STEMS),
        "train_includes_mmsu_eval_stem": MMSU_EVAL_STEM in TRAIN_STEMS
        or EVAL_STEM in TRAIN_STEMS,
        "r_para_teacher_privilege": R_PARA_TEACHER_TASK_PRIVILEGE,
        "r_cnt_teacher_privilege": R_CNT_TEACHER_TASK_PRIVILEGE,
    }
