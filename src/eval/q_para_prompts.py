"""Frozen q_para prompt pool aligned to published speech-emotion instruction designs.

"""

from __future__ import annotations

import random
import re
from typing import Sequence

# Dynamic-SUPERB docs (task_submission.md emotion examples).
DYNAMIC_SUPERB_DOC_STEMS = (
    "Determine the emotion in the speech. The answer could be {labels}.",
    "What is the emotion in the audio. The answer could be {labels}.",
    "Can you tell me the emotion of the speaker? The answer could be {labels}.",
)

# Dynamic-SUPERB paper example + their required closed answer-space clause.
DYNAMIC_SUPERB_PAPER_STEM = (
    "Identify the emotion conveyed in the utterance. The answer could be {labels}."
)

# MMSU Emotion Recognition published example (Choices -> Choose one for label spit).
MMSU_EVAL_STEM = (
    "How does the speaker feel in the recording?\n"
    "Choose one: {labels}.\n"
    "Answer with one label only."
)

TRAIN_STEMS = DYNAMIC_SUPERB_DOC_STEMS + (
    DYNAMIC_SUPERB_PAPER_STEM,
    MMSU_EVAL_STEM,
)
EVAL_STEM = MMSU_EVAL_STEM

P_EMOTION_LABELS = ("neutral", "happy", "angry", "sad", "surprise")
R_EMOTION_LABELS = ("neutral", "happy")
ONE_LABEL_INSTRUCTION = "Answer with one label only."
CLOSED_SET_LABEL_NAMES = P_EMOTION_LABELS

R_PARA_TEACHER_TASK_PRIVILEGE = (
    "Ignore the linguistic content of the speech, including any problem "
    "being read aloud. Classify only the speaker's affect from speaking style."
)


def format_label_list(labels: Sequence[str]) -> str:
    """Join closed-set labels as a comma-separated list for prompt stems."""

    return ", ".join(labels)


def render_q_para_prompt(stem: str, labels: Sequence[str]) -> str:
    """Fill one published stem with a layer-specific label list."""

    return stem.format(labels=format_label_list(labels))


def ensure_one_label_instruction(prompt: str) -> str:
    """Append the one-label instruction unless the stem already has it."""

    stripped = prompt.rstrip()
    if ONE_LABEL_INSTRUCTION.lower() in stripped.lower():
        return stripped
    return f"{stripped}\n{ONE_LABEL_INSTRUCTION}"


def sample_train_q_para_prompt(
    labels: Sequence[str],
    *,
    rng: random.Random | None = None,
) -> str:
    """Uniformly sample one train stem, including the frozen MMSU eval stem."""

    chooser = rng.choice if rng is not None else random.choice
    stem = chooser(TRAIN_STEMS)
    return ensure_one_label_instruction(render_q_para_prompt(stem, labels))


def assert_task_privilege_has_no_closed_set_label(privilege: str) -> None:
    """Refuse privilege text that names a closed-set emotion label."""

    lowered = privilege.lower()
    for label in CLOSED_SET_LABEL_NAMES:
        if re.search(rf"(?<![a-z]){re.escape(label)}(?![a-z])", lowered):
            raise ValueError(
                "R-para teacher privilege must not name a closed-set label: "
                f"{label!r}"
            )


def attach_r_para_teacher_privilege(public_prompt: str) -> str:
    """Prepend the task-level R-para privilege to the student's public prompt."""

    assert_task_privilege_has_no_closed_set_label(R_PARA_TEACHER_TASK_PRIVILEGE)
    privilege = R_PARA_TEACHER_TASK_PRIVILEGE.strip()
    public = public_prompt.strip()
    if public.startswith(privilege):
        return public
    return f"{privilege}\n{public}"


def eval_q_para_prompt(labels: Sequence[str]) -> str:
    """Return the frozen MMSU-aligned eval / ceiling prompt."""

    return render_q_para_prompt(EVAL_STEM, labels)
