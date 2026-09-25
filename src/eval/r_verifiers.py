"""Answer verifiers for R-layer Teacher-text screening.

"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from realmg.data.config import load_local_paths, resolve_from_repo_root


GENERAL_VERIFIER_HF_REPO = "TIGER-Lab/general-verifier"
GENERAL_VERIFIER_PROMPT = (
    "User: ### Question: {question}\n\n"
    "### Ground Truth Answer: {ground_truth}\n\n"
    "### Student Answer: {student_answer}\n\n"
    "For the above question, please verify if the student's answer is equivalent to "
    "the ground truth answer.\n"
    "Do not solve the question by yourself; just check if the student's answer is "
    "equivalent to the ground truth answer.\n"
    'If the student\'s answer is correct, output "Final Decision: Yes". '
    'If the student\'s answer is incorrect, output "Final Decision: No". Assistant:'
)
MCQ_REFERENCE_PATTERN = re.compile(r"^[A-Da-d]$")
MCQ_PROMPT_HINTS = (
    "multiple choice",
    "multiple-choice",
    "choose one:",
    "choose the correct",
    "answer with a or b",
    "answer with a, b",
    "options:",
    "option a",
    "option b",
    "(a)",
    "(b)",
    "(c)",
    "(d)",
)


@dataclass
class VerificationResult:
    """One Teacher-text correctness decision."""

    is_correct: bool | None
    verifier_method: str
    normalized_prediction: str | None = None
    detail: str | None = None


def resolve_general_verifier_root() -> Path:
    """Resolve the local General Verifier checkpoint root."""

    local_paths = load_local_paths()
    configured_root = local_paths.get("models", {}).get("general_verifier_root", "")
    if configured_root:
        return resolve_from_repo_root(configured_root)
    return resolve_from_repo_root(f"Data/models/{GENERAL_VERIFIER_HF_REPO.split('/')[-1]}")


class GeneralVerifierScorer:
    """Lazy GPU wrapper around TIGER-Lab/general-verifier."""

    def __init__(self, model_root: str | None = None) -> None:
        self.model_root = model_root
        self._tokenizer = None
        self._model = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _ensure_loaded(self) -> None:
        """Load the verifier model on first use."""

        if self._model is not None:
            return

        model_path = self.model_root or str(resolve_general_verifier_root())
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
        ).to(self._device)
        self._model.eval()

    def verify(self, question: str, ground_truth: str, student_answer: str) -> bool:
        """Return whether the student answer matches the reference."""

        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None

        prompt = GENERAL_VERIFIER_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            student_answer=student_answer,
        )
        # Truncate from the end of the prompt when the combined input exceeds the
        # verifier's context window.  The fixed instructions live at the start of
        # GENERAL_VERIFIER_PROMPT, and the final "Assistant:" trigger is also at the
        # start of the suffix, so truncation keeps the most critical framing while
        # only trimming the trailing student answer.  Cap at 2048 tokens to avoid
        # blowing up the KV cache on a small laptop GPU.
        max_new_tokens = 256
        max_model_len = getattr(self._tokenizer, "model_max_length", None)
        if max_model_len is not None and isinstance(max_model_len, int) and max_model_len > 0:
            max_input_len = min(max_model_len - max_new_tokens - 10, 2048)
        else:
            max_input_len = 2048
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
        ).to(self._device)
        prompt_length = int(inputs["input_ids"].shape[-1])
        try:
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            generated = outputs[0][prompt_length:]
            decoded = self._tokenizer.decode(generated, skip_special_tokens=True)
        except Exception as exc:  # noqa: BLE001
            # Gracefully degrade on GPU OOM / context edge cases during long
            # R eval scoring runs.  Treat a failed verifier call as incorrect.
            print(
                f"[GeneralVerifierScorer] verifier generation failed: {exc}; "
                f"question_len={len(question)} gt_len={len(ground_truth)} "
                f"answer_len={len(student_answer)} input_len={prompt_length}",
                flush=True,
            )
            return False
        lowered = decoded.lower()
        yes_hit = "final decision: yes" in lowered
        no_hit = "final decision: no" in lowered
        if yes_hit and not no_hit:
            return True
        if no_hit and not yes_hit:
            return False
        # Prefer the last explicit decision if both appear in the chain.
        yes_pos = lowered.rfind("final decision: yes")
        no_pos = lowered.rfind("final decision: no")
        if yes_pos < 0 and no_pos < 0:
            return False
        return yes_pos > no_pos


@lru_cache(maxsize=1)
def pedant_scorer():
    """Return a cached PEDANT scorer."""

    from qa_metrics.pedant import PEDANT

    return PEDANT()


def looks_like_mcq_prompt(prompt_text: str) -> bool:
    """Detect VoiceBench-style multiple-choice prompts."""

    lowered = prompt_text.lower()
    return any(hint in lowered for hint in MCQ_PROMPT_HINTS)


def assign_verifier_method(
    source_name: str,
    prompt_text: str,
    reference_answer: str,
) -> str:
    """Route one example to the frozen per-source verifier."""

    if source_name == "natural_reasoning":
        return "general_verifier"
    reference = reference_answer.strip()
    if MCQ_REFERENCE_PATTERN.fullmatch(reference) and looks_like_mcq_prompt(prompt_text):
        return "mcq"
    return "pedant"


def extract_mcq_option(response_text: str) -> str | None:
    """Extract A/B/C/D from a model response using VoiceBench mcq rules."""

    from realmg.eval.voicebench_mcq import extract_mcq_answer

    return extract_mcq_answer(response_text)


def verify_teacher_prediction(
    *,
    source_name: str,
    prompt_text: str,
    reference_answer: str,
    prediction_text: str,
    general_verifier: GeneralVerifierScorer | None = None,
) -> VerificationResult:
    """Score one Teacher-text prediction with the routed verifier."""

    method = assign_verifier_method(source_name, prompt_text, reference_answer)
    prediction = prediction_text.strip()
    if not prediction:
        return VerificationResult(
            is_correct=False,
            verifier_method=method,
            normalized_prediction=None,
            detail="empty_prediction",
        )

    if method == "general_verifier":
        scorer = general_verifier or GeneralVerifierScorer()
        is_correct = scorer.verify(prompt_text, reference_answer, prediction)
        return VerificationResult(
            is_correct=is_correct,
            verifier_method=method,
            normalized_prediction=prediction,
            detail="general_verifier",
        )

    if method == "mcq":
        extracted = extract_mcq_option(prediction)
        reference = reference_answer.strip().upper()
        is_correct = extracted == reference if extracted is not None else False
        return VerificationResult(
            is_correct=is_correct,
            verifier_method=method,
            normalized_prediction=extracted,
            detail="voicebench_mcq",
        )

    pedant = pedant_scorer()
    is_correct = pedant.evaluate(
        [reference_answer.lower()],
        prediction.lower(),
        prompt_text.lower(),
    )
    return VerificationResult(
        is_correct=bool(is_correct),
        verifier_method=method,
        normalized_prediction=prediction.lower(),
        detail="pedant",
    )
