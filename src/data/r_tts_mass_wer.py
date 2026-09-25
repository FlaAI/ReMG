"""WER scoring for the R-layer mass TTS pool."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import write_jsonl, write_report
from realmg.data.r_tts_gate_wer import (
    WHISPER_MODEL_NAME,
    append_wer_cache,
    load_audio_for_whisper,
    load_wer_cache,
    reference_text_for_gate_job,
)
from realmg.data.r_tts_mass_synth import load_mass_jobs
from realmg.data.wer import load_whisper_model, wer_percent

MASS_SINGLE_WER_MAX_PERCENT = 20.0
MASS_PAIRED_DELTA_WER_MAX_PERCENT = 10.0
EVAL_PAIRED_DELTA_WER_MAX_PERCENT = 5.0
MASS_WER_SCORES_PATH = "Data/cards/r_tts_mass_wer_scores.jsonl"
MASS_WER_PAIRS_PATH = "Data/cards/r_tts_mass_wer_pairs.jsonl"
MASS_WER_SURVIVORS_PATH = "Data/manifests/r_tts_mass_wer_survivor_units.jsonl"
MASS_WER_REPORT_PATH = "Data/cards/r_tts_mass_wer_report.json"


def apply_mass_single_wer_pass(row: dict[str, Any]) -> dict[str, Any]:
    """Recompute single_pass from wer_percent under the mass cutoff."""

    row = dict(row)
    row["single_pass"] = float(row["wer_percent"]) <= MASS_SINGLE_WER_MAX_PERCENT
    return row


def paired_delta_threshold_percent(provisional_ref_split: str) -> float:
    """Return the |ΔWER| cutoff for one provisional ref pool."""

    if provisional_ref_split == "eval":
        return EVAL_PAIRED_DELTA_WER_MAX_PERCENT
    return MASS_PAIRED_DELTA_WER_MAX_PERCENT


def transcribe_with_whisper(audio_path: Path, *, model: Any) -> str:
    """Transcribe one wav with a loaded Whisper model."""

    audio = load_audio_for_whisper(audio_path)
    result = model.transcribe(
        audio,
        language="en",
        fp16=torch.cuda.is_available(),
    )
    return str(result["text"]).strip()


def score_mass_jobs_with_whisper(
    jobs: list[dict[str, Any]],
    *,
    limit: int | None = None,
    resume: bool = True,
    flush_every: int = 50,
) -> list[dict[str, Any]]:
    """Transcribe and score done mass jobs, resuming from cache."""

    cache_path = resolve_from_repo_root(MASS_WER_SCORES_PATH)
    cached = load_wer_cache(cache_path) if resume else {}
    scored: list[dict[str, Any]] = []
    model = None

    done_jobs = [job for job in jobs if job.get("status") == "done"]
    processed_new = 0
    for index, job in enumerate(done_jobs, start=1):
        job_id = job["job_id"]
        if job_id in cached:
            scored.append(apply_mass_single_wer_pass(cached[job_id]))
            continue
        if limit is not None and processed_new >= limit:
            break

        if model is None:
            model = load_whisper_model(WHISPER_MODEL_NAME)

        audio_path = resolve_from_repo_root(job["output_wav_path"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing mass wav: {audio_path}")

        reference = reference_text_for_gate_job(job)
        hypothesis = transcribe_with_whisper(audio_path, model=model)
        row = apply_mass_single_wer_pass(
            {
                "job_id": job_id,
                "unit_id": job["unit_id"],
                "engine": job["engine"],
                "speaker_id": job["speaker_id"],
                "mass_text_id": job["mass_text_id"],
                "content_id": job["content_id"],
                "z_label": job["z_label"],
                "provisional_ref_split": job["provisional_ref_split"],
                "output_wav_path": job["output_wav_path"],
                "reference_text": reference,
                "hypothesis_text": hypothesis,
                "wer_percent": wer_percent(reference, hypothesis),
                "whisper_model": WHISPER_MODEL_NAME,
            }
        )
        append_wer_cache(cache_path, row)
        cached[job_id] = row
        scored.append(row)
        processed_new += 1
        if processed_new % flush_every == 0 or processed_new == 1:
            print(
                f"[score-r-tts-mass-wer] new={processed_new} "
                f"total_scored={len(scored)}/{len(done_jobs)} "
                f"{job_id} wer={row['wer_percent']:.2f}%",
                flush=True,
            )

    # Include cached rows not revisited when limit stops early.
    if limit is None:
        return scored
    merged = {row["job_id"]: row for row in scored}
    return list(merged.values())


def evaluate_paired_mass_wer(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach paired |WER_neutral - WER_happy| pass flags per mass unit."""

    by_unit: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    unit_meta: dict[str, dict[str, Any]] = {}
    for row in scores:
        by_unit[row["unit_id"]][row["z_label"]] = row
        unit_meta[row["unit_id"]] = {
            "engine": row["engine"],
            "speaker_id": row["speaker_id"],
            "mass_text_id": row["mass_text_id"],
            "content_id": row["content_id"],
            "provisional_ref_split": row["provisional_ref_split"],
        }

    pair_rows: list[dict[str, Any]] = []
    for unit_id in sorted(by_unit):
        z_map = by_unit[unit_id]
        if "neutral" not in z_map or "happy" not in z_map:
            continue
        meta = unit_meta[unit_id]
        neutral = z_map["neutral"]["wer_percent"]
        happy = z_map["happy"]["wer_percent"]
        abs_diff = abs(neutral - happy)
        delta_max = paired_delta_threshold_percent(meta["provisional_ref_split"])
        single_pass_both = bool(
            neutral <= MASS_SINGLE_WER_MAX_PERCENT
            and happy <= MASS_SINGLE_WER_MAX_PERCENT
        )
        paired_pass = abs_diff <= delta_max
        pair_rows.append(
            {
                "unit_id": unit_id,
                "engine": meta["engine"],
                "speaker_id": meta["speaker_id"],
                "mass_text_id": meta["mass_text_id"],
                "content_id": meta["content_id"],
                "provisional_ref_split": meta["provisional_ref_split"],
                "neutral_wer_percent": neutral,
                "happy_wer_percent": happy,
                "abs_delta_wer_percent": abs_diff,
                "paired_delta_max_percent": delta_max,
                "single_pass_both": single_pass_both,
                "paired_pass": paired_pass,
                "wer_pass": single_pass_both and paired_pass,
            }
        )
    return pair_rows


def build_survivor_units(pair_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep unit rows that pass single and paired WER rules."""

    survivors: list[dict[str, Any]] = []
    for row in pair_rows:
        if not row["wer_pass"]:
            continue
        survivors.append(
            {
                "unit_id": row["unit_id"],
                "engine": row["engine"],
                "mass_text_id": row["mass_text_id"],
                "content_id": row["content_id"],
                "speaker_id": row["speaker_id"],
                "provisional_ref_split": row["provisional_ref_split"],
                "neutral_wer_percent": row["neutral_wer_percent"],
                "happy_wer_percent": row["happy_wer_percent"],
                "abs_delta_wer_percent": row["abs_delta_wer_percent"],
            }
        )
    return survivors


def summarize_mass_wer(
    scores: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    survivors: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the mass WER report card."""

    by_engine_scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_engine_scores[row["engine"]].append(row)

    by_engine_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_engine_pairs[row["engine"]].append(row)

    by_pool_survivors: Counter[str] = Counter(
        row["provisional_ref_split"] for row in survivors
    )

    engine_summaries: dict[str, Any] = {}
    for engine, engine_scores in sorted(by_engine_scores.items()):
        pairs = by_engine_pairs.get(engine, [])
        wers = [row["wer_percent"] for row in engine_scores]
        engine_summaries[engine] = {
            "utterance_count": len(engine_scores),
            "single_pass_count": sum(1 for row in engine_scores if row["single_pass"]),
            "single_pass_rate": (
                sum(1 for row in engine_scores if row["single_pass"]) / len(engine_scores)
                if engine_scores
                else None
            ),
            "wer_mean": sum(wers) / len(wers) if wers else None,
            "pair_count": len(pairs),
            "complete_pair_count": len(pairs),
            "wer_pass_count": sum(1 for row in pairs if row["wer_pass"]),
            "wer_pass_rate": (
                sum(1 for row in pairs if row["wer_pass"]) / len(pairs) if pairs else None
            ),
        }

    return {
        "stage": "tts_mass_wer",
        "whisper_model": WHISPER_MODEL_NAME,
        "single_wer_max_percent": MASS_SINGLE_WER_MAX_PERCENT,
        "paired_delta_wer_max_percent_train_dev": MASS_PAIRED_DELTA_WER_MAX_PERCENT,
        "paired_delta_wer_max_percent_eval": EVAL_PAIRED_DELTA_WER_MAX_PERCENT,
        "utterance_count": len(scores),
        "pair_count": len(pair_rows),
        "survivor_unit_count": len(survivors),
        "survivor_provisional_ref_split_counts": dict(sorted(by_pool_survivors.items())),
        "engines": engine_summaries,
        "outputs": {
            "scores": MASS_WER_SCORES_PATH,
            "pairs": MASS_WER_PAIRS_PATH,
            "survivors": MASS_WER_SURVIVORS_PATH,
            "report": MASS_WER_REPORT_PATH,
        },
        "notes": [
            "Reference text uses spoken_text when present, else de-LaTeX prompt text.",
            "Incomplete units (missing one z or failed synthesis) are excluded from pairs.",
        ],
    }


def score_r_tts_mass_wer(*, limit: int | None = None, resume: bool = True) -> None:
    """Run Whisper WER on done mass TTS wavs and write survivor units."""

    jobs = load_mass_jobs()
    scores = score_mass_jobs_with_whisper(jobs, limit=limit, resume=resume)
    if limit is None:
        write_jsonl(scores, resolve_from_repo_root(MASS_WER_SCORES_PATH))
    pair_rows = evaluate_paired_mass_wer(scores)
    survivors = build_survivor_units(pair_rows)
    write_jsonl(pair_rows, resolve_from_repo_root(MASS_WER_PAIRS_PATH))
    write_jsonl(survivors, resolve_from_repo_root(MASS_WER_SURVIVORS_PATH))
    write_report(
        summarize_mass_wer(scores, pair_rows, survivors),
        resolve_from_repo_root(MASS_WER_REPORT_PATH),
    )
