"""R-layer mass TTS draw: 10k shortlist texts with stratified engine/ref assignment.

"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report
from realmg.data.r_text_filter import FILTERED_MANIFEST_PATH
from realmg.data.r_tts_gate import (
    COSYVOICE_INSTRUCT_PROMPTS,
    GATE_ENGINES,
    GATE_Z_LABELS,
    INDEXTTS_EMO_ORDER,
    INDEXTTS_EMO_VECTORS,
    REFERENCE_CLIP_MANIFEST_PATH,
    SPLIT_REPORT_PATH,
    engine_control_payload,
    load_reference_clip_index,
    load_split_report,
)


MASS_TEXT_COUNT = 10_000
MASS_TEXT_SEED = 45
MASS_ASSIGN_SEED = 46
# 750 eval @ ~0.40 pair pass ⇒ ~300 survivors; user accepted vs frozen 400 target.
MASS_EVAL_TEXT_QUOTA = 750
MASS_CONSTRUCTION_DEV_TEXT_QUOTA = 1_000
MASS_UNITS_PER_ENGINE = 5_000
MASS_HALF_SUBSAMPLE_SEED = 47
MASS_AUDIO_ROOT = "Data/audio/r_tts_mass"
MASS_UNITS_PATH = "Data/manifests/r_tts_mass_units.jsonl"
MASS_JOBS_PATH = "Data/manifests/r_tts_mass_jobs.jsonl"
MASS_UNITS_20K_BACKUP_PATH = "Data/manifests/r_tts_mass_units_20k.jsonl"
MASS_JOBS_20K_BACKUP_PATH = "Data/manifests/r_tts_mass_jobs_20k.jsonl"
MASS_REPORT_PATH = "Data/cards/r_tts_mass_plan.json"

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_content_token(content_id: str) -> str:
    """Make content_id safe for filenames."""

    return _SAFE_ID_RE.sub("__", content_id)


def sample_mass_texts(*, seed: int = MASS_TEXT_SEED, count: int = MASS_TEXT_COUNT) -> list[dict[str, Any]]:
    """Uniformly sample shortlist texts for mass TTS."""

    records = read_jsonl(resolve_from_repo_root(FILTERED_MANIFEST_PATH))
    if len(records) < count:
        raise ValueError(f"Shortlist has {len(records)} rows; need {count}.")
    rng = random.Random(seed)
    chosen = rng.sample(records, count)
    chosen.sort(key=lambda row: row["content_id"])
    texts: list[dict[str, Any]] = []
    for index, record in enumerate(chosen):
        texts.append(
            {
                "mass_text_id": f"mass_text_{index:05d}",
                "content_id": record["content_id"],
                "source_name": record["source_name"],
                "prompt_text": record["prompt_text"],
                "answer_text": record["answer_text"],
            }
        )
    return texts


def assign_ref_pool_labels(
    count: int,
    *,
    eval_quota: int = MASS_EVAL_TEXT_QUOTA,
    construction_dev_quota: int = MASS_CONSTRUCTION_DEV_TEXT_QUOTA,
) -> list[str]:
    """Build provisional split labels for stratified VCTK ref assignment."""

    if eval_quota + construction_dev_quota > count:
        raise ValueError("Ref-pool quotas exceed mass text count.")
    labels = (
        ["eval"] * eval_quota
        + ["construction-dev"] * construction_dev_quota
        + ["train"] * (count - eval_quota - construction_dev_quota)
    )
    return labels


def build_mass_units(
    texts: list[dict[str, Any]],
    *,
    split_report: dict[str, Any],
    reference_clips: dict[str, dict[str, Any]],
    assign_seed: int = MASS_ASSIGN_SEED,
) -> list[dict[str, Any]]:
    """Assign one engine and one VCTK ref to each mass text (two z later)."""

    rng = random.Random(assign_seed)
    pool_labels = assign_ref_pool_labels(len(texts))
    rng.shuffle(pool_labels)

    engines = list(GATE_ENGINES)
    engine_labels = (engines * ((len(texts) + 1) // 2))[: len(texts)]
    rng.shuffle(engine_labels)

    speakers_by_pool = {
        "eval": [sid for sid in split_report["eval_speakers"] if sid in reference_clips],
        "construction-dev": [
            sid
            for sid in split_report["construction_dev_speakers"]
            if sid in reference_clips
        ],
        "train": [sid for sid in split_report["train_speakers"] if sid in reference_clips],
    }
    for pool_name, speaker_ids in speakers_by_pool.items():
        if not speaker_ids:
            raise ValueError(f"No reference clips for pool {pool_name}.")

    units: list[dict[str, Any]] = []
    for text, pool_label, engine in zip(texts, pool_labels, engine_labels, strict=True):
        speaker_id = rng.choice(speakers_by_pool[pool_label])
        clip = reference_clips[speaker_id]
        units.append(
            {
                "unit_id": f"{engine}::{text['mass_text_id']}::{speaker_id}",
                "mass_text_id": text["mass_text_id"],
                "content_id": text["content_id"],
                "source_name": text["source_name"],
                "prompt_text": text["prompt_text"],
                "answer_text": text["answer_text"],
                "engine": engine,
                "provisional_ref_split": pool_label,
                "speaker_id": speaker_id,
                "speaker_gender": clip["gender"],
                "reference_wav_path": clip["reference_wav_path"],
            }
        )
    return units


def build_mass_jobs(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Expand each mass unit into neutral/happy synthesis jobs."""

    jobs: list[dict[str, Any]] = []
    for unit in units:
        content_token = safe_content_token(unit["content_id"])
        for z_label in GATE_Z_LABELS:
            relative_wav = (
                f"{MASS_AUDIO_ROOT}/{unit['engine']}/"
                f"{unit['speaker_id']}/{unit['mass_text_id']}_{z_label}.wav"
            )
            jobs.append(
                {
                    "job_id": f"{unit['unit_id']}::{z_label}",
                    "stage": "tts_mass",
                    "unit_id": unit["unit_id"],
                    "mass_text_id": unit["mass_text_id"],
                    "content_id": unit["content_id"],
                    "content_token": content_token,
                    "source_name": unit["source_name"],
                    "prompt_text": unit["prompt_text"],
                    "engine": unit["engine"],
                    "z_label": z_label,
                    "provisional_ref_split": unit["provisional_ref_split"],
                    "speaker_id": unit["speaker_id"],
                    "speaker_gender": unit["speaker_gender"],
                    "reference_wav_path": unit["reference_wav_path"],
                    "output_wav_path": relative_wav,
                    "control": engine_control_payload(unit["engine"], z_label),
                    "status": "pending",
                }
            )
    return jobs


def build_mass_report(
    texts: list[dict[str, Any]],
    units: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the frozen mass TTS draw and expected yield."""

    gate_pass_rate = 0.40
    expected_survivors = int(round(len(units) * gate_pass_rate))
    return {
        "stage": "tts_mass_draw",
        "text_count": len(texts),
        "unit_count": len(units),
        "job_count": len(jobs),
        "text_seed": MASS_TEXT_SEED,
        "assign_seed": MASS_ASSIGN_SEED,
        "engines": list(GATE_ENGINES),
        "z_labels": list(GATE_Z_LABELS),
        "engine_counts": dict(sorted(Counter(unit["engine"] for unit in units).items())),
        "provisional_ref_split_counts": dict(
            sorted(Counter(unit["provisional_ref_split"] for unit in units).items())
        ),
        "source_counts": dict(
            sorted(Counter(text["source_name"] for text in texts).items())
        ),
        "yield_estimate": {
            "gate_pair_pass_rate_assumed": gate_pass_rate,
            "expected_surviving_units_before_ser": expected_survivors,
            "eval_target": 300,
            "eval_text_quota": MASS_EVAL_TEXT_QUOTA,
            "expected_eval_survivors_before_ser": int(
                round(MASS_EVAL_TEXT_QUOTA * gate_pass_rate)
            ),
            "notes": [
                "Assumes both engines remain after human forced-choice.",
                "SER attrition is not included yet.",
            ],
        },
        "control_defaults": {
            "indextts2_5": {
                "emo_order": list(INDEXTTS_EMO_ORDER),
                "emo_vectors": INDEXTTS_EMO_VECTORS,
            },
            "cosyvoice2": COSYVOICE_INSTRUCT_PROMPTS,
        },
        "outputs": {
            "units": MASS_UNITS_PATH,
            "jobs": MASS_JOBS_PATH,
            "audio_root": MASS_AUDIO_ROOT,
            "report": MASS_REPORT_PATH,
        },
    }


def _jobs_by_unit_id(jobs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group synthesis jobs by unit_id."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        grouped.setdefault(job["unit_id"], []).append(job)
    return grouped


def _unit_done_count(unit_id: str, jobs_by_unit: dict[str, list[dict[str, Any]]]) -> int:
    """Count finished z jobs for one unit."""

    return sum(job.get("status") == "done" for job in jobs_by_unit.get(unit_id, []))


def _per_engine_pool_targets() -> dict[str, int]:
    """Split ref-pool quotas evenly across the two engines."""

    if MASS_EVAL_TEXT_QUOTA % 2 or MASS_CONSTRUCTION_DEV_TEXT_QUOTA % 2:
        raise ValueError("Eval/dev quotas must be even for balanced engine assignment.")
    train_per_engine = (
        MASS_UNITS_PER_ENGINE
        - MASS_EVAL_TEXT_QUOTA // 2
        - MASS_CONSTRUCTION_DEV_TEXT_QUOTA // 2
    )
    return {
        "eval": MASS_EVAL_TEXT_QUOTA // 2,
        "construction-dev": MASS_CONSTRUCTION_DEV_TEXT_QUOTA // 2,
        "train": train_per_engine,
    }


def subsample_mass_units(
    units: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    *,
    per_engine: int = MASS_UNITS_PER_ENGINE,
    subsample_seed: int = MASS_HALF_SUBSAMPLE_SEED,
) -> list[dict[str, Any]]:
    """Keep a stratified half-scale subset, preferring units with finished audio."""

    jobs_by_unit = _jobs_by_unit_id(jobs)
    pool_targets = _per_engine_pool_targets()
    if sum(pool_targets.values()) != per_engine:
        raise ValueError("Per-engine pool targets do not sum to per_engine quota.")

    selected: list[dict[str, Any]] = []
    for engine in GATE_ENGINES:
        engine_units = [unit for unit in units if unit["engine"] == engine]
        for pool_label, target in pool_targets.items():
            pool_units = [
                unit
                for unit in engine_units
                if unit["provisional_ref_split"] == pool_label
            ]
            if len(pool_units) < target:
                raise ValueError(
                    f"Engine {engine} pool {pool_label} has {len(pool_units)} units; "
                    f"need {target}."
                )
            pool_units.sort(
                key=lambda unit: (
                    -_unit_done_count(unit["unit_id"], jobs_by_unit),
                    unit["mass_text_id"],
                )
            )
            selected.extend(pool_units[:target])
    selected.sort(key=lambda unit: unit["unit_id"])
    return selected


def rebuild_mass_jobs_preserving_status(
    units: list[dict[str, Any]],
    prior_jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand units to jobs while keeping any existing synthesis status."""

    prior_by_id = {job["job_id"]: job for job in prior_jobs}
    jobs = build_mass_jobs(units)
    for job in jobs:
        old = prior_by_id.get(job["job_id"])
        if old is None:
            continue
        job["status"] = old.get("status", "pending")
        if "error" in old:
            job["error"] = old["error"]
    return jobs


def downscale_r_tts_mass() -> None:
    """Halve the frozen mass draw to 5k units per engine, keeping finished audio."""

    units_path = resolve_from_repo_root(MASS_UNITS_PATH)
    jobs_path = resolve_from_repo_root(MASS_JOBS_PATH)
    if not units_path.exists() or not jobs_path.exists():
        raise FileNotFoundError(
            "Missing mass TTS manifests. Run `realmg prepare-r-tts-mass` first."
        )

    units = read_jsonl(units_path)
    jobs = read_jsonl(jobs_path)
    if len(units) != 20_000:
        raise ValueError(
            f"Expected 20k units before downscale; found {len(units)}. "
            "Already downscaled or manifests changed."
        )

    write_jsonl(units, resolve_from_repo_root(MASS_UNITS_20K_BACKUP_PATH))
    write_jsonl(jobs, resolve_from_repo_root(MASS_JOBS_20K_BACKUP_PATH))

    kept_units = subsample_mass_units(units, jobs)
    kept_jobs = rebuild_mass_jobs_preserving_status(kept_units, jobs)
    texts = [
        {
            "mass_text_id": unit["mass_text_id"],
            "content_id": unit["content_id"],
            "source_name": unit["source_name"],
            "prompt_text": unit["prompt_text"],
            "answer_text": unit["answer_text"],
        }
        for unit in kept_units
    ]
    write_jsonl(kept_units, units_path)
    write_jsonl(kept_jobs, jobs_path)
    report = build_mass_report(texts, kept_units, kept_jobs)
    report["downscale"] = {
            "from_unit_count": len(units),
            "to_unit_count": len(kept_units),
            "subsample_seed": MASS_HALF_SUBSAMPLE_SEED,
            "backup_units": MASS_UNITS_20K_BACKUP_PATH,
            "backup_jobs": MASS_JOBS_20K_BACKUP_PATH,
            "preserved_done_jobs": sum(
                job.get("status") == "done" for job in kept_jobs
            ),
        }
    write_report(report, resolve_from_repo_root(MASS_REPORT_PATH))


def prepare_r_tts_mass() -> None:
    """Freeze the 10k mass TTS draw, engine/ref assignment, and job grid."""

    split_report = load_split_report()
    reference_clips = load_reference_clip_index()
    texts = sample_mass_texts(count=MASS_TEXT_COUNT)
    units = build_mass_units(
        texts,
        split_report=split_report,
        reference_clips=reference_clips,
    )
    jobs = build_mass_jobs(units)
    write_jsonl(units, resolve_from_repo_root(MASS_UNITS_PATH))
    write_jsonl(jobs, resolve_from_repo_root(MASS_JOBS_PATH))
    write_report(
        build_mass_report(texts, units, jobs),
        resolve_from_repo_root(MASS_REPORT_PATH),
    )
