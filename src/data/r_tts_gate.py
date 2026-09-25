"""R-layer TTS gate: sample texts/refs and expand the synthesis grid.

"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from realmg.data.config import load_local_paths, resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report
from realmg.data.r_text_filter import FILTERED_MANIFEST_PATH
from realmg.data.r_vctk import parse_vctk_speaker_info, find_vctk_speaker_info_file


GATE_TEXT_COUNT = 20
GATE_TEXT_SEED = 43
GATE_SPEAKER_SEED = 42
GATE_SPEAKERS_PER_GENDER = 2
GATE_ENGINES = ("indextts2_5", "cosyvoice2")
GATE_Z_LABELS = ("neutral", "happy")

SPLIT_REPORT_PATH = "Data/cards/r_vctk_reference_split.json"
REFERENCE_CLIP_MANIFEST_PATH = "Data/manifests/r_vctk_reference_clips.jsonl"
GATE_TEXTS_PATH = "Data/manifests/r_tts_gate_texts.jsonl"
GATE_SPEAKERS_PATH = "Data/manifests/r_tts_gate_speakers.jsonl"
GATE_JOBS_PATH = "Data/manifests/r_tts_gate_jobs.jsonl"
GATE_REPORT_PATH = "Data/cards/r_tts_gate_plan.json"
GATE_LISTEN_SHEET_PATH = "Data/cards/r_tts_gate_listen_sheet.jsonl"
GATE_AUDIO_ROOT = "Data/audio/r_tts_gate"

INDEXTTS_EMO_ORDER = (
    "happy",
    "angry",
    "sad",
    "afraid",
    "disgusted",
    "melancholic",
    "surprised",
    "calm",
)
INDEXTTS_EMO_VECTORS = {
    "happy": [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "neutral": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
}
COSYVOICE_INSTRUCT_PROMPTS = {
    "happy": "Speak in a happy tone.<|endofprompt|>",
    "neutral": "Speak in a calm, neutral tone.<|endofprompt|>",
}


def load_split_report() -> dict[str, Any]:
    """Load the frozen VCTK reference-speaker split report."""

    path = resolve_from_repo_root(SPLIT_REPORT_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "Missing frozen VCTK split report. Run `realmg prepare-r-vctk` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_reference_clip_index() -> dict[str, dict[str, Any]]:
    """Index frozen VCTK reference clips by speaker_id."""

    path = resolve_from_repo_root(REFERENCE_CLIP_MANIFEST_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "Missing VCTK reference clips. Run `realmg prepare-r-vctk-refs` first."
        )
    return {record["speaker_id"]: record for record in read_jsonl(path)}


def sample_train_gate_speakers(
    *,
    train_speaker_ids: list[str],
    speaker_meta: dict[str, dict[str, Any]],
    reference_clips: dict[str, dict[str, Any]],
    seed: int = GATE_SPEAKER_SEED,
    per_gender: int = GATE_SPEAKERS_PER_GENDER,
) -> list[dict[str, Any]]:
    """Sample 2 female + 2 male speakers from the frozen train pool only."""

    rng = random.Random(seed)
    by_gender: dict[str, list[str]] = {"female": [], "male": []}
    for speaker_id in train_speaker_ids:
        if speaker_id not in reference_clips:
            continue
        gender = speaker_meta[speaker_id]["gender"]
        by_gender[gender].append(speaker_id)

    selected: list[dict[str, Any]] = []
    for gender in ("female", "male"):
        pool = list(by_gender[gender])
        rng.shuffle(pool)
        if len(pool) < per_gender:
            raise ValueError(
                f"Not enough train {gender} speakers with reference clips "
                f"(need {per_gender}, have {len(pool)})."
            )
        for speaker_id in pool[:per_gender]:
            clip = reference_clips[speaker_id]
            selected.append(
                {
                    "speaker_id": speaker_id,
                    "split": "train",
                    "gender": gender,
                    "accent": speaker_meta[speaker_id].get("accent", ""),
                    "region": speaker_meta[speaker_id].get("region", ""),
                    "reference_wav_path": clip["reference_wav_path"],
                    "reference_utterance_id": clip["utterance_id"],
                    "reference_duration_seconds": clip["clip_duration_seconds"],
                    "reference_wer_percent": clip["wer_percent"],
                }
            )
    return selected


def sample_gate_texts(
    *,
    seed: int = GATE_TEXT_SEED,
    count: int = GATE_TEXT_COUNT,
) -> list[dict[str, Any]]:
    """Sample shortlist texts uniformly for the TTS gate."""

    records = read_jsonl(resolve_from_repo_root(FILTERED_MANIFEST_PATH))
    if len(records) < count:
        raise ValueError(
            f"Shortlist has only {len(records)} rows; need {count} for the gate."
        )
    rng = random.Random(seed)
    chosen = rng.sample(records, count)
    chosen.sort(key=lambda row: row["content_id"])
    texts: list[dict[str, Any]] = []
    for index, record in enumerate(chosen):
        texts.append(
            {
                "gate_text_id": f"gate_text_{index:02d}",
                "content_id": record["content_id"],
                "source_name": record["source_name"],
                "prompt_text": record["prompt_text"],
                "answer_text": record["answer_text"],
                "prompt_word_count": record.get("metadata", {}).get(
                    "prompt_word_count"
                ),
                "answer_word_count": record.get("metadata", {}).get(
                    "answer_word_count"
                ),
            }
        )
    return texts


def engine_control_payload(engine: str, z_label: str) -> dict[str, Any]:
    """Return the frozen emotion-control payload for one engine and z."""

    if engine == "indextts2_5":
        return {
            "emo_order": list(INDEXTTS_EMO_ORDER),
            "emo_vector": list(INDEXTTS_EMO_VECTORS[z_label]),
            "use_random": False,
            "lang": "EN",
        }
    if engine == "cosyvoice2":
        return {
            "instruct_text": COSYVOICE_INSTRUCT_PROMPTS[z_label],
        }
    raise ValueError(f"Unknown gate engine: {engine}")


def build_gate_jobs(
    texts: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand texts x speakers x engines x z into synthesis jobs."""

    jobs: list[dict[str, Any]] = []
    for text in texts:
        for speaker in speakers:
            for engine in GATE_ENGINES:
                for z_label in GATE_Z_LABELS:
                    relative_wav = (
                        f"{GATE_AUDIO_ROOT}/{engine}/"
                        f"{speaker['speaker_id']}/{text['gate_text_id']}_{z_label}.wav"
                    )
                    jobs.append(
                        {
                            "job_id": (
                                f"{engine}::{speaker['speaker_id']}::"
                                f"{text['gate_text_id']}::{z_label}"
                            ),
                            "stage": "tts_gate_pilot",
                            "engine": engine,
                            "z_label": z_label,
                            "gate_text_id": text["gate_text_id"],
                            "content_id": text["content_id"],
                            "source_name": text["source_name"],
                            "prompt_text": text["prompt_text"],
                            "speaker_id": speaker["speaker_id"],
                            "speaker_gender": speaker["gender"],
                            "reference_wav_path": speaker["reference_wav_path"],
                            "output_wav_path": relative_wav,
                            "control": engine_control_payload(engine, z_label),
                            "status": "pending",
                        }
                    )
    return jobs


def build_listen_sheet(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build forced-choice listen rows grouped by text/speaker/engine."""

    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for job in jobs:
        key = (job["gate_text_id"], job["speaker_id"], job["engine"])
        group = groups.setdefault(
            key,
            {
                "listen_id": f"{job['engine']}::{job['speaker_id']}::{job['gate_text_id']}",
                "engine": job["engine"],
                "speaker_id": job["speaker_id"],
                "gate_text_id": job["gate_text_id"],
                "content_id": job["content_id"],
                "prompt_text": job["prompt_text"],
                "wav_by_z": {},
                "forced_choice_options": ["neutral", "happy"],
                "human_choice": None,
                "human_correct": None,
                "notes": "",
            },
        )
        group["wav_by_z"][job["z_label"]] = job["output_wav_path"]
    return [groups[key] for key in sorted(groups)]


def build_gate_report(
    texts: list[dict[str, Any]],
    speakers: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the frozen TTS gate plan."""

    return {
        "stage": "tts_gate_pilot",
        "text_count": len(texts),
        "speaker_count": len(speakers),
        "engine_count": len(GATE_ENGINES),
        "z_count": len(GATE_Z_LABELS),
        "job_count": len(jobs),
        "expected_job_count": (
            GATE_TEXT_COUNT
            * GATE_SPEAKERS_PER_GENDER
            * 2
            * len(GATE_ENGINES)
            * len(GATE_Z_LABELS)
        ),
        "text_seed": GATE_TEXT_SEED,
        "speaker_seed": GATE_SPEAKER_SEED,
        "engines": list(GATE_ENGINES),
        "z_labels": list(GATE_Z_LABELS),
        "speakers": [
            {
                "speaker_id": speaker["speaker_id"],
                "gender": speaker["gender"],
                "reference_wav_path": speaker["reference_wav_path"],
            }
            for speaker in speakers
        ],
        "source_counts": dict(
            sorted(Counter(text["source_name"] for text in texts).items())
        ),
        "pass_criteria": {
            "human_forced_choice_min": 0.80,
            "single_utterance_wer_max_percent": 10.0,
            "paired_z_wer_abs_diff_train_dev_max_percent": 5.0,
        },
        "control_defaults": {
            "indextts2_5": INDEXTTS_EMO_VECTORS,
            "cosyvoice2": COSYVOICE_INSTRUCT_PROMPTS,
        },
        "outputs": {
            "texts": GATE_TEXTS_PATH,
            "speakers": GATE_SPEAKERS_PATH,
            "jobs": GATE_JOBS_PATH,
            "listen_sheet": GATE_LISTEN_SHEET_PATH,
            "audio_root": GATE_AUDIO_ROOT,
        },
        "notes": [
            "Refs are train-only (2F+2M). Eval and construction-dev refs are unused.",
            "Run `realmg run-r-tts-gate` after this prepare step to synthesize wavs.",
        ],
    }


def prepare_r_tts_gate() -> None:
    """Freeze the R-layer TTS gate sample and job grid."""

    split_report = load_split_report()
    local_paths = load_local_paths()
    vctk_root = resolve_from_repo_root(local_paths["sources"]["vctk_root"])
    speaker_meta = parse_vctk_speaker_info(find_vctk_speaker_info_file(vctk_root))
    reference_clips = load_reference_clip_index()

    speakers = sample_train_gate_speakers(
        train_speaker_ids=list(split_report["train_speakers"]),
        speaker_meta=speaker_meta,
        reference_clips=reference_clips,
    )
    texts = sample_gate_texts()
    jobs = build_gate_jobs(texts, speakers)
    listen_sheet = build_listen_sheet(jobs)

    write_jsonl(texts, resolve_from_repo_root(GATE_TEXTS_PATH))
    write_jsonl(speakers, resolve_from_repo_root(GATE_SPEAKERS_PATH))
    write_jsonl(jobs, resolve_from_repo_root(GATE_JOBS_PATH))
    write_jsonl(listen_sheet, resolve_from_repo_root(GATE_LISTEN_SHEET_PATH))
    write_report(build_gate_report(texts, speakers, jobs), resolve_from_repo_root(GATE_REPORT_PATH))
