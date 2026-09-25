"""WER scoring for the R-layer TTS gate."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import torchaudio.functional as F

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import write_jsonl, write_report
from realmg.data.r_tts_gate import GATE_JOBS_PATH
from realmg.data.r_tts_gate_synth import normalize_text_for_cosyvoice
from realmg.data.wer import load_whisper_model, wer_percent


WHISPER_MODEL_NAME = "medium"
SINGLE_WER_MAX_PERCENT = 10.0
PAIRED_DELTA_WER_MAX_PERCENT = 5.0
GATE_WER_SCORES_PATH = "Data/cards/r_tts_gate_wer_scores.jsonl"
GATE_WER_REPORT_PATH = "Data/cards/r_tts_gate_wer_report.json"


def ensure_parent(path: Path) -> None:
    """Create the parent directory for one output path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def load_gate_jobs() -> list[dict[str, Any]]:
    """Load frozen TTS gate jobs."""

    path = resolve_from_repo_root(GATE_JOBS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "Missing TTS gate jobs. Run `realmg prepare-r-tts-gate` first."
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_audio_for_whisper(audio_path: Path) -> np.ndarray:
    """Load arbitrary wav/flac and resample to mono 16 kHz float32."""

    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = F.resample(waveform, sample_rate, 16000)
    return waveform.squeeze(0).detach().cpu().numpy().astype(np.float32)


def transcribe_gate_audio(audio_path: Path, model_name: str = WHISPER_MODEL_NAME) -> str:
    """Transcribe one gate wav with Whisper-medium."""

    model = load_whisper_model(model_name)
    audio = load_audio_for_whisper(audio_path)
    result = model.transcribe(
        audio,
        language="en",
        fp16=torch.cuda.is_available(),
    )
    return str(result["text"]).strip()


def reference_text_for_gate_job(job: dict[str, Any]) -> str:
    """Choose the WER reference text for one gate job.

    engines. CosyVoice may have stored spoken_text; IndexTTS used raw prompt_text.
    """

    spoken = job.get("spoken_text")
    if isinstance(spoken, str) and spoken.strip():
        return spoken.strip()
    return normalize_text_for_cosyvoice(job["prompt_text"])


def load_wer_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load cached gate WER rows keyed by job_id."""

    if not cache_path.exists():
        return {}
    cached: dict[str, dict[str, Any]] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cached[row["job_id"]] = row
    return cached


def append_wer_cache(cache_path: Path, row: dict[str, Any]) -> None:
    """Append one scored gate WER row."""

    ensure_parent(cache_path)
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def score_gate_jobs_with_whisper(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transcribe and score all done gate jobs, resuming from cache."""

    cache_path = resolve_from_repo_root(GATE_WER_SCORES_PATH)
    cached = load_wer_cache(cache_path)
    scored: list[dict[str, Any]] = []

    done_jobs = [job for job in jobs if job.get("status") == "done"]
    for index, job in enumerate(done_jobs, start=1):
        job_id = job["job_id"]
        if job_id in cached:
            scored.append(cached[job_id])
            continue

        audio_path = resolve_from_repo_root(job["output_wav_path"])
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing gate wav: {audio_path}")

        reference = reference_text_for_gate_job(job)
        hypothesis = transcribe_gate_audio(audio_path)
        row = {
            "job_id": job_id,
            "engine": job["engine"],
            "speaker_id": job["speaker_id"],
            "gate_text_id": job["gate_text_id"],
            "z_label": job["z_label"],
            "content_id": job["content_id"],
            "output_wav_path": job["output_wav_path"],
            "reference_text": reference,
            "hypothesis_text": hypothesis,
            "wer_percent": wer_percent(reference, hypothesis),
            "whisper_model": WHISPER_MODEL_NAME,
            "single_pass": None,
        }
        row["single_pass"] = row["wer_percent"] <= SINGLE_WER_MAX_PERCENT
        append_wer_cache(cache_path, row)
        scored.append(row)
        print(
            f"[score-r-tts-gate-wer] {index}/{len(done_jobs)} "
            f"{job_id} wer={row['wer_percent']:.2f}%",
            flush=True,
        )
    return scored


def evaluate_paired_gate_wer(scores: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach paired |WER_neutral - WER_happy| pass flags."""

    by_pair: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scores:
        key = (row["engine"], row["speaker_id"], row["gate_text_id"])
        by_pair[key][row["z_label"]] = row

    pair_rows: list[dict[str, Any]] = []
    for (engine, speaker_id, gate_text_id), z_map in sorted(by_pair.items()):
        if "neutral" not in z_map or "happy" not in z_map:
            continue
        neutral = z_map["neutral"]["wer_percent"]
        happy = z_map["happy"]["wer_percent"]
        abs_diff = abs(neutral - happy)
        pair_rows.append(
            {
                "engine": engine,
                "speaker_id": speaker_id,
                "gate_text_id": gate_text_id,
                "neutral_wer_percent": neutral,
                "happy_wer_percent": happy,
                "abs_delta_wer_percent": abs_diff,
                "single_pass_both": bool(
                    z_map["neutral"]["single_pass"] and z_map["happy"]["single_pass"]
                ),
                "paired_pass": abs_diff <= PAIRED_DELTA_WER_MAX_PERCENT,
                "gate_pass": bool(
                    z_map["neutral"]["single_pass"]
                    and z_map["happy"]["single_pass"]
                    and abs_diff <= PAIRED_DELTA_WER_MAX_PERCENT
                ),
            }
        )
    return pair_rows


def summarize_gate_wer(
    scores: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the gate WER report card."""

    by_engine_scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_engine_scores[row["engine"]].append(row)

    by_engine_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_engine_pairs[row["engine"]].append(row)

    engine_summaries: dict[str, Any] = {}
    for engine, engine_scores in sorted(by_engine_scores.items()):
        wers = [row["wer_percent"] for row in engine_scores]
        pairs = by_engine_pairs.get(engine, [])
        engine_summaries[engine] = {
            "utterance_count": len(engine_scores),
            "single_pass_count": sum(1 for row in engine_scores if row["single_pass"]),
            "single_pass_rate": (
                sum(1 for row in engine_scores if row["single_pass"]) / len(engine_scores)
                if engine_scores
                else None
            ),
            "wer_mean": sum(wers) / len(wers) if wers else None,
            "wer_max": max(wers) if wers else None,
            "pair_count": len(pairs),
            "paired_pass_count": sum(1 for row in pairs if row["paired_pass"]),
            "paired_pass_rate": (
                sum(1 for row in pairs if row["paired_pass"]) / len(pairs)
                if pairs
                else None
            ),
            "gate_pass_count": sum(1 for row in pairs if row["gate_pass"]),
            "gate_pass_rate": (
                sum(1 for row in pairs if row["gate_pass"]) / len(pairs)
                if pairs
                else None
            ),
        }

    return {
        "stage": "tts_gate_pilot_wer",
        "whisper_model": WHISPER_MODEL_NAME,
        "single_wer_max_percent": SINGLE_WER_MAX_PERCENT,
        "paired_delta_wer_max_percent": PAIRED_DELTA_WER_MAX_PERCENT,
        "utterance_count": len(scores),
        "pair_count": len(pair_rows),
        "engines": engine_summaries,
        "outputs": {
            "scores": GATE_WER_SCORES_PATH,
            "pairs": "Data/cards/r_tts_gate_wer_pairs.jsonl",
            "report": GATE_WER_REPORT_PATH,
        },
        "notes": [
            "Reference text uses de-LaTeX spoken form (spoken_text if present).",
            "Human forced-choice is scored separately on listen trials.",
        ],
    }


def score_r_tts_gate_wer() -> None:
    """Run Whisper WER on all done TTS gate wavs and write the report."""

    jobs = load_gate_jobs()
    scores = score_gate_jobs_with_whisper(jobs)
    # Rewrite cache as a clean jsonl in job order for readability.
    write_jsonl(scores, resolve_from_repo_root(GATE_WER_SCORES_PATH))
    pair_rows = evaluate_paired_gate_wer(scores)
    write_jsonl(pair_rows, resolve_from_repo_root("Data/cards/r_tts_gate_wer_pairs.jsonl"))
    write_report(summarize_gate_wer(scores, pair_rows), resolve_from_repo_root(GATE_WER_REPORT_PATH))
