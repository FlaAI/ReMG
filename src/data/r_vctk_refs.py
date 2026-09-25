"""Select neutral VCTK reference clips for R-layer zero-shot TTS cloning.

"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from realmg.data.config import load_local_paths, resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report
from realmg.data.r_vctk import (
    VCTK_SPEAKERS_WITHOUT_TEXT,
    find_vctk_speaker_info_file,
)
from realmg.data.wer import load_whisper_model, wer_percent


WHISPER_MODEL_NAME = "medium"
REFERENCE_WER_MAX_PERCENT = 5.0
REFERENCE_DURATION_MIN_SECONDS = 3.0
REFERENCE_DURATION_MAX_SECONDS = 5.0
REFERENCE_TRIM_SECONDS = 4.0
REFERENCE_FALLBACK_MAX_SECONDS = 10.0
MAX_NATIVE_CANDIDATES_TO_SCORE = 30
MAX_FALLBACK_CANDIDATES_TO_SCORE = 15
VCTK_REFERENCE_MIC = "mic1"
REFERENCE_WER_CACHE_PATH = "Data/cards/r_vctk_reference_wer_scores.jsonl"
REFERENCE_CLIP_MANIFEST_PATH = "Data/manifests/r_vctk_reference_clips.jsonl"
REFERENCE_CLIP_REPORT_PATH = "Data/cards/r_vctk_reference_clips.json"
REFERENCE_WAV_DIR = "Data/audio/refs/vctk"
SPLIT_REPORT_PATH = "Data/cards/r_vctk_reference_split.json"


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def load_frozen_vctk_split_report() -> dict[str, Any]:
    """Load the frozen R-layer VCTK speaker split report."""

    report_path = resolve_from_repo_root(SPLIT_REPORT_PATH)
    if not report_path.exists():
        raise FileNotFoundError(
            "Missing frozen VCTK split report. Run `realmg prepare-r-vctk` first."
        )
    return json.loads(report_path.read_text(encoding="utf-8"))


def vctk_wav_root(vctk_root: Path) -> Path:
    """Return the trimmed 48 kHz FLAC tree for VCTK 0.92."""

    return vctk_root / "wav48_silence_trimmed"


def vctk_txt_root(vctk_root: Path) -> Path:
    """Return the transcript tree for VCTK 0.92."""

    return vctk_root / "txt"


def flac_duration_seconds(flac_path: Path) -> float:
    """Read one FLAC file duration with ffprobe."""

    output = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(flac_path),
        ],
        text=True,
    )
    payload = json.loads(output)
    return float(payload["format"]["duration"])


def load_flac_mono_16k(
    flac_path: Path,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> np.ndarray:
    """Decode a FLAC segment to mono 16 kHz float32 samples with ffmpeg."""

    command = [
        "ffmpeg",
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        str(start_seconds),
        "-i",
        str(flac_path),
    ]
    if duration_seconds is not None:
        command.extend(["-t", str(duration_seconds)])
    command.extend(
        [
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ]
    )
    raw_bytes = subprocess.check_output(command)
    if not raw_bytes:
        raise ValueError(f"ffmpeg returned empty audio for {flac_path}")
    return np.frombuffer(raw_bytes, dtype=np.float32).copy()


def write_mono_16k_wav(audio: np.ndarray, output_path: Path) -> None:
    """Write mono float32 samples as a 16 kHz PCM WAV file."""

    import wave

    ensure_parent(output_path)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(pcm.tobytes())


def transcribe_audio_array(audio: np.ndarray, model_name: str) -> str:
    """Transcribe one in-memory audio array with Whisper."""

    import torch

    model = load_whisper_model(model_name)
    result = model.transcribe(
        audio,
        language="en",
        fp16=torch.cuda.is_available(),
    )
    return str(result["text"]).strip()


def load_reference_wer_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    """Load cached VCTK reference WER rows keyed by cache id."""

    if not cache_path.exists():
        return {}
    return {row["cache_id"]: row for row in read_jsonl(cache_path)}


def append_reference_wer_cache(cache_path: Path, row: dict[str, Any]) -> None:
    """Append one cached VCTK reference WER row."""

    ensure_parent(cache_path)
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def relative_repo_path(path: Path) -> str:
    """Convert an absolute path to a repository-relative POSIX path."""

    repo_root = resolve_from_repo_root(".")
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def read_vctk_transcript(txt_root: Path, speaker_id: str, utterance_id: str) -> str:
    """Read one official VCTK transcript file."""

    transcript_path = txt_root / speaker_id / f"{utterance_id}.txt"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Missing VCTK transcript: {transcript_path}")
    return transcript_path.read_text(encoding="utf-8").strip()


def build_reference_candidate(
    speaker_id: str,
    utterance_id: str,
    flac_path: Path,
    source_duration_seconds: float,
    clip_start_seconds: float,
    clip_duration_seconds: float,
) -> dict[str, Any]:
    """Build one selectable VCTK reference clip candidate."""

    return {
        "speaker_id": speaker_id,
        "utterance_id": utterance_id,
        "source_flac_path": relative_repo_path(flac_path),
        "source_duration_seconds": source_duration_seconds,
        "clip_start_seconds": clip_start_seconds,
        "clip_duration_seconds": clip_duration_seconds,
        "selection_mode": (
            "native_duration"
            if (
                REFERENCE_DURATION_MIN_SECONDS
                <= source_duration_seconds
                <= REFERENCE_DURATION_MAX_SECONDS
            )
            else "trimmed_prefix"
        ),
    }


def list_native_duration_candidates(
    wav_root: Path,
    speaker_id: str,
) -> list[dict[str, Any]]:
    """List mic1 utterances whose full duration already fits the 3-5 s window."""

    speaker_dir = wav_root / speaker_id
    candidates: list[dict[str, Any]] = []

    for flac_path in sorted(speaker_dir.glob(f"{speaker_id}_*_{VCTK_REFERENCE_MIC}.flac")):
        utterance_id = flac_path.stem.rsplit("_", 1)[0]
        duration_seconds = flac_duration_seconds(flac_path)
        if not (
            REFERENCE_DURATION_MIN_SECONDS
            <= duration_seconds
            <= REFERENCE_DURATION_MAX_SECONDS
        ):
            continue
        candidates.append(
            build_reference_candidate(
                speaker_id,
                utterance_id,
                flac_path,
                duration_seconds,
                clip_start_seconds=0.0,
                clip_duration_seconds=duration_seconds,
            )
        )

    candidates.sort(
        key=lambda item: (
            abs(item["clip_duration_seconds"] - 3.5),
            item["utterance_id"],
        )
    )
    return candidates


def list_trimmed_fallback_candidates(
    wav_root: Path,
    speaker_id: str,
) -> list[dict[str, Any]]:
    """List mic1 utterances that can supply a trimmed 4 s prefix clip."""

    speaker_dir = wav_root / speaker_id
    candidates: list[dict[str, Any]] = []

    for flac_path in sorted(speaker_dir.glob(f"{speaker_id}_*_{VCTK_REFERENCE_MIC}.flac")):
        utterance_id = flac_path.stem.rsplit("_", 1)[0]
        duration_seconds = flac_duration_seconds(flac_path)
        if duration_seconds <= REFERENCE_DURATION_MAX_SECONDS:
            continue
        if duration_seconds > REFERENCE_FALLBACK_MAX_SECONDS:
            continue
        candidates.append(
            build_reference_candidate(
                speaker_id,
                utterance_id,
                flac_path,
                duration_seconds,
                clip_start_seconds=0.0,
                clip_duration_seconds=REFERENCE_TRIM_SECONDS,
            )
        )

    candidates.sort(key=lambda item: (item["source_duration_seconds"], item["utterance_id"]))
    return candidates


def candidate_cache_id(candidate: dict[str, Any]) -> str:
    """Build a stable cache id for one scored clip candidate."""

    return (
        f"{candidate['speaker_id']}::{candidate['utterance_id']}::"
        f"{candidate['clip_start_seconds']:.3f}::"
        f"{candidate['clip_duration_seconds']:.3f}"
    )


def score_reference_candidate(
    candidate: dict[str, Any],
    txt_root: Path,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
) -> dict[str, Any]:
    """Transcribe one candidate clip and attach WER against the official text."""

    cache_id = candidate_cache_id(candidate)
    if cache_id in cache:
        return cache[cache_id]

    flac_path = resolve_from_repo_root(candidate["source_flac_path"])
    reference_transcript = read_vctk_transcript(
        txt_root,
        candidate["speaker_id"],
        candidate["utterance_id"],
    )
    audio = load_flac_mono_16k(
        flac_path,
        start_seconds=candidate["clip_start_seconds"],
        duration_seconds=candidate["clip_duration_seconds"],
    )
    hypothesis = transcribe_audio_array(audio, WHISPER_MODEL_NAME)
    score_row = {
        **candidate,
        "cache_id": cache_id,
        "reference_transcript": reference_transcript,
        "hypothesis_transcript": hypothesis,
        "wer_percent": wer_percent(reference_transcript, hypothesis),
        "whisper_model": WHISPER_MODEL_NAME,
    }
    cache[cache_id] = score_row
    append_reference_wer_cache(cache_path, score_row)
    return score_row


def pick_best_valid_candidate(
    candidates: list[dict[str, Any]],
    txt_root: Path,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
    max_to_score: int,
) -> dict[str, Any] | None:
    """Return the lowest-WER candidate within a bounded scoring budget."""

    best_valid: dict[str, Any] | None = None
    for candidate in candidates[:max_to_score]:
        scored_row = score_reference_candidate(candidate, txt_root, cache, cache_path)
        if scored_row["wer_percent"] > REFERENCE_WER_MAX_PERCENT:
            continue
        if best_valid is None or scored_row["wer_percent"] < best_valid["wer_percent"]:
            best_valid = scored_row
        if scored_row["wer_percent"] == 0.0:
            break
    return best_valid


def export_selected_reference_clip(
    selected: dict[str, Any],
    split_name: str,
    speaker_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write the selected clip to wav and return one manifest row."""

    flac_path = resolve_from_repo_root(selected["source_flac_path"])
    audio = load_flac_mono_16k(
        flac_path,
        start_seconds=selected["clip_start_seconds"],
        duration_seconds=selected["clip_duration_seconds"],
    )
    output_wav_path = resolve_from_repo_root(
        f"{REFERENCE_WAV_DIR}/{selected['speaker_id']}.wav"
    )
    write_mono_16k_wav(audio, output_wav_path)

    return {
        "speaker_id": selected["speaker_id"],
        "split": split_name,
        "gender": speaker_metadata["gender"],
        "accent": speaker_metadata.get("accent", ""),
        "region": speaker_metadata.get("region", ""),
        "utterance_id": selected["utterance_id"],
        "source_flac_path": selected["source_flac_path"],
        "reference_wav_path": relative_repo_path(output_wav_path),
        "clip_start_seconds": selected["clip_start_seconds"],
        "clip_duration_seconds": selected["clip_duration_seconds"],
        "source_duration_seconds": selected["source_duration_seconds"],
        "selection_mode": selected["selection_mode"],
        "reference_transcript": selected["reference_transcript"],
        "hypothesis_transcript": selected["hypothesis_transcript"],
        "wer_percent": selected["wer_percent"],
        "whisper_model": WHISPER_MODEL_NAME,
    }


def select_reference_clip_for_speaker(
    wav_root: Path,
    txt_root: Path,
    speaker_id: str,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
) -> dict[str, Any] | None:
    """Pick one valid 3-5 s neutral reference clip for a VCTK speaker."""

    native_candidates = list_native_duration_candidates(wav_root, speaker_id)
    selected = pick_best_valid_candidate(
        native_candidates,
        txt_root,
        cache,
        cache_path,
        MAX_NATIVE_CANDIDATES_TO_SCORE,
    )
    if selected is not None:
        return selected

    fallback_candidates = list_trimmed_fallback_candidates(wav_root, speaker_id)
    return pick_best_valid_candidate(
        fallback_candidates,
        txt_root,
        cache,
        cache_path,
        MAX_FALLBACK_CANDIDATES_TO_SCORE,
    )


def build_reference_clip_report(
    manifest_rows: list[dict[str, Any]],
    missing_speakers: list[str],
) -> dict[str, Any]:
    """Summarize VCTK reference clip selection."""

    split_counts: dict[str, int] = {}
    wer_values: list[float] = []
    selection_modes: dict[str, int] = {}

    for row in manifest_rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        wer_values.append(float(row["wer_percent"]))
        mode = row["selection_mode"]
        selection_modes[mode] = selection_modes.get(mode, 0) + 1

    return {
        "source": "VCTK-Corpus-0.92",
        "whisper_model": WHISPER_MODEL_NAME,
        "wer_max_percent": REFERENCE_WER_MAX_PERCENT,
        "duration_window_seconds": [
            REFERENCE_DURATION_MIN_SECONDS,
            REFERENCE_DURATION_MAX_SECONDS,
        ],
        "max_native_candidates_to_score": MAX_NATIVE_CANDIDATES_TO_SCORE,
        "max_fallback_candidates_to_score": MAX_FALLBACK_CANDIDATES_TO_SCORE,
        "trim_seconds": REFERENCE_TRIM_SECONDS,
        "mic": VCTK_REFERENCE_MIC,
        "selected_speaker_count": len(manifest_rows),
        "missing_speaker_count": len(missing_speakers),
        "missing_speakers": missing_speakers,
        "split_counts": split_counts,
        "selection_mode_counts": selection_modes,
        "wer_summary": {
            "min": min(wer_values) if wer_values else None,
            "max": max(wer_values) if wer_values else None,
            "mean": sum(wer_values) / len(wer_values) if wer_values else None,
        },
    }


def prepare_r_vctk_reference_clips() -> None:
    """Select and export one neutral reference clip per frozen VCTK speaker."""

    local_paths = load_local_paths()
    vctk_root = resolve_from_repo_root(local_paths["sources"]["vctk_root"])
    find_vctk_speaker_info_file(vctk_root)
    split_report = load_frozen_vctk_split_report()
    wav_root = vctk_wav_root(vctk_root)
    txt_root = vctk_txt_root(vctk_root)
    cache_path = resolve_from_repo_root(REFERENCE_WER_CACHE_PATH)
    cache = load_reference_wer_cache(cache_path)

    manifest_rows: list[dict[str, Any]] = []
    missing_speakers: list[str] = []
    speaker_items = sorted(split_report["speakers"].items())

    for index, (speaker_id, speaker_metadata) in enumerate(speaker_items, start=1):
        if speaker_id in VCTK_SPEAKERS_WITHOUT_TEXT:
            continue

        selected = select_reference_clip_for_speaker(
            wav_root,
            txt_root,
            speaker_id,
            cache,
            cache_path,
        )
        if selected is None:
            missing_speakers.append(speaker_id)
            print(
                f"No valid reference clip for {speaker_id} "
                f"({index}/{len(speaker_items)})",
                flush=True,
            )
            continue

        manifest_rows.append(
            export_selected_reference_clip(
                selected,
                speaker_metadata["split"],
                speaker_metadata,
            )
        )
        print(
            f"Selected {speaker_id} "
            f"({selected['utterance_id']}, WER={selected['wer_percent']:.2f}%) "
            f"[{index}/{len(speaker_items)}]",
            flush=True,
        )

    write_jsonl(
        manifest_rows,
        resolve_from_repo_root(REFERENCE_CLIP_MANIFEST_PATH),
    )
    write_report(
        build_reference_clip_report(manifest_rows, missing_speakers),
        resolve_from_repo_root(REFERENCE_CLIP_REPORT_PATH),
    )
