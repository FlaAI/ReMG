"""Word error rate helpers for RealMG data filtering."""

from __future__ import annotations

import re
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from jiwer import wer as compute_word_error_rate


def normalize_transcript_text(text: str) -> str:
    """Normalize transcript text before WER computation."""

    lowered = text.lower().strip()
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def wer_percent(reference: str, hypothesis: str) -> float:
    """Compute word error rate as a percentage."""

    normalized_reference = normalize_transcript_text(reference)
    normalized_hypothesis = normalize_transcript_text(hypothesis)
    if not normalized_reference:
        return 100.0
    return compute_word_error_rate(normalized_reference, normalized_hypothesis) * 100.0


@lru_cache(maxsize=1)
def load_whisper_model(model_name: str) -> Any:
    """Load and cache a Whisper model."""

    import whisper

    return whisper.load_model(model_name)


def load_audio_array(audio_path: Path) -> np.ndarray:
    """Load mono 16 kHz float32 audio for Whisper without ffmpeg."""

    with wave.open(str(audio_path), "rb") as handle:
        sample_rate = handle.getframerate()
        num_channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        num_frames = handle.getnframes()
        raw_bytes = handle.readframes(num_frames)

    if sample_width == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    if num_channels > 1:
        audio = audio.reshape(-1, num_channels).mean(axis=1)

    if sample_rate != 16000:
        raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz: {audio_path}")

    return audio


def transcribe_audio_file(audio_path: Path, model_name: str) -> str:
    """Transcribe one audio file with Whisper."""

    import torch

    model = load_whisper_model(model_name)
    audio = load_audio_array(audio_path)
    result = model.transcribe(
        audio,
        language="en",
        fp16=torch.cuda.is_available(),
    )
    return str(result["text"]).strip()
