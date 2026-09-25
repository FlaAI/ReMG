"""Synthesize wavs for the frozen R-layer TTS gate jobs."""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from realmg.data.config import load_local_paths, resolve_from_repo_root, repository_root
from realmg.data.esd import write_jsonl, write_report
from realmg.data.r_tts_gate import GATE_JOBS_PATH, GATE_REPORT_PATH


_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_MATH_SPAN_RE = re.compile(r"\$\$([^$]+)\$\$|\$([^$]+)\$")


def ensure_parent(path: Path) -> None:
    """Create the parent directory for one output file."""

    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_text_for_cosyvoice(text: str) -> str:
    """Strip LaTeX math markup that CosyVoice frontend rejects with empty asserts.

    """

    def replace_math(match: re.Match[str]) -> str:
        inner = match.group(1) if match.group(1) is not None else match.group(2)
        replacements = {
            r"\ddot": " double-dot ",
            r"\dot": " dot ",
            r"\cos": " cosine ",
            r"\sin": " sine ",
            r"\tan": " tangent ",
            r"\max": " max ",
            r"\min": " min ",
            r"\exp": " exp ",
            r"\mid": " given ",
            r"\cdot": " times ",
            r"\times": " times ",
            r"\leq": " less or equal ",
            r"\geq": " greater or equal ",
            r"\neq": " not equal ",
            r"\infty": " infinity ",
        }
        # Longer command names first so \ddot is not eaten by \dot.
        for source, target in sorted(replacements.items(), key=lambda item: -len(item[0])):
            inner = inner.replace(source, target)
        inner = _LATEX_CMD_RE.sub(" ", inner)
        inner = re.sub(r"[{}^_\\]", " ", inner)
        return " " + " ".join(inner.split()) + " "

    spoken = _MATH_SPAN_RE.sub(replace_math, text)
    spoken = spoken.replace("\\", " ")
    spoken = spoken.replace("[", " ").replace("]", " ")
    return " ".join(spoken.split())


def resolve_model_root(config_key: str, default_relative: str) -> Path:
    """Resolve one local model root from local_paths or a default."""

    local_paths = load_local_paths()
    configured = local_paths.get("models", {}).get(config_key, "")
    if configured:
        return resolve_from_repo_root(configured)
    return resolve_from_repo_root(default_relative)


def load_gate_jobs() -> list[dict[str, Any]]:
    """Load the frozen TTS gate job list."""

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


def rewrite_gate_jobs(jobs: list[dict[str, Any]]) -> None:
    """Persist updated job statuses back to the gate jobs manifest."""

    write_jsonl(jobs, resolve_from_repo_root(GATE_JOBS_PATH))


class IndexTTSGateSynthesizer:
    """Lazy wrapper around IndexTTS-2.5 for gate jobs."""

    def __init__(self) -> None:
        self._tts = None

    def _ensure_loaded(self) -> None:
        if self._tts is not None:
            return

        index_tts_root = repository_root() / "third_party" / "index-tts"
        if str(index_tts_root) not in sys.path:
            sys.path.insert(0, str(index_tts_root))

        from indextts.infer_v2_5 import IndexTTS2

        model_dir = resolve_model_root("indextts2_root", "Data/models/indextts2_5")
        cfg_path = model_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"IndexTTS-2.5 config missing at {cfg_path}. "
                "Set models.indextts2_root in Data/configs/local_paths.toml."
            )
        self._tts = IndexTTS2(
            cfg_path=str(cfg_path),
            model_dir=str(model_dir),
            use_bf16=torch.cuda.is_available(),
        )

    def synthesize(self, job: dict[str, Any]) -> Path:
        """Synthesize one IndexTTS gate job to its output wav path."""

        self._ensure_loaded()
        assert self._tts is not None
        output_path = resolve_from_repo_root(job["output_wav_path"])
        ensure_parent(output_path)
        reference_wav = resolve_from_repo_root(job["reference_wav_path"])
        control = job["control"]
        self._tts.infer(
            spk_audio_prompt=str(reference_wav),
            text=job["prompt_text"],
            output_path=str(output_path),
            emo_vector=control["emo_vector"],
            use_random=control.get("use_random", False),
            lang=control.get("lang", "EN"),
            verbose=False,
        )
        return output_path

    def warm_up(self) -> None:
        """Load model weights once before the job loop."""

        self._ensure_loaded()


class CosyVoiceGateSynthesizer:
    """Lazy wrapper around CosyVoice2 Instruct for gate jobs."""

    def __init__(self) -> None:
        self._tts = None
        self._sample_rate = None

    def _ensure_loaded(self) -> None:
        if self._tts is not None:
            return

        cosy_root = repository_root() / "third_party" / "CosyVoice"
        matcha_root = cosy_root / "third_party" / "Matcha-TTS"
        for path in (cosy_root, matcha_root):
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from cosyvoice.cli.cosyvoice import AutoModel

        model_dir = resolve_model_root("cosyvoice2_root", "Data/models/cosyvoice2")
        if not (model_dir / "cosyvoice2.yaml").exists():
            raise FileNotFoundError(
                f"CosyVoice2 config missing under {model_dir}. "
                "Set models.cosyvoice2_root in Data/configs/local_paths.toml."
            )
        self._tts = AutoModel(model_dir=str(model_dir))
        self._sample_rate = self._tts.sample_rate

    def synthesize(self, job: dict[str, Any]) -> Path:
        """Synthesize one CosyVoice2 gate job to its output wav path."""

        self._ensure_loaded()
        assert self._tts is not None
        assert self._sample_rate is not None
        output_path = resolve_from_repo_root(job["output_wav_path"])
        ensure_parent(output_path)
        reference_wav = resolve_from_repo_root(job["reference_wav_path"])
        instruct_text = job["control"]["instruct_text"]

        spoken_text = normalize_text_for_cosyvoice(job["prompt_text"])
        job["spoken_text"] = spoken_text
        # prompts; CosyVoice inference still works with text_frontend=False.
        speech_chunks = []
        for chunk in self._tts.inference_instruct2(
            spoken_text,
            instruct_text,
            str(reference_wav),
            text_frontend=False,
        ):
            speech_chunks.append(chunk["tts_speech"])
        if not speech_chunks:
            raise RuntimeError(f"CosyVoice2 returned no audio for {job['job_id']}")
        import torchaudio

        waveform = torch.cat(speech_chunks, dim=-1)
        torchaudio.save(str(output_path), waveform, self._sample_rate)
        return output_path

    def warm_up(self) -> None:
        """Load model weights once before the job loop."""

        self._ensure_loaded()


def run_r_tts_gate(
    *,
    engine: str = "all",
    limit: int | None = None,
    resume: bool = True,
) -> None:
    """Synthesize pending TTS gate jobs for one or both engines."""

    jobs = load_gate_jobs()
    if engine not in {"all", "indextts2_5", "cosyvoice2"}:
        raise ValueError("--engine must be all, indextts2_5, or cosyvoice2")

    indextts = IndexTTSGateSynthesizer() if engine in {"all", "indextts2_5"} else None
    cosyvoice = CosyVoiceGateSynthesizer() if engine in {"all", "cosyvoice2"} else None

    # Fail fast on missing engine deps / checkpoints before marking jobs failed.
    if indextts is not None:
        indextts.warm_up()
    if cosyvoice is not None:
        cosyvoice.warm_up()

    processed = 0
    skipped = 0
    failed = 0
    attempted = 0

    for job in jobs:
        if engine != "all" and job["engine"] != engine:
            continue
        output_path = resolve_from_repo_root(job["output_wav_path"])
        if resume and job.get("status") == "done" and output_path.exists():
            skipped += 1
            continue
        if limit is not None and attempted >= limit:
            break

        attempted += 1
        try:
            if job["engine"] == "indextts2_5":
                assert indextts is not None
                indextts.synthesize(job)
            elif job["engine"] == "cosyvoice2":
                assert cosyvoice is not None
                cosyvoice.synthesize(job)
            else:
                raise ValueError(f"Unknown engine in job: {job['engine']}")
            job["status"] = "done"
            job.pop("error", None)
            processed += 1
            print(
                f"[run-r-tts-gate] done {processed} job_id={job['job_id']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - per-job failure isolation
            job["status"] = "failed"
            detail = repr(exc) if str(exc) == "" else str(exc)
            job["error"] = f"{detail}\n{traceback.format_exc(limit=3)}"
            failed += 1
            print(
                f"[run-r-tts-gate] failed job_id={job['job_id']} error={detail}",
                flush=True,
            )

        if attempted % 10 == 0:
            rewrite_gate_jobs(jobs)

    rewrite_gate_jobs(jobs)
    status_counts: dict[str, int] = {}
    for job in jobs:
        if engine != "all" and job["engine"] != engine:
            continue
        status_counts[job.get("status", "pending")] = (
            status_counts.get(job.get("status", "pending"), 0) + 1
        )
    write_report(
        {
            "jobs_manifest": GATE_JOBS_PATH,
            "engine_filter": engine,
            "processed_count": processed,
            "skipped_existing_count": skipped,
            "failed_count": failed,
            "attempted_count": attempted,
            "status_counts": status_counts,
            "requested_limit": limit,
            "resume": resume,
        },
        resolve_from_repo_root("Data/cards/r_tts_gate_run_report.json"),
    )
