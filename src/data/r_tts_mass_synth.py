"""Synthesize wavs for the frozen R-layer mass TTS jobs."""

from __future__ import annotations

import json
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import write_jsonl, write_report
from realmg.data.r_tts_gate_synth import CosyVoiceGateSynthesizer, IndexTTSGateSynthesizer
from realmg.data.r_tts_mass import MASS_JOBS_PATH


def load_mass_jobs() -> list[dict[str, Any]]:
    """Load the frozen mass TTS job list."""

    path = resolve_from_repo_root(MASS_JOBS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "Missing mass TTS jobs. Run `realmg prepare-r-tts-mass` first."
        )
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rewrite_mass_jobs(jobs: list[dict[str, Any]]) -> None:
    """Persist mass job statuses."""

    write_jsonl(jobs, resolve_from_repo_root(MASS_JOBS_PATH))


def run_r_tts_mass(
    *,
    engine: str = "all",
    limit: int | None = None,
    resume: bool = True,
) -> None:
    """Synthesize pending mass TTS jobs for one or both engines."""

    jobs = load_mass_jobs()
    if engine not in {"all", "indextts2_5", "cosyvoice2"}:
        raise ValueError("--engine must be all, indextts2_5, or cosyvoice2")

    indextts = IndexTTSGateSynthesizer() if engine in {"all", "indextts2_5"} else None
    cosyvoice = CosyVoiceGateSynthesizer() if engine in {"all", "cosyvoice2"} else None
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
            if processed % 10 == 0 or processed == 1:
                print(
                    f"[run-r-tts-mass] done {processed} attempted={attempted} "
                    f"job_id={job['job_id']}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - per-job failure isolation
            job["status"] = "failed"
            detail = repr(exc) if str(exc) == "" else str(exc)
            job["error"] = f"{detail}\n{traceback.format_exc(limit=3)}"
            failed += 1
            print(
                f"[run-r-tts-mass] failed job_id={job['job_id']} error={detail}",
                flush=True,
            )

        if attempted % 20 == 0:
            rewrite_mass_jobs(jobs)

    rewrite_mass_jobs(jobs)
    status_counts = Counter(
        job.get("status", "pending")
        for job in jobs
        if engine == "all" or job["engine"] == engine
    )
    write_report(
        {
            "jobs_manifest": MASS_JOBS_PATH,
            "engine_filter": engine,
            "processed_count": processed,
            "skipped_existing_count": skipped,
            "failed_count": failed,
            "attempted_count": attempted,
            "status_counts": dict(status_counts),
            "requested_limit": limit,
            "resume": resume,
        },
        resolve_from_repo_root("Data/cards/r_tts_mass_run_report.json"),
    )
