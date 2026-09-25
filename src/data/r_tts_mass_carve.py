"""Promote mass SER survivor provisional_ref_split to the official R carve.

"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report
from realmg.data.r_tts_gate import SPLIT_REPORT_PATH, load_split_report
from realmg.data.r_tts_mass import MASS_JOBS_PATH, MASS_UNITS_PATH
from realmg.data.r_tts_mass_ser import MASS_SER_SURVIVORS_PATH


CARVE_UNITS_PATH = "Data/manifests/r_tts_mass_carved_units.jsonl"
CARVE_JOBS_PATH = "Data/manifests/r_tts_mass_carved_jobs.jsonl"
CARVE_REPORT_PATH = "Data/cards/r_tts_mass_carve_report.json"
OFFICIAL_SPLITS = ("eval", "construction-dev", "train")


def load_ser_survivors() -> list[dict[str, Any]]:
    """Load contrastive-frozen SER survivor units."""

    path = resolve_from_repo_root(MASS_SER_SURVIVORS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing SER survivors at {path}. Run score-r-tts-mass-ser first."
        )
    return read_jsonl(path)


def validate_carve_invariants(
    survivors: list[dict[str, Any]],
    vctk_split: dict[str, Any],
) -> list[str]:
    """Return human-readable invariant failures (empty means OK)."""

    failures: list[str] = []
    speakers_by_official = {
        "eval": set(vctk_split["eval_speakers"]),
        "construction-dev": set(vctk_split["construction_dev_speakers"]),
        "train": set(vctk_split["train_speakers"]),
    }

    content_to_splits: dict[str, set[str]] = defaultdict(set)
    speaker_to_splits: dict[str, set[str]] = defaultdict(set)
    for row in survivors:
        split = row["provisional_ref_split"]
        if split not in OFFICIAL_SPLITS:
            failures.append(f"Unknown provisional_ref_split on {row['unit_id']}: {split}")
            continue
        content_to_splits[row["content_id"]].add(split)
        speaker_to_splits[row["speaker_id"]].add(split)
        expected_pool = speakers_by_official[split]
        if row["speaker_id"] not in expected_pool:
            failures.append(
                f"Speaker {row['speaker_id']} on {row['unit_id']} not in VCTK {split} pool."
            )

    multi_content = {
        content_id: sorted(splits)
        for content_id, splits in content_to_splits.items()
        if len(splits) > 1
    }
    if multi_content:
        failures.append(
            f"{len(multi_content)} content_id values appear in multiple splits "
            f"(first: {next(iter(multi_content.items()))})."
        )

    multi_speaker = {
        speaker_id: sorted(splits)
        for speaker_id, splits in speaker_to_splits.items()
        if len(splits) > 1
    }
    if multi_speaker:
        failures.append(
            f"{len(multi_speaker)} speakers appear in multiple survivor splits."
        )

    return failures


def build_carved_units(
    survivors: list[dict[str, Any]],
    mass_units_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join SER survivors with mass unit metadata and set official split."""

    carved: list[dict[str, Any]] = []
    for survivor in survivors:
        unit_id = survivor["unit_id"]
        mass_unit = mass_units_by_id.get(unit_id)
        if mass_unit is None:
            raise KeyError(f"SER survivor unit missing from mass units: {unit_id}")
        split = survivor["provisional_ref_split"]
        carved.append(
            {
                "unit_id": unit_id,
                "split": split,
                "provisional_ref_split": split,
                "engine": survivor["engine"],
                "mass_text_id": survivor["mass_text_id"],
                "content_id": survivor["content_id"],
                "source_name": mass_unit["source_name"],
                "prompt_text": mass_unit["prompt_text"],
                "answer_text": mass_unit["answer_text"],
                "speaker_id": survivor["speaker_id"],
                "speaker_gender": mass_unit["speaker_gender"],
                "reference_wav_path": mass_unit["reference_wav_path"],
                "neutral_wer_percent": survivor["neutral_wer_percent"],
                "happy_wer_percent": survivor["happy_wer_percent"],
                "abs_delta_wer_percent": survivor["abs_delta_wer_percent"],
                "ser_pass_reason": survivor.get("pass_reason"),
                "absolute_pair_pass": survivor.get("absolute_pair_pass"),
                "contrastive_pass": survivor.get("contrastive_pass"),
                "emotion2vec_contrastive_delta": survivor.get(
                    "emotion2vec_contrastive_delta"
                ),
                "speechbrain_contrastive_delta": survivor.get(
                    "speechbrain_contrastive_delta"
                ),
                "contrastive_delta_min": survivor.get("contrastive_delta_min"),
            }
        )
    carved.sort(key=lambda row: (row["split"], row["engine"], row["unit_id"]))
    return carved


def build_carved_jobs(
    carved_units: list[dict[str, Any]],
    jobs_by_unit: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Keep done mass jobs for carved units; stamp official split."""

    carved_jobs: list[dict[str, Any]] = []
    for unit in carved_units:
        unit_jobs = jobs_by_unit.get(unit["unit_id"], [])
        z_labels = {job["z_label"] for job in unit_jobs if job.get("status") == "done"}
        if z_labels != {"neutral", "happy"}:
            raise ValueError(
                f"Carved unit {unit['unit_id']} missing done jobs for both z "
                f"(have {sorted(z_labels)})."
            )
        for job in unit_jobs:
            if job.get("status") != "done":
                continue
            carved_jobs.append(
                {
                    **job,
                    "split": unit["split"],
                    "provisional_ref_split": unit["provisional_ref_split"],
                }
            )
    carved_jobs.sort(key=lambda row: (row["split"], row["job_id"]))
    return carved_jobs


def summarize_carve(
    carved_units: list[dict[str, Any]],
    carved_jobs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the carve report card."""

    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in carved_units:
        by_split[row["split"]].append(row)

    split_summaries: dict[str, Any] = {}
    for split in OFFICIAL_SPLITS:
        rows = by_split.get(split, [])
        split_summaries[split] = {
            "unit_count": len(rows),
            "unique_content_count": len({row["content_id"] for row in rows}),
            "unique_speaker_count": len({row["speaker_id"] for row in rows}),
            "engine_counts": dict(sorted(Counter(row["engine"] for row in rows).items())),
            "ser_pass_reason_counts": dict(
                sorted(Counter(row.get("ser_pass_reason") for row in rows).items())
            ),
        }

    return {
        "stage": "tts_mass_carve_a",
        "policy": "promote_provisional_ref_split",
        "unit_count": len(carved_units),
        "job_count": len(carved_jobs),
        "splits": split_summaries,
        "outputs": {
            "units": CARVE_UNITS_PATH,
            "jobs": CARVE_JOBS_PATH,
            "report": CARVE_REPORT_PATH,
            "ser_survivors_source": MASS_SER_SURVIVORS_PATH,
            "vctk_split_source": SPLIT_REPORT_PATH,
        },
        "notes": [
            "Official split equals provisional_ref_split from the mass draw.",
            "No reshuffle: speaker pools already disjoint via frozen VCTK split.",
        ],
    }


def carve_r_tts_mass_from_provisional() -> None:
    """Freeze carve by promoting provisional_ref_split on SER survivors."""

    survivors = load_ser_survivors()
    vctk_split = load_split_report()
    failures = validate_carve_invariants(survivors, vctk_split)
    if failures:
        raise RuntimeError("Carve invariant failures:\n- " + "\n- ".join(failures))

    mass_units = read_jsonl(resolve_from_repo_root(MASS_UNITS_PATH))
    mass_units_by_id = {unit["unit_id"]: unit for unit in mass_units}
    carved_units = build_carved_units(survivors, mass_units_by_id)

    jobs = read_jsonl(resolve_from_repo_root(MASS_JOBS_PATH))
    jobs_by_unit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for job in jobs:
        jobs_by_unit[job["unit_id"]].append(job)
    carved_jobs = build_carved_jobs(carved_units, jobs_by_unit)

    write_jsonl(carved_units, resolve_from_repo_root(CARVE_UNITS_PATH))
    write_jsonl(carved_jobs, resolve_from_repo_root(CARVE_JOBS_PATH))
    report = summarize_carve(carved_units, carved_jobs)
    write_report(report, resolve_from_repo_root(CARVE_REPORT_PATH))
    print(
        f"[carve-r-tts-mass] units={report['unit_count']} jobs={report['job_count']} "
        f"eval={report['splits']['eval']['unit_count']} "
        f"dev={report['splits']['construction-dev']['unit_count']} "
        f"train={report['splits']['train']['unit_count']}",
        flush=True,
    )
