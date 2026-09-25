"""Build R-layer utterances and 2x2 quadruplets from the carved mass pool.

"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report
from realmg.data.r_tts_mass_carve import CARVE_JOBS_PATH, CARVE_UNITS_PATH


R_PARALINGUISTIC_PAIR = ("neutral", "happy")
R_PAIRING_SEED = 42
R_UTTERANCES_PATH = "Data/manifests/r_utterances_mass_carved.jsonl"
R_QUADRUPLETS_PATH = "Data/manifests/r_quadruplets_mass_carved.jsonl"
R_UNPAIRED_UNITS_PATH = "Data/manifests/r_quadruplet_unpaired_units.jsonl"
R_QUADRUPLET_REPORT_PATH = "Data/cards/r_quadruplet_mass_carved_report.json"
SPLITS = ("eval", "construction-dev", "train")


def load_carved_units_and_jobs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load official carved units and jobs."""

    units_path = resolve_from_repo_root(CARVE_UNITS_PATH)
    jobs_path = resolve_from_repo_root(CARVE_JOBS_PATH)
    if not units_path.exists() or not jobs_path.exists():
        raise FileNotFoundError(
            "Missing carved mass manifests. Run `realmg carve-r-tts-mass` first."
        )
    return read_jsonl(units_path), read_jsonl(jobs_path)


def build_r_utterance_from_job(
    job: dict[str, Any],
    unit: dict[str, Any],
) -> dict[str, Any]:
    """Map one done mass job to the shared utterance schema."""

    z_label = job["z_label"]
    transcript = job.get("spoken_text") or job["prompt_text"]
    return {
        "utterance_id": job["job_id"],
        "layer": "R",
        "source_name": job["source_name"],
        "split": job["split"],
        "content_id": job["content_id"],
        "paralinguistic_label": z_label,
        "audio_path": job["output_wav_path"],
        "transcript": transcript,
        "speaker_id": job["speaker_id"],
        "reference_speaker_id": job["speaker_id"],
        "tts_engine": job["engine"],
        "metadata": {
            "unit_id": job["unit_id"],
            "mass_text_id": job["mass_text_id"],
            "job_id": job["job_id"],
            "prompt_text": unit["prompt_text"],
            "answer_text": unit["answer_text"],
            "spoken_text": job.get("spoken_text"),
            "reference_wav_path": job["reference_wav_path"],
            "control": job.get("control"),
            "ser_pass_reason": unit.get("ser_pass_reason"),
            "neutral_wer_percent": unit.get("neutral_wer_percent"),
            "happy_wer_percent": unit.get("happy_wer_percent"),
            "abs_delta_wer_percent": unit.get("abs_delta_wer_percent"),
        },
    }


def build_r_utterances(
    units: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand carved jobs into utterance rows (2 per unit)."""

    units_by_id = {unit["unit_id"]: unit for unit in units}
    utterances: list[dict[str, Any]] = []
    for job in jobs:
        if job.get("status") != "done":
            continue
        unit = units_by_id.get(job["unit_id"])
        if unit is None:
            raise KeyError(f"Job {job['job_id']} missing carved unit.")
        utterances.append(build_r_utterance_from_job(job, unit))
    utterances.sort(key=lambda row: (row["split"], row["utterance_id"]))
    return utterances


def pair_units_within_group(
    units: list[dict[str, Any]],
    *,
    seed: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    """Greedy pair distinct content units; return pairs and leftovers."""

    pool = list(units)
    rng = random.Random(seed)
    rng.shuffle(pool)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    leftover: list[dict[str, Any]] = []
    index = 0
    while index + 1 < len(pool):
        left = pool[index]
        right = pool[index + 1]
        if left["content_id"] == right["content_id"]:
            # Same content should not appear twice in one group; skip right.
            leftover.append(right)
            index += 1
            continue
        pairs.append((left, right))
        index += 2
    if index < len(pool):
        leftover.append(pool[index])
    return pairs, leftover


def build_r_quadruplet_record(
    left: dict[str, Any],
    right: dict[str, Any],
    utterance_lookup: dict[tuple[str, str, str, str], str],
) -> dict[str, Any]:
    """Build one R quadruplet: same speaker+engine, two contents, two z."""

    split_name = left["split"]
    speaker_id = left["speaker_id"]
    engine = left["engine"]
    content_pair = [left["content_id"], right["content_id"]]
    utterance_ids: list[str] = []
    for content_id in content_pair:
        for z_label in R_PARALINGUISTIC_PAIR:
            key = (engine, speaker_id, content_id, z_label)
            if key not in utterance_lookup:
                raise KeyError(f"Missing R utterance for {key}")
            utterance_ids.append(utterance_lookup[key])

    left_token = left["mass_text_id"]
    right_token = right["mass_text_id"]
    quadruplet_id = (
        f"r-{split_name}-{engine}-{speaker_id}-{left_token}-{right_token}-"
        f"{R_PARALINGUISTIC_PAIR[0]}-{R_PARALINGUISTIC_PAIR[1]}"
    )
    return {
        "quadruplet_id": quadruplet_id,
        "layer": "R",
        "split": split_name,
        "content_pair": content_pair,
        "paralinguistic_pair": list(R_PARALINGUISTIC_PAIR),
        "utterance_ids": utterance_ids,
        "metadata": {
            "speaker_id": speaker_id,
            "reference_speaker_id": speaker_id,
            "engine": engine,
            "unit_ids": [left["unit_id"], right["unit_id"]],
            "mass_text_ids": [left["mass_text_id"], right["mass_text_id"]],
            "prompt_texts": {
                left["content_id"]: left["prompt_text"],
                right["content_id"]: right["prompt_text"],
            },
            "answer_texts": {
                left["content_id"]: left["answer_text"],
                right["content_id"]: right["answer_text"],
            },
        },
    }


def build_r_quadruplets(
    units: list[dict[str, Any]],
    utterances: list[dict[str, Any]],
    *,
    seed: int = R_PAIRING_SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Pair carved units into quadruplets; return quads and unpaired units."""

    utterance_lookup: dict[tuple[str, str, str, str], str] = {}
    for row in utterances:
        key = (
            row["tts_engine"],
            row["speaker_id"],
            row["content_id"],
            row["paralinguistic_label"],
        )
        utterance_lookup[key] = row["utterance_id"]

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        groups[(unit["split"], unit["speaker_id"], unit["engine"])].append(unit)

    quadruplets: list[dict[str, Any]] = []
    unpaired: list[dict[str, Any]] = []
    for (split_name, speaker_id, engine), group_units in sorted(groups.items()):
        # Stable per-group seed from frozen global seed + ids.
        group_seed = seed + sum(ord(ch) for ch in f"{split_name}:{speaker_id}:{engine}")
        pairs, leftover = pair_units_within_group(group_units, seed=group_seed)
        for left, right in pairs:
            quadruplets.append(
                build_r_quadruplet_record(left, right, utterance_lookup)
            )
        for unit in leftover:
            unpaired.append(
                {
                    "unit_id": unit["unit_id"],
                    "split": unit["split"],
                    "speaker_id": unit["speaker_id"],
                    "engine": unit["engine"],
                    "content_id": unit["content_id"],
                    "reason": "odd_one_out_in_speaker_engine_group",
                }
            )

    quadruplets.sort(key=lambda row: (row["split"], row["quadruplet_id"]))
    unpaired.sort(key=lambda row: (row["split"], row["unit_id"]))
    return quadruplets, unpaired


def summarize_r_quadruplets(
    units: list[dict[str, Any]],
    utterances: list[dict[str, Any]],
    quadruplets: list[dict[str, Any]],
    unpaired: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the quadruplet construction card with eval capacity callouts."""

    def split_block(split_name: str) -> dict[str, Any]:
        split_units = [row for row in units if row["split"] == split_name]
        split_quads = [row for row in quadruplets if row["split"] == split_name]
        split_unpaired = [row for row in unpaired if row["split"] == split_name]
        engines = Counter(row["metadata"]["engine"] for row in split_quads)
        speakers = {row["metadata"]["speaker_id"] for row in split_quads}
        # Paired-metric sample counts (one score per quadruplet).
        return {
            "unit_count": len(split_units),
            "utterance_count": len(split_units) * 2,
            "quadruplet_count": len(split_quads),
            "unpaired_unit_count": len(split_unpaired),
            "unique_speaker_count_in_quads": len(speakers),
            "engine_counts_in_quads": dict(sorted(engines.items())),
            "paired_metric_n_content_both_z_correct": len(split_quads),
            "paired_metric_n_para_both_s_correct": len(split_quads),
            "content_query_count_if_eval_expanded": len(split_quads) * 4,
            "paralinguistic_query_count_if_eval_expanded": len(split_quads) * 4,
        }

    splits = {name: split_block(name) for name in SPLITS}
    return {
        "stage": "r_quadruplets_mass_carved",
        "pairing_policy": "same_split_speaker_engine_greedy_seed42",
        "paralinguistic_pair": list(R_PARALINGUISTIC_PAIR),
        "unit_count": len(units),
        "utterance_count": len(utterances),
        "quadruplet_count": len(quadruplets),
        "unpaired_unit_count": len(unpaired),
        "splits": splits,
        "eval_capacity_note": (
            "Eval 'quadruplet all-correct' metrics use n="
            f"{splits['eval']['quadruplet_count']} quadruplets "
            f"({splits['eval']['content_query_count_if_eval_expanded']} "
            "content queries / same for para)."
        ),
        "outputs": {
            "utterances": R_UTTERANCES_PATH,
            "quadruplets": R_QUADRUPLETS_PATH,
            "unpaired_units": R_UNPAIRED_UNITS_PATH,
            "report": R_QUADRUPLET_REPORT_PATH,
        },
    }


def prepare_r_quadruplets() -> None:
    """Create R utterances + quadruplets from the official carve and write the capacity card."""

    units, jobs = load_carved_units_and_jobs()
    utterances = build_r_utterances(units, jobs)
    quadruplets, unpaired = build_r_quadruplets(units, utterances)
    report = summarize_r_quadruplets(units, utterances, quadruplets, unpaired)

    write_jsonl(utterances, resolve_from_repo_root(R_UTTERANCES_PATH))
    write_jsonl(quadruplets, resolve_from_repo_root(R_QUADRUPLETS_PATH))
    write_jsonl(unpaired, resolve_from_repo_root(R_UNPAIRED_UNITS_PATH))
    write_report(report, resolve_from_repo_root(R_QUADRUPLET_REPORT_PATH))

    eval_n = report["splits"]["eval"]["quadruplet_count"]
    print(
        f"[prepare-r-quadruplets] utterances={report['utterance_count']} "
        f"quads={report['quadruplet_count']} unpaired={report['unpaired_unit_count']} "
        f"eval_quads={eval_n} "
        f"(paired-metric n={eval_n}; content/para queries={eval_n * 4} each)",
        flush=True,
    )
