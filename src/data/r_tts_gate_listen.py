"""Blinded forced-choice listen trials for the R-layer TTS gate."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report
from realmg.data.r_tts_gate import GATE_LISTEN_SHEET_PATH


LISTEN_TRIAL_SEED = 44
LISTEN_TRIALS_PATH = "Data/cards/r_tts_gate_listen_trials.jsonl"
LISTEN_WORKSHEET_PATH = "Data/cards/r_tts_gate_listen_worksheet.csv"
LISTEN_GUIDE_PATH = "Data/cards/r_tts_gate_listen_guide.json"


def build_blind_listen_trials(
    listen_rows: list[dict[str, Any]],
    *,
    seed: int = LISTEN_TRIAL_SEED,
) -> list[dict[str, Any]]:
    """Shuffle A/B presentation so the listener does not see true z labels."""

    rng = random.Random(seed)
    trials: list[dict[str, Any]] = []
    for index, row in enumerate(sorted(listen_rows, key=lambda item: item["listen_id"])):
        neutral_path = row["wav_by_z"]["neutral"]
        happy_path = row["wav_by_z"]["happy"]
        if rng.random() < 0.5:
            clip_a, clip_b = neutral_path, happy_path
            gold_happy_slot = "B"
        else:
            clip_a, clip_b = happy_path, neutral_path
            gold_happy_slot = "A"
        trials.append(
            {
                "trial_id": f"trial_{index:03d}",
                "listen_id": row["listen_id"],
                "engine": row["engine"],
                "speaker_id": row["speaker_id"],
                "gate_text_id": row["gate_text_id"],
                "content_id": row["content_id"],
                "prompt_text": row["prompt_text"],
                "clip_a_path": clip_a,
                "clip_b_path": clip_b,
                "gold_happy_slot": gold_happy_slot,
                "human_choice_slot": None,
                "human_correct": None,
                "notes": "",
            }
        )
    return trials


def write_listener_worksheet(trials: list[dict[str, Any]], path: Path) -> None:
    """Write a CSV worksheet without gold labels for human filling."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "engine",
        "speaker_id",
        "gate_text_id",
        "prompt_text",
        "clip_a_path",
        "clip_b_path",
        "human_choice_slot",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            writer.writerow({key: trial.get(key, "") for key in fieldnames})


def prepare_r_tts_gate_listen() -> None:
    """Export blinded listen trials and a fillable worksheet."""

    listen_path = resolve_from_repo_root(GATE_LISTEN_SHEET_PATH)
    if not listen_path.exists():
        raise FileNotFoundError(
            "Missing listen sheet. Run `realmg prepare-r-tts-gate` first."
        )
    listen_rows = read_jsonl(listen_path)
    trials = build_blind_listen_trials(listen_rows)
    write_jsonl(trials, resolve_from_repo_root(LISTEN_TRIALS_PATH))
    write_listener_worksheet(trials, resolve_from_repo_root(LISTEN_WORKSHEET_PATH))
    write_report(
        {
            "stage": "tts_gate_pilot_listen",
            "trial_count": len(trials),
            "trials_per_engine": {
                engine: sum(1 for trial in trials if trial["engine"] == engine)
                for engine in sorted({trial["engine"] for trial in trials})
            },
            "seed": LISTEN_TRIAL_SEED,
            "pass_criterion": {
                "forced_choice_accuracy_min": 0.80,
                "unit": "one trial = one (engine, speaker, text) pair of clips",
                "question": (
                    "Which clip sounds happier / more positively affective? "
                    "Answer A or B only."
                ),
            },
            "how_to_fill": [
                "Open Data/cards/r_tts_gate_listen_worksheet.csv",
                "Play clip_a_path then clip_b_path (absolute under repo root).",
                "Set human_choice_slot to A or B (forced choice; no ties).",
                "Optional notes for artifacts / unintelligible speech.",
                "Do not open r_tts_gate_listen_trials.jsonl while labeling "
                "(it contains gold_happy_slot).",
                "When finished, we score against gold_happy_slot; engine passes if "
                "accuracy >= 80%.",
            ],
            "outputs": {
                "trials_with_gold": LISTEN_TRIALS_PATH,
                "worksheet_no_gold": LISTEN_WORKSHEET_PATH,
            },
        },
        resolve_from_repo_root(LISTEN_GUIDE_PATH),
    )
