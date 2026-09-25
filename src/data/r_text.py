"""R-layer text-source normalization for Tulu 3 and NaturalReasoning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from realmg.data.config import load_local_paths, resolve_from_repo_root
from realmg.data.esd import write_jsonl, write_report


def ensure_parent(path: Path) -> None:
    """Create the parent directory for one output file."""

    path.parent.mkdir(parents=True, exist_ok=True)


def load_or_download_dataset_snapshot(
    hf_repo: str,
    local_root: Path,
) -> DatasetDict:
    """Load a local dataset snapshot or download it from Hugging Face."""

    cache_dir = local_root / "hf_cache"
    dataset_dir = local_root / "hf_snapshot"

    if dataset_dir.exists():
        dataset = load_from_disk(str(dataset_dir))
    else:
        local_root.mkdir(parents=True, exist_ok=True)
        dataset = load_dataset(hf_repo, cache_dir=str(cache_dir))
        dataset.save_to_disk(str(dataset_dir))

    if isinstance(dataset, DatasetDict):
        return dataset

    if isinstance(dataset, Dataset):
        return DatasetDict({"train": dataset})

    raise TypeError(f"Unexpected dataset type for {hf_repo}: {type(dataset)!r}")


def normalize_text(value: Any) -> str:
    """Convert one dataset field into a compact single-line string."""

    if isinstance(value, str):
        return " ".join(value.split()).strip()
    return ""


def build_tulu_records(dataset: DatasetDict) -> list[dict[str, Any]]:
    """Convert Tulu 3 SFT messages into a unified R-text pool."""

    records: list[dict[str, Any]] = []
    train_split = dataset["train"]

    for row in train_split:
        messages = row["messages"]
        if not isinstance(messages, list) or len(messages) != 2:
            continue

        user_message = messages[0]
        assistant_message = messages[1]
        if user_message.get("role") != "user":
            continue
        if assistant_message.get("role") != "assistant":
            continue

        prompt_text = normalize_text(user_message.get("content"))
        answer_text = normalize_text(assistant_message.get("content"))
        if not prompt_text or not answer_text:
            continue

        records.append(
            {
                "content_id": f"tulu3::{row['id']}",
                "source_name": "tulu3",
                "prompt_text": prompt_text,
                "answer_text": answer_text,
                "metadata": {
                    "hf_repo": "allenai/tulu-3-sft-mixture",
                    "source_subset": row["source"],
                    "prompt_word_count": len(prompt_text.split()),
                    "answer_word_count": len(answer_text.split()),
                },
            }
        )

    return records


def build_natural_reasoning_records(dataset: DatasetDict) -> list[dict[str, Any]]:
    """Convert NaturalReasoning rows into a unified R-text pool."""

    records: list[dict[str, Any]] = []
    train_split = dataset["train"]

    for index, row in enumerate(train_split):
        prompt_text = normalize_text(row["question"])
        answer_text = normalize_text(row["reference_answer"])
        if not prompt_text or not answer_text:
            continue

        records.append(
            {
                "content_id": f"natural_reasoning::{index:07d}",
                "source_name": "natural_reasoning",
                "prompt_text": prompt_text,
                "answer_text": answer_text,
                "metadata": {
                    "hf_repo": "facebook/natural_reasoning",
                    "response_count": len(row["responses"]),
                    "prompt_word_count": len(prompt_text.split()),
                    "answer_word_count": len(answer_text.split()),
                },
            }
        )

    return records


def build_report(
    tulu_records: list[dict[str, Any]],
    natural_reasoning_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize the draft R-text pool."""

    all_records = tulu_records + natural_reasoning_records
    return {
        "sources": {
            "tulu3": {
                "hf_repo": "allenai/tulu-3-sft-mixture",
                "record_count": len(tulu_records),
            },
            "natural_reasoning": {
                "hf_repo": "facebook/natural_reasoning",
                "record_count": len(natural_reasoning_records),
            },
        },
        "record_count_total": len(all_records),
        "output_manifest": "Data/manifests/r_text_candidates_draft.jsonl",
        "notes": [
            "This is a normalization stage only.",
            "Short/medium-length filtering and teacher-text screening happen later.",
        ],
    }


def prepare_r_text_sources() -> None:
    """Normalize Tulu 3 and NaturalReasoning into one draft R-text pool."""

    local_paths = load_local_paths()
    tulu_root = resolve_from_repo_root(local_paths["sources"]["tulu3_root"])
    natural_reasoning_root = resolve_from_repo_root(
        local_paths["sources"]["natural_reasoning_root"]
    )

    tulu_dataset = load_or_download_dataset_snapshot(
        "allenai/tulu-3-sft-mixture",
        tulu_root,
    )
    natural_reasoning_dataset = load_or_download_dataset_snapshot(
        "facebook/natural_reasoning",
        natural_reasoning_root,
    )

    tulu_records = build_tulu_records(tulu_dataset)
    natural_reasoning_records = build_natural_reasoning_records(
        natural_reasoning_dataset
    )
    all_records = tulu_records + natural_reasoning_records

    manifest_path = resolve_from_repo_root("Data/manifests/r_text_candidates_draft.jsonl")
    report_path = resolve_from_repo_root("Data/cards/r_text_candidates_report.json")
    ensure_parent(manifest_path)
    write_jsonl(all_records, manifest_path)
    write_report(build_report(tulu_records, natural_reasoning_records), report_path)
