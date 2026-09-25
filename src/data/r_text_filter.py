"""Short-item filtering for the draft R-layer text pool.

"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report


PROMPT_WORD_MIN = 8
PROMPT_WORD_MAX = 50
PROMPT_CHAR_MAX = 350
ANSWER_WORD_MIN = 1
ANSWER_WORD_MAX = 20
ENGLISH_ASCII_LETTER_MIN = 0.90
ENGLISH_LETTER_MIN = 5
ENGLISH_LANGDETECT_MIN_PROB = 0.85
TULU_EXCLUDED_SOURCE_SUBSETS = frozenset(
    {
        "ai2-adapt-dev/tulu_v3.9_aya_100k",
    }
)

DRAFT_MANIFEST_PATH = "Data/manifests/r_text_candidates_draft.jsonl"
FILTERED_MANIFEST_PATH = "Data/manifests/r_text_candidates_filtered.jsonl"
FILTER_REPORT_PATH = "Data/cards/r_text_filter_report.json"

FORBIDDEN_SCRIPT_PATTERNS = (
    re.compile(r"[\u0400-\u04FF]"),
    re.compile(r"[\u4e00-\u9fff]"),
    re.compile(r"[\u0600-\u06FF]"),
    re.compile(r"[\u0900-\u097F]"),
)
CODE_FENCE_PATTERN = re.compile(r"```")


def word_count(text: str) -> int:
    """Count whitespace-delimited words."""

    return len(text.split())


def is_english_prompt(text: str) -> bool:
    """Heuristic plus langdetect check for speakable English prompts."""

    if any(pattern.search(text) for pattern in FORBIDDEN_SCRIPT_PATTERNS):
        return False

    letters = [char for char in text if char.isalpha()]
    if len(letters) < ENGLISH_LETTER_MIN:
        return False

    ascii_letters = sum(1 for char in letters if "a" <= char.lower() <= "z")
    if ascii_letters / len(letters) < ENGLISH_ASCII_LETTER_MIN:
        return False

    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0
    try:
        language_scores = detect_langs(text)
    except LangDetectException:
        return False
    for language_score in language_scores:
        if language_score.lang == "en" and language_score.prob >= ENGLISH_LANGDETECT_MIN_PROB:
            return True
    return False


def is_excluded_tulu_subset(record: dict[str, Any]) -> bool:
    """Drop known multilingual Tulu subsets before language detection."""

    if record["source_name"] != "tulu3":
        return False
    subset = record.get("metadata", {}).get("source_subset", "")
    return subset in TULU_EXCLUDED_SOURCE_SUBSETS


def rejection_reason(record: dict[str, Any]) -> str | None:
    """Return one rejection label or None if the record should be kept."""

    prompt = record["prompt_text"]
    answer = record["answer_text"]
    prompt_words = word_count(prompt)
    answer_words = word_count(answer)

    if is_excluded_tulu_subset(record):
        return "excluded_tulu_subset"
    if not is_english_prompt(prompt):
        return "non_english_prompt"
    if len(prompt) > PROMPT_CHAR_MAX:
        return "prompt_char_count"
    if CODE_FENCE_PATTERN.search(prompt) or CODE_FENCE_PATTERN.search(answer):
        return "code_fence"
    if not (PROMPT_WORD_MIN <= prompt_words <= PROMPT_WORD_MAX):
        return "prompt_word_count"
    if not (ANSWER_WORD_MIN <= answer_words <= ANSWER_WORD_MAX):
        return "answer_word_count"
    return None


def enrich_filtered_record(record: dict[str, Any]) -> dict[str, Any]:
    """Attach final word counts to one kept R-text record."""

    prompt_words = word_count(record["prompt_text"])
    answer_words = word_count(record["answer_text"])
    metadata = dict(record.get("metadata", {}))
    metadata["prompt_word_count"] = prompt_words
    metadata["answer_word_count"] = answer_words
    metadata["prompt_char_count"] = len(record["prompt_text"])
    return {
        **record,
        "metadata": metadata,
    }


def build_filter_report(
    input_count: int,
    kept_records: list[dict[str, Any]],
    rejection_counts: Counter[str],
) -> dict[str, Any]:
    """Summarize short-item filtering."""

    kept_by_source = Counter(record["source_name"] for record in kept_records)
    return {
        "input_manifest": DRAFT_MANIFEST_PATH,
        "output_manifest": FILTERED_MANIFEST_PATH,
        "input_count": input_count,
        "kept_count": len(kept_records),
        "rejected_count": input_count - len(kept_records),
        "kept_by_source": dict(sorted(kept_by_source.items())),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "frozen_rules": {
            "english_prompt": (
                f">= {ENGLISH_ASCII_LETTER_MIN:.0%} ASCII letters, "
                "langdetect en "
                f">= {ENGLISH_LANGDETECT_MIN_PROB:.2f}, "
                "no CJK/Cyrillic/Arabic/Devanagari"
            ),
            "excluded_tulu_subsets": sorted(TULU_EXCLUDED_SOURCE_SUBSETS),
            "prompt_word_count": [PROMPT_WORD_MIN, PROMPT_WORD_MAX],
            "prompt_char_max": PROMPT_CHAR_MAX,
            "answer_word_count": [ANSWER_WORD_MIN, ANSWER_WORD_MAX],
            "exclude_code_fences": True,
        },
        "notes": [
            "Prompt text is the spoken content s; audio is the question.",
        ],
    }


def filter_r_text_candidates() -> None:
    """Apply frozen short-item rules to the draft R-text pool."""

    draft_path = resolve_from_repo_root(DRAFT_MANIFEST_PATH)
    draft_records = read_jsonl(draft_path)

    kept_records: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    seen_prompts: set[str] = set()

    for record in draft_records:
        reason = rejection_reason(record)
        if reason is not None:
            rejection_counts[reason] += 1
            continue

        prompt_key = record["prompt_text"].casefold()
        if prompt_key in seen_prompts:
            rejection_counts["duplicate_prompt"] += 1
            continue
        seen_prompts.add(prompt_key)

        kept_records.append(enrich_filtered_record(record))

    write_jsonl(kept_records, resolve_from_repo_root(FILTERED_MANIFEST_PATH))
    write_report(
        build_filter_report(len(draft_records), kept_records, rejection_counts),
        resolve_from_repo_root(FILTER_REPORT_PATH),
    )
