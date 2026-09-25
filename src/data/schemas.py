"""Manifest schema templates for RealMG datasets."""

from __future__ import annotations

import json
from pathlib import Path


SCHEMA_HEADER = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
}

UTTERANCE_SCHEMA = {
    **SCHEMA_HEADER,
    "required": [
        "utterance_id",
        "layer",
        "source_name",
        "split",
        "content_id",
        "paralinguistic_label",
        "audio_path",
        "transcript",
    ],
    "properties": {
        "utterance_id": {"type": "string"},
        "layer": {"enum": ["R", "P"]},
        "source_name": {"type": "string"},
        "split": {"enum": ["train", "construction-dev", "eval"]},
        "content_id": {"type": "string"},
        "paralinguistic_label": {"type": "string"},
        "audio_path": {"type": "string"},
        "transcript": {"type": "string"},
        "speaker_id": {"type": ["string", "null"]},
        "reference_speaker_id": {"type": ["string", "null"]},
        "tts_engine": {"type": ["string", "null"]},
        "metadata": {"type": "object"},
    },
}

QUADRUPLET_SCHEMA = {
    **SCHEMA_HEADER,
    "required": [
        "quadruplet_id",
        "layer",
        "split",
        "content_pair",
        "paralinguistic_pair",
        "utterance_ids",
    ],
    "properties": {
        "quadruplet_id": {"type": "string"},
        "layer": {"enum": ["R", "P"]},
        "split": {"enum": ["construction-dev", "eval"]},
        "content_pair": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        },
        "paralinguistic_pair": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "maxItems": 2,
        },
        "utterance_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 4,
            "maxItems": 4,
        },
        "metadata": {"type": "object"},
    },
}


def dump_schema(path: Path, schema: dict) -> None:
    """Write a JSON schema with stable formatting."""

    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
