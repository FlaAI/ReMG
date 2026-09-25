"""Bootstrap helpers for the RealMG data workspace."""

from __future__ import annotations

from pathlib import Path

from realmg.data.layout import DATA_SUBDIRECTORIES, ROOT_FILES
from realmg.data.schemas import QUADRUPLET_SCHEMA, UTTERANCE_SCHEMA, dump_schema


def write_file_if_missing(path: Path, contents: str) -> None:
    """Write a file only when it does not already exist."""

    if path.exists():
        return
    path.write_text(contents, encoding="utf-8")


def bootstrap_data_workspace(root: str) -> None:
    """Create the directory skeleton and template files for `Data/`."""

    data_root = Path(root)
    data_root.mkdir(parents=True, exist_ok=True)

    for relative_directory in DATA_SUBDIRECTORIES:
        (data_root / relative_directory).mkdir(parents=True, exist_ok=True)

    for relative_path, contents in ROOT_FILES.items():
        write_file_if_missing(data_root / relative_path, contents)

    dump_schema(data_root / "manifests" / "utterance.schema.json", UTTERANCE_SCHEMA)
    dump_schema(data_root / "manifests" / "quadruplet.schema.json", QUADRUPLET_SCHEMA)
