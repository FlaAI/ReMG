"""Configuration helpers for local RealMG data paths."""

from __future__ import annotations

from pathlib import Path
import tomllib


def repository_root() -> Path:
    """Return the repository root from the installed package location."""

    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    """Return the default `Data/` directory under the repository root."""

    return repository_root() / "Data"


def local_paths_file() -> Path:
    """Return the local path configuration file."""

    return data_root() / "configs" / "local_paths.toml"


def load_local_paths() -> dict:
    """Load the local path configuration from TOML."""

    config_path = local_paths_file()
    with config_path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_from_repo_root(value: str) -> Path:
    """Resolve a path string relative to the repository root when needed."""

    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return repository_root() / candidate
