"""Load local OpenAI-compatible inference server settings from TOML.

"""

from __future__ import annotations

from pathlib import Path
import tomllib

from openai import OpenAI

from realmg.data.config import resolve_from_repo_root


LOCAL_SERVER_CONFIG_RELATIVE_PATH = "Data/configs/local_server_api.toml"
LOCAL_SERVER_CONFIG_EXAMPLE_RELATIVE_PATH = (
    "Data/configs/local_server_api.example.toml"
)


def local_server_config_path() -> Path:
    """Return the local inference-server config path."""

    return resolve_from_repo_root(LOCAL_SERVER_CONFIG_RELATIVE_PATH)


def local_server_config_example_path() -> Path:
    """Return the tracked local-server config template path."""

    return resolve_from_repo_root(LOCAL_SERVER_CONFIG_EXAMPLE_RELATIVE_PATH)


def load_local_server_config() -> dict:
    """Load local-server settings from the local config file."""

    config_path = local_server_config_path()
    if not config_path.exists():
        raise FileNotFoundError(
            "Missing local-server config. Copy "
            f"{LOCAL_SERVER_CONFIG_EXAMPLE_RELATIVE_PATH} to "
            f"{LOCAL_SERVER_CONFIG_RELATIVE_PATH} and set base_url/model."
        )

    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    api_section = config.get("api", {})
    required_fields = ("base_url", "api_key", "model")
    missing = [field for field in required_fields if not api_section.get(field)]
    if missing:
        raise ValueError(
            "Local-server config is incomplete. Missing fields under [api]: "
            + ", ".join(missing)
        )
    return config


def build_local_server_client() -> OpenAI:
    """Create an OpenAI-compatible client for the local inference server."""

    config = load_local_server_config()
    api_section = config["api"]
    client_kwargs: dict = {
        "api_key": api_section["api_key"],
        "base_url": api_section["base_url"],
    }
    timeout = api_section.get("timeout_seconds")
    if timeout is not None:
        client_kwargs["timeout"] = float(timeout)
    return OpenAI(**client_kwargs)


def local_server_model_name() -> str:
    """Return the configured default local-server model id."""

    return load_local_server_config()["api"]["model"]


def default_local_server_request_options() -> dict:
    """Return generation defaults from the local-server config."""

    config = load_local_server_config()
    request_section = config.get("request", {})
    return {
        "temperature": float(request_section.get("temperature", 0.0)),
        "max_tokens": int(request_section.get("max_tokens", 256)),
    }


def local_server_audio_content_format() -> str:
    """Return multimodal audio part format: input_audio or audio_url.

    vLLM typically expects audio_url with a data: URI. Configure per model.
    """

    config = load_local_server_config()
    api_section = config.get("api", {})
    fmt = str(api_section.get("audio_content_format", "input_audio")).strip()
    if fmt not in {"input_audio", "audio_url"}:
        raise ValueError(
            "api.audio_content_format must be 'input_audio' or 'audio_url', "
            f"got {fmt!r}"
        )
    return fmt
