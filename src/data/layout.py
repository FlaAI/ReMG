"""Directory layout and template contents for `Data/`."""

from __future__ import annotations

DATA_SUBDIRECTORIES = (
    "raw",
    "refs",
    "manifests",
    "audio",
    "cards",
    "configs",
)

ROOT_FILES = {
    "README.md": """# Data Workspace

This directory stores local data artifacts for RealMG.

## Tracked

- `manifests/`: JSONL manifests and split lists
- `cards/`: data cards, filtering reports, and construction notes
- `configs/`: local path templates and build configuration

## Ignored

- `raw/`: downloaded corpora and unpacked sources
- `audio/`: synthesized audio and derived waveforms

## Notes

Only the workspace skeleton, templates, and schemas are created by bootstrap.
No corpora, checkpoints, or generated audio should be committed.
""",
    "configs/local_paths.example.toml": """# Copy to `local_paths.toml` and fill real local paths.

[storage]
raw_root = "Data/raw"
audio_root = "Data/audio"

[sources]
esd_root = ""
vctk_root = ""
tulu3_root = ""
natural_reasoning_root = ""

[models]
indextts2_root = "Data/models/indextts2_5"
cosyvoice2_root = "Data/models/cosyvoice2"
asr_root = ""
ser_root = "Data/models/emotion2vec_plus_large"
speechbrain_ser_root = "Data/models/speechbrain_emotion_iemocap"
general_verifier_root = "Data/models/general-verifier"
""",
    "configs/bailian_api.example.toml": """# Copy to `bailian_api.toml` and fill in your local Bailian credentials.
# This file is gitignored; never commit real API keys.

[api]
base_url = "https://YOUR-WORKSPACE.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
api_key = "sk-xxx"
model = "qwen2.5-omni-7b"

[request]
temperature = 0.0
max_tokens = 256
modalities = ["text"]
""",
    "manifests/README.md": """# Manifest Conventions

## Invariants

- one stable `utterance_id` per audio item
- one stable `quadruplet_id` per 2x2 unit
- all relative paths resolved from the repository root
""",
}
