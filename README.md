# ReMG: Rethinking, Revealing, and Reducing Real Modality Gap of Audio Language Models under Linguistic-Paralinguistic Trade-Off

This package contains:

- Frozen ReMG **train / validation (construction-dev) / eval** manifests
- Self-built **R-layer TTS audio** referenced by those manifests
- Data-construction, Ours training, and local evaluation code
- Main-table LoRA adapters:
  - `Data/ckpts/ours_qwen25_omni_7b_v5_brief/epoch_0` (Qwen2.5-Omni-7B)
  - `Data/ckpts/ours_phi4_mm_v6_textkd/epoch_0` (Phi-4-multimodal)

It does **not** contain third-party baseline trainers, transfer-benchmark
harnesses, or vendor API evaluation drivers.

## Anonymity

This release is intended for anonymous peer review. Do not add author names,
affiliations, acknowledgements, or identifiable cloud credentials before the
deanonymization stage.

## License note on third-party audio (P layer)

P-layer waveforms come from the **Emotional Speech Database (ESD)**.
We do **not** redistribute ESD wav files in this package.

1. Obtain ESD under its original license (HF mirror used by this code:
   `jspaulsen/esd`).
2. Run `realmg prepare-esd` (or place English wavs so that paths match
   `Data/raw/esd/wav/en/{speaker}/{emotion}/{id}.wav`).
3. See `Data/esd_required_paths.txt` for the exact relative paths referenced
   by the frozen P manifests in this release.

R-layer audio in `Data/audio/r_tts_mass/` is synthesized for this work and is
included (5875 wav files, about 5.9 GB).

## Setup

Python 3.12 and [uv](https://github.com/astral-sh/uv) are required.
CUDA 12.4 wheels for `torch==2.6.0+cu124` / `torchaudio==2.6.0+cu124` are
pinned in `pyproject.toml`.

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.6.0+cu124 True
```

Copy config templates:

```bash
cp Data/configs/local_paths.example.toml Data/configs/local_paths.toml
cp Data/configs/local_server_api.example.toml Data/configs/local_server_api.toml
# optional teacher text screening via OpenAI-compatible endpoint:
cp Data/configs/bailian_api.example.toml Data/configs/bailian_api.toml
```

Download backbone weights locally (examples):

- `Qwen/Qwen2.5-Omni-7B`
- `microsoft/Phi-4-multimodal-instruct`

## Data layout

```
Data/
  manifests/     # frozen JSONL + schemas + split lists
  cards/         # construction summary cards
  audio/
    r_tts_mass/  # included R TTS wavs
    refs/        # VCTK reference clips used by R construction
  raw/esd/       # NOT shipped; prepare locally (see above)
  ckpts/         # main-table LoRA adapters (epoch_0 only)
  configs/       # *.example.toml templates only
  esd_required_paths.txt
```

Splits:

| Split | Role |
|-------|------|
| train | Ours training utterances |
| construction-dev (validation) | Held-out construction / validation split |
| eval | Frozen 2×2 paper evaluation |

The main reported runs train for **one epoch** and do not use construction-dev
for checkpoint selection. The split is still released as part of the dataset.

## Train (main table)

Omni (`v5_brief`):

```bash
uv run realmg train-ours \
  --backbone qwen25_omni \
  --model-dir /path/to/Qwen2.5-Omni-7B \
  --output-dir Data/ckpts/ours_qwen25_omni_7b_v5_brief \
  --epochs 1
```

Phi-4 (`v6_textkd`):

```bash
uv run realmg train-ours \
  --backbone phi4_mm \
  --model-dir /path/to/Phi-4-multimodal-instruct \
  --output-dir Data/ckpts/ours_phi4_mm_v6_textkd \
  --epochs 1 \
  --r-train-utterances Data/manifests/r_carved_train_utterances_teacher_correct_phi4_mm.jsonl
```

Dry-run data inventory only:

```bash
uv run realmg train-ours --backbone qwen25_omni --model-dir unused --output-dir unused --dry-run-data
```

## Evaluate (local OpenAI-compatible server)

Serve a backbone (optionally with the released LoRA merged or loaded), point
`Data/configs/local_server_api.toml` at the endpoint, then:

```bash
uv run realmg run-p-eval-local-server --protocol paper
uv run realmg run-r-eval-local-server
uv run realmg run-r-eval-text-local-server
uv run realmg score-r-eval --predictions-file Data/manifests/<pred>.jsonl
uv run realmg score-locked-eval --slug <model_slug>
```

## Reconstruct data (optional)

Frozen manifests and R audio are enough to train and evaluate.
To rebuild from upstream corpora, use the `prepare-*` / `run-*` / `score-*`
construction commands (`realmg -h`). You will need ESD, VCTK, text sources,
TTS engines, ASR/SER models, and sufficient disk/GPU. Downstream teacher-text
screening expects an OpenAI-compatible endpoint configured via
`bailian_api.toml` (template only; no keys are shipped).

## What is intentionally omitted

- ESD waveforms (obtain yourself; path list provided)
- Baseline trainers and checkpoints beyond the two main-table LoRAs
- Transfer evaluations on external benchmarks
- Vendor cloud evaluation drivers and secret configs
- Intermediate draft text corpora and prediction dumps

## Citation

Anonymous during review. Citation will be added after deanonymization.
