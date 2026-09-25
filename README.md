# ReMG

This repository contains:

- ReMG train / validation / eval manifests
- Source code for ReMG data construction, fine-tuning, and evaluation
- LoRA adapters by ReMG fine-tuning:
  - Qwen2.5-Omni-7B: `ckpts/ours_qwen2_5_omni`
  - Phi-4-Multimodal: `ckpts/ours_phi4_mm`

\* The release of raw audio data (5875 files, about 5.9 GB) needs (non-anonymous) HF, thus would be supplemented after the anonymous peer review process.


## Environment

We conduct all experiments with Python 3.12 and PyTorch 2.6.0 + cu124. More details can be found in `pyproject.toml`. The [uv](https://github.com/astral-sh/uv) are required.


## Data layout

```
data/
  manifests/                # frozen JSONL + schemas + split lists
  cards/                    # construction summary cards
  audio/
    tier_r/                 # TTS wavs
      indextts2_5/
      cosyvoice2/
    refs/                   # VCTK reference clips
      vctk/          
    esd_required_paths.txt  # ESD index
```


## Fine-tuning command

Qwen2.5-Omni-7B:

```bash
uv run realmg train-ours \
  --backbone qwen25_omni \
  --model-dir /path/to/Qwen2.5-Omni-7B \
  --output-dir ckpts/ours_qwen2_5_omni_7b
```

Phi-4-Multimodal:

```bash
uv run realmg train-ours \
  --backbone phi4_mm \
  --model-dir /path/to/Phi-4-multimodal-instruct \
  --output-dir ckpts/ours_phi4_mm \
  --r-train-utterances data/manifests/r_carved_train_utterances_teacher_correct_phi4_mm.jsonl
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
