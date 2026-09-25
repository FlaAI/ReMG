"""SER scoring for R-layer mass TTS WER survivors.

"""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torchaudio

from realmg.data.config import load_local_paths, resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_jsonl, write_report
from realmg.data.r_tts_gate_wer import append_wer_cache, load_wer_cache
from realmg.data.r_tts_mass import MASS_JOBS_PATH, MASS_UNITS_PATH
from realmg.data.r_tts_mass_wer import MASS_WER_SURVIVORS_PATH


SER_MARGIN_MIN = 0.2
# Unit-level contrastive gate: (P_h-P_n)_happy_wav - (P_h-P_n)_neutral_wav.
# Stored on every pair row so δ can be retuned without rescoring models.
CONTRASTIVE_DELTA_MIN = 0.10
SER_BACKEND_EMOTION2VEC = "emotion2vec"
SER_BACKEND_SPEECHBRAIN = "speechbrain_iemocap"
SER_MODEL_DEFAULT = "Data/models/emotion2vec_plus_large"
SPEECHBRAIN_SER_DEFAULT = "Data/models/speechbrain_emotion_iemocap"
WAV2VEC2_BASE_HUB = "facebook/wav2vec2-base"

# emotion2vec scores keep the historical path for resume compatibility.
MASS_SER_SCORES_PATH = "Data/cards/r_tts_mass_ser_scores.jsonl"
MASS_SER_SPEECHBRAIN_SCORES_PATH = "Data/cards/r_tts_mass_ser_speechbrain_scores.jsonl"
MASS_SER_PAIRS_PATH = "Data/cards/r_tts_mass_ser_pairs.jsonl"
MASS_SER_SURVIVORS_PATH = "Data/manifests/r_tts_mass_ser_survivor_units.jsonl"
MASS_SER_REPORT_PATH = "Data/cards/r_tts_mass_ser_report.json"

Z_TO_SER_LABEL = {
    "happy": "happy",
    "neutral": "neutral",
}

# SpeechBrain IEMOCAP label_encoder.txt order.
SPEECHBRAIN_INDEX_TO_LABEL = ("neu", "ang", "hap", "sad")
SPEECHBRAIN_TO_PLAN_LABEL = {
    "neu": "neutral",
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
}


def resolve_ser_model_path() -> Path:
    """Resolve the local emotion2vec checkpoint directory."""

    configured = load_local_paths().get("models", {}).get("ser_root", "")
    path = resolve_from_repo_root(configured or SER_MODEL_DEFAULT)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing emotion2vec model at {path}. "
            "Set [models].ser_root in Data/configs/local_paths.toml."
        )
    if not (path / "model.pt").exists():
        raise FileNotFoundError(f"Missing model.pt under {path}")
    return path


def resolve_speechbrain_ser_model_path() -> Path:
    """Resolve the local SpeechBrain IEMOCAP SER checkpoint directory."""

    configured = load_local_paths().get("models", {}).get("speechbrain_ser_root", "")
    path = resolve_from_repo_root(configured or SPEECHBRAIN_SER_DEFAULT)
    required = ("model.ckpt", "wav2vec2.ckpt", "label_encoder.txt")
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing SpeechBrain SER files under {path}: {missing}. "
            "Download speechbrain/emotion-recognition-wav2vec2-IEMOCAP into "
            "Data/models/speechbrain_emotion_iemocap (or set "
            "[models].speechbrain_ser_root)."
        )
    return path


@lru_cache(maxsize=1)
def load_emotion2vec_model() -> Any:
    """Load FunASR AutoModel for emotion2vec_plus_large once."""

    from funasr import AutoModel

    model_path = str(resolve_ser_model_path())
    return AutoModel(
        model=model_path,
        hub="hf",
        disable_update=True,
    )


class SpeechBrainIemocapSerModel:
    """Utterance SER using SpeechBrain IEMOCAP wav2vec2 + MLP checkpoints."""

    def __init__(self, model_dir: Path, device: str) -> None:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        self.device = device
        self.labels = list(SPEECHBRAIN_INDEX_TO_LABEL)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            WAV2VEC2_BASE_HUB
        )
        encoder = Wav2Vec2Model.from_pretrained(WAV2VEC2_BASE_HUB)
        wav2vec_state = torch.load(
            model_dir / "wav2vec2.ckpt",
            map_location="cpu",
            weights_only=True,
        )
        remapped = {
            (key[6:] if key.startswith("model.") else key): value
            for key, value in wav2vec_state.items()
        }
        encoder.load_state_dict(remapped, strict=True)
        self.encoder = encoder.eval().to(device)
        # Matches hyperparams output_norm: True (affine LN default init).
        self.output_norm = torch.nn.LayerNorm(768).to(device).eval()
        mlp = torch.nn.Linear(768, 4, bias=False)
        mlp_state = torch.load(
            model_dir / "model.ckpt",
            map_location="cpu",
            weights_only=True,
        )
        mlp.weight.data.copy_(mlp_state["0.w.weight"])
        self.mlp = mlp.eval().to(device)

    @torch.inference_mode()
    def classify_file(self, audio_path: Path) -> dict[str, Any]:
        """Return plan-normalized labels, scores, and top1-top2 margin."""

        waveform, sample_rate = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        inputs = self.feature_extractor(
            waveform.squeeze(0).cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)
        hidden = self.encoder(input_values).last_hidden_state
        hidden = self.output_norm(hidden)
        pooled = hidden.mean(dim=1)
        probs = torch.softmax(self.mlp(pooled), dim=-1)[0]
        score_by_raw = {
            label: float(probs[index].item())
            for index, label in enumerate(self.labels)
        }
        score_by_label = {
            SPEECHBRAIN_TO_PLAN_LABEL[raw]: score
            for raw, score in score_by_raw.items()
        }
        ranked = sorted(score_by_label.items(), key=lambda item: item[1], reverse=True)
        top1_label, top1_score = ranked[0]
        top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
        return {
            "labels": [label for label, _ in ranked],
            "scores": [score for _, score in ranked],
            "score_by_label": score_by_label,
            "predicted_label": top1_label,
            "top1_score": top1_score,
            "top2_score": top2_score,
            "margin": top1_score - top2_score,
        }


@lru_cache(maxsize=1)
def load_speechbrain_ser_model() -> SpeechBrainIemocapSerModel:
    """Load the SpeechBrain IEMOCAP SER once."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SpeechBrainIemocapSerModel(resolve_speechbrain_ser_model_path(), device)


def normalize_ser_label(raw_label: str) -> str:
    """Map bilingual emotion2vec labels to English plan labels."""

    text = raw_label.strip().lower()
    if "/" in text:
        text = text.split("/")[-1].strip()
    aliases = {
        "happiness": "happy",
        "joy": "happy",
        "calm": "neutral",
        "disgust": "disgusted",
        "fear": "fearful",
        "surprise": "surprised",
        "unk": "unknown",
        "<unk>": "unknown",
    }
    return aliases.get(text, text)


def parse_emotion2vec_output(raw: Any) -> dict[str, Any]:
    """Parse FunASR generate() output into labels, scores, top-1/2 margin."""

    if isinstance(raw, list):
        if not raw:
            raise ValueError("Empty emotion2vec output.")
        payload = raw[0]
    elif isinstance(raw, dict):
        payload = raw
    else:
        raise TypeError(f"Unexpected emotion2vec output type: {type(raw)}")

    labels_raw = payload.get("labels") or payload.get("label")
    scores_raw = payload.get("scores") or payload.get("score")
    if labels_raw is None or scores_raw is None:
        raise KeyError(f"emotion2vec output missing labels/scores: {payload.keys()}")

    labels = [normalize_ser_label(str(label)) for label in labels_raw]
    scores = [float(score) for score in scores_raw]
    if len(labels) != len(scores) or not labels:
        raise ValueError("emotion2vec labels/scores length mismatch or empty.")

    ranked = sorted(zip(labels, scores, strict=True), key=lambda item: item[1], reverse=True)
    top1_label, top1_score = ranked[0]
    top2_score = ranked[1][1] if len(ranked) > 1 else 0.0
    return {
        "labels": labels,
        "scores": scores,
        "score_by_label": {label: score for label, score in zip(labels, scores, strict=True)},
        "predicted_label": top1_label,
        "top1_score": top1_score,
        "top2_score": top2_score,
        "margin": top1_score - top2_score,
    }


def score_utterance_emotion2vec(audio_path: Path, *, model: Any) -> dict[str, Any]:
    """Run emotion2vec on one wav and return parsed scores."""

    result = model.generate(
        input=str(audio_path),
        granularity="utterance",
        extract_embedding=False,
    )
    return parse_emotion2vec_output(result)


def utterance_ser_pass(
    *,
    z_label: str,
    predicted_label: str,
    margin: float,
) -> bool:
    """Apply SER pass rule for one utterance on one backend."""

    expected = Z_TO_SER_LABEL[z_label]
    return predicted_label == expected and margin >= SER_MARGIN_MIN


def load_wer_survivor_units() -> list[dict[str, Any]]:
    """Load WER-surviving mass units."""

    path = resolve_from_repo_root(MASS_WER_SURVIVORS_PATH)
    if not path.exists():
        raise FileNotFoundError(
            "Missing WER survivors. Run `realmg score-r-tts-mass-wer` first."
        )
    return read_jsonl(path)


def load_mass_jobs_by_unit() -> dict[str, dict[str, dict[str, Any]]]:
    """Index mass jobs by unit_id then z_label."""

    path = resolve_from_repo_root(MASS_JOBS_PATH)
    by_unit: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for job in read_jsonl(path):
        by_unit[job["unit_id"]][job["z_label"]] = job
    return by_unit


def build_score_row(
    *,
    job: dict[str, Any],
    unit: dict[str, Any],
    z_label: str,
    parsed: dict[str, Any],
    ser_backend: str,
    ser_model: str,
) -> dict[str, Any]:
    """Assemble one cached SER score row."""

    expected = Z_TO_SER_LABEL[z_label]
    return {
        "job_id": job["job_id"],
        "unit_id": unit["unit_id"],
        "engine": unit["engine"],
        "speaker_id": unit["speaker_id"],
        "mass_text_id": unit["mass_text_id"],
        "content_id": unit["content_id"],
        "z_label": z_label,
        "provisional_ref_split": unit["provisional_ref_split"],
        "output_wav_path": job["output_wav_path"],
        "expected_label": expected,
        "predicted_label": parsed["predicted_label"],
        "top1_score": parsed["top1_score"],
        "top2_score": parsed["top2_score"],
        "margin": parsed["margin"],
        "score_by_label": parsed["score_by_label"],
        "label_match": parsed["predicted_label"] == expected,
        "margin_pass": parsed["margin"] >= SER_MARGIN_MIN,
        "ser_pass": utterance_ser_pass(
            z_label=z_label,
            predicted_label=parsed["predicted_label"],
            margin=parsed["margin"],
        ),
        "ser_backend": ser_backend,
        "ser_model": ser_model,
    }


def score_mass_ser_with_backend(
    units: list[dict[str, Any]],
    jobs_by_unit: dict[str, dict[str, dict[str, Any]]],
    *,
    backend: str,
    cache_path: Path,
    score_fn: Any,
    ser_model: str,
    limit: int | None = None,
    resume: bool = True,
    flush_every: int = 50,
) -> list[dict[str, Any]]:
    """Score both z wavs for each WER-surviving unit with one SER backend."""

    cached = load_wer_cache(cache_path) if resume else {}
    scored: list[dict[str, Any]] = []
    processed_new = 0

    for unit in units:
        unit_id = unit["unit_id"]
        z_jobs = jobs_by_unit.get(unit_id, {})
        for z_label in ("neutral", "happy"):
            job = z_jobs.get(z_label)
            if job is None or job.get("status") != "done":
                continue
            job_id = job["job_id"]
            if job_id in cached:
                scored.append(cached[job_id])
                continue
            if limit is not None and processed_new >= limit:
                return scored

            audio_path = resolve_from_repo_root(job["output_wav_path"])
            if not audio_path.exists():
                raise FileNotFoundError(f"Missing mass wav: {audio_path}")

            parsed = score_fn(audio_path)
            row = build_score_row(
                job=job,
                unit=unit,
                z_label=z_label,
                parsed=parsed,
                ser_backend=backend,
                ser_model=ser_model,
            )
            append_wer_cache(cache_path, row)
            cached[job_id] = row
            scored.append(row)
            processed_new += 1
            if processed_new % flush_every == 0 or processed_new == 1:
                print(
                    f"[score-r-tts-mass-ser:{backend}] new={processed_new} "
                    f"total_scored={len(scored)} "
                    f"{job_id} pred={row['predicted_label']} "
                    f"margin={row['margin']:.3f} pass={row['ser_pass']}",
                    flush=True,
                )
    return scored


def index_scores_by_job(scores: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index SER rows by job_id."""

    return {row["job_id"]: row for row in scores}


def index_scores_by_unit_z(
    scores: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Index SER rows by unit_id then z_label."""

    by_unit: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in scores:
        by_unit[row["unit_id"]][row["z_label"]] = row
    return by_unit


def happy_minus_neutral_score(row: dict[str, Any]) -> float:
    """Return P(happy) - P(neutral) from one utterance SER row."""

    scores = row.get("score_by_label") or {}
    return float(scores.get("happy", 0.0)) - float(scores.get("neutral", 0.0))


def contrastive_delta_for_backend(
    neutral_row: dict[str, Any] | None,
    happy_row: dict[str, Any] | None,
) -> float | None:
    """(P_h-P_n) on happy wav minus the same on neutral wav."""

    if neutral_row is None or happy_row is None:
        return None
    return happy_minus_neutral_score(happy_row) - happy_minus_neutral_score(neutral_row)


def absolute_z_pass(row: dict[str, Any] | None) -> bool:
    """Absolute scheme-A utterance pass (label match + margin)."""

    return bool(row is not None and row.get("ser_pass"))


def evaluate_paired_mass_ser_dual(
    emotion2vec_scores: list[dict[str, Any]],
    speechbrain_scores: list[dict[str, Any]],
    *,
    contrastive_delta_min: float = CONTRASTIVE_DELTA_MIN,
) -> list[dict[str, Any]]:
    """Unit pass = absolute A (OR backends per z) OR contrastive (any backend).

    Pair rows always store per-backend contrastive deltas and absolute flags so
    δ can be retuned later without rerunning SER models.
    """

    e2v_by_unit = index_scores_by_unit_z(emotion2vec_scores)
    sb_by_unit = index_scores_by_unit_z(speechbrain_scores)
    unit_ids = sorted(set(e2v_by_unit) | set(sb_by_unit))

    pair_rows: list[dict[str, Any]] = []
    for unit_id in unit_ids:
        e_z = e2v_by_unit.get(unit_id, {})
        s_z = sb_by_unit.get(unit_id, {})
        if "neutral" not in e_z and "neutral" not in s_z:
            continue
        if "happy" not in e_z and "happy" not in s_z:
            continue
        base = e_z.get("neutral") or s_z.get("neutral") or e_z.get("happy") or s_z.get("happy")
        assert base is not None

        e_n, e_h = e_z.get("neutral"), e_z.get("happy")
        s_n, s_h = s_z.get("neutral"), s_z.get("happy")

        neutral_e2v_pass = absolute_z_pass(e_n)
        happy_e2v_pass = absolute_z_pass(e_h)
        neutral_sb_pass = absolute_z_pass(s_n)
        happy_sb_pass = absolute_z_pass(s_h)
        neutral_abs_pass = neutral_e2v_pass or neutral_sb_pass
        happy_abs_pass = happy_e2v_pass or happy_sb_pass
        absolute_pair_pass = neutral_abs_pass and happy_abs_pass

        e2v_delta = contrastive_delta_for_backend(e_n, e_h)
        sb_delta = contrastive_delta_for_backend(s_n, s_h)
        e2v_contrast_pass = (
            e2v_delta is not None and e2v_delta >= contrastive_delta_min
        )
        sb_contrast_pass = sb_delta is not None and sb_delta >= contrastive_delta_min
        contrastive_pass = e2v_contrast_pass or sb_contrast_pass
        ser_pass = absolute_pair_pass or contrastive_pass

        if absolute_pair_pass:
            pass_reason = "absolute"
        elif contrastive_pass:
            pass_reason = "contrastive"
        else:
            pass_reason = None

        # Winning absolute labels (prefer emotion2vec when both pass).
        if neutral_e2v_pass:
            neutral_win_backend = SER_BACKEND_EMOTION2VEC
            neutral_pred = e_n["predicted_label"]
            neutral_margin = e_n["margin"]
        elif neutral_sb_pass:
            neutral_win_backend = SER_BACKEND_SPEECHBRAIN
            neutral_pred = s_n["predicted_label"]
            neutral_margin = s_n["margin"]
        else:
            neutral_win_backend = None
            neutral_src = e_n or s_n
            neutral_pred = neutral_src["predicted_label"] if neutral_src else None
            neutral_margin = neutral_src["margin"] if neutral_src else None

        if happy_e2v_pass:
            happy_win_backend = SER_BACKEND_EMOTION2VEC
            happy_pred = e_h["predicted_label"]
            happy_margin = e_h["margin"]
        elif happy_sb_pass:
            happy_win_backend = SER_BACKEND_SPEECHBRAIN
            happy_pred = s_h["predicted_label"]
            happy_margin = s_h["margin"]
        else:
            happy_win_backend = None
            happy_src = e_h or s_h
            happy_pred = happy_src["predicted_label"] if happy_src else None
            happy_margin = happy_src["margin"] if happy_src else None

        pair_rows.append(
            {
                "unit_id": unit_id,
                "engine": base["engine"],
                "speaker_id": base["speaker_id"],
                "mass_text_id": base["mass_text_id"],
                "content_id": base["content_id"],
                "provisional_ref_split": base["provisional_ref_split"],
                "neutral_predicted_label": neutral_pred,
                "happy_predicted_label": happy_pred,
                "neutral_margin": neutral_margin,
                "happy_margin": happy_margin,
                "neutral_ser_pass": neutral_abs_pass,
                "happy_ser_pass": happy_abs_pass,
                "neutral_emotion2vec_pass": neutral_e2v_pass,
                "happy_emotion2vec_pass": happy_e2v_pass,
                "neutral_speechbrain_pass": neutral_sb_pass,
                "happy_speechbrain_pass": happy_sb_pass,
                "neutral_winning_backend": neutral_win_backend,
                "happy_winning_backend": happy_win_backend,
                "absolute_pair_pass": absolute_pair_pass,
                "emotion2vec_contrastive_delta": e2v_delta,
                "speechbrain_contrastive_delta": sb_delta,
                "emotion2vec_contrastive_pass": e2v_contrast_pass,
                "speechbrain_contrastive_pass": sb_contrast_pass,
                "contrastive_pass": contrastive_pass,
                "contrastive_delta_min": contrastive_delta_min,
                "pass_reason": pass_reason,
                "ser_pass": ser_pass,
            }
        )
    return pair_rows


def build_ser_survivor_units(
    pair_rows: list[dict[str, Any]],
    wer_units_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep units that pass both WER (already filtered) and dual SER."""

    survivors: list[dict[str, Any]] = []
    for row in pair_rows:
        if not row["ser_pass"]:
            continue
        wer_unit = wer_units_by_id[row["unit_id"]]
        survivors.append(
            {
                "unit_id": row["unit_id"],
                "engine": row["engine"],
                "mass_text_id": row["mass_text_id"],
                "content_id": row["content_id"],
                "speaker_id": row["speaker_id"],
                "provisional_ref_split": row["provisional_ref_split"],
                "neutral_wer_percent": wer_unit["neutral_wer_percent"],
                "happy_wer_percent": wer_unit["happy_wer_percent"],
                "abs_delta_wer_percent": wer_unit["abs_delta_wer_percent"],
                "neutral_margin": row["neutral_margin"],
                "happy_margin": row["happy_margin"],
                "neutral_winning_backend": row["neutral_winning_backend"],
                "happy_winning_backend": row["happy_winning_backend"],
                "absolute_pair_pass": row["absolute_pair_pass"],
                "contrastive_pass": row["contrastive_pass"],
                "pass_reason": row["pass_reason"],
                "emotion2vec_contrastive_delta": row["emotion2vec_contrastive_delta"],
                "speechbrain_contrastive_delta": row["speechbrain_contrastive_delta"],
                "contrastive_delta_min": row["contrastive_delta_min"],
            }
        )
    return survivors


def summarize_backend_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize utterance-level pass rates for one SER backend."""

    by_engine: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_engine[row["engine"]].append(row)
    engine_summaries: dict[str, Any] = {}
    for engine, engine_scores in sorted(by_engine.items()):
        engine_summaries[engine] = {
            "utterance_count": len(engine_scores),
            "utterance_ser_pass_count": sum(1 for row in engine_scores if row["ser_pass"]),
            "utterance_ser_pass_rate": (
                sum(1 for row in engine_scores if row["ser_pass"]) / len(engine_scores)
                if engine_scores
                else None
            ),
            "label_match_rate": (
                sum(1 for row in engine_scores if row["label_match"]) / len(engine_scores)
                if engine_scores
                else None
            ),
            "margin_pass_rate": (
                sum(1 for row in engine_scores if row["margin_pass"]) / len(engine_scores)
                if engine_scores
                else None
            ),
        }
    return {
        "utterance_count": len(scores),
        "utterance_ser_pass_count": sum(1 for row in scores if row["ser_pass"]),
        "engines": engine_summaries,
    }


def summarize_mass_ser(
    emotion2vec_scores: list[dict[str, Any]],
    speechbrain_scores: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    survivors: list[dict[str, Any]],
    *,
    contrastive_delta_min: float,
) -> dict[str, Any]:
    """Build the dual-SER mass report card."""

    by_engine_pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_engine_pairs[row["engine"]].append(row)

    engine_summaries: dict[str, Any] = {}
    for engine, pairs in sorted(by_engine_pairs.items()):
        engine_summaries[engine] = {
            "pair_count": len(pairs),
            "ser_pass_count": sum(1 for row in pairs if row["ser_pass"]),
            "ser_pass_rate": (
                sum(1 for row in pairs if row["ser_pass"]) / len(pairs) if pairs else None
            ),
            "absolute_pair_pass_count": sum(
                1 for row in pairs if row.get("absolute_pair_pass")
            ),
            "contrastive_pass_count": sum(
                1 for row in pairs if row.get("contrastive_pass")
            ),
        }

    pass_reasons = Counter(
        row.get("pass_reason") for row in survivors if row.get("pass_reason")
    )

    return {
        "stage": "tts_mass_ser_dual_contrastive",
        "ser_rule": (
            "absolute A (OR backends per z, both z) OR "
            "contrastive ((P_h-P_n)_happy - (P_h-P_n)_neutral >= delta on any backend)"
        ),
        "margin_min": SER_MARGIN_MIN,
        "contrastive_delta_min": contrastive_delta_min,
        "z_to_ser_label": Z_TO_SER_LABEL,
        "emotion2vec_model": str(resolve_ser_model_path().as_posix()),
        "speechbrain_model": str(resolve_speechbrain_ser_model_path().as_posix()),
        "backends": {
            SER_BACKEND_EMOTION2VEC: summarize_backend_scores(emotion2vec_scores),
            SER_BACKEND_SPEECHBRAIN: summarize_backend_scores(speechbrain_scores),
        },
        "pair_count": len(pair_rows),
        "survivor_unit_count": len(survivors),
        "survivor_pass_reason_counts": dict(sorted(pass_reasons.items())),
        "survivor_provisional_ref_split_counts": dict(
            sorted(Counter(row["provisional_ref_split"] for row in survivors).items())
        ),
        "engines": engine_summaries,
        "outputs": {
            "emotion2vec_scores": MASS_SER_SCORES_PATH,
            "speechbrain_scores": MASS_SER_SPEECHBRAIN_SCORES_PATH,
            "pairs": MASS_SER_PAIRS_PATH,
            "survivors": MASS_SER_SURVIVORS_PATH,
            "report": MASS_SER_REPORT_PATH,
        },
        "notes": [
            "Input units are WER survivors only.",
            "Absolute pass: predicted label == constructed z and top1-top2 >= 0.2.",
            "SpeechBrain IEMOCAP weights loaded via transformers (not foreign_class).",
        ],
    }


def load_cached_ser_scores() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load both SER utterance score caches."""

    e2v_path = resolve_from_repo_root(MASS_SER_SCORES_PATH)
    sb_path = resolve_from_repo_root(MASS_SER_SPEECHBRAIN_SCORES_PATH)
    if not e2v_path.exists() or not sb_path.exists():
        raise FileNotFoundError(
            "Missing SER score caches. Run `realmg score-r-tts-mass-ser` first."
        )
    return read_jsonl(e2v_path), read_jsonl(sb_path)


def write_ser_pair_outputs(
    emotion2vec_scores: list[dict[str, Any]],
    speechbrain_scores: list[dict[str, Any]],
    wer_units_by_id: dict[str, dict[str, Any]],
    *,
    contrastive_delta_min: float,
) -> None:
    """Evaluate pairs at one δ and write pairs / survivors / report."""

    pair_rows = evaluate_paired_mass_ser_dual(
        emotion2vec_scores,
        speechbrain_scores,
        contrastive_delta_min=contrastive_delta_min,
    )
    survivors = build_ser_survivor_units(pair_rows, wer_units_by_id)
    write_jsonl(pair_rows, resolve_from_repo_root(MASS_SER_PAIRS_PATH))
    write_jsonl(survivors, resolve_from_repo_root(MASS_SER_SURVIVORS_PATH))
    write_report(
        summarize_mass_ser(
            emotion2vec_scores,
            speechbrain_scores,
            pair_rows,
            survivors,
            contrastive_delta_min=contrastive_delta_min,
        ),
        resolve_from_repo_root(MASS_SER_REPORT_PATH),
    )
    print(
        f"[score-r-tts-mass-ser] delta={contrastive_delta_min}: "
        f"pairs={len(pair_rows)} survivors={len(survivors)} "
        f"pools={dict(sorted(Counter(r['provisional_ref_split'] for r in survivors).items()))}",
        flush=True,
    )


def score_r_tts_mass_ser(
    *,
    limit: int | None = None,
    resume: bool = True,
    refilter_only: bool = False,
    contrastive_delta_min: float = CONTRASTIVE_DELTA_MIN,
) -> None:
    """Run dual SER on WER-surviving mass units and write survivors."""

    units = load_wer_survivor_units()
    wer_units_by_id = {unit["unit_id"]: unit for unit in units}

    if refilter_only:
        emotion2vec_scores, speechbrain_scores = load_cached_ser_scores()
        # Restrict to current WER survivors when refiltering.
        keep = set(wer_units_by_id)
        emotion2vec_scores = [row for row in emotion2vec_scores if row["unit_id"] in keep]
        speechbrain_scores = [row for row in speechbrain_scores if row["unit_id"] in keep]
        write_ser_pair_outputs(
            emotion2vec_scores,
            speechbrain_scores,
            wer_units_by_id,
            contrastive_delta_min=contrastive_delta_min,
        )
        return

    jobs_by_unit = load_mass_jobs_by_unit()
    emotion2vec_cache = resolve_from_repo_root(MASS_SER_SCORES_PATH)
    speechbrain_cache = resolve_from_repo_root(MASS_SER_SPEECHBRAIN_SCORES_PATH)

    def score_emotion2vec(path: Path) -> dict[str, Any]:
        return score_utterance_emotion2vec(path, model=load_emotion2vec_model())

    emotion2vec_scores = score_mass_ser_with_backend(
        units,
        jobs_by_unit,
        backend=SER_BACKEND_EMOTION2VEC,
        cache_path=emotion2vec_cache,
        score_fn=score_emotion2vec,
        ser_model=str(resolve_ser_model_path().as_posix()),
        limit=limit,
        resume=resume,
    )

    speechbrain_scores = score_mass_ser_with_backend(
        units,
        jobs_by_unit,
        backend=SER_BACKEND_SPEECHBRAIN,
        cache_path=speechbrain_cache,
        score_fn=lambda path: load_speechbrain_ser_model().classify_file(path),
        ser_model=str(resolve_speechbrain_ser_model_path().as_posix()),
        limit=limit,
        resume=resume,
    )

    if limit is not None:
        print(
            f"[score-r-tts-mass-ser] limit={limit}: scored "
            f"emotion2vec={len(emotion2vec_scores)} "
            f"speechbrain={len(speechbrain_scores)}; "
            "skip writing pairs/survivors/report.",
            flush=True,
        )
        return

    write_jsonl(emotion2vec_scores, emotion2vec_cache)
    write_jsonl(speechbrain_scores, speechbrain_cache)
    write_ser_pair_outputs(
        emotion2vec_scores,
        speechbrain_scores,
        wer_units_by_id,
        contrastive_delta_min=contrastive_delta_min,
    )
