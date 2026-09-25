"""Score the locked paper catalog from saved P/R scored JSONL files.

"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from realmg.data.config import resolve_from_repo_root
from realmg.data.esd import read_jsonl, write_report
from realmg.eval.locked_metrics import (
    MetricPoint,
    P_EXPECTED_QUADS,
    P_EXPECTED_STEMS,
    P_LOCKED_METRIC_NAMES,
    QuadUnit,
    R_EXPECTED_QUADS,
    R_EXPECTED_STEMS,
    R_FINETUNE_TRADEOFF_METRIC_NAMES,
    R_LOCKED_METRIC_NAMES,
    assert_expected_eval_size,
    build_quad_units,
    complete_quads_for_sampled_s,
    compute_p_locked_metrics,
    compute_r_locked_metrics,
    finetune_tradeoff_from_base,
    format_paper_rate,
    index_text_correct,
    parse_rate_from_rows,
    quads_by_speaker_id,
    stems_by_content_id,
    stems_from_quads,
)
from realmg.eval.p_harness import P_BOOTSTRAP_SAMPLES, P_BOOTSTRAP_SEED, percentile
from realmg.eval.r_harness import R_BOOTSTRAP_SAMPLES, R_BOOTSTRAP_SEED

# Paper Base catalog. Qwen2.5-Omni-7B is the same-stack local vLLM run,
# not Bailian API (stack-confounded; kept on disk, not in --all-base).
BASE_MODEL_SLUGS = (
    "Qwen_Qwen2_5_Omni_7B_local_server_base",
    "qwen3_omni_flash_bailian_base",
    "qwen35_omni_flash_bailian_base",
    "gemini_3_7_flash_poloapi_base",
    "gemini_3_1_pro_preview_poloapi_base",
    "gpt_audio_2025_08_28_ohmygpt_base",
    "bytedance_doubao_seed_2_0_lite_260428_ohmygpt_base",
    "microsoft_Phi_4_multimodal_instruct_local_server_base",
    "Qwen_Qwen2_Audio_7B_Instruct_local_server_base",
    "yuantuo666_TARS_Qwen2_5_Omni_7B_local_server_base",
)

BASE_MODEL_DISPLAY_NAMES = {
    "Qwen_Qwen2_5_Omni_7B_local_server_base": "Qwen2.5-Omni-7B (Local)",
    "qwen25_omni_7b_bailian_base": "Qwen2.5-Omni-7B (Bailian)",
    "qwen3_omni_flash_bailian_base": "Qwen3-Omni-Flash",
    "qwen35_omni_flash_bailian_base": "Qwen3.5-Omni-Flash",
    "gemini_3_7_flash_poloapi_base": "Gemini 3.7 Flash",
    "gemini_3_1_pro_preview_poloapi_base": "Gemini 3.1 Pro Preview",
    "gpt_audio_2025_08_28_ohmygpt_base": "gpt-audio-2025-08-28",
    "bytedance_doubao_seed_2_0_lite_260428_ohmygpt_base": "Doubao Seed 2.0 Lite",
    "microsoft_Phi_4_multimodal_instruct_local_server_base": "Phi-4-MM",
    "Qwen_Qwen2_Audio_7B_Instruct_local_server_base": "Qwen2-Audio-7B",
    "yuantuo666_TARS_Qwen2_5_Omni_7B_local_server_base": "TARS-Qwen2.5-Omni-7B",
}

CATALOG_SOURCE = "locked paper metrics catalog"


def p_paper_scored_path(slug: str) -> Path:
    """Return the paper-P scored JSONL for one model slug."""

    return resolve_from_repo_root(
        f"Data/manifests/p_eval_predictions_paper_{slug}_scored.jsonl"
    )


def r_audio_scored_path(slug: str) -> Path:
    """Return the R audio scored JSONL for one model slug."""

    return resolve_from_repo_root(
        f"Data/manifests/r_eval_predictions_{slug}_scored.jsonl"
    )


def r_text_scored_path(slug: str) -> Path:
    """Return the R text-baseline scored JSONL for one model slug."""

    return resolve_from_repo_root(
        f"Data/manifests/r_eval_text_predictions_{slug}_scored.jsonl"
    )


def p_locked_report_path(slug: str) -> Path:
    """Return the locked P report path for one model slug."""

    return resolve_from_repo_root(
        f"Data/manifests/p_eval_predictions_paper_{slug}_locked_report.json"
    )


def r_locked_report_path(slug: str) -> Path:
    """Return the locked R report path for one model slug."""

    return resolve_from_repo_root(
        f"Data/manifests/r_eval_predictions_{slug}_locked_report.json"
    )


def locked_eval_combined_report_path(slug: str) -> Path:
    """Return the combined P+R locked report for one model slug."""

    return resolve_from_repo_root(f"Data/manifests/locked_eval_{slug}_report.json")


def locked_eval_summary_path() -> Path:
    """Return the stacked Base locked-eval summary path."""

    return resolve_from_repo_root("Data/manifests/locked_eval_base_summary.json")


def prediction_stem_without_scored(path: Path) -> str:
    """Strip .jsonl and an optional _scored suffix from a predictions path."""

    stem = path.stem
    if stem.endswith("_scored"):
        return stem[: -len("_scored")]
    return stem


def slug_from_p_paper_path(path: Path) -> str:
    """Parse the model slug from a paper-P predictions or scored filename."""

    stem = prediction_stem_without_scored(path)
    prefix = "p_eval_predictions_paper_"
    if not stem.startswith(prefix) or stem == prefix:
        raise ValueError(f"Not a paper P predictions path: {path.name}")
    return stem[len(prefix) :]


def slug_from_r_audio_path(path: Path) -> str:
    """Parse the model slug from an R audio predictions or scored filename."""

    stem = prediction_stem_without_scored(path)
    if stem.startswith("r_eval_text_predictions_"):
        raise ValueError(f"Expected R audio predictions path, got: {path.name}")
    prefix = "r_eval_predictions_"
    if not stem.startswith(prefix) or stem == prefix:
        raise ValueError(f"Not an R audio predictions path: {path.name}")
    return stem[len(prefix) :]


def slug_from_r_text_path(path: Path) -> str:
    """Parse the model slug from an R text predictions or scored filename."""

    stem = prediction_stem_without_scored(path)
    prefix = "r_eval_text_predictions_"
    if not stem.startswith(prefix) or stem == prefix:
        raise ValueError(f"Not an R text predictions path: {path.name}")
    return stem[len(prefix) :]


LOCAL_OMNI_FINETUNE_BASE_SLUG = "Qwen_Qwen2_5_Omni_7B_local_server_base"


def display_name_for_slug(slug: str) -> str:
    """Return a known display name, or the slug itself for a new model."""

    return BASE_MODEL_DISPLAY_NAMES.get(slug, slug)


def finetune_base_slug_for(slug: str) -> str | None:
    """Return the designated same-stack base for Ours / TARS / Local Omni.

    API rows and other local backbones (Qwen2-Audio, Phi-4-MM) have no
    finetune-vs-base composites.
    """

    if "Qwen2_Audio" in slug or "Phi_4" in slug:
        return None
    if "Qwen2_5_Omni" in slug and "local_server" in slug:
        return LOCAL_OMNI_FINETUNE_BASE_SLUG
    if slug.startswith("RealMG_Ours_") and "local_server" in slug:
        return LOCAL_OMNI_FINETUNE_BASE_SLUG
    return None


def load_r_locked_para_and_gap(slug: str) -> tuple[float, float] | None:
    """Read para_score and modality_gap point estimates from a locked R report."""

    path = r_locked_report_path(slug)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or {}
    para = metrics.get("para_score") or {}
    gap = metrics.get("modality_gap") or {}
    if "point_estimate" not in para or "point_estimate" not in gap:
        return None
    return float(para["point_estimate"]), float(gap["point_estimate"])


def maybe_finetune_tradeoff_points(
    slug: str,
    points: dict[str, MetricPoint],
) -> tuple[str, dict[str, MetricPoint]] | None:
    """Build score_net vs the designated base when that report exists."""

    base_slug = finetune_base_slug_for(slug)
    if base_slug is None:
        return None
    if slug == base_slug:
        para_base = points["para_score"].value
        gap_base = points["modality_gap"].value
    else:
        loaded = load_r_locked_para_and_gap(base_slug)
        if loaded is None:
            print(
                f"[score-locked-eval] skip score_net {slug}: "
                f"missing base R report {base_slug}",
                flush=True,
            )
            return None
        para_base, gap_base = loaded
    tradeoff = finetune_tradeoff_from_base(
        points["para_score"],
        points["modality_gap"],
        para_base=para_base,
        gap_base=gap_base,
    )
    return base_slug, tradeoff


def _metric_entry(point: MetricPoint, ci95: dict[str, float] | None) -> dict[str, Any]:
    """Serialize one metric with optional cluster-bootstrap CI."""

    entry: dict[str, Any] = {
        "point_estimate": point.value,
        "denominator": point.denominator,
    }
    if ci95 is not None:
        entry["ci95"] = ci95
    return entry


def _bundle_entries(
    points: dict[str, MetricPoint],
    names: tuple[str, ...],
    intervals: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """Keep catalog order and attach CI when the bootstrap produced one."""

    return {
        name: _metric_entry(points[name], intervals.get(name))
        for name in names
    }


def _ci_from_samples(
    samples_by_name: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """Percentile 95% CI from bootstrap replicates, skipping empty series."""

    intervals: dict[str, dict[str, float]] = {}
    for name, values in samples_by_name.items():
        if not values:
            continue
        ordered = sorted(values)
        intervals[name] = {
            "low": percentile(ordered, 0.025),
            "high": percentile(ordered, 0.975),
        }
    return intervals


def _record_defined_metrics(
    points: dict[str, MetricPoint],
    samples_by_name: dict[str, list[float]],
) -> None:
    """Keep replicates whose catalog denominator is defined (non-empty)."""

    for name, point in points.items():
        if point.denominator <= 0:
            continue
        samples_by_name[name].append(point.value)


def cluster_bootstrap_p_locked(
    quads: list[QuadUnit],
    *,
    seed: int = P_BOOTSTRAP_SEED,
    samples: int = P_BOOTSTRAP_SAMPLES,
) -> dict[str, dict[str, float]]:
    """Speaker-cluster bootstrap for the P locked catalog."""

    grouped = quads_by_speaker_id(quads)
    speakers = sorted(grouped)
    rng = random.Random(seed)
    samples_by_name: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        boot_quads = [
            quad
            for speaker in (rng.choice(speakers) for _ in speakers)
            for quad in grouped[speaker]
        ]
        points = compute_p_locked_metrics(stems_from_quads(boot_quads), boot_quads)
        _record_defined_metrics(points, samples_by_name)
    return _ci_from_samples(samples_by_name)


def cluster_bootstrap_r_locked(
    quads: list[QuadUnit],
    *,
    seed: int = R_BOOTSTRAP_SEED,
    samples: int = R_BOOTSTRAP_SAMPLES,
) -> dict[str, dict[str, float]]:
    """s-cluster bootstrap for the R locked catalog."""

    stem_index = stems_by_content_id(stems_from_quads(quads))
    content_ids = sorted(stem_index)
    rng = random.Random(seed)
    samples_by_name: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        sampled_ids = [rng.choice(content_ids) for _ in content_ids]
        boot_stems = [stem_index[content_id] for content_id in sampled_ids]
        boot_quads = complete_quads_for_sampled_s(quads, set(sampled_ids))
        points = compute_r_locked_metrics(boot_stems, boot_quads)
        _record_defined_metrics(points, samples_by_name)
    return _ci_from_samples(samples_by_name)


def build_p_locked_report(
    slug: str,
    audio_rows: list[dict[str, Any]],
    scored_path: Path,
) -> dict[str, Any]:
    """Build the paper-P locked report for one saved scored file."""

    quads = build_quad_units(audio_rows, require_text=False)
    assert_expected_eval_size(
        quads,
        expected_quads=P_EXPECTED_QUADS,
        expected_stems=P_EXPECTED_STEMS,
        layer="P",
    )
    stems = stems_from_quads(quads)
    points = compute_p_locked_metrics(stems, quads)
    intervals = cluster_bootstrap_p_locked(quads)
    return {
        "catalog": "locked_paper_metrics",
        "catalog_source": CATALOG_SOURCE,
        "slug": slug,
        "display_name": display_name_for_slug(slug),
        "layer": "P",
        "protocol": "paper",
        "split": "eval",
        "scored_source": str(scored_path),
        "bootstrap": {
            "cluster_unit": "speaker_id",
            "samples": P_BOOTSTRAP_SAMPLES,
            "seed": P_BOOTSTRAP_SEED,
        },
        "n": {
            "speakers": len({quad.speaker_id for quad in quads}),
            "quadruplets": len(quads),
            "stems": len(stems),
            "content_audios": len(stems) * 2,
            "paralinguistic_audios": len(stems) * 2,
        },
        "metrics": _bundle_entries(points, P_LOCKED_METRIC_NAMES, intervals),
        "diagnostics": {"parse_rate": parse_rate_from_rows(audio_rows)},
    }


def build_r_locked_report(
    slug: str,
    audio_rows: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
    audio_path: Path,
    text_path: Path,
) -> dict[str, Any]:
    """Build the R locked report from saved audio and text scored files."""

    text_by_s = index_text_correct(text_rows)
    quads = build_quad_units(audio_rows, text_by_s=text_by_s, require_text=True)
    assert_expected_eval_size(
        quads,
        expected_quads=R_EXPECTED_QUADS,
        expected_stems=R_EXPECTED_STEMS,
        layer="R",
    )
    extra_text = sorted(set(text_by_s) - {stem.content_id for stem in stems_from_quads(quads)})
    if extra_text:
        raise ValueError(f"Text rows not in R audio eval: {extra_text[:5]}")
    stems = stems_from_quads(quads)
    points = compute_r_locked_metrics(stems, quads)
    intervals = cluster_bootstrap_r_locked(quads)
    supporting_names = ("acc_text",)
    metrics = _bundle_entries(points, R_LOCKED_METRIC_NAMES, intervals)
    tradeoff_meta: dict[str, Any] | None = None
    attached = maybe_finetune_tradeoff_points(slug, points)
    if attached is not None:
        base_slug, tradeoff_points = attached
        metrics.update(
            _bundle_entries(tradeoff_points, R_FINETUNE_TRADEOFF_METRIC_NAMES, {})
        )
        tradeoff_meta = {
            "base_slug": base_slug,
            "base_treated_as_constant": True,
            "larger_better": True,
        }
    report = {
        "catalog": "locked_paper_metrics",
        "catalog_source": CATALOG_SOURCE,
        "slug": slug,
        "display_name": display_name_for_slug(slug),
        "layer": "R",
        "split": "eval",
        "audio_scored_source": str(audio_path),
        "text_scored_source": str(text_path),
        "bootstrap": {
            "cluster_unit": "content_id",
            "samples": R_BOOTSTRAP_SAMPLES,
            "seed": R_BOOTSTRAP_SEED,
            "column_and_unit_rule": (
                "unit_success and column metrics keep a quadruplet only when "
                "both s appear in the resampled content_id list"
            ),
        },
        "n": {
            "unique_s": len(stems),
            "quadruplets": len(quads),
            "content_audios": len(stems) * 2,
            "paralinguistic_audios": len(stems) * 2,
            "text_queries": len(stems),
            "success_cnt_invariability_gate_s": points[
                "success_cnt_invariability"
            ].denominator,
            "success_cnt_sensitivity_gate_columns": points[
                "success_cnt_sensitivity"
            ].denominator,
        },
        "metrics": metrics,
        "supporting": _bundle_entries(points, supporting_names, intervals),
        "diagnostics": {"parse_rate": parse_rate_from_rows(audio_rows)},
    }
    if tradeoff_meta is not None:
        report["finetune_vs_base"] = tradeoff_meta
    return report


def _print_p_locked_oneliner(report: dict[str, Any]) -> None:
    """Print the P locked headline metrics for one slug."""

    cnt = report["metrics"]["cnt_score"]["point_estimate"]
    para = report["metrics"]["para_score"]["point_estimate"]
    score_am = report["metrics"]["score_am"]["point_estimate"]
    score_hm = report["metrics"]["score_hm"]["point_estimate"]
    print(
        f"[score-locked-eval] {report['slug']} P "
        f"cnt={format_paper_rate(cnt)} para={format_paper_rate(para)} "
        f"score_am={format_paper_rate(score_am)} "
        f"score_hm={format_paper_rate(score_hm)}",
        flush=True,
    )


def _print_r_locked_oneliner(report: dict[str, Any]) -> None:
    """Print the R locked headline metrics for one slug."""

    gap = report["metrics"]["modality_gap"]["point_estimate"]
    score_am = report["metrics"]["score_am"]["point_estimate"]
    score_hm = report["metrics"]["score_hm"]["point_estimate"]
    unit = report["metrics"]["unit_success"]["point_estimate"]
    extra = ""
    net = report["metrics"].get("score_net")
    if net is not None:
        extra = f" score_net={format_paper_rate(net['point_estimate'])}"
    print(
        f"[score-locked-eval] {report['slug']} R "
        f"gap={format_paper_rate(gap)} "
        f"score_am={format_paper_rate(score_am)} "
        f"score_hm={format_paper_rate(score_hm)} "
        f"unit_success={format_paper_rate(unit)}"
        f"{extra}",
        flush=True,
    )


def maybe_write_combined_locked_report(slug: str) -> dict[str, Any] | None:
    """Write the combined P+R report when both locked layer files exist."""

    p_path = p_locked_report_path(slug)
    r_path = r_locked_report_path(slug)
    if not p_path.is_file() or not r_path.is_file():
        return None
    combined = {
        "catalog": "locked_paper_metrics",
        "catalog_source": CATALOG_SOURCE,
        "slug": slug,
        "display_name": display_name_for_slug(slug),
        "P": json.loads(p_path.read_text(encoding="utf-8")),
        "R": json.loads(r_path.read_text(encoding="utf-8")),
    }
    write_report(combined, locked_eval_combined_report_path(slug))
    return combined


def write_p_locked_report_from_scored(
    *,
    slug: str,
    scored_rows: list[dict[str, Any]],
    scored_path: Path,
) -> dict[str, Any]:
    """Write the paper-P locked report from already-scored rows."""

    report = build_p_locked_report(slug, scored_rows, scored_path)
    write_report(report, p_locked_report_path(slug))
    maybe_write_combined_locked_report(slug)
    _print_p_locked_oneliner(report)
    return report


def maybe_write_r_locked_report_for_slug(slug: str) -> dict[str, Any] | None:
    """Write the R locked report when both audio and text scored files exist."""

    audio_path = r_audio_scored_path(slug)
    text_path = r_text_scored_path(slug)
    if not audio_path.is_file() or not text_path.is_file():
        missing = "audio" if not audio_path.is_file() else "text"
        print(
            f"[score-locked-eval] skip R {slug}: missing {missing} scored file",
            flush=True,
        )
        return None
    report = build_r_locked_report(
        slug,
        read_jsonl(audio_path),
        read_jsonl(text_path),
        audio_path,
        text_path,
    )
    write_report(report, r_locked_report_path(slug))
    maybe_write_combined_locked_report(slug)
    _print_r_locked_oneliner(report)
    return report


def score_locked_eval_for_slug(slug: str) -> dict[str, Any]:
    """Write whichever locked reports exist for one model slug."""

    p_path = p_paper_scored_path(slug)
    p_report = None
    if p_path.is_file():
        p_report = write_p_locked_report_from_scored(
            slug=slug,
            scored_rows=read_jsonl(p_path),
            scored_path=p_path,
        )
    else:
        print(f"[score-locked-eval] skip P {slug}: missing {p_path.name}", flush=True)
    r_report = maybe_write_r_locked_report_for_slug(slug)
    if p_report is None and r_report is None:
        raise FileNotFoundError(
            f"No paper-P or R scored files found for slug {slug!r}"
        )
    return {
        "catalog": "locked_paper_metrics",
        "catalog_source": CATALOG_SOURCE,
        "slug": slug,
        "display_name": display_name_for_slug(slug),
        "P": p_report,
        "R": r_report,
    }


def _flat_points(report_layer: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    """Extract point estimates in catalog order for the summary table."""

    metrics = report_layer["metrics"]
    supporting = report_layer.get("supporting", {})
    out: dict[str, Any] = {}
    for name in names:
        if name in metrics:
            out[name] = metrics[name]["point_estimate"]
        elif name in supporting:
            out[name] = supporting[name]["point_estimate"]
    out["parse_rate"] = report_layer["diagnostics"]["parse_rate"]
    return out


def build_base_summary(combined_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack catalog Base models into one summary JSON."""

    return {
        "catalog": "locked_paper_metrics",
        "catalog_source": CATALOG_SOURCE,
        "n_models": len(combined_reports),
        "models": [
            {
                "slug": report["slug"],
                "display_name": report["display_name"],
                "P": _flat_points(report["P"], P_LOCKED_METRIC_NAMES),
                "R": _flat_points(
                    report["R"],
                    ("acc_text",)
                    + R_LOCKED_METRIC_NAMES
                    + R_FINETUNE_TRADEOFF_METRIC_NAMES,
                ),
            }
            for report in combined_reports
        ],
    }


def score_locked_eval(slug: str | None = None, *, all_base: bool = False) -> None:
    """CLI entry: score one slug or the frozen Base catalog."""

    if slug and all_base:
        raise ValueError("Pass either --slug or --all-base, not both.")
    slugs = list(BASE_MODEL_SLUGS) if (all_base or slug is None) else [slug]
    combined_reports = [score_locked_eval_for_slug(item) for item in slugs]
    complete_reports = [
        report
        for report in combined_reports
        if report.get("P") is not None and report.get("R") is not None
    ]
    if len(complete_reports) > 1:
        write_report(build_base_summary(complete_reports), locked_eval_summary_path())
