"""Locked paper metric catalog from scored 0/1 flags.

"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from realmg.eval.p_harness import safe_mean

NEUTRAL = "neutral"
HAPPY = "happy"
Z_LABELS = (NEUTRAL, HAPPY)

R_EXPECTED_QUADS = 88
R_EXPECTED_STEMS = 176
P_EXPECTED_QUADS = 220
P_EXPECTED_STEMS = 440

P_LOCKED_METRIC_NAMES = (
    "cnt_score",
    "para_score",
    "score_am",
    "score_hm",
    "para_n_recall",
    "para_h_recall",
    "para_invariability",
    "success_para_invariability",
    "success_para_sensitivity",
)
R_LOCKED_METRIC_NAMES = (
    "modality_gap",
    "cnt_score",
    "para_score",
    "score_am",
    "score_hm",
    "para_n_recall",
    "para_h_recall",
    "text_retention",
    "modality_consistency",
    "success_modality_consistency",
    "cnt_invariability",
    "success_cnt_invariability",
    "success_cnt_sensitivity",
    "para_invariability",
    "success_para_invariability",
    "success_para_sensitivity",
    "unit_success",
)

# R-only composites vs a designated same-stack finetune base. Not for API
# rows. Not s-cluster bootstrapped. Larger is better for score_net.
R_FINETUNE_TRADEOFF_METRIC_NAMES = (
    "delta_para_damage",
    "delta_gap_gain",
    "score_net",
)

# Rates stay full precision in JSON. Tables and logs use four decimals so
# 0.5352 can be written as 53.52%.
PAPER_RATE_DECIMALS = 4


def format_paper_rate(value: float) -> str:
    """Format a rate or gap as four decimals (0.5352 equals 53.52 percent)."""

    return f"{value:.{PAPER_RATE_DECIMALS}f}"


@dataclass(frozen=True)
class StemUnit:
    """One s inside one quadruplet: two z, content and para 0/1."""

    quadruplet_id: str
    content_id: str
    speaker_id: str
    text_correct: bool | None
    content_neutral: bool
    content_happy: bool
    para_neutral: bool
    para_happy: bool


@dataclass(frozen=True)
class QuadUnit:
    """One 2x2 quadruplet with two stems (two s)."""

    quadruplet_id: str
    speaker_id: str
    stem_a: StemUnit
    stem_b: StemUnit


@dataclass(frozen=True)
class MetricPoint:
    """One locked metric: mean of 0/1 bits and the catalog denominator."""

    value: float
    denominator: int


def mean_of_flags(flags: list[bool]) -> MetricPoint:
    """Average a list of 0/1 flags. Empty list is undefined (denom 0)."""

    return MetricPoint(safe_mean([float(flag) for flag in flags]), len(flags))


def _require_z_slot(slot: dict[str, bool], *, context: str) -> None:
    """Require both content and para 0/1 on one (quad, s, z) audio."""

    missing = [name for name in ("content", "paralinguistic") if name not in slot]
    if missing:
        raise ValueError(f"Incomplete audio slot {context}: missing {missing}")


def _stem_from_z_slots(
    quadruplet_id: str,
    content_id: str,
    speaker_id: str,
    z_slots: dict[str, dict[str, bool]],
    text_correct: bool | None,
) -> StemUnit:
    """Build one stem after checking both N and H audios exist."""

    missing_z = [label for label in Z_LABELS if label not in z_slots]
    if missing_z:
        raise ValueError(
            f"Stem {quadruplet_id}/{content_id} missing z={missing_z}"
        )
    for label in Z_LABELS:
        _require_z_slot(
            z_slots[label],
            context=f"{quadruplet_id}/{content_id}/{label}",
        )
    return StemUnit(
        quadruplet_id=quadruplet_id,
        content_id=content_id,
        speaker_id=speaker_id,
        text_correct=text_correct,
        content_neutral=z_slots[NEUTRAL]["content"],
        content_happy=z_slots[HAPPY]["content"],
        para_neutral=z_slots[NEUTRAL]["paralinguistic"],
        para_happy=z_slots[HAPPY]["paralinguistic"],
    )


def index_text_correct(text_rows: list[dict[str, Any]]) -> dict[str, bool]:
    """Map unique s (content_id) to Teacher-text verifier 0/1."""

    text_by_s: dict[str, bool] = {}
    for row in text_rows:
        content_id = str(row["content_id"])
        if content_id in text_by_s:
            raise ValueError(f"Duplicate text scored row for content_id={content_id}")
        text_by_s[content_id] = bool(row["is_correct"])
    return text_by_s


def parse_rate_from_rows(rows: list[dict[str, Any]]) -> float:
    """Diagnostic parse rate. Not a locked catalog name."""

    return safe_mean([float(bool(row.get("is_parseable"))) for row in rows])


def build_quad_units(
    audio_rows: list[dict[str, Any]],
    *,
    text_by_s: dict[str, bool] | None = None,
    require_text: bool = False,
) -> list[QuadUnit]:
    """Group scored audio rows into frozen 2x2 quadruplets.

    overwrite 0/1 flags.
    """

    slots: dict[str, dict[str, dict[str, dict[str, bool]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    speaker_by_stem: dict[tuple[str, str], str] = {}
    for row in audio_rows:
        quadruplet_id = str(row["quadruplet_id"])
        content_id = str(row["content_id"])
        z_label = str(row["paralinguistic_label"])
        query_type = str(row["query_type"])
        if query_type not in ("content", "paralinguistic"):
            raise ValueError(f"Unexpected query_type={query_type}")
        cell = slots[quadruplet_id][content_id][z_label]
        if query_type in cell:
            raise ValueError(
                "Duplicate scored row for "
                f"{quadruplet_id}/{content_id}/{z_label}/{query_type}"
            )
        cell[query_type] = bool(row["is_correct"])
        speaker_by_stem[(quadruplet_id, content_id)] = str(row["speaker_id"])

    return [
        _quad_from_stem_slots(
            quadruplet_id,
            stem_slots,
            speaker_by_stem,
            text_by_s=text_by_s,
            require_text=require_text,
        )
        for quadruplet_id, stem_slots in sorted(slots.items())
    ]


def _quad_from_stem_slots(
    quadruplet_id: str,
    stem_slots: dict[str, dict[str, dict[str, bool]]],
    speaker_by_stem: dict[tuple[str, str], str],
    *,
    text_by_s: dict[str, bool] | None,
    require_text: bool,
) -> QuadUnit:
    """Convert one quadruplet's nested slots into two StemUnit values."""

    content_ids = sorted(stem_slots)
    if len(content_ids) != 2:
        raise ValueError(
            f"Quadruplet {quadruplet_id} has {len(content_ids)} stems, expected 2"
        )
    stems = [
        _stem_from_z_slots(
            quadruplet_id,
            content_id,
            speaker_by_stem[(quadruplet_id, content_id)],
            stem_slots[content_id],
            _text_flag_for_stem(content_id, text_by_s, require_text=require_text),
        )
        for content_id in content_ids
    ]
    speakers = {stem.speaker_id for stem in stems}
    if len(speakers) != 1:
        raise ValueError(
            f"Quadruplet {quadruplet_id} has mixed speakers {sorted(speakers)}"
        )
    return QuadUnit(
        quadruplet_id=quadruplet_id,
        speaker_id=stems[0].speaker_id,
        stem_a=stems[0],
        stem_b=stems[1],
    )


def _text_flag_for_stem(
    content_id: str,
    text_by_s: dict[str, bool] | None,
    *,
    require_text: bool,
) -> bool | None:
    """Return text 0/1 when required; otherwise leave the stem unaligned."""

    if not require_text:
        return None
    if text_by_s is None or content_id not in text_by_s:
        raise KeyError(f"Missing text correctness for content_id={content_id}")
    return bool(text_by_s[content_id])


def stems_from_quads(quads: list[QuadUnit]) -> list[StemUnit]:
    """Flatten quadruplets into the 176/440 stem list."""

    return [stem for quad in quads for stem in (quad.stem_a, quad.stem_b)]


def stems_by_content_id(stems: list[StemUnit]) -> dict[str, StemUnit]:
    """Index stems by unique s. Duplicate s across quads is an error."""

    indexed: dict[str, StemUnit] = {}
    for stem in stems:
        if stem.content_id in indexed:
            raise ValueError(
                f"content_id {stem.content_id} appears in more than one quadruplet"
            )
        indexed[stem.content_id] = stem
    return indexed


def quads_by_speaker_id(quads: list[QuadUnit]) -> dict[str, list[QuadUnit]]:
    """Group complete quadruplets by speaker for P cluster bootstrap."""

    grouped: dict[str, list[QuadUnit]] = defaultdict(list)
    for quad in quads:
        grouped[quad.speaker_id].append(quad)
    return dict(grouped)


def assert_expected_eval_size(
    quads: list[QuadUnit],
    *,
    expected_quads: int,
    expected_stems: int,
    layer: str,
) -> None:
    """Refuse to score a table that is not the frozen eval size."""

    stem_count = len(stems_from_quads(quads))
    if len(quads) != expected_quads or stem_count != expected_stems:
        raise ValueError(
            f"{layer} locked eval size mismatch: got {len(quads)} quads / "
            f"{stem_count} stems, expected {expected_quads} / {expected_stems}"
        )


def cnt_score_from_stems(stems: list[StemUnit]) -> MetricPoint:
    """Content mean Acc over audios (two z per s)."""

    flags = [
        flag
        for stem in stems
        for flag in (stem.content_neutral, stem.content_happy)
    ]
    return mean_of_flags(flags)


def para_score_from_stems(stems: list[StemUnit]) -> MetricPoint:
    """Paralinguistic mean Acc over audios (two z per s)."""

    flags = [
        flag
        for stem in stems
        for flag in (stem.para_neutral, stem.para_happy)
    ]
    return mean_of_flags(flags)


def para_n_recall_from_stems(stems: list[StemUnit]) -> MetricPoint:
    """Accuracy on gold-neutral para queries."""

    return mean_of_flags([stem.para_neutral for stem in stems])


def para_h_recall_from_stems(stems: list[StemUnit]) -> MetricPoint:
    """Accuracy on gold-happy para queries."""

    return mean_of_flags([stem.para_happy for stem in stems])


def para_invariability_from_quads(quads: list[QuadUnit]) -> MetricPoint:
    """Same z, two s: para correctness equal (including both wrong)."""

    flags = [
        flag
        for quad in quads
        for flag in (
            quad.stem_a.para_neutral == quad.stem_b.para_neutral,
            quad.stem_a.para_happy == quad.stem_b.para_happy,
        )
    ]
    return mean_of_flags(flags)


def success_para_invariability_from_quads(quads: list[QuadUnit]) -> MetricPoint:
    """Same z, two s: both para answers correct."""

    flags = [
        flag
        for quad in quads
        for flag in (
            quad.stem_a.para_neutral and quad.stem_b.para_neutral,
            quad.stem_a.para_happy and quad.stem_b.para_happy,
        )
    ]
    return mean_of_flags(flags)


def success_para_sensitivity_from_stems(stems: list[StemUnit]) -> MetricPoint:
    """Same s, two z: both N and H para answers correct."""

    return mean_of_flags(
        [stem.para_neutral and stem.para_happy for stem in stems]
    )


def acc_text_from_stems(stems: list[StemUnit]) -> MetricPoint:
    """Mean text Acc over the supplied s list (multiplicity preserved)."""

    flags: list[bool] = []
    for stem in stems:
        if stem.text_correct is None:
            raise ValueError(f"Stem {stem.content_id} has no text correctness")
        flags.append(stem.text_correct)
    return mean_of_flags(flags)


def modality_consistency_from_stems(stems: list[StemUnit]) -> MetricPoint:
    """Per content audio: audio 0/1 equals text(s) 0/1."""

    flags: list[bool] = []
    for stem in stems:
        if stem.text_correct is None:
            raise ValueError(f"Stem {stem.content_id} has no text correctness")
        flags.append(stem.content_neutral == stem.text_correct)
        flags.append(stem.content_happy == stem.text_correct)
    return mean_of_flags(flags)


def success_modality_consistency_from_stems(stems: list[StemUnit]) -> MetricPoint:
    """Per content audio: text(s) correct and that audio correct."""

    flags: list[bool] = []
    for stem in stems:
        if stem.text_correct is None:
            raise ValueError(f"Stem {stem.content_id} has no text correctness")
        flags.append(bool(stem.text_correct and stem.content_neutral))
        flags.append(bool(stem.text_correct and stem.content_happy))
    return mean_of_flags(flags)


def cnt_invariability_from_stems(stems: list[StemUnit]) -> MetricPoint:
    """Same s, two z: content correctness equal (including both wrong)."""

    return mean_of_flags(
        [stem.content_neutral == stem.content_happy for stem in stems]
    )


def success_cnt_invariability_from_stems(stems: list[StemUnit]) -> MetricPoint:
    """Among text-correct s: both z content audios correct.

    """

    flags = []
    for stem in stems:
        if stem.text_correct is None:
            raise ValueError(f"Stem {stem.content_id} has no text correctness")
        if not stem.text_correct:
            continue
        flags.append(stem.content_neutral and stem.content_happy)
    return mean_of_flags(flags)


def success_cnt_sensitivity_from_quads(quads: list[QuadUnit]) -> MetricPoint:
    """Column (fixed z, two s): if both texts correct, both content audios correct.

    unprefixed cnt_sensitivity.
    """

    flags: list[bool] = []
    for quad in quads:
        text_a = quad.stem_a.text_correct
        text_b = quad.stem_b.text_correct
        if text_a is None or text_b is None:
            raise ValueError(f"Quad {quad.quadruplet_id} missing text flags")
        if not (text_a and text_b):
            continue
        flags.append(
            quad.stem_a.content_neutral and quad.stem_b.content_neutral
        )
        flags.append(quad.stem_a.content_happy and quad.stem_b.content_happy)
    return mean_of_flags(flags)


def unit_success_from_quads(quads: list[QuadUnit]) -> MetricPoint:
    """One if all 8 audio answers in the quadruplet are correct, else 0."""

    flags = [_quad_all_eight_correct(quad) for quad in quads]
    return mean_of_flags(flags)


def _quad_all_eight_correct(quad: QuadUnit) -> bool:
    """True only when both stems have all four audio answers correct."""

    return all(
        flag
        for stem in (quad.stem_a, quad.stem_b)
        for flag in (
            stem.content_neutral,
            stem.content_happy,
            stem.para_neutral,
            stem.para_happy,
        )
    )


def ratio_or_zero(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, or 0 when the denominator is 0."""

    if denominator <= 0:
        return 0.0
    return numerator / denominator


def score_am_from_cnt_para(
    cnt_score: MetricPoint, para_score: MetricPoint
) -> MetricPoint:
    """Total score: arithmetic mean of content and para mean Acc.

    equals the micro-average of all eight answers in each quadruplet. It is
    not unit_success and not min of the two task scores.
    """

    return MetricPoint(
        (cnt_score.value + para_score.value) / 2.0,
        cnt_score.denominator + para_score.denominator,
    )


def score_hm_from_cnt_para(
    cnt_score: MetricPoint, para_score: MetricPoint
) -> MetricPoint:
    """Harmonic mean of content and para mean Acc: 2ab / (a + b).

    """

    combined = cnt_score.value + para_score.value
    return MetricPoint(
        2.0 * ratio_or_zero(cnt_score.value * para_score.value, combined),
        cnt_score.denominator + para_score.denominator,
    )


def delta_para_damage_from_scores(para: float, para_base: float) -> float:
    """One-sided para drop vs a designated base. Zero when para did not fall."""

    return max(0.0, para_base - para)


def delta_gap_gain_from_scores(gap: float, gap_base: float) -> float:
    """One-sided gap shrinkage vs a designated base. Zero when gap did not fall."""

    return max(0.0, gap_base - gap)


def score_net_from_gain_and_damage(
    delta_gap_gain: MetricPoint,
    delta_para_damage: MetricPoint,
) -> MetricPoint:
    """Gap shrinkage minus para drop. Larger is better.

    cost scores -0.010. 0.005 gap closed at 0 para cost scores +0.005.
    """

    return MetricPoint(
        delta_gap_gain.value - delta_para_damage.value,
        delta_para_damage.denominator,
    )


def finetune_tradeoff_from_base(
    para_score: MetricPoint,
    modality_gap: MetricPoint,
    *,
    para_base: float,
    gap_base: float,
) -> dict[str, MetricPoint]:
    """R composites vs a frozen same-stack base point estimate."""

    para_damage = MetricPoint(
        delta_para_damage_from_scores(para_score.value, para_base),
        para_score.denominator,
    )
    gap_gain = MetricPoint(
        delta_gap_gain_from_scores(modality_gap.value, gap_base),
        modality_gap.denominator,
    )
    return {
        "delta_para_damage": para_damage,
        "delta_gap_gain": gap_gain,
        "score_net": score_net_from_gain_and_damage(gap_gain, para_damage),
    }


def compute_p_locked_metrics(
    stems: list[StemUnit],
    quads: list[QuadUnit],
) -> dict[str, MetricPoint]:
    """P paper catalog: task scores, AM/HM totals, and the shared para suite."""

    cnt_score = cnt_score_from_stems(stems)
    para_score = para_score_from_stems(stems)
    return {
        "cnt_score": cnt_score,
        "para_score": para_score,
        "score_am": score_am_from_cnt_para(cnt_score, para_score),
        "score_hm": score_hm_from_cnt_para(cnt_score, para_score),
        "para_n_recall": para_n_recall_from_stems(stems),
        "para_h_recall": para_h_recall_from_stems(stems),
        "para_invariability": para_invariability_from_quads(quads),
        "success_para_invariability": success_para_invariability_from_quads(quads),
        "success_para_sensitivity": success_para_sensitivity_from_stems(stems),
    }


def compute_r_locked_metrics(
    stems: list[StemUnit],
    quads: list[QuadUnit],
) -> dict[str, MetricPoint]:
    """R catalog: gap, task scores, AM/HM totals, pairing suite, unit_success."""

    acc_audio = cnt_score_from_stems(stems)
    acc_text = acc_text_from_stems(stems)
    para_score = para_score_from_stems(stems)
    gap = acc_text.value - acc_audio.value
    retention = ratio_or_zero(acc_audio.value, acc_text.value)
    return {
        "modality_gap": MetricPoint(gap, acc_audio.denominator),
        "cnt_score": acc_audio,
        "para_score": para_score,
        "score_am": score_am_from_cnt_para(acc_audio, para_score),
        "score_hm": score_hm_from_cnt_para(acc_audio, para_score),
        "para_n_recall": para_n_recall_from_stems(stems),
        "para_h_recall": para_h_recall_from_stems(stems),
        "text_retention": MetricPoint(retention, acc_text.denominator),
        "modality_consistency": modality_consistency_from_stems(stems),
        "success_modality_consistency": success_modality_consistency_from_stems(
            stems
        ),
        "cnt_invariability": cnt_invariability_from_stems(stems),
        "success_cnt_invariability": success_cnt_invariability_from_stems(stems),
        "success_cnt_sensitivity": success_cnt_sensitivity_from_quads(quads),
        "para_invariability": para_invariability_from_quads(quads),
        "success_para_invariability": success_para_invariability_from_quads(quads),
        "success_para_sensitivity": success_para_sensitivity_from_stems(stems),
        "unit_success": unit_success_from_quads(quads),
        "acc_text": acc_text,
    }


def complete_quads_for_sampled_s(
    quads: list[QuadUnit],
    sampled_content_ids: set[str],
) -> list[QuadUnit]:
    """Keep quadruplets whose both s appear in the resampled s set.

    are dropped from those metrics in an s-cluster replicate.
    """

    return [
        quad
        for quad in quads
        if quad.stem_a.content_id in sampled_content_ids
        and quad.stem_b.content_id in sampled_content_ids
    ]
