"""Anonymous CLI for ReMG data construction, Ours training, and local eval."""

from __future__ import annotations

import argparse
from pathlib import Path

from realmg.data.bootstrap import bootstrap_data_workspace
from realmg.data.esd import (
    download_and_verify_esd,
    prepare_p_unseen_content_split,
    prepare_p_unseen_speaker_split,
)
from realmg.data.p_layer import prepare_p_quadruplets
from realmg.data.p_lexical_filter import filter_p_layer_lexical
from realmg.data.p_wer_filter import filter_p_layer_wer
from realmg.data.r_quadruplets import prepare_r_quadruplets
from realmg.data.r_text import prepare_r_text_sources
from realmg.data.r_text_filter import filter_r_text_candidates
from realmg.data.r_text_teacher import (
    filter_r_teacher_text_predictions,
    prepare_r_teacher_text_examples,
    run_r_teacher_text_api,
)
from realmg.data.r_teacher_text_carved import (
    filter_r_teacher_text_carved_train,
    prepare_r_teacher_text_carved_train,
    run_r_teacher_text_carved_train,
    set_carved_teacher_backbone,
)
from realmg.data.r_tts_gate import prepare_r_tts_gate
from realmg.data.r_tts_gate_listen import prepare_r_tts_gate_listen
from realmg.data.r_tts_gate_synth import run_r_tts_gate
from realmg.data.r_tts_gate_wer import score_r_tts_gate_wer
from realmg.data.r_tts_mass import downscale_r_tts_mass, prepare_r_tts_mass
from realmg.data.r_tts_mass_carve import carve_r_tts_mass_from_provisional
from realmg.data.r_tts_mass_ser import score_r_tts_mass_ser
from realmg.data.r_tts_mass_synth import run_r_tts_mass
from realmg.data.r_tts_mass_wer import score_r_tts_mass_wer
from realmg.data.r_vctk import prepare_r_vctk_reference_speakers
from realmg.data.r_vctk_refs import prepare_r_vctk_reference_clips
from realmg.eval.local_server_audio_eval import (
    run_p_eval_local_server,
    run_r_eval_local_server,
    run_r_eval_text_local_server,
    smoke_local_server_audio,
)
from realmg.eval.p_harness import score_p_eval_predictions
from realmg.eval.p_paper_eval import prepare_p_paper_eval
from realmg.eval.r_eval_text import (
    prepare_r_eval_text_examples,
    score_r_eval_text_predictions,
)
from realmg.eval.r_harness import prepare_r_eval_examples, score_r_eval_predictions
from realmg.eval.score_locked_eval import score_locked_eval
from realmg.train.ours_model import BACKBONE_PHI4_MM, BACKBONE_QWEN25_OMNI
from realmg.train.ours_train import dry_run_ours_data, run_ours_training


def build_parser() -> argparse.ArgumentParser:
    """Build the anonymous ReMG CLI."""

    parser = argparse.ArgumentParser(prog="realmg", description="ReMG toolkit")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("bootstrap-data", help="Create Data/ skeleton.")
    sub.add_parser("prepare-esd", help="Download and verify ESD (English) via HF.")
    sub.add_parser("prepare-p-unseen-speaker")
    sub.add_parser("prepare-p-unseen-content")
    sub.add_parser("prepare-p-quadruplets")
    sub.add_parser("filter-p-wer")
    sub.add_parser("filter-p-lexical")
    sub.add_parser("prepare-p-paper-eval")
    sub.add_parser("prepare-ours-train-construction-dev")
    sub.add_parser("prepare-r-vctk")
    sub.add_parser("prepare-r-vctk-refs")
    sub.add_parser("prepare-r-text")
    sub.add_parser("filter-r-text")
    sub.add_parser("prepare-r-teacher-text")
    run_teacher = sub.add_parser("run-r-teacher-text")
    run_teacher.add_argument("--limit", type=int, default=None)
    run_teacher.add_argument("--sample-size", type=int, default=None)
    filter_teacher = sub.add_parser("filter-r-teacher-text")
    filter_teacher.add_argument("--predictions-file", default=None)
    carved = sub.add_parser("prepare-r-teacher-text-carved-train")
    carved.add_argument(
        "--backbone",
        default=BACKBONE_QWEN25_OMNI,
        choices=(BACKBONE_QWEN25_OMNI, BACKBONE_PHI4_MM),
    )
    run_carved = sub.add_parser("run-r-teacher-text-carved-train")
    run_carved.add_argument(
        "--backbone",
        default=BACKBONE_QWEN25_OMNI,
        choices=(BACKBONE_QWEN25_OMNI, BACKBONE_PHI4_MM),
    )
    run_carved.add_argument("--limit", type=int, default=None)
    filter_carved = sub.add_parser("filter-r-teacher-text-carved-train")
    filter_carved.add_argument(
        "--backbone",
        default=BACKBONE_QWEN25_OMNI,
        choices=(BACKBONE_QWEN25_OMNI, BACKBONE_PHI4_MM),
    )
    filter_carved.add_argument("--predictions-file", default=None)
    sub.add_parser("prepare-r-tts-gate")
    run_gate = sub.add_parser("run-r-tts-gate")
    run_gate.add_argument("--engine", default=None)
    sub.add_parser("score-r-tts-gate-wer")
    sub.add_parser("prepare-r-tts-gate-listen")
    sub.add_parser("prepare-r-tts-mass")
    run_mass = sub.add_parser("run-r-tts-mass")
    run_mass.add_argument("--engine", default=None)
    sub.add_parser("score-r-tts-mass-wer")
    sub.add_parser("score-r-tts-mass-ser")
    sub.add_parser("carve-r-tts-mass")
    sub.add_parser("downscale-r-tts-mass")
    sub.add_parser("prepare-r-quadruplets")
    sub.add_parser("prepare-r-eval")
    sub.add_parser("prepare-r-eval-text")

    train = sub.add_parser(
        "train-ours",
        help="Train Ours (Omni v5_brief or Phi-4 v6_textkd).",
    )
    train.add_argument(
        "--backbone",
        default=BACKBONE_QWEN25_OMNI,
        choices=(BACKBONE_QWEN25_OMNI, BACKBONE_PHI4_MM),
    )
    train.add_argument("--model-dir", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--micro-batch-size", type=int, default=4)
    train.add_argument("--grad-accum", type=int, default=32)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-new-tokens-r", type=int, default=512)
    train.add_argument("--max-new-tokens-p", type=int, default=64)
    train.add_argument("--clip-eps", type=float, default=0.2)
    train.add_argument("--fkl-top-k", type=int, default=16)
    train.add_argument("--max-optimizer-steps", type=int, default=None)
    train.add_argument("--no-prefetch", action="store_true")
    train.add_argument("--dry-run-data", action="store_true")
    train.add_argument("--r-train-utterances", default=None)

    sub.add_parser("smoke-local-server-audio")
    run_p = sub.add_parser("run-p-eval-local-server")
    run_p.add_argument("--protocol", default="paper")
    sub.add_parser("run-r-eval-local-server")
    sub.add_parser("run-r-eval-text-local-server")
    score_p = sub.add_parser("score-p-eval")
    score_p.add_argument("--predictions-file", type=str, required=True)
    score_r = sub.add_parser("score-r-eval")
    score_r.add_argument("--predictions-file", type=str, required=True)
    score_rt = sub.add_parser("score-r-eval-text")
    score_rt.add_argument("--predictions-file", type=str, required=True)
    score_locked = sub.add_parser("score-locked-eval")
    score_locked.add_argument("--slug", default=None)
    score_locked.add_argument("--all-base", action="store_true")
    return parser


def main() -> None:
    """Dispatch CLI commands."""

    args = build_parser().parse_args()
    cmd = args.command
    if cmd == "bootstrap-data":
        bootstrap_data_workspace(".")
    elif cmd == "prepare-esd":
        download_and_verify_esd()
    elif cmd == "prepare-p-unseen-speaker":
        prepare_p_unseen_speaker_split()
    elif cmd == "prepare-p-unseen-content":
        prepare_p_unseen_content_split()
    elif cmd == "prepare-p-quadruplets":
        prepare_p_quadruplets()
    elif cmd == "filter-p-wer":
        filter_p_layer_wer()
    elif cmd == "filter-p-lexical":
        filter_p_layer_lexical()
    elif cmd == "prepare-p-paper-eval":
        prepare_p_paper_eval(merge_predictions=True)
    elif cmd == "prepare-ours-train-construction-dev":
        from realmg.data.freeze_ours_train_manifests import (
            prepare_ours_train_construction_dev_manifests,
        )

        prepare_ours_train_construction_dev_manifests()
    elif cmd == "prepare-r-vctk":
        prepare_r_vctk_reference_speakers()
    elif cmd == "prepare-r-vctk-refs":
        prepare_r_vctk_reference_clips()
    elif cmd == "prepare-r-text":
        prepare_r_text_sources()
    elif cmd == "filter-r-text":
        filter_r_text_candidates()
    elif cmd == "prepare-r-teacher-text":
        prepare_r_teacher_text_examples()
    elif cmd == "run-r-teacher-text":
        run_r_teacher_text_api(limit=args.limit, sample_size=args.sample_size)
    elif cmd == "filter-r-teacher-text":
        filter_r_teacher_text_predictions(predictions_file=args.predictions_file)
    elif cmd == "prepare-r-teacher-text-carved-train":
        set_carved_teacher_backbone(args.backbone)
        prepare_r_teacher_text_carved_train()
    elif cmd == "run-r-teacher-text-carved-train":
        set_carved_teacher_backbone(args.backbone)
        run_r_teacher_text_carved_train(limit=args.limit)
    elif cmd == "filter-r-teacher-text-carved-train":
        set_carved_teacher_backbone(args.backbone)
        filter_r_teacher_text_carved_train(predictions_file=args.predictions_file)
    elif cmd == "prepare-r-tts-gate":
        prepare_r_tts_gate()
    elif cmd == "run-r-tts-gate":
        run_r_tts_gate(engine=args.engine or "all")
    elif cmd == "score-r-tts-gate-wer":
        score_r_tts_gate_wer()
    elif cmd == "prepare-r-tts-gate-listen":
        prepare_r_tts_gate_listen()
    elif cmd == "prepare-r-tts-mass":
        prepare_r_tts_mass()
    elif cmd == "run-r-tts-mass":
        run_r_tts_mass(engine=args.engine or "all")
    elif cmd == "score-r-tts-mass-wer":
        score_r_tts_mass_wer()
    elif cmd == "score-r-tts-mass-ser":
        score_r_tts_mass_ser()
    elif cmd == "carve-r-tts-mass":
        carve_r_tts_mass_from_provisional()
    elif cmd == "downscale-r-tts-mass":
        downscale_r_tts_mass()
    elif cmd == "prepare-r-quadruplets":
        prepare_r_quadruplets()
    elif cmd == "prepare-r-eval":
        prepare_r_eval_examples()
    elif cmd == "prepare-r-eval-text":
        prepare_r_eval_text_examples()
    elif cmd == "train-ours":
        if args.r_train_utterances:
            import os

            os.environ["REALMG_R_TRAIN_UTTERANCES"] = str(args.r_train_utterances)
        if args.dry_run_data:
            dry_run_ours_data()
            return
        if args.backbone == BACKBONE_PHI4_MM:
            from realmg.train.ours_train_v6 import run_ours_training as run_v6

            run_v6(
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,
                micro_batch_size=args.micro_batch_size,
                grad_accum=args.grad_accum,
                learning_rate=args.learning_rate,
                seed=args.seed,
                max_new_tokens_r=args.max_new_tokens_r,
                max_new_tokens_p=args.max_new_tokens_p,
                clip_eps=args.clip_eps,
                fkl_top_k=args.fkl_top_k,
                max_optimizer_steps=args.max_optimizer_steps,
                prefetch=not args.no_prefetch,
                backbone=BACKBONE_PHI4_MM,
            )
        else:
            run_ours_training(
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                epochs=args.epochs,
                micro_batch_size=args.micro_batch_size,
                grad_accum=args.grad_accum,
                learning_rate=args.learning_rate,
                seed=args.seed,
                max_new_tokens_r=args.max_new_tokens_r,
                max_new_tokens_p=args.max_new_tokens_p,
                clip_eps=args.clip_eps,
                fkl_top_k=args.fkl_top_k,
                max_optimizer_steps=args.max_optimizer_steps,
                prefetch=not args.no_prefetch,
                backbone=BACKBONE_QWEN25_OMNI,
            )
    elif cmd == "smoke-local-server-audio":
        smoke_local_server_audio()
    elif cmd == "run-p-eval-local-server":
        run_p_eval_local_server(protocol=args.protocol)
    elif cmd == "run-r-eval-local-server":
        run_r_eval_local_server()
    elif cmd == "run-r-eval-text-local-server":
        run_r_eval_text_local_server()
    elif cmd == "score-p-eval":
        score_p_eval_predictions(args.predictions_file)
    elif cmd == "score-r-eval":
        score_r_eval_predictions(args.predictions_file)
    elif cmd == "score-r-eval-text":
        score_r_eval_text_predictions(args.predictions_file)
    elif cmd == "score-locked-eval":
        score_locked_eval(slug=args.slug, all_base=args.all_base)
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
