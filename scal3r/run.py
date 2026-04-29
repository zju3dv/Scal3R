import argparse
from os.path import abspath, expanduser, join

from scal3r.engine import load_config
from scal3r.engine.path import get_default_output_dir, resolve_release_path

from scal3r.utils.console_utils import get_logger
from scal3r.pipelines.inference import InferenceRequest, run_inference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SCAL3R release inference entrypoint")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/models/scal3r.yaml",
        help="Release config path relative to the release root.",
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Optional custom result directory. Defaults to data/result/custom/<tag>.",
    )
    parser.add_argument(
        "--runtime_dir",
        type=str,
        default="",
        help="Optional runtime directory. Defaults to <output_dir>/runtime.",
    )
    parser.add_argument("--tag", type=str, default="run", help="Tag used for default output layout under data/result/custom/<tag>.")
    parser.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint path.")
    parser.add_argument("--device", type=str, default="", help="Optional device override.")
    parser.add_argument("--max_images", type=int, default=-1, help="Optional image-count cap.")
    parser.add_argument(
        "--preprocess_workers",
        type=int,
        default=-1,
        help="Optional image-preprocess worker override.",
    )
    parser.add_argument("--block_size", type=int, default=-1, help="Optional block-size override.")
    parser.add_argument(
        "--overlap_size",
        type=int,
        default=-1,
        help="Optional overlap-size override.",
    )
    parser.add_argument("--use_loop", type=int, default=-1, help="Optional loop toggle override.")
    parser.add_argument(
        "--use_xyz_align",
        type=int,
        default=-1,
        help="Optional xyz-alignment override.",
    )
    parser.add_argument(
        "--max_align_points_per_frame",
        type=int,
        default=-1,
        help="Optional per-overlap-frame point cap for PGO alignment. Negative keeps config/default behavior.",
    )
    parser.add_argument(
        "--pgo_workers",
        type=int,
        default=-1,
        help="Optional worker count for adjacent-block PGO alignment. Negative keeps config/default behavior; 0 uses auto.",
    )
    parser.add_argument("--test_use_amp", action="store_true", help="Enable AMP during inference.")
    parser.add_argument("--save_dpt", type=int, default=-1, help="Optional depth-save override.")
    parser.add_argument("--save_xyz", type=int, default=-1, help="Optional point-save override.")
    parser.add_argument(
        "--streaming_state",
        type=int,
        default=-1,
        help="Whether to stream aggregator/decoder states through disk instead of keeping them in memory.",
    )
    parser.add_argument("--offload_batches", type=int, default=-1, help="Optional batch-offload override.")
    parser.add_argument("--offload_outputs", type=int, default=-1, help="Optional output-offload override.")
    parser.add_argument("--cleanup_offload", type=int, default=-1, help="Optional offload-cleanup override.")
    parser.add_argument(
        "--offload_dir",
        type=str,
        default="",
        help="Optional offload directory override. Defaults under the runtime directory.",
    )
    parser.add_argument(
        "--probe_dir",
        type=str,
        default="",
        help="Optional directory for step-by-step probe JSON outputs.",
    )
    parser.add_argument("--stop_after_stage", type=str, default="", help="Optional exact stage name after which the backend exits.")
    parser.add_argument("--dry_run", action="store_true", help="Print backend command without executing.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logger = get_logger("scal3r.run")

    config = load_config(args.config)
    output_dir = resolve_release_path(args.output_dir) if args.output_dir else get_default_output_dir(args.tag)
    runtime_dir = resolve_release_path(args.runtime_dir) if args.runtime_dir else join(output_dir, "runtime")
    request = InferenceRequest(
        config_path=resolve_release_path(args.config),
        input_dir=abspath(expanduser(args.input_dir)),
        output_dir=output_dir,
        runtime_dir=runtime_dir,
        checkpoint=resolve_release_path(args.checkpoint) if args.checkpoint else None,
        device=args.device or None,
        max_images=args.max_images if args.max_images > 0 else None,
        preprocess_workers=args.preprocess_workers if args.preprocess_workers > 0 else None,
        block_size=args.block_size if args.block_size > 0 else None,
        overlap_size=args.overlap_size if args.overlap_size > 0 else None,
        use_loop=args.use_loop if args.use_loop >= 0 else None,
        use_xyz_align=args.use_xyz_align if args.use_xyz_align >= 0 else None,
        max_align_points_per_frame=args.max_align_points_per_frame if args.max_align_points_per_frame >= 0 else None,
        pgo_workers=args.pgo_workers if args.pgo_workers >= 0 else None,
        test_use_amp=args.test_use_amp,
        save_dpt=args.save_dpt if args.save_dpt >= 0 else None,
        save_xyz=args.save_xyz if args.save_xyz >= 0 else None,
        streaming_state=args.streaming_state if args.streaming_state >= 0 else None,
        offload_batches=args.offload_batches if args.offload_batches >= 0 else None,
        offload_outputs=args.offload_outputs if args.offload_outputs >= 0 else None,
        cleanup_offload=args.cleanup_offload if args.cleanup_offload >= 0 else None,
        offload_dir=resolve_release_path(args.offload_dir) if args.offload_dir else None,
        probe_dir=resolve_release_path(args.probe_dir) if args.probe_dir else None,
        stop_after_stage=args.stop_after_stage or None,
        dry_run=args.dry_run,
    )
    result = run_inference(config, request)
    logger.info("Backend command plan: %s", result["plan_path"])
    logger.info("Image count: %s", result["image_count"])
    logger.info("Executed: %s", result["executed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
