"""Create an MP4 from an image directory with a target duration.

Usage:
    python scripts/visualize/tools/images_to_video.py \
        data/datasets/zju/zjg/images \
        40 \
        data/datasets/zju/zjg/video.mp4
"""

import os
import shutil
import tempfile
import argparse
import subprocess
from os.path import abspath, dirname, expanduser, isdir, isfile, join, splitext


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", help="Directory containing input images")
    parser.add_argument("duration", type=float, help="Target video duration in seconds")
    parser.add_argument("output_path", help="Output video path")
    parser.add_argument("--codec", default="libx264", help="FFmpeg video codec")
    parser.add_argument("--pix_fmt", default="yuv420p", help="FFmpeg pixel format")
    parser.add_argument("--crf", type=int, default=18, help="FFmpeg CRF")
    parser.add_argument("--preset", default="slow", help="FFmpeg preset")
    parser.add_argument(
        "--powerpoint_safe",
        type=int,
        default=1,
        help="Pad to macroblock-friendly size and set SAR=1 for PowerPoint compatibility",
    )
    parser.add_argument(
        "--pad_color",
        default="white",
        help="Padding color used when --powerpoint_safe=1",
    )
    parser.add_argument(
        "--pad_multiple",
        type=int,
        default=16,
        help="Alignment multiple used when --powerpoint_safe=1",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=1,
        help="Whether to overwrite an existing output file",
    )
    parser.add_argument(
        "--dry_run",
        type=int,
        default=0,
        help="Only print the resolved FFmpeg command without running it",
    )
    return parser.parse_args()


def collect_images(image_dir: str) -> list[str]:
    images = [
        join(image_dir, name)
        for name in os.listdir(image_dir)
        if isfile(join(image_dir, name)) and splitext(name)[1].lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def detect_numbered_sequence(images: list[str]) -> tuple[int, int, str] | None:
    if not images:
        return None

    suffix = splitext(images[0])[1].lower()
    stems = []
    for image in images:
        stem = splitext(os.path.basename(image))[0]
        if splitext(image)[1].lower() != suffix or not stem.isdigit():
            return None
        stems.append(stem)

    widths = {len(stem) for stem in stems}
    if len(widths) != 1:
        return None

    numbers = [int(stem) for stem in stems]
    start = numbers[0]
    expected = list(range(start, start + len(numbers)))
    if numbers != expected:
        return None

    return start, next(iter(widths)), suffix


def quote_ffconcat_path(path: str) -> str:
    escaped = abspath(path).replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def write_ffconcat_manifest(images: list[str], fps: float) -> str:
    frame_duration = 1.0 / fps
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".ffconcat",
        prefix="images_to_video_",
        delete=False,
    )
    manifest_path = handle.name
    with handle:
        handle.write("ffconcat version 1.0\n")
        for image in images[:-1]:
            handle.write(f"file {quote_ffconcat_path(image)}\n")
            handle.write(f"duration {frame_duration:.12f}\n")
        last_image = images[-1]
        handle.write(f"file {quote_ffconcat_path(last_image)}\n")
        handle.write(f"duration {frame_duration:.12f}\n")
        handle.write(f"file {quote_ffconcat_path(last_image)}\n")
    return manifest_path


def build_ffmpeg_command(
    image_dir: str,
    images: list[str],
    fps: float,
    output_path: str,
    codec: str,
    pix_fmt: str,
    crf: int,
    preset: str,
    powerpoint_safe: bool,
    pad_color: str,
    pad_multiple: int,
    overwrite: bool,
) -> tuple[list[str], str | None]:
    overwrite_flag = "-y" if overwrite else "-n"
    video_filters = []
    if powerpoint_safe:
        if pad_multiple <= 1:
            raise ValueError(
                f"pad_multiple must be > 1 when powerpoint_safe is enabled, got {pad_multiple}"
            )
        video_filters.append(
            f"pad=ceil(iw/{pad_multiple})*{pad_multiple}:"
            f"ceil(ih/{pad_multiple})*{pad_multiple}:0:0:{pad_color}"
        )
        video_filters.append("setsar=1")

    sequence = detect_numbered_sequence(images)
    if sequence is not None:
        start_number, width, suffix = sequence
        input_pattern = join(image_dir, f"%0{width}d{suffix}")
        command = [
            "ffmpeg",
            overwrite_flag,
            "-framerate",
            f"{fps:.12f}",
            "-start_number",
            str(start_number),
            "-i",
            input_pattern,
        ]
        if video_filters:
            command.extend(["-vf", ",".join(video_filters)])
        command.extend(
            [
                "-c:v",
                codec,
                "-pix_fmt",
                pix_fmt,
                "-crf",
                str(crf),
                "-preset",
                preset,
                output_path,
            ]
        )
        return command, None

    manifest_path = write_ffconcat_manifest(images, fps)
    command = [
        "ffmpeg",
        overwrite_flag,
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        manifest_path,
        "-vsync",
        "vfr",
    ]
    if video_filters:
        command.extend(["-vf", ",".join(video_filters)])
    command.extend(
        [
            "-c:v",
            codec,
            "-pix_fmt",
            pix_fmt,
            "-crf",
            str(crf),
            "-preset",
            preset,
            output_path,
        ]
    )
    return command, manifest_path


def main():
    args = parse_args()

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH")

    image_dir = abspath(expanduser(args.image_dir))
    output_path = abspath(expanduser(args.output_path))

    if not isdir(image_dir):
        raise NotADirectoryError(f"Not a directory: {image_dir}")
    if args.duration <= 0:
        raise ValueError(f"Duration must be positive, got {args.duration}")

    images = collect_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No supported images found in {image_dir}")

    fps = len(images) / args.duration
    output_dir = dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    command, manifest_path = build_ffmpeg_command(
        image_dir=image_dir,
        images=images,
        fps=fps,
        output_path=output_path,
        codec=args.codec,
        pix_fmt=args.pix_fmt,
        crf=args.crf,
        preset=args.preset,
        powerpoint_safe=bool(args.powerpoint_safe),
        pad_color=args.pad_color,
        pad_multiple=args.pad_multiple,
        overwrite=bool(args.overwrite),
    )

    print(f"Image count: {len(images)}")
    print(f"Target duration: {args.duration:.6f}s")
    print(f"Resolved FPS: {fps:.6f}")
    print(f"Output path: {output_path}")
    print(f"PowerPoint safe: {bool(args.powerpoint_safe)}")
    if args.powerpoint_safe:
        print(f"Pad multiple: {args.pad_multiple}")
        print(f"Pad color: {args.pad_color}")
    if manifest_path is None:
        print("Input mode: numbered sequence")
    else:
        print(f"Input mode: ffconcat manifest ({manifest_path})")
    print("FFmpeg command:")
    print(" ".join(command))

    try:
        if not args.dry_run:
            subprocess.run(command, check=True)
    finally:
        if manifest_path is not None and os.path.exists(manifest_path):
            os.remove(manifest_path)


if __name__ == "__main__":
    main()
