"""Downsample a binary PLY point cloud by keeping 1/N of the vertices.

Usage:
    python scripts/visualize/geometry/utils/downsample_ply.py \
        input.ply \
        output.ply \
        --factor 10
"""

from __future__ import annotations

import os
import shutil
import argparse
import numpy as np
from os.path import abspath, basename, dirname, splitext


PLY_SCALAR_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample a binary PLY point cloud while preserving vertex properties."
    )
    parser.add_argument("input", type=str, help="Input binary PLY file.")
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        help="Output PLY path. Defaults to <input>_<mode>_1of<factor>.ply",
    )
    parser.add_argument(
        "--factor",
        type=int,
        required=True,
        help="Keep 1 out of every N points.",
    )
    parser.add_argument(
        "--mode",
        choices=("sequential", "random"),
        default="sequential",
        help="Sequential keeps every Nth point; random keeps an exact 1/N sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used only when --mode=random.",
    )
    parser.add_argument(
        "--chunk-points",
        type=int,
        default=1_000_000,
        help="How many vertices to process per chunk.",
    )
    args = parser.parse_args()

    if args.factor < 1:
        parser.error("--factor must be >= 1")
    if args.chunk_points < 1:
        parser.error("--chunk-points must be >= 1")

    args.input = abspath(args.input)
    if args.output is None:
        stem, suffix = splitext(basename(args.input))
        args.output = os.path.join(dirname(args.input), f"{stem}_{args.mode}_1of{args.factor}{suffix}")
    else:
        args.output = abspath(args.output)

    if args.input == args.output:
        parser.error("input and output must be different files")

    return args


def read_ply_header(path: str) -> tuple[list[str], int, str, dict[str, object], np.dtype]:
    with open(path, "rb") as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError("PLY header is incomplete")
            decoded = line.decode("ascii").rstrip("\n")
            header_lines.append(decoded)
            if decoded == "end_header":
                header_size = f.tell()
                break

    if not header_lines or header_lines[0] != "ply":
        raise ValueError("Not a PLY file")

    format_name = None
    elements: list[dict[str, object]] = []
    current_element: dict[str, object] | None = None

    for line in header_lines[1:]:
        if not line or line.startswith("comment ") or line.startswith("obj_info "):
            continue
        parts = line.split()
        if parts[0] == "format":
            format_name = parts[1]
        elif parts[0] == "element":
            current_element = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elements.append(current_element)
        elif parts[0] == "property":
            if current_element is None:
                raise ValueError("Found property before any element definition")
            if parts[1] == "list":
                current_element["properties"].append(
                    {
                        "kind": "list",
                        "count_type": parts[2],
                        "item_type": parts[3],
                        "name": parts[4],
                    }
                )
            else:
                current_element["properties"].append(
                    {"kind": "scalar", "type": parts[1], "name": parts[2]}
                )

    if format_name not in {"binary_little_endian", "binary_big_endian"}:
        raise ValueError("Only binary_little_endian and binary_big_endian PLY files are supported")
    if not elements:
        raise ValueError("PLY file does not define any elements")
    if elements[0]["name"] != "vertex":
        raise ValueError("This script expects the vertex element to be the first PLY element")

    vertex_element = elements[0]
    vertex_properties = vertex_element["properties"]
    if any(prop["kind"] != "scalar" for prop in vertex_properties):
        raise ValueError("Vertex element with list properties is not supported")

    endian = "<" if format_name == "binary_little_endian" else ">"
    dtype_fields = []
    for prop in vertex_properties:
        prop_type = prop["type"]
        if prop_type not in PLY_SCALAR_TYPES:
            raise ValueError(f"Unsupported PLY property type: {prop_type}")
        dtype_fields.append((prop["name"], endian + PLY_SCALAR_TYPES[prop_type]))
    vertex_dtype = np.dtype(dtype_fields)

    return header_lines, header_size, format_name, vertex_element, vertex_dtype


def build_header(header_lines: list[str], new_vertex_count: int) -> bytes:
    updated_lines = []
    replaced = False
    for line in header_lines:
        if not replaced and line.startswith("element vertex "):
            updated_lines.append(f"element vertex {new_vertex_count}")
            replaced = True
        else:
            updated_lines.append(line)
    if not replaced:
        raise ValueError("Failed to locate vertex count in PLY header")
    return ("\n".join(updated_lines) + "\n").encode("ascii")


def downsample_sequential(
    fin,
    fout,
    vertex_dtype: np.dtype,
    vertex_count: int,
    factor: int,
    chunk_points: int,
) -> int:
    written = 0
    global_index = 0
    remaining = vertex_count

    while remaining > 0:
        current_count = min(chunk_points, remaining)
        chunk = np.fromfile(fin, dtype=vertex_dtype, count=current_count)
        if chunk.size != current_count:
            raise ValueError("Unexpected EOF while reading vertex data")

        start = (-global_index) % factor
        sampled = chunk[start::factor]
        sampled.tofile(fout)

        written += sampled.size
        global_index += chunk.size
        remaining -= chunk.size

    return written


def downsample_random(
    fin,
    fout,
    vertex_dtype: np.dtype,
    vertex_count: int,
    keep_count: int,
    chunk_points: int,
    seed: int,
) -> int:
    written = 0
    remaining_total = vertex_count
    remaining_keep = keep_count
    rng = np.random.default_rng(seed)

    while remaining_total > 0:
        current_count = min(chunk_points, remaining_total)
        chunk = np.fromfile(fin, dtype=vertex_dtype, count=current_count)
        if chunk.size != current_count:
            raise ValueError("Unexpected EOF while reading vertex data")

        if remaining_keep == 0:
            remaining_total -= current_count
            continue

        if current_count == remaining_total:
            take_count = remaining_keep
        else:
            take_count = int(
                rng.hypergeometric(
                    ngood=current_count,
                    nbad=remaining_total - current_count,
                    nsample=remaining_keep,
                )
            )

        if take_count == current_count:
            chunk.tofile(fout)
        elif take_count > 0:
            indices = np.sort(rng.choice(current_count, size=take_count, replace=False))
            chunk[indices].tofile(fout)

        written += take_count
        remaining_keep -= take_count
        remaining_total -= current_count

    return written


def main() -> None:
    args = parse_args()
    header_lines, header_size, _format_name, vertex_element, vertex_dtype = read_ply_header(
        args.input
    )

    vertex_count = int(vertex_element["count"])
    keep_count = (vertex_count + args.factor - 1) // args.factor
    output_header = build_header(header_lines, keep_count)

    with args.input.open("rb") as fin, args.output.open("wb") as fout:
        fin.seek(header_size)
        fout.write(output_header)

        if args.mode == "sequential":
            written = downsample_sequential(
                fin=fin,
                fout=fout,
                vertex_dtype=vertex_dtype,
                vertex_count=vertex_count,
                factor=args.factor,
                chunk_points=args.chunk_points,
            )
        else:
            written = downsample_random(
                fin=fin,
                fout=fout,
                vertex_dtype=vertex_dtype,
                vertex_count=vertex_count,
                keep_count=keep_count,
                chunk_points=args.chunk_points,
                seed=args.seed,
            )

        shutil.copyfileobj(fin, fout, length=1024 * 1024)

    if written != keep_count:
        raise ValueError(f"Written point count mismatch: expected {keep_count}, got {written}")

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"mode={args.mode}")
    print(f"factor={args.factor}")
    print(f"source_points={vertex_count}")
    print(f"written_points={written}")
    if args.mode == "random":
        print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
