"""Command-line interface for PVD steganography.

Usage examples:
    python -m stego embed   --cover img.png --message msg.txt --stego out.png \\
                            --params out.json
    python -m stego extract --stego out.png --params out.json --output msg.txt
    python -m stego capacity --cover img.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from . import pvd
from . import metrics as M


def _load_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def _save_image(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


def cmd_embed(args: argparse.Namespace) -> int:
    cover = _load_image(Path(args.cover))
    message = Path(args.message).read_bytes()

    stego, params = pvd.embed(cover, message)

    _save_image(stego, Path(args.stego))
    Path(args.params).write_text(json.dumps(params.to_dict()))

    print(f"cover  : {args.cover}  shape={cover.shape}")
    print(f"message: {len(message)} bytes ({len(message) * 8} bits)")
    print(f"stego  : {args.stego}")
    print(f"params : {args.params}")
    print(f"PSNR   : {M.psnr(cover, stego):.4f} dB")
    print(f"MSE    : {M.mse(cover, stego):.4f}")
    print(f"RMSE   : {M.rmse(cover, stego):.4f}")
    print(f"SSIM   : {M.ssim(cover, stego):.6f}")
    print(f"EC     : {M.capacity_bpp(len(message) * 8, cover):.4f} bpp")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    stego = _load_image(Path(args.stego))
    params = pvd.EmbedParams.from_dict(json.loads(Path(args.params).read_text()))
    message = pvd.extract(stego, params)
    Path(args.output).write_bytes(message)
    print(f"extracted {len(message)} bytes -> {args.output}")
    return 0


def cmd_capacity(args: argparse.Namespace) -> int:
    cover = _load_image(Path(args.cover))
    cap_bits = pvd.estimate_capacity_bits(cover)
    print(f"cover  : {args.cover}  shape={cover.shape}")
    print(f"upper-bound capacity: {cap_bits} bits = {cap_bits // 8} bytes")
    print(f"upper-bound bpp    : {cap_bits / (cover.shape[0] * cover.shape[1]):.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stego",
        description="PVD steganography for RGB images",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("embed", help="embed a message into a cover image")
    pe.add_argument("--cover", required=True)
    pe.add_argument("--message", required=True, help="path to a binary file")
    pe.add_argument("--stego", required=True)
    pe.add_argument("--params", required=True)
    pe.set_defaults(func=cmd_embed)

    px = sub.add_parser("extract", help="extract a message from a stego image")
    px.add_argument("--stego", required=True)
    px.add_argument("--params", required=True)
    px.add_argument("--output", required=True)
    px.set_defaults(func=cmd_extract)

    pc = sub.add_parser("capacity", help="estimate embedding capacity")
    pc.add_argument("--cover", required=True)
    pc.set_defaults(func=cmd_capacity)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
