"""Computational experiments for the PVD report.

Run as ``python -m stego.experiments``. The script writes:

* PNG figures to ``images/pvd/`` (cover, stego, histograms, robustness plots);
* a CSV table with the metrics (``images/pvd/metrics.csv``);
* a Markdown summary (``images/pvd/summary.md``) used as a draft for the
  report tables.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from skimage import data, util

from . import pvd
from . import metrics as M


OUT_DIR = Path("images/pvd")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- helpers ----------------------------------------------------------

def make_payload(n_bytes: int, seed: int = 0xC0FFEE) -> bytes:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=n_bytes, dtype=np.uint8).tobytes()


def safe_embed(cover: np.ndarray, target_bytes: int) -> tuple[np.ndarray, pvd.EmbedParams, bytes]:
    """Try to embed `target_bytes` random bytes. If capacity is too low,
    halve the payload until it fits. Returns the stego, params and the
    actually embedded message."""
    n = target_bytes
    while n > 0:
        msg = make_payload(n)
        try:
            stego, params = pvd.embed(cover, msg)
            return stego, params, msg
        except ValueError:
            n = int(n * 0.95) - 1
    raise RuntimeError("could not embed any payload")


def find_max_capacity_bytes(cover: np.ndarray, hi: int | None = None) -> int:
    """Binary-search the largest payload size (in bytes) that the cover
    accepts."""
    upper = hi if hi is not None else pvd.estimate_capacity_bits(cover) // 8
    lo = 1
    hi_v = max(upper, 1)
    # ensure hi_v is too big
    last_ok = 0
    for _ in range(40):
        if lo > hi_v:
            break
        mid = (lo + hi_v) // 2
        try:
            pvd.embed(cover, make_payload(mid))
            last_ok = mid
            lo = mid + 1
        except ValueError:
            hi_v = mid - 1
    return last_ok


def save_image(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


# ---------- experiments ------------------------------------------------------

def experiment_max_capacity(cover: np.ndarray) -> dict:
    print("[exp 1] searching for maximum capacity ...")
    max_bytes = find_max_capacity_bytes(cover)
    msg = make_payload(max_bytes)
    stego, params = pvd.embed(cover, msg)
    recovered = pvd.extract(stego, params)
    assert recovered == msg, "max-capacity roundtrip failed"

    metrics = {
        "payload_bytes": max_bytes,
        "payload_bits": max_bytes * 8,
        "psnr_db": M.psnr(cover, stego),
        "mse": M.mse(cover, stego),
        "rmse": M.rmse(cover, stego),
        "ssim": M.ssim(cover, stego),
        "capacity_bpp": M.capacity_bpp(max_bytes * 8, cover),
        "pairs_used": int(sum(params.used_mask)),
        "pairs_total": len(params.used_mask),
    }
    save_image(stego, OUT_DIR / "stego_max.png")
    print(f"        max payload: {max_bytes} bytes, PSNR={metrics['psnr_db']:.2f} dB, "
          f"SSIM={metrics['ssim']:.4f}")
    return metrics


def experiment_varying_payload(cover: np.ndarray, max_bytes: int) -> list[dict]:
    print("[exp 2] varying payload sizes ...")
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    rows = []
    for f in fractions:
        target = max(1, int(max_bytes * f))
        stego, params, msg = safe_embed(cover, target)
        actual_bytes = len(msg)
        recovered = pvd.extract(stego, params)
        assert recovered == msg
        rows.append({
            "fraction": f,
            "payload_bytes": actual_bytes,
            "payload_bits": actual_bytes * 8,
            "psnr_db": M.psnr(cover, stego),
            "mse": M.mse(cover, stego),
            "rmse": M.rmse(cover, stego),
            "ssim": M.ssim(cover, stego),
            "capacity_bpp": M.capacity_bpp(actual_bytes * 8, cover),
        })
        save_image(stego, OUT_DIR / f"stego_{int(f*100):03d}pc.png")
        print(f"        payload {actual_bytes} B  PSNR={rows[-1]['psnr_db']:.2f} dB  "
              f"SSIM={rows[-1]['ssim']:.4f}")
    return rows


def experiment_robustness(cover: np.ndarray, payload_bytes: int) -> list[dict]:
    print("[exp 3] robustness to post-processing ...")
    msg = make_payload(payload_bytes)
    stego, params = pvd.embed(cover, msg)
    msg_bits = np.unpackbits(np.frombuffer(msg, dtype=np.uint8))

    def evaluate(name: str, attacked: np.ndarray) -> dict:
        try:
            recovered = pvd.extract(attacked, params)
            ok = recovered == msg
        except Exception as exc:
            recovered = b""
            ok = False
        rec_bits = np.unpackbits(np.frombuffer(recovered, dtype=np.uint8))
        bit_err = M.ber(msg_bits.tolist(), rec_bits.tolist())
        return {
            "attack": name,
            "psnr_stego_vs_attacked": M.psnr(stego, attacked),
            "ber": bit_err,
            "exact_match": ok,
        }

    rows = []
    rows.append(evaluate("none (clean stego)", stego))

    # brightness +5
    bright = np.clip(stego.astype(np.int16) + 5, 0, 255).astype(np.uint8)
    rows.append(evaluate("brightness +5", bright))

    # additive Gaussian noise sigma=2
    rng = np.random.default_rng(7)
    noisy = np.clip(stego.astype(np.int16) + rng.normal(0, 2, stego.shape).astype(np.int16),
                    0, 255).astype(np.uint8)
    rows.append(evaluate("Gaussian noise sigma=2", noisy))

    # salt-and-pepper 0.5%
    sp_rng = np.random.default_rng(11)
    sp = stego.copy()
    n_pix = sp.size
    n_pepper = int(n_pix * 0.005 / 2)
    n_salt = int(n_pix * 0.005 / 2)
    flat_sp = sp.reshape(-1)
    idx_p = sp_rng.choice(n_pix, size=n_pepper, replace=False)
    idx_s = sp_rng.choice(n_pix, size=n_salt, replace=False)
    flat_sp[idx_p] = 0
    flat_sp[idx_s] = 255
    rows.append(evaluate("salt-and-pepper 0.5%", sp))

    # JPEG compression q=90
    buf = io.BytesIO()
    Image.fromarray(stego).save(buf, format="JPEG", quality=90)
    buf.seek(0)
    jpg = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
    rows.append(evaluate("JPEG quality 90", jpg))

    # JPEG compression q=70
    buf = io.BytesIO()
    Image.fromarray(stego).save(buf, format="JPEG", quality=70)
    buf.seek(0)
    jpg2 = np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)
    rows.append(evaluate("JPEG quality 70", jpg2))

    for r in rows:
        print(f"        {r['attack']:30s}  BER={r['ber']:.4f}  "
              f"PSNR(stego↔attacked)={r['psnr_stego_vs_attacked']:.2f} dB  "
              f"exact={r['exact_match']}")
    return rows


def experiment_histograms(cover: np.ndarray, stego: np.ndarray, name: str) -> Path:
    print(f"[exp 4] histograms ({name}) ...")
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), sharex=True, sharey=True)
    channel_names = ["Red", "Green", "Blue"]
    for c in range(3):
        cov = cover[..., c].ravel()
        ste = stego[..., c].ravel()
        axes[0, c].hist(cov, bins=256, range=(0, 255), color="steelblue")
        axes[0, c].set_title(f"{channel_names[c]} — cover")
        axes[1, c].hist(ste, bins=256, range=(0, 255), color="indianred")
        axes[1, c].set_title(f"{channel_names[c]} — stego ({name})")
        axes[1, c].set_xlabel("pixel value")
    axes[0, 0].set_ylabel("count")
    axes[1, 0].set_ylabel("count")
    fig.tight_layout()
    out_path = OUT_DIR / f"histograms_{name}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def experiment_diff_map(cover: np.ndarray, stego: np.ndarray, name: str) -> Path:
    diff = np.abs(cover.astype(np.int16) - stego.astype(np.int16)).astype(np.uint8)
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    # amplify for visibility
    scale = max(1, 255 // max(1, int(diff.max())))
    ax.imshow(np.clip(diff * scale, 0, 255).astype(np.uint8))
    ax.set_title(f"|cover − stego| × {scale}")
    ax.axis("off")
    fig.tight_layout()
    out_path = OUT_DIR / f"diffmap_{name}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


# ---------- entry-point ------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = np.random.default_rng(0)
    cover = data.astronaut()
    if cover.shape[2] == 4:
        cover = cover[..., :3]
    save_image(cover, OUT_DIR / "cover.png")
    print(f"cover image: {cover.shape}")

    # exp 1: max capacity
    max_metrics = experiment_max_capacity(cover)
    max_bytes = max_metrics["payload_bytes"]

    # exp 2: varying payload
    rows_var = experiment_varying_payload(cover, max_bytes)
    write_csv(OUT_DIR / "varying_payload.csv", rows_var)

    # build the max-capacity stego again for histograms/diff-map (cheap)
    msg_max = make_payload(max_bytes)
    stego_max, params_max = pvd.embed(cover, msg_max)
    save_image(stego_max, OUT_DIR / "stego_max.png")

    # build a small-payload stego for comparison
    small_target = max(1, int(max_bytes * 0.10))
    stego_small, params_small, msg_small = safe_embed(cover, small_target)
    save_image(stego_small, OUT_DIR / "stego_small.png")

    # exp 3: robustness (use ~50% of max for a meaningful BER number)
    rob_target = max(1, int(max_bytes * 0.50))
    rows_rob = experiment_robustness(cover, rob_target)
    write_csv(OUT_DIR / "robustness.csv", rows_rob)

    # exp 4: histograms before/after (max + small payload)
    experiment_histograms(cover, stego_max, "max")
    experiment_histograms(cover, stego_small, "small")
    experiment_diff_map(cover, stego_max, "max")
    experiment_diff_map(cover, stego_small, "small")

    # write final summary
    summary = {
        "cover_shape": list(cover.shape),
        "experiment_1_max": max_metrics,
        "experiment_2_varying": rows_var,
        "experiment_3_robustness": rows_rob,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # quick markdown summary
    lines = ["# PVD experiments summary", ""]
    lines.append(f"Cover image shape: {cover.shape}")
    lines.append("")
    lines.append("## Experiment 1 — maximum payload")
    for k, v in max_metrics.items():
        lines.append(f"* {k}: {v}")
    lines.append("")
    lines.append("## Experiment 2 — varying payload")
    lines.append("| fraction | bytes | PSNR (dB) | SSIM | bpp |")
    lines.append("|---|---|---|---|---|")
    for r in rows_var:
        lines.append(
            f"| {r['fraction']:.2f} | {r['payload_bytes']} | "
            f"{r['psnr_db']:.2f} | {r['ssim']:.4f} | {r['capacity_bpp']:.4f} |"
        )
    lines.append("")
    lines.append("## Experiment 3 — robustness")
    lines.append("| attack | BER | PSNR(stego↔att) | exact |")
    lines.append("|---|---|---|---|")
    for r in rows_rob:
        lines.append(
            f"| {r['attack']} | {r['ber']:.4f} | "
            f"{r['psnr_stego_vs_attacked']:.2f} | {r['exact_match']} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print(f"\nDone. Output written to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
