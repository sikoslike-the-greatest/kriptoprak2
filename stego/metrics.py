"""Imperceptibility and robustness metrics for stego images."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from skimage.metrics import structural_similarity as sk_ssim


def mse(cover: np.ndarray, stego: np.ndarray) -> float:
    a = cover.astype(np.float64)
    b = stego.astype(np.float64)
    return float(np.mean((a - b) ** 2))


def rmse(cover: np.ndarray, stego: np.ndarray) -> float:
    return math.sqrt(mse(cover, stego))


def psnr(cover: np.ndarray, stego: np.ndarray) -> float:
    err = mse(cover, stego)
    if err == 0:
        return float("inf")
    return 10.0 * math.log10((255.0 ** 2) / err)


def ssim(cover: np.ndarray, stego: np.ndarray) -> float:
    if cover.ndim == 3:
        return float(sk_ssim(cover, stego, channel_axis=-1, data_range=255))
    return float(sk_ssim(cover, stego, data_range=255))


def ber(message: Iterable[int], extracted: Iterable[int]) -> float:
    """Bit-error rate between two equal-length bit sequences."""
    a = np.asarray(list(message), dtype=np.uint8)
    b = np.asarray(list(extracted), dtype=np.uint8)
    if a.shape != b.shape:
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
    if a.size == 0:
        return 0.0
    return float(np.mean(a != b))


def capacity_bpp(bits_embedded: int, image: np.ndarray) -> float:
    """Embedding capacity expressed in bits per pixel of the cover image."""
    h, w = image.shape[:2]
    return bits_embedded / (h * w)


def ncc(original_msg: np.ndarray, extracted_msg: np.ndarray) -> float:
    """Normalized cross-correlation, useful when the message is itself an
    image; returns 1.0 for identical inputs."""
    a = original_msg.astype(np.float64).ravel()
    b = extracted_msg.astype(np.float64).ravel()
    if a.size == 0 or b.size == 0:
        return 0.0
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    num = float(np.sum(a * b))
    den = float(math.sqrt(np.sum(a * a) * np.sum(b * b)))
    return num / den if den != 0 else 0.0
