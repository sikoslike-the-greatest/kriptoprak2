"""Pixel Value Differencing (PVD) steganography for RGB images.

Algorithm reference: Wu D.-C., Tsai W.-H. "A steganographic method for images
by pixel-value differencing" // Pattern Recognition Letters. 2003. Vol. 24,
No. 9-10. P. 1613-1626.

The cover image is processed channel by channel in raster order. Every
channel is split into non-overlapping pairs of consecutive pixels. For
each pair (P_i, P_{i+1}) the absolute difference |d| = |P_i - P_{i+1}|
falls into one of six fixed sub-ranges; the width of the sub-range
defines the number of bits n that may be hidden in the pair.

A "fall-off-boundary" check is performed before embedding: if the new
pair would leave the [0, 255] range it is left untouched and no bits
are consumed. The list of pairs that were actually used is returned as
side information; the receiver applies the same list to retrieve the
original bit stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

# Six fixed sub-ranges as in Wu-Tsai (2003). The width of every range is a
# power of two so that the number of embeddable bits is integer.
RANGES: List[Tuple[int, int]] = [
    (0, 7),       # 3 bits
    (8, 15),      # 3 bits
    (16, 31),     # 4 bits
    (32, 63),     # 5 bits
    (64, 127),    # 6 bits
    (128, 255),   # 7 bits
]


def range_for(abs_d: int) -> Tuple[int, int]:
    for l, u in RANGES:
        if l <= abs_d <= u:
            return l, u
    raise ValueError(f"|d| = {abs_d} is outside [0, 255]")


def bits_for(abs_d: int) -> Tuple[int, int, int]:
    l, u = range_for(abs_d)
    n = int(round(math.log2(u - l + 1)))
    return n, l, u


@dataclass
class EmbedParams:
    """Side information required for extraction."""
    height: int
    width: int
    channels: int
    message_bits: int          # length of the original message in bits
    used_mask: List[bool]      # per pair: True if the pair carries data

    def to_dict(self) -> dict:
        # The mask is converted to a bytes-packed string for compact storage.
        bits = np.array(self.used_mask, dtype=np.uint8)
        packed = np.packbits(bits).tobytes().hex()
        return {
            "height": self.height,
            "width": self.width,
            "channels": self.channels,
            "message_bits": self.message_bits,
            "used_mask_packed_hex": packed,
            "used_mask_length": len(self.used_mask),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmbedParams":
        packed = bytes.fromhex(d["used_mask_packed_hex"])
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
        bits = bits[: d["used_mask_length"]]
        return cls(
            height=d["height"],
            width=d["width"],
            channels=d["channels"],
            message_bits=d["message_bits"],
            used_mask=[bool(b) for b in bits.tolist()],
        )


def _bytes_to_bits(data: bytes) -> List[int]:
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr).tolist()


def _bits_to_bytes(bits: List[int]) -> bytes:
    if len(bits) % 8 != 0:
        bits = bits + [0] * (8 - len(bits) % 8)
    return np.packbits(np.array(bits, dtype=np.uint8)).tobytes()


def _split_delta(diff_total: int, d_is_odd: bool) -> Tuple[int, int]:
    """Split (d* - d) between the two pixels of a pair.

    Returns (delta_first, delta_second) so that
        P_i'   = P_i   + delta_first
        P_{i+1}' = P_{i+1} + delta_second
    and  delta_first - delta_second = d* - d, which keeps the new
    difference equal to d*.
    """
    half_ceil = math.ceil(diff_total / 2)
    half_floor = math.floor(diff_total / 2)
    if d_is_odd:
        return half_ceil, -half_floor
    return half_floor, -half_ceil


def estimate_capacity_bits(cover: np.ndarray) -> int:
    """Approximate maximum payload (bits) the cover may hold.

    The estimate ignores the fall-off-boundary check, so it is an upper bound
    on the real capacity.
    """
    if cover.ndim == 2:
        cover = cover[..., None]
    h, w, c = cover.shape
    pairs = (w // 2) * h * c
    total = 0
    flat = cover.reshape(-1)
    pair_count = (flat.size // 2)
    for k in range(pair_count):
        d = int(flat[2 * k]) - int(flat[2 * k + 1])
        n, _, _ = bits_for(abs(d))
        total += n
    # subtract the 32-bit length header
    return max(0, total - 32)


def embed(cover: np.ndarray, message: bytes) -> Tuple[np.ndarray, EmbedParams]:
    """Embed `message` into `cover` and return the stego image and parameters.

    Raises ValueError if the message does not fit.
    """
    if cover.dtype != np.uint8:
        raise TypeError("cover must be a uint8 array")
    if cover.ndim == 2:
        cover_work = cover[..., None]
    else:
        cover_work = cover
    h, w, c = cover_work.shape

    # bit stream = 32-bit big-endian length + payload bits
    payload_bits = _bytes_to_bits(message)
    length_header = [int(b) for b in format(len(payload_bits), "032b")]
    bit_stream: List[int] = length_header + payload_bits

    # work on a flat int16 array to avoid uint8 overflow during arithmetic
    stego = cover_work.astype(np.int16).copy()
    flat = stego.reshape(-1)
    n_pairs = flat.size // 2

    used_mask: List[bool] = [False] * n_pairs
    bit_idx = 0

    for k in range(n_pairs):
        if bit_idx >= len(bit_stream):
            break

        i = 2 * k
        p1 = int(flat[i])
        p2 = int(flat[i + 1])
        d = p1 - p2
        n, l, _u = bits_for(abs(d))

        chunk = bit_stream[bit_idx : bit_idx + n]
        if len(chunk) < n:
            chunk = chunk + [0] * (n - len(chunk))
        m_value = 0
        for b in chunk:
            m_value = (m_value << 1) | int(b)

        d_star = (l + m_value) if d >= 0 else -(l + m_value)
        delta_total = d_star - d
        delta1, delta2 = _split_delta(delta_total, d % 2 != 0)
        p1_new = p1 + delta1
        p2_new = p2 + delta2

        if 0 <= p1_new <= 255 and 0 <= p2_new <= 255:
            flat[i] = p1_new
            flat[i + 1] = p2_new
            used_mask[k] = True
            bit_idx += n
        # else: skip the pair, leave it unchanged, do not consume bits

    if bit_idx < len(bit_stream):
        raise ValueError(
            f"cover capacity exhausted: embedded {bit_idx} of "
            f"{len(bit_stream)} bits (message too long)"
        )

    stego = stego.astype(np.uint8)
    if cover.ndim == 2:
        stego = stego[..., 0]

    params = EmbedParams(
        height=h,
        width=w,
        channels=c,
        message_bits=len(payload_bits),
        used_mask=used_mask,
    )
    return stego, params


def extract(stego: np.ndarray, params: EmbedParams) -> bytes:
    """Recover the message embedded into `stego` using `params`."""
    if stego.dtype != np.uint8:
        raise TypeError("stego must be a uint8 array")
    if stego.ndim == 2:
        stego_work = stego[..., None]
    else:
        stego_work = stego
    h, w, c = stego_work.shape
    if (h, w, c) != (params.height, params.width, params.channels):
        raise ValueError("stego image shape does not match parameters")

    flat = stego_work.reshape(-1)
    n_pairs = flat.size // 2

    bits: List[int] = []
    target_bits = 32 + params.message_bits

    for k in range(n_pairs):
        if not params.used_mask[k]:
            continue
        if len(bits) >= target_bits:
            break

        i = 2 * k
        p1 = int(flat[i])
        p2 = int(flat[i + 1])
        d_star = p1 - p2
        n, l, _u = bits_for(abs(d_star))
        m_value = abs(d_star) - l
        chunk = [(m_value >> (n - 1 - j)) & 1 for j in range(n)]
        bits.extend(chunk)

    if len(bits) < target_bits:
        raise ValueError("not enough bits could be recovered from stego image")

    header_bits = bits[:32]
    declared_len = 0
    for b in header_bits:
        declared_len = (declared_len << 1) | b
    if declared_len != params.message_bits:
        # tolerate noise: prefer params if declared length disagrees
        pass

    payload_bits = bits[32 : 32 + params.message_bits]
    return _bits_to_bytes(payload_bits)
