import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kvbytepack.codec import (available_methods, byte_merge, byte_split, decode, encode,  # noqa: E402
                         payload_size, transpose_bytes, untranspose_bytes)


def _bf16_like(S, D, seed=0):
    """BF16 bit patterns of Gaussian values with per-channel scale (KV/activation-like)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((S, D)).astype(np.float32) * rng.uniform(0.1, 8, D).astype(np.float32)
    return (x.view(np.uint32) >> 16).astype(np.uint16).tobytes()


@pytest.mark.parametrize("bpe", [1, 2, 4])
def test_layout_inverses(bpe):
    S, D = 7, 5
    raw = np.random.default_rng(1).integers(0, 256, S * D * bpe, dtype=np.uint8).tobytes()
    assert untranspose_bytes(transpose_bytes(raw, S, D, bpe), S, D, bpe) == raw
    assert byte_merge(byte_split(raw, bpe)) == raw


def test_transpose_is_channel_major():
    S, D = 3, 2
    raw = bytes(range(S * D * 2))  # element (s, d) = bytes [2*(s*D+d), 2*(s*D+d)+1]
    t = transpose_bytes(raw, S, D, 2)
    assert t[:6] == bytes([0, 1, 4, 5, 8, 9])  # channel 0 across tokens 0..2


@pytest.mark.parametrize("method", available_methods())
@pytest.mark.parametrize("bpe", [2, 4])
def test_round_trip(method, bpe):
    S, D = 64, 48
    raw = _bf16_like(S, D) if bpe == 2 else np.random.default_rng(2).standard_normal(S * D).astype(np.float32).tobytes()
    blob = encode(raw, S, D, bpe, method)
    assert decode(blob, S, D, bpe, method) == raw
    assert 0 < payload_size(blob, method) <= len(blob)


def test_sub_byte_formats_round_trip():
    raw = np.random.default_rng(3).integers(0, 256, 1000, dtype=np.uint8).tobytes()
    for bpe in (0.5, 0.25, 1):
        for m in ("raw_zstd-1", "pipeline_zstd-1"):
            assert decode(encode(raw, 40, 25, bpe, m), 40, 25, bpe, m) == raw


def test_pipeline_beats_raw_on_bf16():
    # High byte (sign+exponent) has low entropy once separated; this is the core effect.
    S, D = 512, 256
    raw = _bf16_like(S, D)
    raw_sz = payload_size(encode(raw, S, D, 2, "raw_zstd-1"), "raw_zstd-1")
    pipe_sz = payload_size(encode(raw, S, D, 2, "pipeline_zstd-1"), "pipeline_zstd-1")
    assert pipe_sz < raw_sz


def test_unknown_method():
    with pytest.raises(KeyError):
        encode(b"\0\0", 1, 1, 2, "pipeline_nope")
