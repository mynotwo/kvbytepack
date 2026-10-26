"""Serialize a [S, D] float32 KV tensor into a storage format.

Returns (bytes, bytes_per_element). BF16/FP32 are lossless w.r.t. the original
BF16 model output; the other formats are lossy conversions, and compression is
lossless only relative to the converted bytes.
"""
import numpy as np

FORMATS = ["bf16", "fp16", "fp32", "fp8_e4m3", "fp8_e5m2", "int8", "int4", "int2"]


def _quantize(t, qmax, qmin):
    """Per-channel symmetric quantization (scale = channel absmax)."""
    import torch

    amax = t.abs().amax(dim=0, keepdim=True).clamp(min=1e-10)
    return (t / amax * qmax).round().clamp(qmin, qmax).to(torch.int8).numpy().flatten()


def _pack_bits(q, bits):
    per = 8 // bits
    mask = (1 << bits) - 1
    pad = (per - len(q) % per) % per
    if pad:
        q = np.append(q, np.zeros(pad, dtype=np.int8))
    out = np.zeros(len(q) // per, dtype=np.uint8)
    for i in range(per):
        out |= (q[i::per].astype(np.uint8) & mask) << (bits * i)
    return out.tobytes()


def to_format_bytes(tensor_fp32, fmt):
    import torch

    if fmt == "bf16":
        return tensor_fp32.to(torch.bfloat16).contiguous().view(torch.int16).numpy().tobytes(), 2
    if fmt == "fp16":
        return tensor_fp32.to(torch.float16).contiguous().view(torch.int16).numpy().tobytes(), 2
    if fmt == "fp32":
        return tensor_fp32.contiguous().numpy().tobytes(), 4
    if fmt in ("fp8_e4m3", "fp8_e5m2"):
        dt = getattr(torch, "float8_e4m3fn" if fmt == "fp8_e4m3" else "float8_e5m2", None)
        if dt is None:
            return None, 0
        return tensor_fp32.to(dt).contiguous().view(torch.uint8).numpy().tobytes(), 1
    if fmt == "int8":
        return _quantize(tensor_fp32, 127, -128).tobytes(), 1
    if fmt == "int4":
        return _pack_bits(_quantize(tensor_fp32, 7, -8), 4), 0.5
    if fmt == "int2":
        return _pack_bits(_quantize(tensor_fp32, 1, -2), 2), 0.25
    return None, 0
