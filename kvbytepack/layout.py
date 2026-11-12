"""Byte permutations that expose structure in fixed-width tensor elements.

A tensor is viewed as a ``rows x cols`` matrix of ``width``-byte elements, where
``cols`` is the last (channel) axis. Every layout is a pure byte permutation, so
``merge(split(x)) == x`` for any bit pattern, including NaN payloads and -0.0.

Layouts
-------
none            one plane: the bytes as given
planes          ``width`` planes, plane i holds byte i of every element (row-major)
channel-planes  as ``planes``, but each plane is channel-major: all rows of channel 0,
                then channel 1, ... Neighbouring bytes then share a channel's scale.
"""
import numpy as np

try:
    from ._native import merge_bf16 as _native_merge_2byte
except ImportError:  # no compiler at install time: NumPy fallback below
    _native_merge_2byte = None

LAYOUTS = ("none", "planes", "channel-planes")


def plane_count(layout, width):
    if layout not in LAYOUTS:
        raise ValueError(f"unknown layout {layout!r}; expected one of {LAYOUTS}")
    return 1 if layout == "none" else width


def split(raw, rows, cols, width, layout):
    """Return the planes for ``raw`` as a list of C-contiguous buffers."""
    if len(raw) != rows * cols * width:
        raise ValueError("buffer length does not match rows * cols * width")
    if layout == "none" or (layout == "planes" and width == 1):
        return [raw]
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(rows, cols, width)
    if layout == "planes":
        planes = np.ascontiguousarray(arr.transpose(2, 0, 1))
    elif layout == "channel-planes":
        planes = np.ascontiguousarray(arr.transpose(2, 1, 0))
    else:
        raise ValueError(f"unknown layout {layout!r}")
    if planes.size == 0:
        return [b""] * width
    return [memoryview(p).cast("B") for p in planes]


def merge(planes, rows, cols, width, layout):
    """Inverse of :func:`split`. Returns ``bytes``."""
    n = plane_count(layout, width)
    if len(planes) != n:
        raise ValueError(f"expected {n} planes, got {len(planes)}")
    if any(len(p) != rows * cols * width // n for p in planes):
        raise ValueError("plane length does not match geometry")
    if n == 1 and layout != "channel-planes":
        return bytes(planes[0])
    if (layout == "channel-planes" and width == 2 and _native_merge_2byte is not None):
        return _native_merge_2byte(bytes(planes[0]), bytes(planes[1]), rows, cols)
    out = np.empty((rows, cols, width), dtype=np.uint8)
    for i, plane in enumerate(planes):
        src = np.frombuffer(plane, dtype=np.uint8)
        if layout == "planes":
            out[:, :, i] = src.reshape(rows, cols)
        else:
            out[:, :, i] = src.reshape(cols, rows).T
    return out.tobytes()


def native_available():
    return _native_merge_2byte is not None
