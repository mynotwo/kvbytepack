"""Read-only support for KVZ1 blobs written by kvbytepack <= 0.2 / kvzip.

KVZ1 = "KVZ1" | u32 n | JSON {m, dtype, shape, kind} | body, where the body
depends on the method string ``<layout><codec>``:

    raw_<c>        c(bytes)
    transpose+<c>  c(channel-major elements)
    bytesplit+<c>  framed planes, row-major
    pipeline_<c>   framed planes, channel-major
    framed         u32 count | u64 size x count | c(plane) x count
"""
import gzip
import json
import lzma
import math
import struct

import numpy as np
import zstandard

from . import layout as _layout

MAGIC = b"KVZ1"


def _codec(name):
    if name.startswith("zstd-"):
        return lambda d: zstandard.ZstdDecompressor().decompress(d)
    if name == "lzma":
        return lzma.decompress
    if name == "gzip":
        return gzip.decompress
    try:
        if name == "lz4":
            import lz4.frame as m
            return m.decompress
        if name == "brotli":
            import brotli as m
            return m.decompress
        if name == "snappy":
            import snappy as m
            return m.decompress
    except ImportError:
        raise ValueError(f"KVZ1 blob needs the optional {name!r} package") from None
    raise ValueError(f"unknown KVZ1 codec {name!r}")


def _unframe(body):
    (n,) = struct.unpack_from("<I", body, 0)
    sizes = struct.unpack_from(f"<{n}Q", body, 4)
    off, out = 4 + 8 * n, []
    for s in sizes:
        out.append(body[off:off + s])
        off += s
    if off != len(body):
        raise ValueError("KVZ1 plane framing mismatch")
    return out


def decode(blob):
    """Return ``(meta, raw_bytes)`` with meta = {dtype, shape, kind}."""
    if bytes(blob[:4]) != MAGIC:
        raise ValueError("not a KVZ1 blob")
    (n,) = struct.unpack_from("<I", blob, 4)
    meta = json.loads(bytes(blob[8:8 + n]))
    body = bytes(blob[8 + n:])
    method, shape = meta["m"], meta["shape"]
    width = np.dtype(meta["dtype"]).itemsize if meta["kind"] == "numpy" else {
        "bfloat16": 2, "float8_e4m3fn": 1, "float8_e5m2": 1}.get(
        meta["dtype"], None) or np.dtype(meta["dtype"]).itemsize
    cols = shape[-1] if shape else 1
    rows = math.prod(shape) // cols if cols else 0
    for prefix, kind in (("raw_", "raw"), ("transpose+", "transpose"),
                         ("bytesplit+", "planes"), ("pipeline_", "channel-planes")):
        if method.startswith(prefix):
            dec = _codec(method[len(prefix):])
            break
    else:
        raise ValueError(f"unknown KVZ1 method {method!r}")
    if kind == "raw":
        raw = dec(body)
    elif kind == "transpose":
        arr = np.frombuffer(dec(body), np.uint8).reshape(cols, rows, width)
        raw = arr.transpose(1, 0, 2).tobytes()
    else:
        raw = _layout.merge([dec(p) for p in _unframe(body)], rows, cols, width, kind)
    if len(raw) != rows * cols * width:
        raise ValueError("KVZ1 payload length mismatch")
    return meta, raw
