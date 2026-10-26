"""Lossless KV cache codecs: layout transform + byte split + general-purpose codec.

A KV tensor of shape [S, D] (tokens x channels) is serialized as S*D elements of
`bpe` bytes each. The layout transforms here are pure byte permutations, so every
method is exactly invertible.

Methods
-------
raw_<codec>          codec(raw bytes)
transpose+<codec>    codec(channel-first bytes)
bytesplit+<codec>    concat(codec(byte plane i) for i in range(bpe))
pipeline_<codec>     byte split applied to channel-first bytes (the main method)
"""
import gzip
import lzma
import struct
import threading

import numpy as np
import zstandard as zstd

try:
    import lz4.frame as _lz4
except ImportError:
    _lz4 = None
try:
    import brotli as _brotli
except ImportError:
    _brotli = None
try:
    import snappy as _snappy
except ImportError:
    _snappy = None


_zstd_local = threading.local()


def _zstd1_compress(data):
    # Contexts are lazy and private to each calling thread, never shared scratch.
    if not hasattr(_zstd_local, "compressor"):
        _zstd_local.compressor = zstd.ZstdCompressor(level=1)
    if getattr(_zstd_local, "compressing", False):
        return zstd.ZstdCompressor(level=1).compress(data)
    _zstd_local.compressing = True
    try:
        return _zstd_local.compressor.compress(data)
    finally:
        _zstd_local.compressing = False


def _zstd_decompress(data, max_output_size=0, allow_extra_data=True):
    if not hasattr(_zstd_local, "decompressor"):
        _zstd_local.decompressor = zstd.ZstdDecompressor()
    if getattr(_zstd_local, "decompressing", False):
        return zstd.ZstdDecompressor().decompress(
            data, max_output_size=max_output_size, allow_extra_data=allow_extra_data)
    _zstd_local.decompressing = True
    try:
        return _zstd_local.decompressor.decompress(
            data, max_output_size=max_output_size, allow_extra_data=allow_extra_data)
    finally:
        _zstd_local.decompressing = False


def _zstd(level):
    if level == 1:
        return _zstd1_compress, _zstd_decompress
    return (lambda d: zstd.ZstdCompressor(level=level).compress(d),
            lambda d: zstd.ZstdDecompressor().decompress(d))


CODECS = {
    "zstd-1": _zstd(1),
    "zstd-9": _zstd(9),
    "zstd-19": _zstd(19),
    "lzma": (lambda d: lzma.compress(d, preset=6), lzma.decompress),
    "gzip": (lambda d: gzip.compress(d, compresslevel=6), gzip.decompress),
}
if _lz4 is not None:
    CODECS["lz4"] = (_lz4.compress, _lz4.decompress)
if _brotli is not None:
    CODECS["brotli"] = (lambda d: _brotli.compress(d, quality=4), _brotli.decompress)
if _snappy is not None:
    CODECS["snappy"] = (_snappy.compress, _snappy.decompress)


def transpose_bytes(raw, S, D, bpe):
    """Token-major [S, D, bpe] -> channel-major [D, S, bpe]."""
    if bpe < 1:
        return raw
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(S, D, int(bpe))
    return arr.transpose(1, 0, 2).tobytes()


def untranspose_bytes(raw, S, D, bpe):
    if bpe < 1:
        return raw
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(D, S, int(bpe))
    return arr.transpose(1, 0, 2).tobytes()


def byte_split(raw, bpe):
    """Split interleaved elements into `bpe` planes (plane i = byte i of each element)."""
    if bpe <= 1:
        return [raw]
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(-1, int(bpe))
    return [arr[:, i].tobytes() for i in range(int(bpe))]


def byte_merge(planes):
    if len(planes) == 1:
        return planes[0]
    cols = [np.frombuffer(p, dtype=np.uint8) for p in planes]
    return np.stack(cols, axis=1).tobytes()


def transpose_split(raw, S, D, bpe):
    """Produce the existing channel-major byte planes in a single layout pass."""
    if bpe < 1:
        return [raw]
    return [p.tobytes() for p in _transpose_plane_array(raw, S, D, bpe)]


def _transpose_plane_array(raw, S, D, bpe):
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(S, D, int(bpe))
    return np.ascontiguousarray(arr.transpose(2, 1, 0))


def merge_untranspose(planes, S, D, bpe):
    """Restore token-major bytes without an intermediate channel-major buffer."""
    if bpe < 1:
        return byte_merge(planes)
    width = int(bpe)
    if len(planes) != width:
        raise ValueError("plane count does not match bytes per element")
    arr = np.empty((S, D, width), dtype=np.uint8)
    for i, plane in enumerate(planes):
        arr[:, :, i] = np.frombuffer(plane, dtype=np.uint8).reshape(D, S).T
    return arr.tobytes()


def _pack(blobs):
    """Length-prefixed concatenation so planes can be split again on decode."""
    head = struct.pack("<I", len(blobs)) + b"".join(struct.pack("<Q", len(b)) for b in blobs)
    return head + b"".join(blobs)


def _unpack(blob):
    (n,) = struct.unpack_from("<I", blob, 0)
    sizes = struct.unpack_from(f"<{n}Q", blob, 4)
    off = 4 + 8 * n
    out = []
    for s in sizes:
        out.append(blob[off:off + s])
        off += s
    return out


def _parse(method):
    for prefix in ("raw_", "transpose+", "bytesplit+", "pipeline_"):
        if method.startswith(prefix):
            name = method[len(prefix):]
            if name not in CODECS:
                raise KeyError(f"codec not available: {name}")
            return prefix.rstrip("_+"), name
    raise KeyError(f"unknown method: {method}")


def encode(raw, S, D, bpe, method):
    """Compress `raw` (S*D elements of `bpe` bytes). Returns a self-delimiting blob."""
    layout, name = _parse(method)
    comp = CODECS[name][0]
    if layout == "raw":
        return comp(raw)
    if layout == "transpose":
        return comp(transpose_bytes(raw, S, D, bpe))
    if layout == "pipeline" and comp is _zstd1_compress and bpe >= 1:
        # Native Zstd accepts contiguous buffers: avoid serializing each plane again.
        planes = [memoryview(p) for p in _transpose_plane_array(raw, S, D, bpe)]
    else:
        planes = transpose_split(raw, S, D, bpe) if layout == "pipeline" else byte_split(raw, bpe)
    return _pack([comp(p) for p in planes])


def decode(blob, S, D, bpe, method):
    layout, name = _parse(method)
    dec = CODECS[name][1]
    if layout == "raw":
        return dec(blob)
    if layout == "transpose":
        return untranspose_bytes(dec(blob), S, D, bpe)
    planes = [dec(p) for p in _unpack(blob)]
    return merge_untranspose(planes, S, D, bpe) if layout == "pipeline" else byte_merge(planes)


def payload_size(blob, method):
    """Compressed bytes excluding the plane-length header (matches the original benchmark)."""
    layout, _ = _parse(method)
    if layout in ("bytesplit", "pipeline"):
        return sum(len(b) for b in _unpack(blob))
    return len(blob)


def available_methods():
    names = list(CODECS)
    pipe = [c for c in ("zstd-1", "zstd-9", "zstd-19", "lz4", "lzma") if c in CODECS]
    return ([f"raw_{c}" for c in names]
            + ["transpose+zstd-1", "bytesplit+zstd-1"]
            + [f"pipeline_{c}" for c in pipe])
