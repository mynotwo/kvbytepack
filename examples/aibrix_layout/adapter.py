"""Downstream byte-layout experiment, not an installed AIBrix feature."""
from dataclasses import dataclass
import hashlib
import hmac
import json
import struct

import numpy as np
import zstandard as zstd

from kvbytepack import zip_tensor
from kvbytepack.codec import _unpack, merge_untranspose
from kvbytepack.tensor import MAGIC

RECORD_MAGIC = b"KBL1"
METHOD = "pipeline_zstd-1"


@dataclass(frozen=True)
class Layout:
    tokens: int
    channels: int
    bytes_per_element: int

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in
               (self.tokens, self.channels, self.bytes_per_element)):
            raise ValueError("layout dimensions must be positive integers")
        if self.bytes_per_element not in (1, 2, 4, 8):
            raise ValueError("only fixed-width 1/2/4/8-byte elements are supported")

    @property
    def raw_bytes(self):
        return self.tokens * self.channels * self.bytes_per_element

    @property
    def dtype(self):
        # Opaque elements preserve bytes without interpreting numeric values.
        return np.dtype(f"V{self.bytes_per_element}")


def pack(raw, layout):
    if not isinstance(raw, bytes) or len(raw) != layout.raw_bytes:
        raise ValueError("expected bytes matching the explicit payload layout")
    array = np.frombuffer(raw, dtype=layout.dtype).reshape(layout.tokens, layout.channels)
    inner = zip_tensor(array, method=METHOD)
    return RECORD_MAGIC + hashlib.sha256(inner).digest() + inner


def unpack(record, layout):
    """Reject damaged or mismatched records before invoking the existing decoder."""
    if not isinstance(record, bytes) or len(record) < 44 or record[:4] != RECORD_MAGIC:
        raise ValueError("invalid layout record/version")
    inner = record[36:]
    if not hmac.compare_digest(record[4:36], hashlib.sha256(inner).digest()):
        raise ValueError("layout record checksum mismatch")
    if inner[:4] != MAGIC:
        raise ValueError("invalid tensor record")
    (n,) = struct.unpack_from("<I", inner, 4)
    if n > 512 or 8 + n > len(inner):
        raise ValueError("invalid metadata length")
    try:
        meta = json.loads(inner[8:8 + n])
    except (ValueError, UnicodeError) as exc:
        raise ValueError("invalid layout metadata") from exc
    expected = {"m": METHOD, "dtype": str(layout.dtype),
                "shape": [layout.tokens, layout.channels], "kind": "numpy"}
    if meta != expected or any(type(v) is not int for v in meta.get("shape", [])):
        raise ValueError("record metadata does not match the expected layout")
    body = inner[8 + n:]
    count = layout.bytes_per_element
    if len(body) < 4 + 8 * count or struct.unpack_from("<I", body)[0] != count:
        raise ValueError("invalid plane count/header")
    sizes = struct.unpack_from(f"<{count}Q", body, 4)
    if 4 + 8 * count + sum(sizes) != len(body):
        raise ValueError("truncated or trailing plane payload")
    planes = _unpack(body)
    decoded = []
    for plane in planes:
        try:
            size = zstd.frame_content_size(plane)
        except zstd.ZstdError as exc:
            raise ValueError("invalid zstd plane") from exc
        if size != layout.tokens * layout.channels:
            raise ValueError("plane size does not match expected layout")
        try:
            decoded.append(zstd.ZstdDecompressor().decompress(
                plane, max_output_size=size, allow_extra_data=False))
        except zstd.ZstdError as exc:
            raise ValueError("cannot decode layout record plane") from exc
    restored = merge_untranspose(decoded, layout.tokens, layout.channels,
                                layout.bytes_per_element)
    if len(restored) != layout.raw_bytes:
        raise ValueError("decoded payload length mismatch")
    return restored


def make_compressor(base_marshaller, layout, marshaller=None):
    """Use the actual AIBrix BaseMarshaller without copying its chaining logic.

    Pass a BaseMarshaller class imported from AIBrix. The optional inner
    marshaller must emit exactly this layout's tensor bytes, without index heads.
    """
    class LayoutCompressor(base_marshaller):
        def _marshal(self, data):
            return pack(data, layout)

        def _unmarshal(self, data):
            return unpack(data, layout)

    return LayoutCompressor(marshaller)
