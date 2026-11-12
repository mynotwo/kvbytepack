"""KVB2: a self-describing, checksummed record for one fixed-width tensor.

Layout (little-endian)::

    "KVB2"
    u8  layout id        0 none, 1 planes, 2 channel-planes
    u8  codec id         see codecs.py
    i8  codec level
    u8  element width    bytes per element, 1..8
    u8  ndim
    u8  len(tag), tag    ascii dtype tag, e.g. "numpy:<f4", "torch:bfloat16", "bytes"
    u64 x ndim           shape
    u8  plane count
    (u8 mode, u32 len) x planes   mode 0 = stored verbatim, 1 = codec
    plane payloads
    u32 crc32            of every byte before it

A plane whose compressed form is not smaller is stored verbatim, so a record
never spends codec time on decode for incompressible planes (e.g. BF16 mantissa
bytes) and grows by at most the header for random data.
"""
import math
import struct
import zlib
from typing import NamedTuple, Tuple

from . import codecs, layout as _layout

MAGIC = b"KVB2"
_LAYOUT_IDS = {name: i for i, name in enumerate(_layout.LAYOUTS)}
_FIXED = struct.Struct("<4sBBbBBB")
_PLANE = struct.Struct("<BI")
_MAX_PLANE = 0xFFFFFFFF


class Header(NamedTuple):
    shape: Tuple[int, ...]
    width: int
    tag: str
    layout: str
    codec: str
    plane_modes: Tuple[int, ...]
    plane_sizes: Tuple[int, ...]
    header_bytes: int
    record_bytes: int

    @property
    def raw_bytes(self):
        return math.prod(self.shape) * self.width


def geometry(shape, width):
    """Rows x cols view used by the layouts: cols is the last (channel) axis."""
    if not shape:
        return 1, 1
    cols = shape[-1]
    rows = math.prod(shape[:-1]) if cols else 0
    return rows, cols


def encode(buf, shape, width, *, tag="bytes", layout="channel-planes", codec="zstd:1"):
    """Pack ``buf`` (C-order bytes of a tensor with ``shape`` and element ``width``)."""
    shape = tuple(int(s) for s in shape)
    if not 1 <= width <= 8:
        raise ValueError("element width must be 1..8 bytes")
    if len(shape) > 255 or any(s < 0 for s in shape):
        raise ValueError("invalid shape")
    tag_b = tag.encode("ascii")
    if len(tag_b) > 255:
        raise ValueError("dtype tag too long")
    c, level = codecs.parse(codec)
    rows, cols = geometry(shape, width)
    planes = _layout.split(buf, rows, cols, width, layout)

    modes, blobs = [], []
    for p in planes:
        z = c.compress(p, level) if c.id else p
        if c.id and len(z) < len(p):
            modes.append(1)
            blobs.append(z)
        else:
            modes.append(0)
            blobs.append(p)
        if len(blobs[-1]) > _MAX_PLANE:
            raise ValueError("plane exceeds 4 GiB; split the tensor")

    head = [_FIXED.pack(MAGIC, _LAYOUT_IDS[layout], c.id, level, width, len(shape), len(tag_b)),
            tag_b, struct.pack(f"<{len(shape)}Q", *shape), bytes([len(planes)])]
    head += [_PLANE.pack(m, len(b)) for m, b in zip(modes, blobs)]
    out = bytearray(b"".join(head))
    for b in blobs:
        out += b
    out += struct.pack("<I", zlib.crc32(out))
    return bytes(out)


def read_header(record):
    """Parse and validate the header without decompressing anything."""
    rec = memoryview(record)
    if len(rec) < _FIXED.size + 5 or bytes(rec[:4]) != MAGIC:
        raise ValueError("not a KVB2 record")
    _, lid, cid, level, width, ndim, tlen = _FIXED.unpack_from(rec, 0)
    off = _FIXED.size
    try:
        tag = bytes(rec[off:off + tlen]).decode("ascii")
        off += tlen
        shape = struct.unpack_from(f"<{ndim}Q", rec, off)
        off += 8 * ndim
        n = rec[off]
        off += 1
        modes, sizes = [], []
        for _ in range(n):
            m, s = _PLANE.unpack_from(rec, off)
            modes.append(m)
            sizes.append(s)
            off += _PLANE.size
    except (struct.error, IndexError, UnicodeDecodeError):
        raise ValueError("truncated KVB2 header") from None
    if lid >= len(_layout.LAYOUTS) or not 1 <= width <= 8 or any(m > 1 for m in modes):
        raise ValueError("invalid KVB2 header field")
    name = _layout.LAYOUTS[lid]
    if n != _layout.plane_count(name, width):
        raise ValueError("plane count does not match layout")
    if off + sum(sizes) + 4 != len(rec):
        raise ValueError("record length does not match header")
    c = codecs.by_id(cid)
    return Header(tuple(shape), width, tag, name, codecs.spec(c, level), tuple(modes),
                  tuple(sizes), off, len(rec))


def decode(record):
    """Return ``(header, raw_bytes)``. Raises ``ValueError`` on any corruption."""
    h = read_header(record)
    rec = memoryview(record)
    (crc,) = struct.unpack_from("<I", rec, len(rec) - 4)
    if zlib.crc32(rec[:-4]) != crc:
        raise ValueError("KVB2 checksum mismatch")
    c, _ = codecs.parse(h.codec)
    rows, cols = geometry(h.shape, h.width)
    n = len(h.plane_sizes)
    plane_raw = rows * cols * h.width // n
    planes, off = [], h.header_bytes
    for mode, size in zip(h.plane_modes, h.plane_sizes):
        chunk = rec[off:off + size]
        off += size
        if mode == 0:
            if size != plane_raw:
                raise ValueError("stored plane has wrong length")
            planes.append(chunk)
        else:
            planes.append(c.decompress(chunk, plane_raw))
    return h, _layout.merge(planes, rows, cols, h.width, h.layout)
