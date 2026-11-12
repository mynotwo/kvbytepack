import struct
import zlib

import numpy as np
import pytest

from kvbytepack import codecs, layout, record


def _rand(n, seed=0):
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


@pytest.mark.parametrize("lay", layout.LAYOUTS)
@pytest.mark.parametrize("width", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("shape", [(7, 5), (1, 1), (3, 1, 4), (0, 6), (5, 0), ()])
def test_layout_is_a_permutation(lay, width, shape):
    rows, cols = record.geometry(shape, width)
    raw = _rand(rows * cols * width)
    planes = layout.split(raw, rows, cols, width, lay)
    assert len(planes) == layout.plane_count(lay, width)
    assert sorted(b"".join(bytes(p) for p in planes)) == sorted(raw)
    assert layout.merge(planes, rows, cols, width, lay) == raw


def test_channel_planes_order():
    # 3 rows x 2 cols of 2-byte elements; element (r, c) = bytes [2k, 2k+1], k = 2r + c
    planes = layout.split(bytes(range(12)), 3, 2, 2, "channel-planes")
    assert bytes(planes[0]) == bytes([0, 4, 8, 2, 6, 10])   # low bytes, channel 0 then 1
    assert bytes(planes[1]) == bytes([1, 5, 9, 3, 7, 11])


def test_native_and_numpy_merge_agree(monkeypatch):
    rows, cols = 37, 29  # not multiples of the 8x8 tile
    raw = _rand(rows * cols * 2, 3)
    planes = [bytes(p) for p in layout.split(raw, rows, cols, 2, "channel-planes")]
    monkeypatch.setattr(layout, "_native_merge_2byte", None)
    assert layout.merge(planes, rows, cols, 2, "channel-planes") == raw


@pytest.mark.parametrize("codec", codecs.available())
def test_record_round_trip(codec):
    raw = _rand(64 * 48 * 2, 1)
    rec = record.encode(raw, (64, 48), 2, codec=codec)
    h, out = record.decode(rec)
    assert out == raw and h.shape == (64, 48) and h.width == 2 and h.tag == "bytes"
    assert h.record_bytes == len(rec)


def test_incompressible_planes_are_stored_and_overhead_is_bounded():
    raw = _rand(4096 * 2, 2)
    h = record.read_header(record.encode(raw, (64, 64), 2))
    assert h.plane_modes == (0, 0)
    assert h.record_bytes - len(raw) == h.header_bytes + 4 < 64


def test_codec_spec_and_level_are_recorded():
    raw = bytes(4096)
    assert record.read_header(record.encode(raw, (64, 32), 2, codec="zstd:-3")).codec == "zstd:-3"
    assert record.read_header(record.encode(raw, (64, 32), 2, codec="zstd")).codec == "zstd:1"
    assert record.read_header(record.encode(raw, (64, 32), 2, codec="store")).codec == "store"


@pytest.mark.parametrize("bad", ["nope", "store:3", "zstd:x", "zstd:500"])
def test_bad_codec_spec(bad):
    with pytest.raises(ValueError):
        record.encode(bytes(4), (2,), 2, codec=bad)


def test_geometry_mismatch_rejected():
    with pytest.raises(ValueError):
        record.encode(bytes(10), (2, 3), 2)


def test_every_single_byte_flip_is_detected():
    rec = bytearray(record.encode(bytes(range(256)) * 4, (32, 16), 2))
    for i in range(len(rec)):
        bad = bytearray(rec)
        bad[i] ^= 0x01
        with pytest.raises(ValueError):
            record.decode(bytes(bad))


def test_truncation_and_extension_rejected():
    rec = record.encode(bytes(range(256)) * 4, (32, 16), 2)
    for cut in (0, 3, 10, len(rec) - 1):
        with pytest.raises(ValueError):
            record.decode(rec[:cut])
    with pytest.raises(ValueError):
        record.decode(rec + b"\0")


def test_lying_header_with_valid_crc_rejected():
    # A well-formed record whose codec plane decodes to the wrong length.
    rec = bytearray(record.encode(bytes(1024), (32, 16), 2))
    h = record.read_header(bytes(rec))
    shape_off = 10 + len(h.tag)
    struct.pack_into("<Q", rec, shape_off, 31)   # claim 31 rows: planes are now oversized
    body = rec[:-4]
    with pytest.raises(ValueError):
        record.decode(bytes(body + struct.pack("<I", zlib.crc32(body))))
