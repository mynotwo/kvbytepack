import hashlib
import json
from pathlib import Path
import struct
import sys

import pytest

from examples.aibrix_layout.adapter import Layout, pack, unpack, make_compressor
from examples.aibrix_layout.benchmark import check_results, upstream_classes, measure, write_results


@pytest.mark.parametrize("shape", [(1, 1), (7, 5), (4, 16)])
@pytest.mark.parametrize("bpe", [1, 2, 4, 8])
def test_layout_roundtrip_including_arbitrary_float_bits(shape, bpe):
    layout = Layout(*shape, bpe)
    raw = bytes(i % 256 for i in range(layout.raw_bytes))
    assert unpack(pack(raw, layout), layout) == raw


@pytest.mark.parametrize("values", [(0, 2, 2), (-1, 2, 2), (2, 0, 2),
                                    (True, 2, 2), (2, 2, 0.5), (2, 2, 3)])
def test_invalid_layout_rejected(values):
    with pytest.raises(ValueError):
        Layout(*values)


def test_length_mismatch_and_index_header_rejected():
    layout = Layout(4, 8, 2)
    for raw in (b"", bytes(63), bytes(65), struct.pack("i", 0) + bytes(64), bytearray(64)):
        with pytest.raises(ValueError):
            pack(raw, layout)


def test_wrong_same_size_layout_rejected():
    blob = pack(bytes(64), Layout(4, 8, 2))
    with pytest.raises(ValueError, match="expected layout"):
        unpack(blob, Layout(8, 4, 2))


def test_bit_patterns_survive_without_numeric_conversion():
    import numpy as np
    bits = np.array([0, 0x8000, 1, 0x7C00, 0xFC00, 0x7E01, 0x7FFF, 0xFFFF], dtype="<u2")
    raw = bits.tobytes()
    layout = Layout(2, 4, 2)
    assert unpack(pack(raw, layout), layout) == raw


@pytest.mark.parametrize("damage", ["version", "checksum", "metadata", "body", "truncated", "trailing"])
def test_corrupted_records_rejected(damage):
    layout = Layout(4, 8, 2)
    record = bytearray(pack(bytes(64), layout))
    if damage == "truncated":
        record = record[:-1]
    elif damage == "trailing":
        record += b"extra"
    else:
        offset = {"version": 3, "checksum": 4, "metadata": 45, "body": -1}[damage]
        record[offset] ^= 1
    with pytest.raises(ValueError):
        unpack(bytes(record), layout)


def resign(inner):
    return b"KBL1" + hashlib.sha256(inner).digest() + inner


@pytest.mark.parametrize("damage", ["metadata_size", "invalid_json", "shape", "kind", "method",
                                    "plane_count", "plane_size", "trailing_payload", "frame_size", "extra_frame_data"])
def test_rechecks_structure_even_with_recomputed_checksum(damage):
    layout = Layout(4, 8, 2)
    inner = bytearray(pack(bytes(64), layout)[36:])
    n = struct.unpack_from("<I", inner, 4)[0]
    meta = json.loads(inner[8:8 + n])
    if damage in ("shape", "kind", "method"):
        key, value = {"shape": ("shape", [True, 32]), "kind": ("kind", "torch"),
                      "method": ("m", "pipeline_lzma")}[damage]
        meta[key] = value
        head = json.dumps(meta).encode()
        inner = bytearray(inner[:4] + struct.pack("<I", len(head)) + head + inner[8 + n:])
    elif damage == "metadata_size":
        struct.pack_into("<I", inner, 4, 2**32 - 1)
    elif damage == "invalid_json":
        inner[8] = ord("!")
    elif damage == "plane_count":
        struct.pack_into("<I", inner, 8 + n, 1000000)
    elif damage == "plane_size":
        struct.pack_into("<Q", inner, 12 + n, 2**63)
    elif damage == "trailing_payload":
        inner += b"tail"
    else:
        import zstandard as zstd
        first = zstd.ZstdCompressor(level=1).compress(bytes(31 if damage == "frame_size" else 32))
        if damage == "extra_frame_data":
            first += b"tail"
        second = zstd.ZstdCompressor(level=1).compress(bytes(32))
        inner = inner[:8 + n] + struct.pack("<IQQ", 2, len(first), len(second)) + first + second
    with pytest.raises(ValueError):
        unpack(resign(bytes(inner)), layout)


@pytest.mark.skipif(sys.version_info < (3, 10), reason="pinned AIBrix module uses Python 3.10 annotations")
def test_actual_upstream_chaining_and_raw_baseline():
    base, zstd_class, provenance = upstream_classes()
    assert provenance["revision"] == "d459142cee4a75a3a84a6e504ea1add5b490a87a"
    layout = Layout(4, 8, 2)
    raw = bytes(range(64))
    # This inner marshaller emits only the known payload, not TensorSerializer's head.
    class RawSerializer(base):
        def _marshal(self, obj):
            return obj["payload"]

        def _unmarshal(self, data):
            return {"payload": data}

    compressor = make_compressor(base, layout, RawSerializer())
    assert isinstance(compressor, base)
    assert compressor.unmarshal(compressor.marshal({"payload": raw})) == {"payload": raw}
    default = zstd_class()
    assert default.unmarshal(default.marshal(raw)) == raw


@pytest.mark.skipif(sys.version_info < (3, 10), reason="pinned AIBrix module requires Python 3.10")
def test_public_synthetic_evidence():
    check_results(Path(__file__).resolve().parents[1] / "results/aibrix_layout")


@pytest.mark.skipif(sys.version_info < (3, 10), reason="pinned AIBrix module requires Python 3.10")
def test_supplied_payload_preserves_provenance_and_complete_cost(tmp_path):
    raw = bytes(range(32))
    rows, samples, manifest = measure([("supplied_array", raw, Layout(2, 8, 2), "user_supplied_npy")], 2, 1)
    assert len(rows) == 4
    assert all(r["input_source"] == "user_supplied_npy" and r["byte_exact"] for r in rows)
    layout_row = next(r for r in rows if r["path"] == "layout_zstd1_record")
    assert layout_row["stored_bytes"] == len(pack(raw, Layout(2, 8, 2)))
    write_results(tmp_path, rows, samples, manifest)
    assert (tmp_path / "samples.json").exists()


@pytest.mark.skipif(sys.version_info < (3, 10), reason="pinned AIBrix module requires Python 3.10")
@pytest.mark.parametrize("damage", ["duplicate", "missing", "input_source", "dimensions", "timing"])
def test_evidence_corruption_rejected_even_with_updated_hashes(tmp_path, damage):
    import csv
    import io
    source = Path(__file__).resolve().parents[1] / "results/aibrix_layout"
    for path in source.iterdir():
        (tmp_path / path.name).write_bytes(path.read_bytes())
    with (tmp_path / "measurements.csv").open() as file:
        rows = list(csv.DictReader(file))
    if damage == "duplicate":
        rows[-1] = rows[0]
    elif damage == "missing":
        rows.pop()
    else:
        key, value = {"input_source": ("input_source", "model_captured"),
                      "dimensions": ("tokens", "999"),
                      "timing": ("decode_median_ms", "nan")}[damage]
        rows[0][key] = value
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    raw = stream.getvalue().encode()
    (tmp_path / "measurements.csv").write_bytes(raw)
    meta = json.loads((tmp_path / "manifest.json").read_text())
    meta["files"]["measurements.csv"] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        check_results(tmp_path)
