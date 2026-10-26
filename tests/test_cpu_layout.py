from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import numpy as np
import pytest

from kvbytepack.codec import (CODECS, _pack, _unpack, available_methods,
                             byte_merge, byte_split, decode, encode,
                             merge_untranspose, transpose_bytes, transpose_split,
                             untranspose_bytes)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (1, 1), (7, 5), (129, 17)])
@pytest.mark.parametrize("width", [1, 2, 3, 4, 8])
def test_fused_planes_match_legacy_order_and_independent_indices(shape, width):
    s, d = shape
    raw = np.random.default_rng(11).integers(0, 256, s * d * width, dtype=np.uint8).tobytes()
    expected = [bytes(raw[(token * d + channel) * width + byte]
                      for channel in range(d) for token in range(s)) for byte in range(width)]
    assert transpose_split(raw, s, d, width) == expected
    assert expected == byte_split(transpose_bytes(raw, s, d, width), width)
    assert merge_untranspose(expected, s, d, width) == raw


@pytest.mark.parametrize("method", [m for m in available_methods() if m.startswith("pipeline_")])
@pytest.mark.parametrize("width", [1, 2, 3, 4, 8])
def test_old_and_new_pipeline_blobs_are_interchangeable(method, width):
    s, d = 7, 5
    raw = np.random.default_rng(19).integers(0, 256, s * d * width, dtype=np.uint8).tobytes()
    comp, dec = CODECS[method[len("pipeline_"):]]
    old_blob = _pack([comp(p) for p in byte_split(transpose_bytes(raw, s, d, width), width)])
    new_blob = encode(raw, s, d, width, method)
    assert old_blob == new_blob
    assert decode(old_blob, s, d, width, method) == raw
    # The unmodified inverse path can still read new records.
    assert untranspose_bytes(byte_merge([dec(p) for p in _unpack(new_blob)]), s, d, width) == raw


@pytest.mark.parametrize("bpe", [0.25, 0.5])
def test_packed_sub_byte_behavior_unchanged(bpe):
    raw = bytes(range(64))
    assert transpose_split(raw, 8, 8, bpe) == [raw]
    assert merge_untranspose([raw], 8, 8, bpe) == raw


@pytest.mark.parametrize("planes", [[], [bytes(15)], [bytes(15)] * 3, [bytes(15), bytes(14)]])
def test_bad_plane_geometry_never_returns_partially_initialized_bytes(planes):
    with pytest.raises(ValueError):
        merge_untranspose(planes, 3, 5, 2)


def test_concurrent_calls_have_no_shared_scratch_buffer():
    def roundtrip(seed):
        width = (1, 2, 3, 4, 8)[seed % 5]
        raw = np.random.default_rng(seed).integers(0, 256, 31 * 17 * width, dtype=np.uint8).tobytes()
        return decode(encode(raw, 31, 17, width, "pipeline_zstd-1"),
                      31, 17, width, "pipeline_zstd-1") == raw
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(roundtrip, range(40)))


def test_paired_summary_uses_full_context_and_rounds():
    from bench.compare_cpu import PATHS, summarize
    rows = []
    for round_id in range(2):
        for tokens in (128, 512):
            for path in PATHS:
                for payload in range(48):
                    for variant, factor in (("before", 1), ("after", .5)):
                        rows.append({"round": round_id, "tokens": tokens, "path": path,
                                     "variant": variant, "encode_median_ms": factor,
                                     "decode_median_ms": 2 * factor})
    summary = summarize(rows)
    assert len(summary) == 16
    for row in summary:
        assert row["rounds"] == 2 and row["payloads"] == 48
        assert row["paired_speedup_median"] == row["paired_speedup_min"] == row["paired_speedup_max"] == 2
        assert row["before_sum_medians_ms"] == (48 if row["phase"] == "encode" else 96)
        assert row["after_sum_medians_ms"] == row["before_sum_medians_ms"] / 2


@pytest.mark.skipif(sys.version_info < (3, 10), reason="pinned AIBrix module requires Python 3.10")
def test_published_cpu_comparison_matches_actual_payloads_and_samples():
    from bench.compare_cpu import check_results
    root = Path(__file__).resolve().parents[1]
    check_results(root / "results/cpu_optimization", root / "results/aibrix_layout/model_capture")


@pytest.mark.skipif(sys.version_info < (3, 10), reason="pinned AIBrix module requires Python 3.10")
@pytest.mark.parametrize("damage", ["missing_row", "wrong_wire_hash", "invalid_sample"])
def test_invalid_comparisons_rejected_even_with_recomputed_file_hash(tmp_path, damage):
    import csv
    import hashlib
    import io
    import json
    from bench.compare_cpu import check_results
    root = Path(__file__).resolve().parents[1]
    source = root / "results/cpu_optimization"
    for name in ("measurements.csv", "summary.csv", "samples.json", "manifest.json"):
        (tmp_path / name).write_bytes((source / name).read_bytes())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    if damage == "invalid_sample":
        path = tmp_path / "samples.json"
        samples = json.loads(path.read_text())
        samples[0]["encode_ms"][0] = float("nan")
        path.write_text(json.dumps(samples))
    else:
        path = tmp_path / "measurements.csv"
        with path.open() as file:
            rows = list(csv.DictReader(file))
        if damage == "missing_row":
            rows.pop()
        else:
            rows[0]["encoded_sha256"] = "0" * 64
        text = io.StringIO(newline="")
        writer = csv.DictWriter(text, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        path.write_text(text.getvalue())
    manifest["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        check_results(tmp_path, root / "results/aibrix_layout/model_capture")
