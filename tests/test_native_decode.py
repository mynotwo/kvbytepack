"""Independent byte-index oracles and safety boundaries for the optional kernel."""
from concurrent.futures import ThreadPoolExecutor
import sys

import numpy as np
import pytest

from kvbytepack import codec


def native():
    if codec._native_merge is None:
        pytest.skip("optional compiled kernel not built")
    return codec._native_merge


def planes(raw, s, d):
    return [bytes(raw[(t * d + c) * 2 + i] for c in range(d) for t in range(s))
            for i in range(2)]


def test_all_small_shapes_and_tile_edges_match_original_bytes():
    merge = native()
    for s in range(34):
        for d in range(34):
            raw = np.random.default_rng(s * 34 + d).integers(0, 256, s * d * 2,
                                                          dtype=np.uint8).tobytes()
            ps = planes(raw, s, d)
            assert merge(*ps, s, d) == raw
            assert codec.merge_untranspose(ps, s, d, 2) == raw


@pytest.mark.parametrize("s,d", [(7, 129), (8, 128), (9, 127), (31, 65), (33, 63),
                                 (127, 131), (128, 128), (129, 125), (511, 127),
                                 (512, 128), (513, 129)])
def test_odd_and_full_size_geometry_with_arbitrary_bits(s, d):
    raw = np.random.default_rng(s * d).integers(0, 256, s * d * 2, dtype=np.uint8).tobytes()
    assert native()(*planes(raw, s, d), s, d) == raw


@pytest.mark.parametrize("shape", [(-1, 8), (8, -1)])
def test_negative_dimensions_rejected(shape):
    with pytest.raises(ValueError):
        native()(b"", b"", *shape)


@pytest.mark.parametrize("shape", [(sys.maxsize, 2), (2, sys.maxsize), (sys.maxsize, 1)])
def test_overflow_rejected_before_allocation(shape):
    with pytest.raises(OverflowError):
        native()(b"", b"", *shape)


@pytest.mark.parametrize("shape", [(0, sys.maxsize), (sys.maxsize, 0), (0, 0)])
def test_empty_extreme_shape_does_not_loop(shape):
    assert native()(b"", b"", *shape) == b""


@pytest.mark.parametrize("low,high", [(b"", b"1234"), (b"1234", b""),
                                      (b"123", b"1234"), (b"12345", b"1234")])
def test_malformed_plane_lengths_rejected(low, high):
    with pytest.raises(ValueError):
        native()(low, high, 2, 2)


@pytest.mark.parametrize("bad", [bytearray(b"1234"), memoryview(b"1234"), None, "1234"])
def test_direct_native_call_rejects_non_bytes(bad):
    with pytest.raises(TypeError):
        native()(bad, b"1234", 2, 2)


def test_numpy_integer_dimensions():
    raw = bytes(range(32))
    assert native()(*planes(raw, 4, 4), np.int64(4), np.int64(4)) == raw


def test_mutable_planes_keep_numpy_path(monkeypatch):
    def forbidden(*args):
        raise AssertionError("mutable planes entered native path")
    raw = bytes(range(32))
    monkeypatch.setattr(codec, "_native_merge", forbidden)
    ps = planes(raw, 4, 4)
    assert codec.merge_untranspose([bytearray(p) for p in ps], 4, 4, 2) == raw
    assert codec.merge_untranspose([memoryview(p) for p in ps], 4, 4, 2) == raw


def test_numpy_fallback_matches_same_public_records(monkeypatch):
    from examples.aibrix_layout import adapter
    raw = bytes(range(256)) * 512
    layout = adapter.Layout(512, 128, 2)
    blob = adapter.pack(raw, layout)
    expected = adapter.unpack(blob, layout)
    monkeypatch.setattr(codec, "_native_merge", None)
    assert adapter.unpack(blob, layout) == expected == raw
    assert adapter.pack(raw, layout) == blob


def test_parallel_native_calls_have_independent_output():
    merge = native()
    def exercise(seed):
        rng = np.random.default_rng(seed)
        for _ in range(20):
            s, d = (int(v) for v in rng.integers(1, 259, 2))
            raw = rng.integers(0, 256, s * d * 2, dtype=np.uint8).tobytes()
            assert merge(*planes(raw, s, d), s, d) == raw
        return True
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(exercise, range(4)))
