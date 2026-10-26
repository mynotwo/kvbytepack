"""Wire compatibility, per-thread ownership and failure recovery for warm contexts."""
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest
import zstandard as zstd

import kvbytepack.codec as codec
from examples.aibrix_layout import adapter


@pytest.fixture(autouse=True)
def fresh_thread_state(monkeypatch):
    monkeypatch.setattr(codec, "_zstd_local", threading.local())


@pytest.mark.parametrize("size", [0, 1, 127, 32768, 131072, 17])
def test_level_one_matches_fresh_context_after_mixed_history(size):
    for n in (131072, 1, 32768, size):
        raw = np.random.default_rng(n).integers(0, 256, n, dtype=np.uint8).tobytes()
        expected = zstd.ZstdCompressor(level=1).compress(raw)
        assert codec.CODECS["zstd-1"][0](raw) == expected
        assert codec.CODECS["zstd-1"][1](expected) == raw
        assert zstd.ZstdDecompressor().decompress(codec.CODECS["zstd-1"][0](raw)) == raw


def test_contexts_constructed_once_per_thread(monkeypatch):
    compressor, decompressor = zstd.ZstdCompressor, zstd.ZstdDecompressor
    calls = []
    def make_compressor(*args, **kwargs):
        calls.append("compress")
        return compressor(*args, **kwargs)
    def make_decompressor(*args, **kwargs):
        calls.append("decode")
        return decompressor(*args, **kwargs)
    monkeypatch.setattr(zstd, "ZstdCompressor", make_compressor)
    monkeypatch.setattr(zstd, "ZstdDecompressor", make_decompressor)
    for raw in (b"", b"abc", bytes(range(256)) * 32):
        assert codec._zstd_decompress(codec._zstd1_compress(raw)) == raw
    assert calls == ["compress", "decode"]


def test_threads_own_distinct_native_contexts():
    barrier = threading.Barrier(4)
    def exercise(seed):
        raw = np.random.default_rng(seed).integers(0, 256, 16384, dtype=np.uint8).tobytes()
        expected = zstd.ZstdCompressor(level=1).compress(raw)
        for _ in range(20):
            assert codec._zstd1_compress(raw) == expected
            assert codec._zstd_decompress(expected) == raw
        barrier.wait(timeout=30)
        return codec._zstd_local.compressor, codec._zstd_local.decompressor
    with ThreadPoolExecutor(max_workers=4) as pool:
        contexts = list(pool.map(exercise, range(4)))
    assert len({id(c) for c, _ in contexts}) == 4
    assert len({id(d) for _, d in contexts}) == 4


def test_strict_decoder_errors_do_not_poison_next_call():
    raw = bytes(range(256)) * 32
    frame = zstd.ZstdCompressor(level=1).compress(raw)
    for broken in (b"not zstd", frame[:-1], frame + b"trailing"):
        with pytest.raises(zstd.ZstdError):
            codec._zstd_decompress(broken, allow_extra_data=False)
        assert codec._zstd_decompress(frame, allow_extra_data=False) == raw
        assert not codec._zstd_local.decompressing
    layout = adapter.Layout(32, 128, 2)
    assert adapter.unpack(adapter.pack(raw, layout), layout) == raw


def test_same_thread_nested_call_uses_separate_context():
    raw = b"same-thread reentry" * 128
    codec._zstd1_compress(raw)
    original = codec._zstd_local.compressor
    class NestedCompressor:
        def compress(self, data):
            return codec._zstd1_compress(data)
    codec._zstd_local.compressor = NestedCompressor()
    blob = codec._zstd1_compress(raw)
    assert blob == zstd.ZstdCompressor(level=1).compress(raw)
    assert not codec._zstd_local.compressing
    codec._zstd_local.compressor = original
    class NestedDecompressor:
        def decompress(self, data, **kwargs):
            return codec._zstd_decompress(data, **kwargs)
    codec._zstd_local.decompressor = NestedDecompressor()
    assert codec._zstd_decompress(blob, allow_extra_data=False) == raw
    assert not codec._zstd_local.decompressing


def test_bad_compressor_input_releases_busy_flag():
    with pytest.raises(TypeError):
        codec._zstd1_compress(None)
    assert not codec._zstd_local.compressing
    assert codec._zstd_decompress(codec._zstd1_compress(b"valid after error")) == b"valid after error"
