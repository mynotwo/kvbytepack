"""Measure layout x codec choices on real KV arrays and apply the cost model.

Sizes are complete KVB2 records (header, plane table and checksum included),
not bare payloads. With ``threads=1`` times are per-block medians summed over
blocks. With ``threads=N`` they are the median wall-clock time to encode (or
decode) every block on an N-thread pool; zstd and the C merge kernel release
the GIL, so this is the multi-core throughput a store could actually get.
"""
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import costmodel, record

DEFAULT_CONFIGS = (
    ("none", "zstd:1"),
    ("planes", "zstd:1"),
    ("channel-planes", "zstd:1"),
    ("channel-planes", "zstd:-1"),
    ("channel-planes", "zstd:3"),
)


def load_arrays(paths):
    """Load ``.npy``/``.npz`` files into an ordered ``{name: ndarray}``."""
    out = {}
    for p in map(Path, paths):
        if p.suffix == ".npz":
            with np.load(p, allow_pickle=False) as z:
                for k in z.files:
                    out[f"{p.stem}/{k}"] = z[k]
        elif p.suffix == ".npy":
            out[p.stem] = np.load(p, allow_pickle=False)
        else:
            raise ValueError(f"expected .npy or .npz, got {p}")
    if not out:
        raise ValueError("no arrays loaded")
    return out


def _median_seconds(fn, repeats, warmups=1):
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        samples.append(time.perf_counter_ns() - t0)
    return statistics.median(samples) / 1e9


def measure(arrays, configs=DEFAULT_CONFIGS, repeats=5, threads=1):
    """Return one aggregate :class:`costmodel.Measurement` per (layout, codec)."""
    blocks = []
    for arr in arrays.values():
        arr = arr if arr.flags.c_contiguous else arr.copy(order="C")
        blocks.append((arr.tobytes(), arr.shape, arr.dtype.itemsize))
    raw_total = sum(len(b[0]) for b in blocks)
    pool = ThreadPoolExecutor(threads) if threads > 1 else None
    results = []
    try:
        for layout, codec in configs:
            def enc_one(b, layout=layout, codec=codec):
                return record.encode(b[0], b[1], b[2], layout=layout, codec=codec)
            recs = [enc_one(b) for b in blocks]
            for b, rec in zip(blocks, recs):
                if record.decode(rec)[1] != b[0]:
                    raise AssertionError(f"round trip failed for {layout}/{codec}")
            if pool is None:
                enc = sum(_median_seconds(lambda b=b: enc_one(b), repeats) for b in blocks)
                dec = sum(_median_seconds(lambda r=r: record.decode(r), repeats) for r in recs)
            else:
                enc = _median_seconds(lambda: list(pool.map(enc_one, blocks)), repeats)
                dec = _median_seconds(lambda: list(pool.map(record.decode, recs)), repeats)
            results.append(costmodel.Measurement(
                f"{layout} {codec}", raw_total, sum(map(len, recs)), enc, dec))
    finally:
        if pool is not None:
            pool.shutdown()
    return results


def _gb(x):
    return "inf" if math.isinf(x) else f"{x / 1e9:.3g}"


def report(results, bandwidths_gbs=(1, 10, 25, 100)):
    """Render a plain-text table: sizes, speeds, and the serial/pipelined verdicts."""
    raw = results[0].raw_bytes
    base = costmodel.uncompressed(raw)
    results = [base] + [m for m in results if m.name != base.name]
    lines = [f"{raw:,} raw bytes; GB/s are decimal and in raw (uncompressed) bytes.",
             "",
             f"{'config':<24}{'ratio':>7}{'enc GB/s':>10}{'dec GB/s':>10}"
             f"{'serial wins if link <':>23}{'pipelined ceiling':>19}"]
    for m in results[1:]:
        c = costmodel.crossover(m, base)
        verdict = {"always": "always", "never": "never"}.get(c.wins, None) or (
            f"{c.wins} {_gb(c.bandwidth)} GB/s")
        lines.append(f"{m.name:<24}{m.ratio:>7.3f}{_gb(raw / m.encode_s):>10}"
                     f"{_gb(raw / m.decode_s):>10}{verdict:>23}"
                     f"{_gb(costmodel.pipelined_ceiling(m)) + ' GB/s':>19}")
    lines += ["", "Pipelined goodput at link speed (GB/s of raw bytes delivered):",
              f"{'config':<24}" + "".join(f"{f'{b} GB/s':>11}" for b in bandwidths_gbs)]
    for m in results:
        lines.append(f"{m.name:<24}" + "".join(
            f"{_gb(costmodel.pipelined_goodput(m, b * 1e9)):>11}" for b in bandwidths_gbs))
    return "\n".join(lines)
