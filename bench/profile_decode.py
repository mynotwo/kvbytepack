"""Separate 512-token decode profile and per-thread native/tracked-memory checks."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import statistics
import sys
import threading
import tracemalloc


def measure(root):
    sys.path.insert(0, str(root))
    import numpy as np
    from kvbytepack import codec
    from examples.aibrix_layout.adapter import Layout, pack, unpack
    source = root / "results/aibrix_layout/model_capture/kv_payloads.npz"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != "8596ac9a5b7475ee61b26c72847966ec846cb02ccee934868725db298143c211":
        raise ValueError("wrong real KV payloads")
    with np.load(source, allow_pickle=False) as arrays:
        cases = [(name, arrays[name].tobytes(), Layout(*arrays[name].shape, 2))
                 for name in sorted(arrays.files)]
    records = [(name, pack(raw, layout), layout, raw) for name, raw, layout in cases]
    selected = [r for r in records if r[2].tokens == 512]
    assert len(cases) == 96 and len(selected) == 48

    def run():
        for _ in range(60):
            for _, record, layout, raw in selected:
                assert unpack(record, layout) == raw

    profiler = cProfile.Profile()
    profiler.runcall(run)
    stats = pstats.Stats(profiler)
    profile = {"scope": "all 48 512-token records, 60 decode-only sweeps",
               "total_seconds": stats.total_tt,
               "functions": [{"name": k[2], "module": Path(k[0]).name, "calls": v[1],
                              "self_seconds": v[2], "cumulative_seconds": v[3]}
                             for k, v in sorted(stats.stats.items(), key=lambda item: item[1][2],
                                                reverse=True)]}
    _, record, layout, raw = next(r for r in records if r[0] == "ctx512_V_23")
    peaks = {}
    for name, operation in (("encode", lambda: pack(raw, layout)),
                            ("decode", lambda: unpack(record, layout))):
        values = []
        for _ in range(7):
            tracemalloc.start()
            result = operation()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert (unpack(result, layout) if name == "encode" else result) == raw
            values.append(peak)
        peaks[name] = statistics.median(values)

    barrier = threading.Barrier(4)
    def thread_memory(index):
        for _, raw, geometry in cases:
            assert unpack(pack(raw, geometry), geometry) == raw
        result = {"thread": index,
                  "compressor_bytes": codec._zstd_local.compressor.memory_size(),
                  "decompressor_bytes": codec._zstd_local.decompressor.memory_size()}
        barrier.wait(timeout=30)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        retained = list(pool.map(thread_memory, range(4)))
    return {"payloads_sha256": digest, "backend": "c-8x8" if getattr(codec, "_native_merge", None) else "numpy",
            "profile": profile, "memory_case": "ctx512_V_23", "raw_bytes": len(raw),
            "tracked_peak_bytes": peaks, "retained_native_contexts": retained,
            "scope": "tracked peak above a warm process in one thread, median of 7 calls; native contexts separately measured in 4 simultaneously alive threads after the full corpus; excludes OS thread stacks and other untracked native allocations"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(measure(args.root.resolve()), indent=2) + "\n")


if __name__ == "__main__":
    main()
