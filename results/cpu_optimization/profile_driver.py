"""Profile the full record path separately from the uninstrumented comparison."""
import cProfile
import hashlib
import json
from pathlib import Path
import pstats
import statistics
import sys
import tracemalloc

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))

import numpy as np  # noqa: E402
from examples.aibrix_layout.adapter import Layout, pack, unpack  # noqa: E402

source = root / "results/aibrix_layout/model_capture/kv_payloads.npz"
with np.load(source, allow_pickle=False) as arrays:
    names = sorted(arrays.files)
    cases = [(arrays[k].tobytes(), Layout(*arrays[k].shape, 2)) for k in names]


def run():
    for _ in range(20):
        for raw, layout in cases:
            assert unpack(pack(raw, layout), layout) == raw


profiler = cProfile.Profile()
profiler.runcall(run)
stats = pstats.Stats(profiler)
functions = []
for (file, line, name), (cc, nc, tt, ct, callers) in stats.stats.items():
    functions.append({"function": name, "module": Path(file).name, "calls": nc,
                      "self_seconds": tt, "cumulative_seconds": ct})
raw, layout = cases[-1]
blob = pack(raw, layout)
peaks = {}
for name, operation in (("encode", lambda: pack(raw, layout)),
                        ("decode", lambda: unpack(blob, layout))):
    values = []
    for _ in range(5):
        tracemalloc.start()
        result = operation()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        values.append(peak)
    peaks[name] = statistics.median(values)
print(json.dumps({"profile_iterations": 20, "profile_total_seconds": stats.total_tt,
                  "functions": sorted(functions, key=lambda r: r["self_seconds"], reverse=True),
                  "payloads_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "memory_case": names[-1], "tracked_memory_raw_bytes": len(raw),
                  "tracked_peak_bytes": peaks,
                  "memory_scope": "Python/NumPy tracemalloc; excludes untracked native allocations; separate from timing"},
                 indent=2))
