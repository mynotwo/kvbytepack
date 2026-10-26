# CPU context reuse and plane buffers

This follow-up compares the current implementation with **v0.2.1**, rather than
reusing the older layout-pass baseline. It reduces warm level-1 Zstd context
setup and avoids serializing contiguous byte planes a second time before native
compression. Method names, KVZ1/KBL1 records, compression settings and integrity
checks are unchanged. It adds no dependency.

## Corpus and protocol

The input is the same 96 original Qwen2.5-0.5B BF16 K/V payloads: all 24 layers,
at overlapping 128/512-token prefixes of one public CPython file. No recapture
or payload selection was performed. The NPZ SHA256 remains
`8596ac9a5b7475ee61b26c72847966ec846cb02ccee934868725db298143c211`.
The [capture and provenance](../results/aibrix_layout/model_capture) remain intact.

Baseline: `65fe9f83680cb8ddc80cb8186ecea86741e76b1f` (v0.2.1).
Measured candidate: `536b4ba5187db44003415ebc08cfd28b7b65c4e2`.
The manifest also pins the actual codec/tensor/adapter file hashes; subsequent
documentation commits do not change those implementations.

Apple M4 Pro, macOS arm64, Python 3.12.13, NumPy 2.5.4, Zstandard 0.25.0 /
libzstd 1.5.7. Separate sequential processes use one environment, the same
seeded case/path shuffle in each pair, and 12 rounds alternating AB/BA.
Each payload/path has five warmups and 41 timed encode/decode repetitions.
These are warm-context calls; first-call allocation is excluded by warmup.
Profiler and memory diagnostics run separately from the final timing comparison.

There are **9,216 measurement rows**, 4,608 before/after pairs and 377,856 timed
round trips. Every timed decode matches the original bytes; every pair matches
raw SHA256, encoded SHA256 and full stored size. This verifies byte preservation
on the fixed corpus, not whole-model output equivalence.

## Results against v0.2.1

Times are milliseconds. For each path/context/round, sum the 48 per-payload
medians, then report the median across 12 rounds. Speedups are medians of paired
round ratios and can differ from ratios of rounded table values.

| Tokens | Path | Encode before → after (ms) | Encode speedup | Decode before → after (ms) | Decode speedup |
|---|---|---:|---:|---:|---:|
| 128 | pipeline_codec | 2.875 → 2.632 | 1.093× | 1.650 → 1.618 | 1.017× |
| 128 | numpy_tensor | 3.083 → 2.858 | 1.079× | 1.886 → 1.856 | 1.020× |
| 128 | layout_record | 3.490 → 3.254 | 1.071× | 2.223 → 2.173 | 1.021× |
| 128 | raw_zstd1_control | 1.177 → 1.174 | 1.001× | 1.220 → 1.217 | 1.002× |
| 512 | pipeline_codec | 8.079 → 7.806 | 1.036× | 4.901 → 4.913 | 0.999× |
| 512 | numpy_tensor | 8.458 → 8.165 | 1.038× | 5.325 → 5.297 | 1.004× |
| 512 | layout_record | 9.956 → 9.655 | 1.032× | 6.681 → 6.653 | 1.005× |
| 512 | raw_zstd1_control | 4.181 → 4.182 | 1.000× | 4.676 → 4.679 | 1.000× |

Complete-record encoding time falls by **6.6% / 3.1%** at 128/512 tokens,
using `1 - 1 / paired_speedup`. Complete-record decode speedups are only
1.021× / 1.005×; both round ranges cross 1.0×, as do several other paths.
The 512-token core decode median is 0.999×. These results do not establish a
robust decode improvement or a win on every payload/round. The unchanged raw
Zstd control stays near 1.0×, with visible round-to-round variation.
All rounds, p95 values and negative observations are retained.

- [Measurements](../results/cpu_context_reuse/measurements.csv), [summary and round ranges](../results/cpu_context_reuse/summary.csv)
- [Timing samples](../results/cpu_context_reuse/samples.json), [source/environment manifest](../results/cpu_context_reuse/manifest.json)
- [Previous layout-pass study at v0.2.1](https://github.com/mynotwo/kvbytepack/blob/v0.2.1/docs/cpu-optimization.md)

## Implementation and tradeoff

Level-1 compressor/decompressor contexts are created lazily per Python thread.
They are never concurrently shared between threads; same-thread nested calls
use separate contexts, and busy flags reset even on errors. Level-9/19 context
behavior is unchanged. The strict example retains checksum, geometry, content
size and trailing-data checks and passes its options independently per call.
The [native API](https://python-zstandard.readthedocs.io/en/latest/compressor.html)
permits reuse but requires non-overlapping operations; its
[decoder contract](https://python-zstandard.readthedocs.io/en/latest/decompressor.html)
has the same ownership restriction.

The native level-1 pipeline receives contiguous memoryviews backed by a local
plane array. Public `transpose_split` still returns byte strings. Mutable inputs
and custom codec callbacks retain the ordinary byte-plane path. This avoids
intermediate plane serialization, not the layout permutation or output copy.

After a sweep of this corpus, one thread retains **501,049 bytes** in its
compressor and **95,968 bytes** in its decompressor, about **0.57 MiB** total,
as reported by the native contexts' `memory_size()`. Contexts live until thread
exit. This is a measured retained allocation, not a universal memory cap or a
memory-reduction claim; memory depends on inputs and active thread count.
See [native-memory diagnostic](../results/cpu_context_reuse/context-memory.json).

The separate 20-sweep full-record profile puts 63.3% of self-time in native
compression/decompression and SHA256. Even eliminating every other category
would give only a loose 1.58× upper bound with those costs fixed; layout and
output copies cannot actually be eliminated wholesale. The binding costs remain
compression, integrity work and required byte movement.
[Diagnostic profiles and driver](../results/cpu_context_reuse/profile-manifest.json)
are separate from the uninstrumented result table.

Encoded bytes and storage savings remain unchanged from v0.2.1. The record path
still costs more CPU than the raw level-1 control. Under the same serial,
non-overlapped single-transfer model, the median necessary bandwidth bounds are
approximately **0.049 / 0.086 GB/s** (decimal) for 128/512 tokens:
saved bytes divided by additional encode-plus-decode seconds. This is not a
measured serving speed or link speed. One small model, correlated layers and
overlapping prefixes do not establish hybrid/GPU generality or production gains.

## Reproduce without a GPU

From this source checkout/archive, use Python 3.12:

```bash
python -m pip install . numpy==2.5.4 zstandard==0.25.0
python bench/compare_cpu.py --check results/cpu_context_reuse
```

The check revalidates original bytes, wire hashes, source fingerprints, all
samples, coverage and aggregation. It does not rerun historical speed claims.
PyTorch, model weights and a GPU are unnecessary. To rerun timings:

```bash
git fetch origin 65fe9f83680cb8ddc80cb8186ecea86741e76b1f
git worktree add --detach ../kvbytepack-before 65fe9f83680cb8ddc80cb8186ecea86741e76b1f
python bench/compare_cpu.py --baseline-dir ../kvbytepack-before \
  --rounds 12 --repeats 41 --warmups 5 --out-dir /tmp/kvbytepack-contexts
git worktree remove ../kvbytepack-before
```

Use a quiet machine. CI verifies correctness and a short fresh paired run;
it does not impose a hardware-specific performance threshold. Regression tests
cover fresh-context wire compatibility, mixed input sizes, four concurrent
threads, same-thread reentry, strict/permissive option isolation, errors followed
by valid calls, odd widths, empty geometry and legacy import compatibility.
