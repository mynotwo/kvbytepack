# Reducing 512-token decode layout cost

This study compares the optional compiled decode path with the preceding
context-reuse revision, **`98f56cc16fe46195d70730e99ac64b754d5f75ad`**.
That baseline already includes both earlier CPU optimizations; it is not the
original v0.2.1 release. The measured candidate is
`c6b4d5e` (full commit and source/binary hashes are in the manifest).

## Fixed data and measurement

The same 96 original Qwen2.5-0.5B BF16 K/V payloads cover all 24 layers at
overlapping 128/512-token prefixes of one public CPython file. No recapture or
case selection was performed. The [original capture](../results/aibrix_layout/model_capture)
NPZ SHA256 remains `8596ac9a5b7475ee61b26c72847966ec846cb02ccee934868725db298143c211`.

Apple M4 Pro, macOS arm64, Python 3.12.13, NumPy 2.5.4, Zstandard 0.25.0 /
libzstd 1.5.7; the optional kernel was built with Apple Clang 17.0.0.
One environment runs isolated, sequential baseline/candidate processes. Twelve
rounds alternate AB/BA; each paired process uses the same seeded case/path order,
five warmups and 41 timed repetitions per payload/path. Diagnostics run separately.

There are 9,216 measurement rows and 377,856 timed round trips. Every timed decode
matches the original bytes; all 4,608 before/after pairs retain identical input
hashes, encoded hashes and complete stored sizes. The baseline uses NumPy layout
reconstruction and the candidate uses the compiled kernel. The manifest records
the loaded binary hash. Later driver edits only strengthen backend provenance
checks outside the unchanged timed worker.

## Results

Times are milliseconds: sum 48 per-payload medians for each context/path/round,
then take the median across 12 rounds. Speedups are medians of paired round ratios.
These are warm CPU calls, not batched serving latency.

| Tokens | Path | Encode before → after (ms) | Decode before → after (ms) | Paired decode speedup |
|---|---|---:|---:|---:|
| 128 | pipeline_codec | 2.663 → 2.671 | 1.637 → 1.312 | 1.247× |
| 128 | numpy_tensor | 2.887 → 2.892 | 1.878 → 1.545 | 1.216× |
| 128 | layout_record | 3.301 → 3.304 | 2.195 → 1.864 | 1.177× |
| 128 | raw_zstd1_control | 1.185 → 1.187 | 1.227 → 1.229 | 0.999× |
| 512 | pipeline_codec | 7.850 → 7.863 | 4.943 → 3.789 | 1.305× |
| 512 | numpy_tensor | 8.235 → 8.230 | 5.352 → 4.219 | 1.268× |
| 512 | layout_record | 9.788 → 9.774 | 6.735 → 5.580 | 1.205× |
| 512 | raw_zstd1_control | 4.222 → 4.232 | 4.719 → 4.727 | 0.999× |

At **512 tokens**, core decode improves **1.305×** and complete-record decode
**1.205×**, a **17.0% time reduction** using `1 - 1 / paired_speedup`.
Complete-record round speedups range from 1.200× to 1.213×; all twelve are above
1.0×. At 128 tokens, complete-record decode improves 1.177×, reducing time by
15.0%. Encoding is unchanged in implementation and shows only small fluctuations;
the unchanged raw Zstd control stays near 1.0×. No round or path is removed.

- [All measurements](../results/cpu_decode/measurements.csv), [summary and round ranges](../results/cpu_decode/summary.csv)
- [Timing samples](../results/cpu_decode/samples.json), [source/binary hashes and environments](../results/cpu_decode/manifest.json)

## Why it helps, and the remaining limit

The baseline writes each decoded byte plane into a strided NumPy output array,
then copies that array to the returned byte string. In a separate decode-only
profile of all 48 512-token records, reconstruction accounts for 30.4% of total
profiled time. Even making that entire phase free would give only a loose 1.44×
upper bound with the rest fixed.

The optional C kernel transposes 8×8 blocks of bytes with unsigned 64-bit
butterfly swaps, interleaves two planes and writes directly to the private
returned byte string. Partial tiles handle arbitrary dimensions. Byte loads and
stores explicitly define lane order; values are never interpreted as BF16 or
converted numerically. It uses no architecture-specific intrinsics, persistent
scratch or new runtime dependency. Other element widths keep the NumPy path.

Zstd calls, framing, metadata, SHA256, content-size and trailing-data checks are
unchanged. The new profile puts 63.4% of self-time in Zstd decompression and
SHA256. Each record still requires two separate Zstd frames plus integrity and
metadata work. The complete record remains more CPU-expensive than the raw
Zstd control. These data do not establish serving speedup, GPU/hybrid generality
or whole-model output equivalence.

[Before](../results/cpu_decode/diagnostic-before.json) /
[after](../results/cpu_decode/diagnostic-after.json) diagnostic traces and
[their manifest](../results/cpu_decode/diagnostic-manifest.json) are separate
from uninstrumented timing samples.

## Per-thread memory

Four simultaneously alive threads each process the full frozen corpus. Before
and after, every thread reports 501,049 compressor bytes plus 95,968 decompressor
bytes: **597,017 bytes, about 0.57 MiB, per thread**. The four-thread total is
2,388,068 native-context bytes. The new kernel retains no per-thread heap buffer;
its two temporary 8-word arrays hold 128 bytes on the C call stack.

For the same 131,072-byte `ctx512_V_23` record, the median additional tracked
decode peak over seven warmed calls falls from **667,966 to 536,834 bytes**,
about **128 KiB / 19.6% lower**. Encoding peaks remain 578,405 bytes. This tracks
Python allocations, including the returned byte string, above a warmed process;
it excludes pre-existing inputs, native Zstd workspace, OS thread stacks and
other untracked allocations. The context counts are measured separately, not
added to a claim about total RSS. See the [memory comparison](../results/cpu_decode/memory.json).

## Install and reproduce without a GPU

The kernel is optional. An ordinary `python -m pip install .` builds it when a
C compiler and Python headers are available. If compilation fails, installation
retains the original NumPy decoder. Wheels containing the kernel are specific
to their Python/platform; the package is no longer necessarily a pure-Python
wheel. No PyPI or new release publication is implied by this source change.

For timings from a source checkout, editable installation places the compiled
kernel alongside the source imported by isolated benchmark workers:

```bash
python -m pip install -e . numpy==2.5.4 zstandard==0.25.0
python -c "from kvbytepack.codec import _native_merge; assert _native_merge is not None"
python bench/compare_cpu.py --check results/cpu_decode
git fetch origin 98f56cc16fe46195d70730e99ac64b754d5f75ad
git worktree add --detach ../kvbytepack-before 98f56cc16fe46195d70730e99ac64b754d5f75ad
python bench/compare_cpu.py --baseline-dir ../kvbytepack-before \
  --rounds 12 --repeats 41 --warmups 5 --out-dir /tmp/kvbytepack-decode
git worktree remove ../kvbytepack-before
```

The offline `--check` also works with the NumPy fallback and prints the current
verification backend; it checks bytes, samples and aggregation, not historical
speed. Fresh timing runs require the compiled candidate. No PyTorch, weights,
GPU, network model download or hardware-specific speed threshold is required.

Memory and profile diagnostics can be rerun separately:

```bash
python bench/profile_decode.py --root . --out /tmp/kvbytepack-decode-diagnostic.json
```

Tests cover an independent byte-index oracle, 1,156 small geometries, tile tails,
full-size arbitrary bits, malformed lengths/types, empty and overflowing shapes,
mutable-plane fallback and parallel calls. Clang static analysis and a separate
UBSan build pass 500 random geometries and overflow checks. Both compiled-wheel
and deliberately failed-compiler fallback installations are checked separately.
