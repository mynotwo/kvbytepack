# CPU layout-pass optimization

This is the historical v0.2.1 study. Run its reproduction commands at tag
`v0.2.1`; the [current follow-up](cpu-context-reuse.md) uses a new baseline and
a separate result directory. Historical measurements below remain unchanged.

Version 0.2.1 fuses channel transposition with byte splitting during encoding,
and interleaves decoded planes directly into token-major output during decoding.
It reuses the existing codecs, method names, KVZ1/KBL1 formats and checksums.
There is no shared scratch buffer or new dependency. The strict AIBrix example
uses the same fused inverse after its existing integrity/framing checks.

## Fixed data and paired protocol

The input is the **same 96 published BF16 KV payloads** from Qwen2.5-0.5B:
24 layers × K/V × overlapping 128/512-token prefixes of one public source file.
There was no new model capture in this optimization experiment. Raw data:
[NPZ](../results/aibrix_layout/model_capture/kv_payloads.npz), SHA256
`8596ac9a5b7475ee61b26c72847966ec846cb02ccee934868725db298143c211`.

The baseline is commit `1d8a4764681e1449cc5e4d2a6681d51a2f2eddd0`
(`aibrix-layout-v1`). The measured implementation commit is
`608dca4285bc57f0992f6988e015105c0fa2d7d8`; file hashes in the manifest bind
the measurements to the codec, tensor and adapter implementations.

Measurements use Apple M4 Pro, macOS arm64, Python 3.12.13, NumPy 2.5.4,
Zstandard 0.25.0 / libzstd 1.5.7. One shared environment runs separate, sequential
baseline/candidate processes. Six rounds alternate before/after and after/before;
each round uses the same seeded payload/path order in both processes. Each
payload/path has three warmups and 21 timed encode/decode repetitions. No model
inference, profiler, tracemalloc, input hashing or equality comparison is inside
the timed region; record SHA256 validation remains part of the record path.

Each timed decode is compared with the original bytes. Every before/after pair
must match input SHA256, encoded SHA256 and full stored size. The experiment
has **4,608 measurement rows**, 2,304 before/after pairs and complete 96-payload
coverage in each process. No layer is selected for reporting.

## Results

Times below are milliseconds: for each context/path/round, sum the medians for
all 48 K/V payloads, then take the median across six rounds. Speedups are the
median of paired per-round before/after ratios. They need not equal a ratio of
the rounded table times. These are sums of individual calls, not a measured
batched serving latency or end-to-end throughput.

| Tokens | Path | Encode before → after (ms) | Paired encode speedup | Decode before → after (ms) | Paired decode speedup |
|---|---|---:|---:|---:|---:|
| 128 | pipeline_codec | 4.637 → 2.874 | 1.61× | 3.471 → 1.644 | 2.11× |
| 128 | numpy_tensor | 4.862 → 3.091 | 1.57× | 3.733 → 1.886 | 1.98× |
| 128 | layout_record | 5.268 → 3.500 | 1.51× | 4.045 → 2.218 | 1.82× |
| 128 | raw_zstd1_control | 1.180 → 1.177 | 1.00× | 1.222 → 1.218 | 1.00× |
| 512 | pipeline_codec | 14.957 → 8.111 | 1.85× | 11.951 → 4.911 | 2.44× |
| 512 | numpy_tensor | 15.304 → 8.490 | 1.81× | 12.345 → 5.309 | 2.33× |
| 512 | layout_record | 16.850 → 9.956 | 1.69× | 13.729 → 6.676 | 2.06× |
| 512 | raw_zstd1_control | 4.193 → 4.184 | 1.00× | 4.693 → 4.680 | 1.00× |

`pipeline_codec` is the low-level pipeline zstd-1 API; `numpy_tensor` includes
the tensor envelope; `layout_record` includes the complete versioned record,
checksum and strict decoder checks. The raw Zstd level-1 control is unchanged.
All 96 payloads improve in both phases on all three layout paths when comparing
per-payload median paired speedups. The full per-round range and all p95 values
are retained in the CSVs; there is no universal timing threshold in CI.

**Encoded bytes and complete stored sizes are unchanged** in every pair.
The earlier 12.07%/13.12% storage reductions versus raw level-1 Zstd therefore
remain the same; this change reduces CPU cost rather than changing compression.

- [All measurements](../results/cpu_optimization/measurements.csv)
- [Paired summary, including min/max across rounds](../results/cpu_optimization/summary.csv)
- [All timing samples](../results/cpu_optimization/samples.json)
- [Source hashes, environment and protocol](../results/cpu_optimization/manifest.json)

## Bottleneck and remaining limit

The old path first materialized `[D,S,bpe]`, then serialized each byte plane.
The old inverse stacked/interleaved the planes before a separate transpose.
The fused encoder constructs each channel-major plane directly; the fused
inverse writes each plane into its final position in a local output array.
It checks plane count and dimensions before returning any result.

Separate diagnostic profiles of 20 sweeps over all 96 full records put 50.6% of baseline self-time in `tobytes` and `stack`. Even eliminating both categories entirely would give a loose Amdahl upper bound of 2.02× for that trace with everything else fixed. The optimization still has to copy output bytes.

After fusion, Zstd compression/decompression and SHA256 account for 63.0% of profiled self-time. The remaining constraint is no longer just the extra layout passes. These instrumented traces explain the bottleneck; the result table uses uninstrumented measurements.

For the same 131,072-byte payload (`ctx512_V_23`), tracemalloc peaks are encode 577,270 → 577,391 bytes and decode 668,231 → 667,966 bytes. This is essentially unchanged tracked memory, not a zero-copy or total-memory reduction claim; untracked native allocations are excluded.

Diagnostic [before](../results/cpu_optimization/profile-before.json) /
[after](../results/cpu_optimization/profile-after.json) records and their
[driver](../results/cpu_optimization/profile_driver.py) are separate from timing samples.

Under a non-overlapped, single-transfer cost model, byte savings must outweigh
additional encode+decode time versus the raw Zstd control. Using each round's
summed per-payload medians, then taking the median bound:

- 128 tokens: bandwidth below approximately 0.045 GB/s (decimal) is necessary for this CPU record path to win under that model.
- 512 tokens: bandwidth below approximately 0.083 GB/s (decimal) is necessary for this CPU record path to win under that model.

This is not a measured link speed. The optimized record path still costs more
CPU time than raw Zstd. One small model and one overlapping-prefix source do
not establish 7B/hybrid/GPU generality, production capacity or serving speedup.

## Reproduce

From a source checkout or source release ZIP, Python 3.12, with the measured
NumPy/Zstd versions for exact compressed-byte checks:

```bash
python -m pip install . numpy==2.5.4 zstandard==0.25.0
python bench/compare_cpu.py --check results/cpu_optimization
```

This checks original bytes, encoded hashes, full coverage, timing/sample
consistency and paired aggregation. It needs neither PyTorch nor model weights,
and does not pretend to rerun historical timings. It also checks that the
current codec/adapter files match the measured implementation.

To rerun timings, prepare the pinned baseline as a Git worktree, sharing the
same environment. The driver does not modify either checkout:

```bash
git fetch origin 1d8a4764681e1449cc5e4d2a6681d51a2f2eddd0
git worktree add --detach ../kvbytepack-before 1d8a4764681e1449cc5e4d2a6681d51a2f2eddd0
python bench/compare_cpu.py --baseline-dir ../kvbytepack-before \
  --rounds 6 --repeats 21 --warmups 3 --out-dir /tmp/kvbytepack-cpu
git worktree remove ../kvbytepack-before
```

Use a quiet machine; repeated warm inputs and machine-specific CPU/cache state
limit how far these numbers transfer to other workloads. The unchanged Zstd
control and balanced order help detect drift; they do not guarantee isolation.
CI performs correctness/wire checks and a short fresh comparison, without a
hardware-specific speed gate. Cross-version tests also exercise old/new decoding,
odd element widths, empty dimensions, sub-byte formats and concurrent calls.
