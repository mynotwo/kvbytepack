# LMCache GPU byte-split Zstd helper: measured snapshot

This directory is an evidence snapshot of a component microbenchmark. It is not
an LMCache integration, serving benchmark, or claim of production support. The
helper consumes the experimental byte-plane framing below; no shared wire-format
registration or storage path is included.

## Input and provenance

The measurement uses three BF16 KV records from one 0.5B CPU capture with all
24 K/V layers: `ctx128`, and the first and second 256-token halves of a `ctx512`
record. These are related slices, not three independent requests or captures.
The raw bytes total 7,864,320 B; the level-1 byte-split payload totals
5,471,264 B in 480 independent 16 KiB Zstd frames.

The measured local input was 6,235,650 B with SHA-256
`8596ac9a5b7475ee61b26c72847966ec846cb02ccee934868725db298143c211`. Its bytes
were checked against the public file at the pinned KVBytePack commit:
https://raw.githubusercontent.com/mynotwo/kvbytepack/d4039cdcaed434ba2f63dd4bc2eed2c6265b79fa/results/aibrix_layout/model_capture/kv_payloads.npz
The public corpus file is not duplicated in this snapshot. The source manifest
is `corpus-manifest.json`.

## Method and result

The CPU path uses one reused Zstd decompressor, the public CPU byte-merge, pinned
raw-host transfer, and synchronization to a preallocated GPU destination. The
GPU helper validates framing/lease metadata, packs and pins compressed input,
uses nvCOMP batch Zstd decode, performs GPU byte-merge, then synchronizes to the
same destination. Output SHA checks are outside timing. The original frozen
benchmark used two warmups followed by 12 alternating CPU/GPU sample groups;
reported values are medians and the long-tail sample remains included.

| Measurement | Result |
|---|---:|
| Original-script rerun, CPU / GPU | 15.557 / 9.328 ms; 1.668x |
| Historical original-script sample | 16.378 / 8.910 ms; 1.838x (not reproduced as stable) |
| Profiling harness uninstrumented control | 12.805 / 8.896 ms; 1.439x |
| Kineto nvCOMP Zstd kernel activity | 2.393 ms total; 1.862 ms main decode kernel |
| Compressed payload / descriptor H2D activity | 0.591 / 0.009 ms |
| GPU byte-merge kernel activity | 0.023 ms |
| CPU raw H2D activity | 1.128 ms |

Kineto and event traces are separate instrumented observations; their stage
activity must not be added to uninstrumented wall time. Kineto profiling
increased measured wall time by about 39.9% CPU and 35.7% GPU; the trace is for
stage attribution, not the headline timing. Transfers use pinned host memory to
GPU. There is no GDS, storage I/O, model serving, or request latency
measurement. CPU system load was not sampled, so the spread among the three
wall-time comparisons has no established cause.

## Environment and dependencies

Measured on NVIDIA H100 80GB with Python 3.11.2, PyTorch 2.7.0+cu126, CUDA
runtime 12.6, nvCOMP 6.0.0.12, and zstandard 0.25.0. The helper source is
SHA-256 `63f4c7e3bf3265eb17ba0c51c2aecd5af9841c4ba7ad93b1daba1827755f06d0`;
benchmark and profile script hashes are in `SHA256SUMS.txt`.

The script snapshot uses LMCache source commit
e9916d1af42319381a8d00b2edd0eee1575b1128, from the open #5480 stack, which
depends on open #5479. This snapshot does not include either LMCache change.
The frozen scripts retain the original task-machine absolute paths and bootstrap
assumptions. They are preserved for review and provenance; they are not promised
to execute directly from this directory. The sample JSON and profiler traces
are preserved byte-for-byte from the local evidence archive and therefore
retain run identifiers and package/task paths. `environment.json` is a separate
concise environment summary. No corpus file, raw run logs, or GPU process
monitor snapshots are included here. A scan of the published files found no
credentials.

## Files

- `source-snapshot/byte_split_zstd.py`: measured helper snapshot.
- `bench_codec_core.executed.py`: exact frozen CPU/GPU benchmark runner.
- `profile_helper.py`: profiler harness; it does not change helper source.
- `summarize_traces.py`: deterministic aggregation and validation of sample
  hashes, traces, transfer byte counts and summary medians.
- `previous-baseline.json`, `original-method-rerun.json`, `profile-result.json`,
  `gpu-trace.json`, `cpu-trace.json`, `derived-summary.json`: measurement inputs
  and recomputed outputs; the five source JSON/trace files are byte-identical to
  the archived measurement evidence.
- `environment.json`: software, input provenance and scope.
- `SHA256SUMS.txt`: hashes for the files in this snapshot, excluding itself.
- `.ruff.toml`: narrowly ignores the two pre-existing F401 unused-import
  findings in the frozen `profile_helper.py`; that script is kept byte-exact.

From this directory, `python3 summarize_traces.py` recomputes
`derived-summary.json`; `sha256sum -c SHA256SUMS.txt` verifies the published
snapshot files. Re-running the GPU benchmark requires the original environment,
LMCache dependency stack, H100-class device, and input at the fixed public URL.
