# Evaluate KVBytePack independently

This is a CPU codec study for people evaluating byte-preserving KV storage or
transfer. The most useful contribution is an independently run result, including
a mismatch, a slowdown or a case where compression expands the record.

[Submit a reproduction report](https://github.com/mynotwo/kvbytepack/issues/new?template=independent-evaluation.yml).
You do not need to deploy a serving engine or share private model data.

## Choose a question

| Route | What you run | What the result establishes |
|---|---|---|
| Verify the public records | Stored-result check on the frozen BF16 corpus | Byte restoration and internal consistency of the published bundle |
| Measure your CPU | Fresh paired runs against the pinned context-reuse baseline | Codec cost on the reported machine and backend |
| Evaluate your own workload | Separate capture with your model, geometry and data provenance | Only the measured workload; keep it separate from the frozen corpus |

The published corpus is 96 Qwen2.5-0.5B BF16 K/V records from all 24 layers at
overlapping 128/512-token prefixes of one public source file. Layers and prefixes
are correlated. It is not a representative sample of production traffic.

## 1. Verify the public records without a GPU

Use Python 3.12 for the pinned real-record study. From a fresh checkout:

```bash
git clone https://github.com/mynotwo/kvbytepack.git
cd kvbytepack
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e . numpy==2.5.4 zstandard==0.25.0
git rev-parse HEAD
python -m kvbytepack
python bench/compare_cpu.py --check results/cpu_decode
```

No PyTorch, model weights or GPU is required. The small CPU entry should report
`byte_exact: true` and `checks: 8`. The real-record check verifies the 9,216
published before/after rows, original bytes, encoded hashes, stored sizes and
aggregation. It reports the current verification backend.

**This verifies the archived experiment; it does not rerun its timing results.**
A compiler is optional for verification: installation can fall back to NumPy.
Keep the output and source commit in your report, including a failure if any.

## 2. Measure your CPU against the pinned baseline

Use the same environment and corpus. Fresh decode comparisons require the
optional C inverse in the candidate; a NumPy-only installation is not a
measurement of that optimization. Check the backend before running:

```bash
python -c "from kvbytepack.codec import _native_merge; assert _native_merge is not None"
git fetch origin 98f56cc16fe46195d70730e99ac64b754d5f75ad
git worktree add --detach ../kvbytepack-before 98f56cc16fe46195d70730e99ac64b754d5f75ad
python bench/compare_cpu.py --baseline-dir ../kvbytepack-before \
  --rounds 12 --repeats 41 --warmups 5 --out-dir ../kvbytepack-cpu-report
git worktree remove ../kvbytepack-before
```

Use an unused output directory. Run on an otherwise idle machine and record
CPU model, OS, compiler, Python, NumPy, Zstandard, active backend and competing
load. The driver alternates AB/BA in isolated sequential processes and checks
original-byte round trips and encoded hashes. Keep all rounds and paths,
including `raw_zstd1_control`. Do not remove slow or negative results.

The comparison baseline already includes earlier CPU optimizations; it is not
the original v0.2.1 release. Report complete-record costs separately from the
core codec. The summed 48-record medians are not request latency or throughput.
See the [decode protocol and memory accounting](cpu-decode.md) for the full
contract and optional memory diagnostics. Python-tracked allocation peaks,
retained native contexts and RSS are different measures.

If compilation fails, report your compiler output and the NumPy verification
result. Do not disable checks or report fallback timings as compiled timings.

## 3. Decide whether the experiment fits your system

People working on CPU serialization, KV persistence or bandwidth-limited
storage can use these records as a reference corpus and measure their own
tradeoff. They are also useful for rejecting an unsuitable codec.

KVBytePack is not a drop-in LMCache, Mooncake or Dynamo integration and has no
GPU decoder. A GPU-offload pipeline with CPU decompression still transfers the
expanded bytes from host to GPU. The measured complete record costs more CPU
than bare raw Zstd, whose control has a different metadata/integrity contract.
A fast network or SSD path may therefore have no latency benefit.

The [cost model](../README.md#decide-whether-compression-pays) is a serial,
non-overlapped comparison, not a serving prediction. Layout geometry must be
explicit; do not apply tensor byte transforms across opaque headers or indices.
The [AIBrix boundary study](aibrix-layout.md) is separate from MEMORY backend
PR #3221. Byte preservation does not establish whole-model equivalence.

## Share a small, useful report

Use the [independent evaluation form](https://github.com/mynotwo/kvbytepack/issues/new?template=independent-evaluation.yml)
with the source commit, environment, route, commands and actual result. For
fresh timings, include the generated manifest, summary and unfiltered results
or a link to them. Include the baseline commit and backend for each side.
For a failed check, the case/path and traceback are more useful than a pass badge.

Report how you found the project if you want to; it helps distinguish a genuine
independent attempt from repository traffic. Share synthetic or already-public
inputs only. Do not upload private KV bytes, prompts, credentials, model weights
or a complete environment dump.
