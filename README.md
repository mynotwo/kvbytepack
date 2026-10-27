# KVBytePack

**Byte-exact compression of KV-cache and activation tensors through layout transforms and general-purpose codecs.**

KV tensors carry structure across channels and byte planes. KVBytePack rearranges
those bytes before compression, then reverses the transformation on decode.
It preserves the supplied representation, including NaN payloads and signed zero.

Formerly **mynotwo/kvzip**. This project is independent of
[SNU's KVzip](https://github.com/snu-mllab/KVzip), which studies query-agnostic
KV-cache eviction. KVBytePack does not select or discard tokens.
The old `kvzip` Python imports remain compatible; the blob format is unchanged.
See [migration and reproduction](docs/reproduce.md).

## Try it without a GPU

Python 3.9+. No model weights, PyTorch, credentials or GPU required:

```bash
git clone https://github.com/mynotwo/kvbytepack.git
cd kvbytepack
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m kvbytepack
```

Expected: a JSON record with `byte_exact: true`, `checks: 8`, and fixture SHA256
`f097ddb619e8f3f09aae9f836e858bff1e948b647a9a6c2075526e1cb2a412b3`.
This checks synthetic bit patterns; it does not measure model compression or speed.
Install from this repository or a release wheel; no PyPI publication is implied.

```python
import numpy as np
from kvbytepack import zip_tensor, unzip_tensor

x = np.arange(1024, dtype=np.float32).reshape(32, 32)
blob = zip_tensor(x, method="pipeline_zstd-1")
y = unzip_tensor(blob)
assert y.dtype == x.dtype and y.shape == x.shape
assert y.tobytes() == x.tobytes()
```

NumPy arrays and PyTorch tensors are supported; PyTorch is optional and needed
only for PyTorch inputs. Decoding a PyTorch-origin blob returns a CPU tensor.
The high-level blob stores method, shape and dtype. The low-level
`encode(raw, S, D, bpe, method)` / `decode(...)` API requires those parameters
from the caller. Neither API is a streaming or random-access storage format.

## Inspect the public experiment results

The repository includes a historical aggregate for Qwen2.5-7B and Qwen3-8B:
2 models × 5 context lengths (512–8192) × 5 input types = **50 cells**.
Each cell has 8 representations, 3 scopes (K, V, KV) and 15 methods.
Inputs repeat short paragraphs to fill context. They are **not independent
real-text documents**, and repetition can inflate compression ratios.

Recompute the exports using Python's standard library only:

```bash
python bench/export_results.py --check
python bench/summarize.py results/kv_aggregate_qwen.json
python bench/export_results.py --out-dir /tmp/kvbytepack-results
```

- [Original aggregate JSON](results/kv_aggregate_qwen.json)
- [18,000 ratios as CSV](results/measurements.csv)
- [Summary CSV](results/summary.csv)
- [Source hash, definitions and missing evidence](results/manifest.json)

Selected KV rows, medians over the same 50 cells:

| Representation | Raw zstd-1 | Byte split + zstd-1 | Channel-first + byte split + zstd-1 | Best of 15 methods |
|---|---:|---:|---:|---:|
| BF16 | 1.292× | 1.505× | 1.596× | 1.713× |
| FP16 | 1.314× | 1.563× | 1.575× | 1.752× |
| FP32 | 2.087× | 3.011× | 3.192× | 3.426× |

Ratio = uncompressed bytes / compressed **payload** bytes. Plane-length headers
are excluded in the historical aggregate; these are not total stored-blob ratios.
“Best” selects a method separately in each cell, then takes the median; it is
not a single deployable codec or a latency comparison. Higher is smaller payload.

BF16 and FP32 originate from a BF16 model representation; the runner widens to
FP32 for analysis. FP32 results therefore do not establish compression of
native FP32 model state. FP16, FP8 and integer conversions may lose information
before compression. Byte-exact codecs do not undo that conversion loss.

The archive contains rounded ratios, not original tensors, timing logs,
per-layer byte counts or per-blob verification records. Recomputing these
statistics verifies the public aggregate, not the original model capture.
The CPU smoke check independently exercises the current codec implementation.

## Capture new model data

[Reproduction guide](docs/reproduce.md) covers a bounded one-cell capture,
expected output, hardware needs and troubleshooting. Model capture requires
PyTorch, Transformers, model downloads and sufficient memory; it is separate
from the lightweight CPU entry above.

```bash
python -m pip install -e '.[bench,test]'
python bench/run_bench.py --models Qwen/Qwen2.5-7B --target kv \
  --ctx 512 --inputs english --formats bf16 \
  --methods raw_zstd-1 pipeline_zstd-1 --device cpu --out results/raw
python bench/summarize.py results/raw
```

## CPU layout performance

[Version 0.2.1 measurements](https://github.com/mynotwo/kvbytepack/blob/v0.2.1/docs/cpu-optimization.md) compare the same 96 real
KV payloads against the previous implementation in six balanced rounds.
Core encoding is 1.61–1.85× faster and decoding 2.11–2.44× faster on the measured
Apple M4 Pro. Complete records retain their metadata/checksum checks and improve
as well. Every before/after encoded byte and stored size matches; all timed
decodes restore the original bytes. These are CPU-call measurements.

The [follow-up against v0.2.1](docs/cpu-context-reuse.md) reuses level-1 contexts
per thread and passes contiguous planes directly to native Zstd. Across 12
paired rounds on the same 96 payloads, complete-record encoding time falls by
6.6% / 3.1% at 128/512 tokens. Decode changes are small and round ranges cross
1.0×. Wire bytes stay identical; warm contexts retain about 0.57 MiB per active
thread on this corpus. Full results include unchanged controls and limitations.

The [512-token decode study](docs/cpu-decode.md) compares an optional compiled
byte-layout kernel with the context-reuse revision above. On the same 96
payloads, complete-record decode is 1.205× faster at 512 tokens (17.0% less time),
with identical bytes and checksums. Measured native contexts remain about
0.57 MiB per thread; the tracked temporary decode peak falls by about 128 KiB
for the measured 512-token record. A NumPy fallback remains available when the
optional C kernel cannot be built. Full results include compiler requirements,
per-thread memory checks and the limits of these CPU measurements.

## AIBrix interface experiment

A [CPU contract experiment](docs/aibrix-layout.md) compares complete record sizes
and encode/decode costs against AIBrix's pinned Zstd marshaller. It includes
nine synthetic cases and a fresh Qwen2.5-0.5B capture of all 24 layers' K/V at
128/512 tokens, with original bytes. All 420 case/path rows round-trip. Layout
reduces complete stored size in 94/96 model payloads but adds CPU cost; this is
an inspectable downstream prototype, not a serving-speed claim.

## How the layout works

```text
[S, D, bytes_per_element]
  -> [D, S, bytes_per_element]     channel-first layout
  -> separate byte planes        group similar byte positions
  -> compress each plane         zstd / optional lz4 / lzma
  -> length-prefix each payload
```

Raw-codec, transpose-only and byte-split-only methods provide ablations.
The codec operates on bytes, without arithmetic on tensor values.

## Scope and related work

The [benchmark website](https://mynotwo.github.io/kvcache-bench/) is a separate
presentation with additional historical tables. This release's reproducible
public evidence is the pinned aggregate and the checks linked above.
Earlier README tables for real-text KV, activations, nvCOMP and AIBrix restore
timings are preserved in [the original revision](https://github.com/mynotwo/kvbytepack/blob/83a478d51d3d12a17d511977ef2af9c5a02bae6d/README.md).
Their underlying captures are not bundled here, so they are not promoted as
reproduced results in this release.

[AIBrix hybrid KV-cache work](https://github.com/mynotwo/aibrix-hybrid-kvcache)
is experimental. Whole-model output equivalence remains under investigation.
This repository does not establish end-to-end serving acceleration, production
capacity, concurrency benefits, or codec/transfer overlap.

Transfer savings help only if codec and layout cost fit the serving path.
For a non-overlapped transfer, a necessary break-even condition is
`encode_time + decode_time < raw_bytes / bandwidth - stored_bytes / bandwidth`.
The public ratio archive alone cannot answer that timing question.

## Tests

```bash
python -m pip install -e '.[test,codecs]'
python -m pytest -q tests
```

CI checks Python 3.9 and 3.12, installed-wheel imports, the CPU entry and
reproducible CSV exports. The 3.12 job also checks CPU PyTorch BF16 round trips.
Licensed under [Apache-2.0](LICENSE).
