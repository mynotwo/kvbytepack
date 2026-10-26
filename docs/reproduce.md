# Reproduce KVBytePack

## Name migration

The canonical repository is https://github.com/mynotwo/kvbytepack and the
Python distribution/import name is `kvbytepack`, starting with version 0.2.0.
The former `mynotwo/kvzip` repository URL redirects after the repository rename.
Update Git remotes with:

```bash
git remote set-url origin https://github.com/mynotwo/kvbytepack.git
```

Change `from kvzip import ...` to `from kvbytepack import ...` in new code.
Existing imports remain supported, including `kvzip.codec`, `.tensor`,
`.formats`, `.inputs` and `.capture`; these alias the canonical modules.
The capture module still requires PyTorch. There is no removal deadline.
The `KVZ1` tensor magic, metadata and codec behavior are unchanged. Existing
blobs remain readable with the same dependencies. No separate `kvzip` wheel
or PyPI release is published by this change.

This project is unrelated to SNU's KVzip:
https://github.com/snu-mllab/KVzip and https://arxiv.org/abs/2505.23416.
That work studies eviction; KVBytePack preserves the bytes supplied to its codec.

## CPU entry: no model or GPU

Install from a checkout as shown in the README, then run `python -m kvbytepack`.
The eight checks cover four layouts through both byte and NumPy tensor APIs.
The fixture includes zero, signed zero, subnormal, infinity and NaN bit patterns.
Comparison uses bytes, since NaNs do not compare numerically equal.
The fixture is synthetic; its output contains no compression-performance claim.

Run `python bench/export_results.py --check` before installing anything to
verify the committed CSVs against the pinned JSON. This command uses only
Python's standard library. `--out-dir PATH` writes an independent copy.
Each row identifies model, context, input, representation, scope and method.
There are no seeds or timing fields in the source archive; none are invented.
K, V and KV rows are related scopes, not independent model runs. Representations
and codecs also reuse the same 50 captured cells.

The exporter checks source SHA256, the full model/context/input grid, all
representation/scope/method combinations and positive finite ratios. It rejects
changed sources and incomplete coverage. Summary values use existing
`bench/summarize.py` aggregation. The source JSON is preserved byte-for-byte.

## Fresh model capture

Requirements: model access over the network, Python, `.[bench,test]`, and enough
RAM or VRAM to load the chosen model and its cache. A 7B BF16 model's weights
alone are roughly 14 GB (7 billion parameters × 2 bytes), before framework,
cache and working memory. Do not use the weight estimate as total memory.
CPU works but model inference and high-level codecs can take substantial time.
The historical H100 environment is not required for the CPU smoke check.

A bounded one-cell run:

```bash
python -m pip install -e '.[bench,test]'
python bench/run_bench.py --models Qwen/Qwen2.5-7B --target kv \
  --ctx 512 --inputs english --formats bf16 \
  --methods raw_zstd-1 pipeline_zstd-1 --device cpu --out results/raw
python bench/summarize.py results/raw
```

Expected artifact: `results/raw/Qwen2.5-7B_512_english.json` with `verified:
true`, per-layer byte counts, aggregate ratios and codec timings. The runner
verifies decode against converted input bytes by default and aborts on mismatch.
Do not use `--no-verify` for evidence intended to support round-trip claims.
Check actual `context_length` rather than assuming the tokenizer reached 512.

For the historical grid, use the two models, context lengths and input types
listed in the README with `--target kv`; omit format/method filters only when
prepared for the complete codec sweep. Some historical optional codecs such as
Snappy are not in the default extras, so the installed method list can differ.
Record `available_methods()` alongside results before comparing “best” values.

New captures are not expected to be byte-identical to the historical archive:
its exact model revisions, capture software versions and source tensors were
not retained publicly. Record the resolved Hugging Face model/tokenizer commit,
`python -m pip freeze`, device, command, actual token count and input protocol
with every new result. Current runner model IDs are not revision-pinned and
use `trust_remote_code=True`; inspect the selected model repository before a
fresh capture. The no-model CPU path does not load remote model code.

## Troubleshooting and interpretation

- `ModuleNotFoundError: kvbytepack`: install the project with the same Python
  interpreter used to run it. Prefer a virtual environment.
- Missing PyTorch: install `.[bench]` for captures or PyTorch-origin blobs;
  it is unnecessary for NumPy and the CPU entry.
- Missing optional codec: install `.[codecs]` for lz4/brotli, or choose zstd.
- Download, access or memory failures: resolve model access and available memory;
  the exported results and CPU entry can still run independently.
- Export mismatch: regenerate to a temporary directory and inspect the diff;
  do not overwrite the pinned source or edit numbers to match a narrative.
- Larger complete blobs than reported payloads: the historical ratios exclude
  plane-length headers. Use `len(blob)` when measuring actual storage cost.
- Poor compression on a new workload: report it. Repeated paragraphs and FP32
  widening can make this archive more compressible than production state.

No fresh model capture or GPU benchmark is included in the 0.2.0 evidence
release. Serving speed, whole-model equivalence, real-text corpus generality
and production benefit require separate measurements.
