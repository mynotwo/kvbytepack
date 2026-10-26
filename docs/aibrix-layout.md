# AIBrix layout-compression contract experiment

This is a downstream CPU experiment, not an AIBrix feature or a serving benchmark.
It asks where layout metadata must live and whether byte savings can justify
layout/codec cost. No AIBrix runtime dependency or default is changed.

## Run it

Use Python 3.12 (the pinned AIBrix modules need Python 3.10+), from this checkout:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python -m examples.aibrix_layout.benchmark --out-dir /tmp/aibrix-layout
python -m examples.aibrix_layout.benchmark --check results/aibrix_layout
```

No GPU, PyTorch, model weights, Kubernetes or network access is required after
installation. The first command writes your own timings. The check command
rebuilds fixture bytes, verifies all 36 case/path sizes and round trips, and
checks historical timing summaries against their samples. It does not claim
to reproduce the original machine's timings. Encoded sizes can change with
Zstd versions; the original dependency/runtime versions are in the manifest.

Four paths run on exactly the same payloads:

1. Uncompressed bytes (identity operation; no storage or transfer is timed).
2. The unmodified upstream `ZstdCompressor`, with its default compressor.
3. The same upstream class with `compressor` explicitly set to Zstd level 1.
4. A layout-aware record using KVBytePack's `pipeline_zstd-1`.

The third path controls codec level when assessing layout cost. Neither raw
baseline carries new layout/checksum metadata. The layout path's complete
stored length includes its tensor envelope, byte-plane headers, version magic
and SHA256; metadata is never subtracted from the reported size. The checksum
is corruption detection, not authentication.

## Observed CPU results

[CSV](../results/aibrix_layout/measurements.csv),
[all timing samples](../results/aibrix_layout/samples.json), and
[environment/provenance manifest](../results/aibrix_layout/manifest.json).
Each path has 11 measured encode/decode repetitions after two warmups.
The three fixtures are zeros, uniform bytes and deliberately constructed
channel-correlated bytes. They are **synthetic, not captured model KV**.

| Synthetic case | Raw bytes | Upstream default Zstd bytes | Level-1 control bytes | Complete layout record bytes |
|---|---:|---:|---:|---:|
| zeros_8x8 | 128 | 17 | 17 | 172 |
| uniform_bytes_8x8 | 128 | 137 | 137 | 284 |
| channel_bytes_8x8 | 128 | 137 | 137 | 249 |
| zeros_512x128 | 131072 | 22 | 22 | 182 |
| uniform_bytes_512x128 | 131072 | 131084 | 131084 | 131234 |
| channel_bytes_512x128 | 131072 | 131084 | 131084 | 65968 |
| zeros_2048x128 | 524288 | 34 | 34 | 195 |
| uniform_bytes_2048x128 | 524288 | 524309 | 524309 | 524461 |
| channel_bytes_2048x128 | 524288 | 524309 | 524309 | 262666 |

All 36 case/path rows restored the original bytes. The layout record is smaller
than the level-1 control in only two of nine cases; the other seven expand.
This is evidence against enabling this prototype by default.

For a serialized encode-transfer-decode path, define `delta_t` as the sum of the layout
encode and decode medians minus the sum of the level-1 control's medians.
If `saved_bytes > 0` and `delta_t > 0`, layout can win only when
`bandwidth < saved_bytes / delta_t`. This is a necessary condition under a
non-overlapped, single-transfer model, not a measured link speed or serving gain.

- `channel_bytes_512x128` saves 65,116 bytes and adds 0.453 ms of median CPU work; the bound is 0.144 GB/s (decimal).
- `channel_bytes_2048x128` saves 261,643 bytes and adds 1.775 ms of median CPU work; the bound is 0.147 GB/s (decimal).

These machine-specific observations do not justify GPU/PCIe or production
acceleration claims. Even favorable synthetic byte ratios can lose to CPU
layout/copy/checksum work. This reference implementation is not a GPU codec.

## Captured model KV: all layers, with original bytes

A separate fresh CPU capture uses
[Qwen2.5-0.5B at revision 060db649](https://huggingface.co/Qwen/Qwen2.5-0.5B/tree/060db6499f32faf8b98477b0a26969ef7d8b9987).
It runs non-repeated 128- and 512-token prefixes of one pinned CPython source
file through the model and captures every layer's K and V in BF16.
There are 24 layers × 2 scopes × 2 contexts = **96 payloads**, compared through
four paths = **384 rows**. The two prefixes overlap; layers, K/V and contexts
are related samples, not independent documents or training runs.

The [raw BF16 bit patterns](../results/aibrix_layout/model_capture/kv_payloads.npz)
are published as non-object uint16 NumPy arrays, one array per context/layer/scope.
The archive preserves bytes; it performs no BF16-to-FP16 conversion.
[Capture metadata and token IDs](../results/aibrix_layout/model_capture/capture.json),
[full CSV](../results/aibrix_layout/model_capture/measurements.csv),
[timing samples](../results/aibrix_layout/model_capture/samples.json), and
[hash manifest](../results/aibrix_layout/model_capture/manifest.json) accompany it.
The public input and its PSF license are included. No user/private prompt is used.

Complete stored bytes summed over all 48 K/V payloads in each context:

| Context | Raw bytes | Upstream default Zstd bytes | Level-1 control bytes | Complete layout records | Smaller than level-1 control |
|---|---:|---:|---:|---:|---:|
| 128 | 1572864 | 1233913 | 1234526 | 1085457 | 47/48 |
| 512 | 6291456 | 4887811 | 4890637 | 4249030 | 47/48 |

All 384 rows restored their original bytes. Relative to the matched level-1
control, this layout record is smaller in 94/96 payloads, but slower to encode
and decode. Sum the per-payload encode/decode medians for a serial cost bound:

- 128 tokens: 149,069 fewer stored bytes (12.07%), with 6.811 ms of additional summed CPU medians; break-even bandwidth 0.022 GB/s.
- 512 tokens: 641,607 fewer stored bytes (13.12%), with 21.349 ms of additional summed CPU medians; break-even bandwidth 0.030 GB/s.

These sums are a cost model, not a measured batched transfer. This small-model,
single-source result supports investigating byte layout, but does not establish
7B/hybrid-model generality, serving gains or whole-model output equivalence.
The measured additional CPU cost still limits the usefulness of this reference
implementation on fast links.

Offline verification uses NumPy/Zstd only, with no model download or PyTorch:

```bash
python -m examples.aibrix_layout.capture --check results/aibrix_layout/model_capture
```

To collect a new run with the same pinned model and input:

```bash
python -m pip install '.[bench]'
python -m examples.aibrix_layout.capture --out-dir /tmp/own-model-capture
```

This downloads approximately 1 GB of model weights and uses CPU PyTorch.
Capture metadata records software versions, source-code hashes, device,
attention implementation, actual token IDs and model revision. Different
software/hardware may produce different model bytes; the offline check verifies
the published bytes rather than claiming cross-platform model determinism.
On the author's machine the download used system TLS trust-store initialization;
no certificate verification was disabled and no HF token was supplied.

## Adapter contract

The caller supplies `Layout(tokens, channels, bytes_per_element)` for token-major
`[S,D]` payload bytes. Only positive dimensions and fixed-width 1/2/4/8-byte
elements are accepted. There is no quantization, pruning or numeric conversion.
Empty, packed sub-byte, strided, mixed-state and prefixed/index-bearing payloads
are outside this prototype. Callers own trustworthy layout dimensions.

The record uses the existing KVZ1 tensor metadata and layout transforms, wrapped
by `KBL1` and a SHA256 digest of the tensor record. Decode checks version,
metadata, expected shape/element width, plane lengths, declared Zstd frame
sizes and trailing data before returning bytes. The prototype always requires
an explicit expected layout on decode; it does not infer model geometry.

With a full AIBrix installation, the bridge is:

```python
from aibrix_kvcache.l2.marshallers import BaseMarshaller
from examples.aibrix_layout.adapter import Layout, make_compressor

compressor = make_compressor(BaseMarshaller, Layout(512, 128, 2))
payload_bytes = bytes(512 * 128 * 2)  # Synthetic; replace with a real payload.
blob = compressor.marshal(payload_bytes)  # exactly 512 * 128 * 2 bytes
assert compressor.unmarshal(blob) == payload_bytes
```

This uses the real BaseMarshaller chaining implementation. `TensorSerializer`
adds an index header, so **do not compose this directly around TensorSerializer**.
The required integration decision is where AIBrix should expose tensor layout
before flattening/serialization, while preserving indices separately.

For the CPU contract check, two upstream modules are pinned and included
unmodified under `tests/fixtures/aibrix_marshallers`, retaining Apache notices.
Their commit, primary-source URLs and hashes are in `provenance.json`. Only the
two modules are loaded, not a fabricated full AIBrix package. This validates
marshaller/chaining behavior, not connector, package-install or model behavior.

## Bring a real tensor

```bash
python -m examples.aibrix_layout.benchmark \
  --input-npy /path/to/your-2d-kv.npy --out-dir /tmp/own-kv-result
```

The input must be a two-dimensional, fixed-width NumPy array with no objects or
structured fields. It is serialized in C order without changing element bits.
Record model/revision, original dtype, layer/head mapping, token source and
capture environment alongside your own results. Output labels it
`user_supplied_npy`; the tool cannot certify that a supplied file is model KV.
It records the exact payload SHA256, complete sizes and timing samples.
The committed `--check` mode applies to the fixed synthetic suite only.

## Upstream decision

AIBrix requires an accepted issue before feature implementation/PR. This
experiment supports the existing design discussion without modifying upstream
code or adding a dependency. An upstream feature needs maintainer agreement on
the metadata boundary and evidence from relevant serving workloads with actual end-to-end cost.
The separate MEMORY backend PR #3221 remains independent.
