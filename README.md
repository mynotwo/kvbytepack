# KVBytePack

Bit-exact compression for LLM KV-cache blocks, plus the arithmetic that tells
you whether compressing is worth it on your link.

A KV block is a matrix of 2-byte (BF16/FP16) or 1-byte (FP8) elements whose
channels each have their own scale. Splitting each element into byte planes and
laying every plane out channel by channel puts the low-entropy exponent bytes
next to each other, and a general-purpose codec then shrinks them. The
transform is a pure byte permutation, so decoding returns the exact input bits,
NaN payloads and `-0.0` included. Nothing is quantized or dropped.

Compression only helps if the bytes it saves are worth more than the CPU time it
costs. `kvbytepack bench` measures both on your arrays and answers that for a
range of link speeds.

## Install

```bash
pip install -e .            # numpy + zstandard; builds an optional C kernel
pip install -e '.[codecs]'  # adds lz4
python -m kvbytepack selftest
```

Python 3.10+. Without a C compiler the install falls back to NumPy.

## Use

```python
import numpy as np
import kvbytepack

kv = np.load("data/qwen2.5-0.5b-bf16-kv.npz")["ctx512_K_05"]   # [tokens, channels] BF16 bits
rec = kvbytepack.pack(kv)                 # channel-planes + zstd:1 by default
assert kvbytepack.unpack(rec).tobytes() == kv.tobytes()
print(kvbytepack.read_header(rec))        # shape, dtype tag, codec, per-plane sizes
```

`pack` accepts NumPy arrays and PyTorch tensors (`bfloat16` and `float8_*`
included) and puts the dtype and shape in the record. The last axis is the
channel axis. Options: `layout="none" | "planes" | "channel-planes"`,
`codec="zstd:<level>" | "lz4" | "zlib:<level>" | "lzma:<preset>" | "store"`.

For raw buffers with an explicit geometry, use `kvbytepack.record.encode` /
`decode`. Blobs written by kvbytepack 0.2 / kvzip (`KVZ1`) still unpack.

### AIBrix

`kvbytepack.integrations.aibrix.make_marshaller(BaseMarshaller, tokens=..., channels=..., width=2)`
returns an AIBrix L2 marshaller built on AIBrix's own base class. It is tested
against unmodified copies of AIBrix's `BaseMarshaller` and `ZstdCompressor`.

## Does it pay? Results on the bundled capture

`data/` holds 96 real KV blocks from Qwen2.5-0.5B (24 layers, K and V, 128- and
512-token prefixes of a CPython source file; provenance in
`data/qwen2.5-0.5b-bf16-kv.json`).

```bash
python -m kvbytepack bench data/qwen2.5-0.5b-bf16-kv.npz --threads 4
```

On an Apple M4 Pro laptop (one run, 2026-11-12; your numbers will differ):

| config | ratio | 1 thread enc / dec GB/s | 4 threads enc / dec GB/s |
|---|---:|---:|---:|
| none + zstd:1 | 1.283 | 1.28 / 1.27 | 3.14 / 3.09 |
| channel-planes + zstd:1 | 1.477 | 0.70 / 1.52 | 1.64 / 2.73 |
| channel-planes + lz4 | 1.203 | 0.75 / 2.06 | 1.65 / 2.46 |

Ratio is raw bytes over complete record bytes (header and checksum included).
GB/s are raw bytes per second.

What this says:

- The layout is worth 15% more compression than plain zstd (1.48x vs 1.28x).
  The exponent plane compresses 2.7x (median over blocks); the mantissa plane is close to random
  and is stored verbatim in 95 of 96 blocks.
- Encoding is the bottleneck. Even on 4 threads the layout path encodes about
  1.6 GB/s, so on any link faster than that, sending uncompressed blocks is
  faster. Under the serial model (encode, send, decode, one block at a time) the
  layout path wins only below about 0.15 GB/s on one thread and 0.33 GB/s on
  four.
- So on RDMA, NVLink or PCIe, do not compress KV on the CPU with this. It can
  pay for capacity (more blocks per GB of host memory or SSD) or on slow
  cross-region links.

## Cost model

`kvbytepack.costmodel` holds the two models the bench uses. For a candidate
against a baseline:

```text
serial:     t = encode + stored_bytes / bandwidth + decode
            candidate wins  <=>  bandwidth < (baseline_bytes - candidate_bytes)
                                             / (candidate_codec_s - baseline_codec_s)
pipelined:  goodput = min(raw / encode, raw / decode, bandwidth * ratio)
```

Neither models queueing, contention with model compute, or memory bandwidth.
They give a necessary condition, not a promised speedup.

## Record format (KVB2)

Magic, layout id, codec id and level, element width, shape, dtype tag, a plane
table (mode and length per plane), the plane payloads, then a CRC32 over
everything. Corruption, truncation and trailing data are rejected before data
is returned. CRC32 detects accidents; it does not authenticate.
Field-by-field layout is in `kvbytepack/record.py`.

## History

Formerly `mynotwo/kvzip`, unrelated to SNU's KVzip (token eviction). The 0.2
line's frozen result archives and scripts are on `main` and tags `v0.2.*`; see
`REDESIGN.md` for what changed.

Apache-2.0.
