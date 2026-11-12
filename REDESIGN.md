# Redesign (branch `redesign`, v0.3.0)

## Purpose

A small library and CLI that answers one engineering question for a KV-cache
store: should this tier compress KV blocks, and with what? It gives a
bit-exact codec that can go into a store (with a checksummed, self-describing
record) and a benchmark that measures it on real KV arrays and turns ratio and
codec speed into a verdict per link speed.

The 0.2 line was mostly an evidence archive: 43 MB of frozen timing samples,
hash-checked result exports and scripts whose job was re-verifying earlier
claims. That made the repo hard to build on. The redesign makes measurement a
command you run on your own data instead of a record of past runs.

## Architecture

```
kvbytepack/
  layout.py       byte permutations: none | planes | channel-planes  (+ optional C merge kernel, _native.c)
  codecs.py       store/zstd/zlib/lzma/lz4 behind stable one-byte ids, "name:level" specs
  record.py       KVB2 format: header, per-plane table, payloads, CRC32
  tensor.py       pack/unpack for NumPy arrays and PyTorch tensors
  legacy.py       read-only decoder for 0.2 / kvzip KVZ1 blobs
  costmodel.py    serial crossover bandwidth, pipelined goodput
  bench.py        measure layouts x codecs on .npy/.npz (1 or N threads), text report
  __main__.py     `python -m kvbytepack selftest | bench | inspect`
  integrations/aibrix.py   AIBrix L2 marshaller on AIBrix's own BaseMarshaller
data/             96 real Qwen2.5-0.5B BF16 KV blocks (the benchmark dataset) + provenance
tests/            unit, corruption, legacy-compat, CLI and AIBrix-chaining tests
```

Record format changes from KVZ1 to KVB2:

- Codec and layout are separate fields with numeric ids. KVZ1 stored a free-form
  method string in JSON.
- Planes that do not shrink are stored verbatim (BF16 mantissa bytes, 95 of 96
  real blocks), so decode skips the codec for them and random data grows by the
  header only.
- CRC32 over the whole record, and decoded plane lengths are checked against the
  header. A bit flip at any byte position, truncation and trailing bytes are
  all rejected (tested at every position of a small record). The 0.2 AIBrix prototype used
  SHA-256, which on this machine runs at about 3 GB/s, the same order as the
  codec; CRC32 runs at about 40 GB/s.
- Sizes reported by the bench are whole records, never bare payloads.

## Removed, and why

- `results/` (43 MB), `bench/*.py`, `docs/*`, `examples/aibrix_layout/`, issue
  template, `.gitattributes`, `MANIFEST.in`: archives and checkers for earlier
  claims. They remain on `main` and tags `v0.2.0`, `v0.2.1`, `aibrix-layout-v1`.
- `kvzip` import alias: the name is two releases old. Old blobs still decode
  through `legacy.py`, checked against 32 fixtures written by 0.2.1.
- `formats.py`, `inputs.py`, `capture.py`: lossy quantization formats and
  repeated-paragraph inputs that inflated ratios. Capture needs PyTorch and
  model weights; it is the first thing to rebuild (below).
- brotli/snappy codecs and zstd 9/19 in the default sweep: slower than the
  link they would serve. Any zstd level remains available as `zstd:N`.
- Python 3.9 support: EOL, and the pinned AIBrix fixtures use 3.10 syntax.

## Kept

- The channel-major byte-plane transform and the C 8x8 transpose kernel
  (renamed `_native`), with a NumPy fallback.
- The real Qwen2.5-0.5B capture, now the bundled benchmark dataset.
- The AIBrix integration idea, rebuilt as a library module and tested by
  chaining with unmodified upstream `BaseMarshaller` and `ZstdCompressor`.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -e '.[test,codecs]'
.venv/bin/python -m kvbytepack selftest
.venv/bin/python -m pytest -q tests
.venv/bin/python -m kvbytepack bench data/qwen2.5-0.5b-bf16-kv.npz --threads 4
```

## Finding on this machine

On an M4 Pro, channel-planes + zstd:1 reaches 1.48x versus 1.28x for plain
zstd:1, but encodes at 0.70 GB/s on one thread and 1.64 GB/s on four. It only
beats sending raw bytes below roughly 0.15 to 0.33 GB/s in the serial model.
CPU-side compression of KV is a capacity tool, not a bandwidth tool, at these
speeds. Full table in README.

## Next

1. `kvbytepack capture`: dump KV from any Hugging Face model to `.npz` (needs
   PyTorch; can't be tested in CI cheaply). Use non-repeated text.
2. Speed up encode, the bottleneck. On a 512-token block the zstd pass over
   the exponent plane takes about 100 of roughly 190 us. That plane uses about
   20 distinct byte values, so a plain Huffman/ANS coder may beat zstd on speed.
   Moving the split into the C kernel saves the ~40 us NumPy transpose.
3. Capacity mode in the cost model: blocks per GB of host DRAM or SSD and the
   hit-rate change that buys, next to the bandwidth verdict.
4. A GPU path (nvCOMP or a Triton byte-plane kernel), where codec throughput is
   high enough to compete with PCIe.
5. Wire the AIBrix marshaller into a real AIBrix deployment and measure
   end-to-end restore latency.
