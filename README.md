# kvzip: lossless compression for LLM KV cache and activations

KV cache and hidden-state activations are arrays of floating-point numbers, not
arbitrary byte streams. In BF16, the high byte of each value (sign + exponent)
takes few distinct values, while the low byte (mantissa) is close to random.
Off-the-shelf codecs see the two bytes interleaved and miss most of that
structure.

kvzip rearranges the bytes before compressing, and the rearrangement is exactly
invertible:

1. **Channel-first layout**: store each channel's values across all tokens
   contiguously (`[S, D]` → `[D, S]`).
2. **Byte split**: write byte 0 of every element to one stream and byte 1 to
   another, and so on for wider formats.
3. **General-purpose codec**: compress each stream separately with zstd, lz4 or
   lzma.

Compression policy = representation + layout + codec.

Project page with interactive results: https://mynotwo.github.io/kvcache-bench/

## Where compression pays off

The offline ratios in Results assume a CPU codec. In a serving system the codec
sits on the restore path, and there CPU decode costs more than it saves.
[aibrix-hybrid-kvcache](https://github.com/mynotwo/aibrix-hybrid-kvcache)
(branch `delivery/hybrid-connector`) offloads the KV cache of
Qwen3.8-Flash-Next (TP8, H800) to AIBrix and restores it into unmodified vLLM.
Restore window, median over 8 GPUs:

| Prompt | No compression | GPU ANS (nvCOMP, ratio 1.34) | CPU zstd (ratio 1.22) |
|---|---|---|---|
| 8K | 6.3 ms | 6.2 ms | 32 ms |
| 128K | 43.8 ms | 34.5 ms | – |
| 256K | 83.9 ms | 64.4 ms | – |

- Above 32K the restore is bound by host-to-device PCIe bandwidth, so fewer
  bytes means a shorter restore only when decode runs on the GPU.
- The zstd number comes from an earlier build, where no compression took
  6.7 ms at 8K. zstd was dropped from the connector after that run.
- At 256K, restoring takes 0.44 s to the first token; recomputing takes
  11–15 s.

The connector compresses raw bytes with ANS; it does not use the layout in
this repo. Measured on H100 with nvCOMP (Qwen2.5-7B KV, BF16, real text,
median of 18 samples):

| GPU codec | Ratio | Decode |
|---|---|---|
| ANS, raw bytes | 1.22× | 47 GB/s |
| ANS, channel-first | 1.23× | 34 GB/s |
| ANS, channel-first + byte split | 1.43× | 19 GB/s |
| GDeflate, channel-first + byte split | 1.44× | 1.1 GB/s |
| zstd, channel-first + byte split | 1.49× | 0.7 GB/s |

Decode is GB/s of BF16 output, on samples of 2K–8K tokens; a larger record
may decode faster. Byte split lifts the ANS ratio by 17%, but decode falls
to 19 GB/s, well under the ~50 GB/s PCIe copy it overlaps with. A 1.6 GB record would take
about 84 ms to decode, against 34.5 ms for the whole restore today, so the
connector keeps ANS on raw bytes.

## Results

### KV cache, real text (Qwen2.5-7B, BF16)

18 samples: English (ShareGPT conversations), Chinese (Wikipedia) and code
(CPython standard library), 2K and 8K tokens, 3 samples each. No document is
used twice. Every blob was decoded and compared byte for byte.

| Method | Median ratio | 2K / 8K |
|---|---|---|
| raw zstd-1 | 1.29× | 1.29× / 1.29× |
| byte split + zstd-1 | 1.48× | 1.48× / 1.48× |
| channel-first + byte split + zstd-1 | 1.49× | 1.49× / 1.49× |
| channel-first + byte split + zstd-19 (best) | **1.52×** | 1.52× / 1.53× |

- The three input types land within 0.4% of each other.
- Context length barely matters: 8K compresses 0.4% better than 2K.
- Against floating-point compressors on the same tensors:

  | Compressor | Median ratio |
  |---|---|
  | kvzip, channel-first + byte split + zstd-1 | 1.49× |
  | XOR with previous token + zstd-1 | 1.46× |
  | zfp, reversible mode | 1.02× |
  | fpzip, lossless | 0.56× |

  zfp and fpzip take FP32, so BF16 was widened first; the ratio is against
  the BF16 bytes. SZ3 has no lossless mode and was left out.

Scripts and per-sample results: `kvzip-evidence` on the H100 test machine,
not in this repo.

### KV cache, repeated-paragraph inputs (Qwen2.5-7B, Qwen3-8B; 512–8K tokens; 5 input types; 50 cells)

These inputs repeat one short paragraph to the target length, which
overstates the ratio: on Qwen2.5-7B, channel-first + byte split + zstd-1
gives 1.62× here against 1.49× on real text.

| Format | Median best ratio | Range | raw zstd-1 | byte split + zstd-1 | Gain over best raw codec |
|---|---|---|---|---|---|
| BF16 | 1.71× | 1.52–1.93× | 1.29× | 1.51× | +19.4% |
| FP16 | 1.75× | 1.54–2.09× | 1.31× | 1.56× | +22.9% |
| FP32 | 3.43× | 3.03–3.86× | 2.09× | 3.01× | +15.9% |
| FP8 E4M3 / E5M2 | 1.56× / 1.95× | | | | +7% |
| INT8 / INT4 / INT2 | 1.27× / 1.92× / 5.00× | | | | ≤6% |

- "Best ratio" is the best of all methods in the cell. "Gain" compares the best
  method with the best codec run on unmodified bytes.
- BF16 and FP32 are lossless relative to the model's original tensors. FP16,
  FP8 and INT formats are lossy conversions, and the compression is lossless
  only relative to the converted bytes.
- The ratio rises with context length here (Qwen2.5-7B: 1.56× at 512 tokens,
  1.86× at 8K). That rise comes from the repetition; on real text it is 0.4%.

Source: `results/kv_aggregate_qwen.json`. Run
`python bench/summarize.py results/kv_aggregate_qwen.json` to regenerate the
table.

### Activations

Residual-stream hidden states (output of every transformer block) from
Qwen2.5-7B, 512 and 2048 tokens, english and code inputs (4 cells). Every
compressed blob was decoded and compared byte for byte with the input.

| Format | Median best ratio | Range | raw zstd-1 | channel-first + byte split + zstd-1 | Gain over best raw codec |
|---|---|---|---|---|---|
| BF16 | **1.67×** | 1.52-1.80× | 1.28× | 1.58× | +17.4% |
| FP16 | 1.71× | 1.54-1.88× | 1.30× | 1.58× | +20.6% |
| FP32 | 3.34× | 3.05-3.59× | 2.10× | 3.17× | +11.4% |
| INT4 | 3.11× | 1.85-4.81× | 1.97× | 1.97× | +0.0% |

- The embedding output is excluded. It can be recomputed from token ids, and
  with the repeated input text its rows repeat exactly, which pushes that one
  layer to ~28×.
- INT4 ratios are high partly because per-channel absmax quantization sends
  most non-outlier values to a few levels. That reflects the lossy conversion,
  not the layout.
- Coverage is smaller than for KV (one model, 4 cells). Run
  `python bench/run_bench.py --target act` for the full matrix.

## Usage

```python
from kvzip import zip_tensor, unzip_tensor

# any numpy array or torch tensor (bfloat16 included); last dim = channels
blob = zip_tensor(kv, method="pipeline_zstd-1")
assert unzip_tensor(blob).equal(kv)          # torch; numpy: array_equal
```

The blob carries method, dtype and shape. The byte-level API underneath:

```python
from kvzip import encode, decode

# raw: bytes of a [S, D] tensor in BF16 (bpe = bytes per element = 2)
blob = encode(raw, S, D, 2, "pipeline_zstd-1")
assert decode(blob, S, D, 2, "pipeline_zstd-1") == raw
```

Methods: `raw_<codec>`, `transpose+<codec>`, `bytesplit+<codec>`, and
`pipeline_<codec>` (channel-first layout plus byte split). The codec can be any
of `zstd-1/9/19`, `lzma`, `gzip`, and `lz4`/`brotli`/`snappy` when installed.

## Reproduce

```bash
pip install -e ".[bench,test]"
pytest -q tests                      # round-trip tests, no model needed

# KV cache + activations, every blob decoded and compared byte for byte
python bench/run_bench.py --models Qwen/Qwen2.5-7B Qwen/Qwen3-8B --target both
python bench/summarize.py results/raw
```

Defaults: context lengths 512–8192, input types english/code/math/chinese/json,
and all 8 formats, on CPU. A full run takes several hours, mostly in lzma and
zstd-19. Use `--ctx`, `--formats` and `--methods` to narrow it down.

Numbers can drift slightly across torch/transformers versions because the
model's forward pass is not bit-identical across versions. Re-running 4
Qwen2.5-7B cells (512/2048 tokens × english/code) with a newer stack
reproduced the best KV ratio within 0.1% in all 16 cell × format pairs. One
method, INT4 `pipeline_zstd-1`, drifted by up to 0.5%.

## Caveats

- `bench/run_bench.py` builds inputs by repeating a short paragraph (see
  `kvzip/inputs.py`). Use the real-text numbers above when quoting a ratio.
- The activation numbers come from repeated inputs only.
- Real text was measured on one model, Qwen2.5-7B, in BF16.

## Not measured

- End-to-end serving throughput, concurrency gains, and production memory savings
- Overlap between copying and compression during serving
- Random access within a compressed block

## License

Apache-2.0
