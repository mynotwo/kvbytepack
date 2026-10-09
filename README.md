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

## Results

### KV cache (Qwen2.5-7B, Qwen3-8B; 512–8K tokens; 5 input types; 50 cells)

| Format | Median best ratio | Range | raw zstd-1 | byte split + zstd-1 | Gain over best raw codec |
|---|---|---|---|---|---|
| BF16 | **1.71×** | 1.52–1.93× | 1.29× | 1.51× | +19.4% |
| FP16 | 1.75× | 1.54–2.09× | 1.31× | 1.56× | +22.9% |
| FP32 | 3.43× | 3.03–3.86× | 2.09× | 3.01× | +15.9% |
| FP8 E4M3 / E5M2 | 1.56× / 1.95× | | | | +7% |
| INT8 / INT4 / INT2 | 1.27× / 1.92× / 5.00× | | | | ≤6% |

- "Best ratio" is the best of all methods in the cell. "Gain" compares the best
  method with the best codec run on unmodified bytes.
- BF16 and FP32 are lossless relative to the model's original tensors. FP16,
  FP8 and INT formats are lossy conversions, and the compression is lossless
  only relative to the converted bytes.
- Compression ratio rises with context length: Qwen2.5-7B goes from 1.56× at
  512 tokens to 1.86× at 8K tokens (BF16, best method, median over the 5 input types).

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

- Inputs are a short paragraph repeated to the target length (see
  `kvzip/inputs.py`). Repeated tokens make layer-0 V rows (a linear function
  of the embedding) repeat exactly. That one layer compresses at ~28× at 2K
  tokens, and it raises the aggregate KV ratio by about 2%. The KV numbers above
  include layer 0. How much the repetition affects deeper layers, or the trend
  with context length, has not been measured with non-repeated text.

## Not measured

- End-to-end serving throughput, concurrency gains, and production memory savings
- Overlap between copying and compression during serving
- Random access within a compressed block
- GPU-side codecs (all codecs here run on CPU)

## License

Apache-2.0
