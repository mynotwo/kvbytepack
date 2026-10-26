#!/usr/bin/env python3
"""Lossless compression benchmark for KV cache and activations.

For each (model, context length, input type, format) cell, every method is run on
every layer's K, V and activation tensor. Each compressed blob is decoded and
compared byte-for-byte with the input; a mismatch aborts the run.

Example:
  python bench/run_bench.py --models Qwen/Qwen2.5-7B --ctx 512 2048 --target both
"""
import argparse
import gc
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kvbytepack.codec import available_methods, decode, encode, payload_size  # noqa: E402
from kvbytepack.formats import FORMATS, to_format_bytes  # noqa: E402
from kvbytepack.inputs import INPUT_TYPES  # noqa: E402


def byte_entropy(data):
    arr = np.frombuffer(data, dtype=np.uint8)
    if len(arr) == 0:
        return 0.0
    p = np.bincount(arr, minlength=256)
    p = p[p > 0] / len(arr)
    return float(-(p * np.log2(p)).sum())


def run_tensor(tensor, fmt, methods, verify):
    S, D = tensor.shape
    raw, bpe = to_format_bytes(tensor, fmt)
    if raw is None:
        return None
    out = {"raw_size": len(raw), "entropy_h0": round(byte_entropy(raw), 4), "methods": {}}
    for m in methods:
        t0 = time.perf_counter()
        blob = encode(raw, S, D, bpe, m)
        t1 = time.perf_counter()
        if verify:
            if decode(blob, S, D, bpe, m) != raw:
                raise RuntimeError(f"round-trip mismatch: {fmt} {m}")
        t2 = time.perf_counter()
        comp = payload_size(blob, m)
        out["methods"][m] = {
            "ratio": round(len(raw) / comp, 4),
            "compressed": comp,
            "blob": len(blob),
            "enc_ms": round((t1 - t0) * 1000, 2),
            "dec_ms": round((t2 - t1) * 1000, 2) if verify else None,
        }
    return out


def run_cell(groups, fmt, methods, verify):
    """groups: {'K': [layers], 'V': [...], 'act': [...]} -> per-layer results + aggregates."""
    per_layer = {g: [run_tensor(t, fmt, methods, verify) for t in ts] for g, ts in groups.items()}
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for g, layers in per_layer.items():
        scopes = [g] + (["KV"] if g in ("K", "V") else [])
        for lr in layers:
            if lr is None:
                continue
            for m, r in lr["methods"].items():
                for s in scopes:
                    agg[s][m][0] += lr["raw_size"]
                    agg[s][m][1] += r["compressed"]
    aggregate = {s: {m: round(raw / comp, 4) for m, (raw, comp) in d.items() if comp}
                 for s, d in agg.items()}
    return {"per_layer": per_layer, "aggregate": aggregate}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-7B", "Qwen/Qwen3-8B"],
                    help="HF model ids or local paths")
    ap.add_argument("--ctx", nargs="+", type=int, default=[512, 1024, 2048, 4096, 8192])
    ap.add_argument("--inputs", nargs="+", default=list(INPUT_TYPES), choices=list(INPUT_TYPES))
    ap.add_argument("--formats", nargs="+", default=FORMATS, choices=FORMATS)
    ap.add_argument("--methods", nargs="+", default=None, help="default: all available")
    ap.add_argument("--target", choices=["kv", "act", "both"], default="both")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-verify", action="store_true", help="skip decode round-trip check")
    ap.add_argument("--out", default="results/raw")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kvbytepack.capture import capture

    methods = args.methods or available_methods()
    want_kv = args.target in ("kv", "both")
    want_act = args.target in ("act", "both")
    os.makedirs(args.out, exist_ok=True)

    for model_id in args.models:
        short = os.path.basename(model_id.rstrip("/"))
        print(f"== {short}", flush=True)
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, trust_remote_code=True).to(args.device).eval()

        for ctx in args.ctx:
            for inp in args.inputs:
                key = f"{short}_{ctx}_{inp}"
                cap = capture(model, tok, INPUT_TYPES[inp], ctx, want_kv, want_act)
                groups = {g: cap[g] for g in ("K", "V", "act") if g in cap}
                res = {"model": short, "context_length": cap["tokens"], "input_type": inp,
                       "n_layers": len(next(iter(groups.values()))),
                       "dims": {g: ts[0].shape[-1] for g, ts in groups.items()},
                       "verified": not args.no_verify, "formats": {}}
                for fmt in args.formats:
                    cell = run_cell(groups, fmt, methods, not args.no_verify)
                    if cell["aggregate"]:
                        res["formats"][fmt] = cell
                        a = cell["aggregate"]
                        line = "  ".join(f"{s}={a[s].get('pipeline_zstd-1', 0):.3f}x"
                                         for s in ("KV", "act") if s in a)
                        print(f"  {key:<32} {fmt:<9} pipeline_zstd-1  {line}", flush=True)
                with open(os.path.join(args.out, f"{key}.json"), "w") as f:
                    json.dump(res, f)
                del cap, groups
                gc.collect()
        del model
        gc.collect()


if __name__ == "__main__":
    main()
