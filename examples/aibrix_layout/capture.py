"""Pinned, small-model CPU capture; offline checking does not import PyTorch."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .adapter import Layout
from .benchmark import PATHS, ROOT, check_rows, measure, write_results

MODEL = "Qwen/Qwen2.5-0.5B"
REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
CONTEXTS = (128, 512)
SOURCE = ROOT / "results/aibrix_layout/model_capture"
PROMPT_SHA = "7cc582dcb3b76e2c957f78a3f74e9809d6c311d6aab7bcaa2c9954d0cbff5f30"
PROMPT_URL = "https://github.com/python/cpython/blob/3bb231a6a5dc02b95658877318bf61501a7209e9/Lib/argparse.py"


def check(out):
    capture = json.loads((out / "capture.json").read_text())
    if (capture["model"] != MODEL or capture["revision"] != REVISION
            or capture["contexts"] != list(CONTEXTS) or capture["layers"] != 24
            or capture["dtype"] != "bfloat16" or capture["device"] != "cpu"
            or capture["prompt_url"] != PROMPT_URL or capture["attention"] != "eager"
            or capture["prompt_sha256"] != PROMPT_SHA
            or hashlib.sha256((out / "prompt.txt").read_bytes()).hexdigest() != PROMPT_SHA):
        raise ValueError("capture provenance/coverage mismatch")
    expected = {}
    keys = {f"ctx{ctx}_{scope}_{layer:02d}"
            for ctx in CONTEXTS for scope in ("K", "V") for layer in range(24)}
    with np.load(out / "kv_payloads.npz", allow_pickle=False) as payloads:
        if len(payloads.files) != len(keys) or set(payloads.files) != keys:
            raise ValueError("missing/duplicate layer, scope or context")
        for key in sorted(keys):
            array = payloads[key]
            ctx = int(key.split("_")[0][3:])
            if array.dtype != np.dtype("<u2") or array.shape != (ctx, 128):
                raise ValueError("unexpected BF16 payload layout")
            raw, layout = array.tobytes(), Layout(ctx, 128, 2)
            for path in PATHS:
                expected[(key, path)] = raw, layout
    for ctx in CONTEXTS:
        if (set(capture["input_ids"]) != {"128", "512"}
                or len(capture["input_ids"][str(ctx)]) != ctx
                or any(type(token) is not int or token < 0 for token in capture["input_ids"][str(ctx)])):
            raise ValueError("actual input token count mismatch")
    if capture["input_ids"]["512"][:128] != capture["input_ids"]["128"]:
        raise ValueError("context prefixes do not share the claimed source")
    check_rows(out, expected, "model_captured_bf16", files=("measurements.csv", "samples.json",
               "kv_payloads.npz", "capture.json", "prompt.txt", "PYTHON-LICENSE"))


def run(out):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from kvbytepack.capture import _iter_kv, _to_sd

    prompt = (SOURCE / "prompt.txt").read_bytes()
    if hashlib.sha256(prompt).hexdigest() != PROMPT_SHA:
        raise ValueError("pinned prompt checksum mismatch")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("prompt.txt", "PYTHON-LICENSE"):
        source = SOURCE / name
        if source.resolve() != (out / name).resolve():
            shutil.copy2(source, out / name)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, token=False,
                                             trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, token=False, trust_remote_code=False,
        use_safetensors=True, dtype=torch.bfloat16, attn_implementation="eager").cpu().eval()
    if model.config.num_hidden_layers != 24:
        raise ValueError("model layer count differs from pinned config")
    metadata = {"model": MODEL, "revision": REVISION, "contexts": list(CONTEXTS),
                "layers": 24, "dtype": "bfloat16", "device": "cpu", "torch_threads": 4,
                "torch": torch.__version__, "transformers": transformers.__version__,
                "attention": "eager", "model_eval": True,
                "prompt_url": PROMPT_URL, "prompt_sha256": PROMPT_SHA,
                "input_protocol": "non-repeated prefixes of one public CPython source file",
                "input_ids": {}, "capture_seconds": {},
                "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in (Path(__file__), Path(__file__).with_name("adapter.py"),
                                          Path(__file__).with_name("benchmark.py"))}}
    arrays = {}
    for ctx in CONTEXTS:
        inputs = tokenizer(prompt.decode(), return_tensors="pt", truncation=True,
                           max_length=ctx, add_special_tokens=False)
        if inputs["input_ids"].shape[1] != ctx:
            raise ValueError("tokenizer did not reach the required context length")
        metadata["input_ids"][str(ctx)] = inputs["input_ids"][0].tolist()
        start = time.perf_counter()
        with torch.inference_mode():
            result = model(**inputs, use_cache=True)
        metadata["capture_seconds"][str(ctx)] = time.perf_counter() - start
        pairs = list(_iter_kv(result.past_key_values))
        if len(pairs) != 24:
            raise ValueError("cache layer coverage mismatch")
        for layer, pair in enumerate(pairs):
            for scope, tensor in zip(("K", "V"), pair):
                if tensor.dtype != torch.bfloat16:
                    raise ValueError("unexpected cache dtype; no silent conversion")
                sd = _to_sd(tensor).contiguous()
                if sd.shape != (ctx, 128):
                    raise ValueError("unexpected cache geometry")
                arrays[f"ctx{ctx}_{scope}_{layer:02d}"] = sd.view(torch.int16).numpy().view("<u2").copy()
        del result, pairs
        print(f"Captured all 24 K/V layers at {ctx} tokens", flush=True)
    del model
    gc.collect()
    np.savez_compressed(out / "kv_payloads.npz", **arrays)
    (out / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    inputs = [(key, a.tobytes(), Layout(*a.shape, a.itemsize), "model_captured_bf16")
              for key, a in sorted(arrays.items())]
    rows, samples, manifest = measure(inputs, repeats=11, warmups=2)
    manifest["limits"] = ["one 0.5B model and one code file, with overlapping context prefixes",
                          "CPU capture, all 24 layers' K/V; not independent documents or runs",
                          "isolated marshaller contract; no complete AIBrix or serving measurement",
                          "timings are machine-specific; no GPU/production acceleration claim"]
    write_results(out, rows, samples, manifest)
    manifest = json.loads((out / "manifest.json").read_text())
    for name in ("kv_payloads.npz", "capture.json", "prompt.txt", "PYTHON-LICENSE"):
        manifest["files"][name] = hashlib.sha256((out / name).read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    check(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/aibrix-model-capture"))
    parser.add_argument("--check", type=Path)
    args = parser.parse_args()
    if args.check:
        check(args.check)
    else:
        run(args.out_dir)


if __name__ == "__main__":
    main()
