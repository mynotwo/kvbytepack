#!/usr/bin/env python3
"""Alternate isolated baseline/candidate processes on the pinned 96 real KV payloads."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics as st
import subprocess
import sys
import tempfile
import time

BASELINE = "98f56cc16fe46195d70730e99ac64b754d5f75ad"
DATA_SHA = "8596ac9a5b7475ee61b26c72847966ec846cb02ccee934868725db298143c211"
PATHS = ("pipeline_codec", "numpy_tensor", "layout_record", "raw_zstd1_control")
SOURCE_FILES = ("kvbytepack/codec.py", "kvbytepack/tensor.py", "examples/aibrix_layout/adapter.py",
                "kvbytepack/_decode.c", "setup.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def percentile(values):
    return sorted(values)[math.ceil(len(values) * .95) - 1]


def worker(root, data, repeats, warmups, seed):
    sys.path.insert(0, str(root))
    import numpy as np
    import zstandard as zstd
    import kvbytepack
    from kvbytepack import codec
    from examples.aibrix_layout import adapter
    from examples.aibrix_layout.capture import check
    if root not in Path(kvbytepack.__file__).resolve().parents or root not in Path(adapter.__file__).resolve().parents:
        raise ValueError("worker imported the wrong checkout")
    if sha(data / "kv_payloads.npz") != DATA_SHA:
        raise ValueError("not the pinned real KV data")
    check(data)
    rows, samples = [], []
    with np.load(data / "kv_payloads.npz", allow_pickle=False) as arrays:
        names = sorted(arrays.files)
        random.Random(seed).shuffle(names)
        for position, name in enumerate(names):
            array = arrays[name]
            raw = array.tobytes()
            layout = adapter.Layout(*array.shape, array.itemsize)
            opaque = np.frombuffer(raw, dtype=f"V{array.itemsize}").reshape(array.shape)
            compressor, decompressor = zstd.ZstdCompressor(level=1), zstd.ZstdDecompressor()
            operations = {
                "pipeline_codec": (lambda: kvbytepack.encode(raw, *array.shape, array.itemsize, "pipeline_zstd-1"),
                                   lambda b: kvbytepack.decode(b, *array.shape, array.itemsize, "pipeline_zstd-1")),
                "numpy_tensor": (lambda: kvbytepack.zip_tensor(opaque), kvbytepack.unzip_tensor),
                "layout_record": (lambda: adapter.pack(raw, layout), lambda b: adapter.unpack(b, layout)),
                "raw_zstd1_control": (lambda: compressor.compress(raw), decompressor.decompress)}
            paths = list(PATHS)
            random.Random(seed + position).shuffle(paths)
            for path in paths:
                encode, decode = operations[path]
                enc, dec = [], []
                blob_sha = None
                for index in range(warmups + repeats):
                    start = time.perf_counter_ns()
                    blob = encode()
                    middle = time.perf_counter_ns()
                    restored = decode(blob)
                    end = time.perf_counter_ns()
                    if (restored.tobytes() if isinstance(restored, np.ndarray) else restored) != raw:
                        raise ValueError(f"byte mismatch: {name}/{path}")
                    digest = hashlib.sha256(blob).hexdigest()
                    if blob_sha is not None and digest != blob_sha:
                        raise ValueError("non-deterministic encoded bytes")
                    blob_sha = digest
                    if index >= warmups:
                        enc.append((middle - start) / 1e6)
                        dec.append((end - middle) / 1e6)
                rows.append({"case_id": name, "path": path, "tokens": array.shape[0],
                             "raw_bytes": len(raw), "stored_bytes": len(blob),
                             "input_sha256": hashlib.sha256(raw).hexdigest(),
                             "encoded_sha256": blob_sha, "byte_exact": True,
                             "encode_median_ms": st.median(enc), "decode_median_ms": st.median(dec),
                             "encode_p95_ms": percentile(enc), "decode_p95_ms": percentile(dec)})
                samples.append({"case_id": name, "path": path, "encode_ms": enc, "decode_ms": dec})
    native = getattr(codec, "_native_merge", None)
    native_file = Path(sys.modules[native.__module__].__file__).resolve() if native else None
    if native_file and root not in native_file.parents:
        raise ValueError("native kernel imported from the wrong checkout")
    return {"rows": rows, "samples": samples,
            "source": {"commit": git(root, "rev-parse", "HEAD") if (root / ".git").exists() else None,
                       "dirty": bool(git(root, "status", "--porcelain", "--untracked-files=no"))
                       if (root / ".git").exists() else None,
                       "files": {name: sha(root / name) if (root / name).exists() else None for name in SOURCE_FILES},
                       "backend": "c-8x8" if native else "numpy",
                       "binary_sha256": sha(native_file) if native_file else None},
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "machine": platform.machine(), "numpy": np.__version__,
                            "zstandard": zstd.__version__, "zstd_runtime": list(zstd.ZSTD_VERSION),
                            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None}}


def summarize(rows):
    result = []
    for tokens in (128, 512):
        for path in PATHS:
            for phase in ("encode", "decode"):
                totals = {}
                for r in rows:
                    if r["tokens"] == tokens and r["path"] == path:
                        key = (r["round"], r["variant"])
                        totals[key] = totals.get(key, 0) + r[phase + "_median_ms"]
                rounds = sorted({key[0] for key in totals})
                before = [totals[(i, "before")] for i in rounds]
                after = [totals[(i, "after")] for i in rounds]
                result.append({"tokens": tokens, "path": path, "phase": phase,
                               "payloads": 48, "rounds": len(rounds),
                               "before_sum_medians_ms": st.median(before),
                               "after_sum_medians_ms": st.median(after),
                               "paired_speedup_median": st.median(b / a for b, a in zip(before, after)),
                               "paired_speedup_min": min(b / a for b, a in zip(before, after)),
                               "paired_speedup_max": max(b / a for b, a in zip(before, after))})
    return result


def csv_data(rows):
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue().encode()


def run(args):
    root = Path(__file__).resolve().parents[1]
    baseline, data = args.baseline_dir.resolve(), args.data_dir.resolve()
    from kvbytepack.codec import _native_merge
    if _native_merge is None:
        raise ValueError("build the optional native kernel before timing this experiment")
    if git(baseline, "rev-parse", "HEAD") != BASELINE or git(baseline, "status", "--porcelain"):
        raise ValueError("baseline must be a clean checkout of the pinned release commit")
    if args.rounds % 2:
        raise ValueError("use an even round count for balanced before/after ordering")
    all_rows, all_samples, sources, environments = [], [], {}, []
    with tempfile.TemporaryDirectory(prefix="kvbytepack-cpu-") as temp:
        for index in range(args.rounds):
            order = ("before", "after") if index % 2 == 0 else ("after", "before")
            paired = {}
            for variant in order:
                output = Path(temp) / f"{index}-{variant}.json"
                command = [sys.executable, str(Path(__file__).resolve()), "--worker-root",
                           str(baseline if variant == "before" else root), "--data-dir", str(data),
                           "--repeats", str(args.repeats), "--warmups", str(args.warmups),
                           "--seed", str(20261026 + index), "--worker-output", str(output)]
                completed = subprocess.run(command, capture_output=True, text=True)
                if completed.returncode:
                    raise RuntimeError(completed.stdout + completed.stderr)
                result = json.loads(output.read_text())
                if variant in sources and result["source"] != sources[variant]:
                    raise ValueError("source changed during measurement")
                sources[variant] = result["source"]
                if environments:
                    for field in ("python", "platform", "machine", "numpy", "zstandard", "zstd_runtime"):
                        if result["environment"][field] != environments[0][field]:
                            raise ValueError("measurement environment changed")
                environments.append({"round": index, "variant": variant, **result["environment"]})
                paired[variant] = {(r["case_id"], r["path"]): r for r in result["rows"]}
                for row in result["rows"]:
                    all_rows.append({"round": index, "order": "/".join(order), "variant": variant, **row})
                for sample in result["samples"]:
                    all_samples.append({"round": index, "variant": variant, **sample})
            if paired["before"].keys() != paired["after"].keys() or len(paired["before"]) != 384:
                raise ValueError("missing case/path coverage")
            for key, before in paired["before"].items():
                after = paired["after"][key]
                for field in ("input_sha256", "encoded_sha256", "stored_bytes", "raw_bytes"):
                    if before[field] != after[field]:
                        raise ValueError(f"wire or data changed: {key}/{field}")
            print(f"round {index + 1}/{args.rounds}: 384 before/after pairs byte-identical", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = {"measurements.csv": csv_data(all_rows), "summary.csv": csv_data(summarize(all_rows)),
             "samples.json": (json.dumps(all_samples, indent=2) + "\n").encode()}
    for name, content in files.items():
        (args.out_dir / name).write_bytes(content)
    manifest = {"schema": 1, "baseline": BASELINE, "payloads_sha256": DATA_SHA,
                "driver_sha256": sha(Path(__file__)),
                "rounds": args.rounds, "repeats": args.repeats, "warmups": args.warmups,
                "sources": sources, "environments": environments,
                "protocol": "isolated sequential processes; AB/BA alternating rounds; same seeded payload/path shuffle",
                "scope": "96 stored BF16 KV payloads, one 0.5B model, overlapping 128/512-token prefixes",
                "metric": "sum of 48 per-payload medians per context/path/round; median paired speedup across rounds",
                "limits": "CPU codec calls, not serving/transfer throughput; no universal performance threshold",
                "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def check_results(out, data):
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["baseline"] != BASELINE or manifest["payloads_sha256"] != DATA_SHA:
        raise ValueError("wrong baseline or real dataset")
    if manifest["sources"]["before"]["commit"] != BASELINE or manifest["sources"]["before"]["dirty"]:
        raise ValueError("invalid measured baseline source")
    if set(manifest["files"]) != {"measurements.csv", "summary.csv", "samples.json"}:
        raise ValueError("missing result files")
    for name, digest in manifest["files"].items():
        if sha(out / name) != digest:
            raise ValueError(f"result checksum mismatch: {name}")
    root = Path(__file__).resolve().parents[1]
    if manifest["sources"]["after"]["files"] != {name: sha(root / name) if (root / name).exists() else None for name in SOURCE_FILES}:
        raise ValueError("current codec files differ from the measured implementation")
    current = worker(root, data, repeats=1, warmups=1, seed=0)
    references = {(r["case_id"], r["path"]): r for r in current["rows"]}
    rounds, repeats = manifest["rounds"], manifest["repeats"]
    if type(rounds) is not int or rounds <= 0 or rounds % 2 or type(repeats) is not int or repeats <= 0:
        raise ValueError("invalid round/repetition count")
    expected = {(i, v, *key) for i in range(rounds) for v in ("before", "after") for key in references}
    samples = json.loads((out / "samples.json").read_text())
    sample_map = {(s["round"], s["variant"], s["case_id"], s["path"]): s for s in samples}
    if set(sample_map) != expected or len(samples) != len(expected):
        raise ValueError("sample coverage/duplicates")
    with (out / "measurements.csv").open() as file:
        rows = list(csv.DictReader(file))
    seen = set()
    for row in rows:
        for field in ("round", "tokens", "raw_bytes", "stored_bytes"):
            row[field] = int(row[field])
        key = (row["round"], row["variant"], row["case_id"], row["path"])
        if key not in expected or key in seen:
            raise ValueError("unknown or duplicate measurement")
        seen.add(key)
        reference = references[(row["case_id"], row["path"])]
        for field in ("tokens", "raw_bytes", "stored_bytes", "input_sha256", "encoded_sha256"):
            if row[field] != reference[field]:
                raise ValueError(f"wire/data mismatch: {field}")
        if row["byte_exact"] != "True" or row["order"] != ("before/after" if row["round"] % 2 == 0 else "after/before"):
            raise ValueError("round trip or balanced-order metadata")
        for phase in ("encode", "decode"):
            values = sample_map[key][phase + "_ms"]
            if len(values) != repeats or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("invalid timing samples")
            for name, value in (("median", st.median(values)), ("p95", percentile(values))):
                field = phase + "_" + name + "_ms"
                row[field] = float(row[field])
                if row[field] != value:
                    raise ValueError("timing summary does not match samples")
    if seen != expected or len(rows) != len(expected):
        raise ValueError("incomplete measurements")
    if csv_data(summarize(rows)) != (out / "summary.csv").read_bytes():
        raise ValueError("paired summary mismatch")
    print("Current verification backend:", current["source"]["backend"])
    print(f"Checked {len(rows)} before/after rows, original bytes and unchanged encodings; historical speed not rerun")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=root / "results/aibrix_layout/model_capture")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cpu-comparison"))
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--check", type=Path, metavar="RESULT_DIR")
    parser.add_argument("--worker-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=20261026, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.rounds, args.repeats, args.warmups) <= 0:
        parser.error("rounds/repeats/warmups must be positive")
    if args.check:
        check_results(args.check, args.data_dir.resolve())
    elif args.worker_root:
        result = worker(args.worker_root.resolve(), args.data_dir.resolve(), args.repeats, args.warmups, args.seed)
        args.worker_output.write_text(json.dumps(result))
    elif args.baseline_dir:
        run(args)
    else:
        parser.error("provide --baseline-dir")


if __name__ == "__main__":
    main()
