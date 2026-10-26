"""CPU contract experiment: synthetic fixtures or a supplied 2-D .npy tensor."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import numpy as np
import zstandard as zstd

from .adapter import Layout, make_compressor

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "tests/fixtures/aibrix_marshallers"
PATHS = ("uncompressed", "aibrix_zstd_default", "raw_zstd1_control", "layout_zstd1_record")


def upstream_classes():
    """Load two hash-verified, unmodified modules; no AIBrix package simulation."""
    provenance = json.loads((UPSTREAM / "provenance.json").read_text())
    for name, entry in provenance["files"].items():
        if hashlib.sha256((UPSTREAM / name).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"upstream fixture checksum mismatch: {name}")
    name = "_kvbytepack_upstream_marshallers"
    spec = importlib.util.spec_from_file_location(name, UPSTREAM / "__init__.py",
                                                submodule_search_locations=[str(UPSTREAM)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.BaseMarshaller, module.ZstdCompressor, provenance


def fixtures():
    for index, (s, d) in enumerate(((8, 8), (512, 128), (2048, 128))):
        layout = Layout(s, d, 2)
        rng = np.random.default_rng(20261026 + index)
        random = rng.integers(0, 256, layout.raw_bytes, dtype=np.uint8).tobytes()
        channels = np.empty((s, d, 2), dtype=np.uint8)
        channels[:, :, 0] = rng.integers(0, 256, (s, d), dtype=np.uint8)
        channels[:, :, 1] = (np.arange(d, dtype=np.uint16) * 17 % 256).astype(np.uint8)
        for kind, raw in (("zeros", bytes(layout.raw_bytes)), ("uniform_bytes", random),
                          ("channel_bytes", channels.tobytes())):
            yield f"{kind}_{s}x{d}", raw, layout, "synthetic"


def codecs(layout, base, zstd_class):
    default = zstd_class()
    control = zstd_class()
    control.compressor = zstd.ZstdCompressor(level=1)
    adapter = make_compressor(base, layout)
    return {"uncompressed": (lambda raw: raw, lambda raw: raw),
            "aibrix_zstd_default": (default.marshal, default.unmarshal),
            "raw_zstd1_control": (control.marshal, control.unmarshal),
            "layout_zstd1_record": (adapter.marshal, adapter.unmarshal)}


def p95(values):
    return sorted(values)[math.ceil(len(values) * 0.95) - 1]


def measure(inputs, repeats, warmups):
    base, zstd_class, upstream = upstream_classes()
    rows, samples = [], []
    for case_id, raw, layout, source in inputs:
        for path, (encode, decode) in codecs(layout, base, zstd_class).items():
            enc, dec, sizes = [], [], set()
            for index in range(warmups + repeats):
                start = time.perf_counter_ns()
                blob = encode(raw)
                middle = time.perf_counter_ns()
                restored = decode(blob)
                end = time.perf_counter_ns()
                if restored != raw:
                    raise RuntimeError(f"byte mismatch: {case_id}/{path}")
                sizes.add(len(blob))
                if index >= warmups:
                    enc.append((middle - start) / 1e6)
                    dec.append((end - middle) / 1e6)
            if len(sizes) != 1:
                raise RuntimeError("non-deterministic encoded size")
            stored = sizes.pop()
            rows.append({"case_id": case_id, "input_source": source, "path": path,
                         "tokens": layout.tokens, "channels": layout.channels,
                         "bytes_per_element": layout.bytes_per_element,
                         "input_sha256": hashlib.sha256(raw).hexdigest(),
                         "raw_bytes": len(raw), "stored_bytes": stored,
                         "full_blob_ratio": len(raw) / stored,
                         "encode_median_ms": statistics.median(enc),
                         "encode_p95_ms": p95(enc),
                         "decode_median_ms": statistics.median(dec),
                         "decode_p95_ms": p95(dec), "repeats": repeats,
                         "byte_exact": True})
            samples.append({"case_id": case_id, "path": path,
                            "encode_ms": enc, "decode_ms": dec})
    manifest = {"schema": 1, "python": platform.python_version(),
                "platform": platform.platform(), "processor": platform.processor(),
                "numpy": np.__version__, "zstandard": zstd.__version__,
                "zstd_runtime": list(zstd.ZSTD_VERSION), "clock": "perf_counter_ns",
                "repeats": repeats, "warmups": warmups, "case_count": len(rows) // 4,
                "paths": list(PATHS), "upstream": upstream,
                "baseline": "unmodified upstream ZstdCompressor; additional raw control sets level=1",
                "layout_costs": "existing tensor envelope + plane headers + KBL1 magic + SHA256; all counted",
                "limits": ["isolated upstream module contract, not a full AIBrix installation",
                           "CPU, no serving/transfer overlap or GPU measurement",
                           "synthetic cases are not model-captured KV data",
                           "timings are machine-specific and not expected to reproduce exactly"]}
    return rows, samples, manifest


def write_results(out, rows, samples, manifest):
    out.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    files = {"measurements.csv": stream.getvalue().encode(),
             "samples.json": (json.dumps(samples, indent=2) + "\n").encode()}
    manifest["files"] = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
    for name, content in files.items():
        (out / name).write_bytes(content)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def check_rows(out, expected, input_source, files=("measurements.csv", "samples.json")):
    """Recheck sizes/round trips and sample summaries, not historical speed."""
    manifest = json.loads((out / "manifest.json").read_text())
    if set(manifest["files"]) != set(files):
        raise ValueError("result file coverage mismatch")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((out / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"result checksum mismatch: {name}")
    base, zstd_class, provenance = upstream_classes()
    if provenance != manifest["upstream"]:
        raise ValueError("upstream provenance mismatch")
    with (out / "measurements.csv").open() as file:
        rows = list(csv.DictReader(file))
    samples = json.loads((out / "samples.json").read_text())
    sample_map = {(s["case_id"], s["path"]): s for s in samples}
    if (len(rows) != len(expected) or manifest["case_count"] != len(expected) // 4
            or len(samples) != len(expected) or set(sample_map) != set(expected)
            or manifest["paths"] != list(PATHS)):
        raise ValueError("case/path coverage mismatch")
    seen = set()
    for row in rows:
        key = (row["case_id"], row["path"])
        if key not in expected or key in seen:
            raise ValueError("unknown or duplicate case/path")
        seen.add(key)
        raw, layout = expected[key]
        if (row["input_source"] != input_source or int(row["tokens"]) != layout.tokens
                or int(row["channels"]) != layout.channels
                or int(row["bytes_per_element"]) != layout.bytes_per_element
                or int(row["repeats"]) != manifest["repeats"]):
            raise ValueError(f"input metadata mismatch: {key}")
        for field in ("encode", "decode"):
            values = sample_map[key][field + "_ms"]
            if (len(values) != manifest["repeats"] or not values
                    or any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in values)
                    or float(row[field + "_median_ms"]) != statistics.median(values)
                    or float(row[field + "_p95_ms"]) != p95(values)):
                raise ValueError(f"timing/sample mismatch: {key}")
        encode, decode = codecs(layout, base, zstd_class)[key[1]]
        blob = encode(raw)
        if (row["input_sha256"] != hashlib.sha256(raw).hexdigest()
                or int(row["raw_bytes"]) != len(raw)
                or int(row["stored_bytes"]) != len(blob)
                or float(row["full_blob_ratio"]) != len(raw) / len(blob)
                or row["byte_exact"] != "True" or decode(blob) != raw):
            raise ValueError(f"size or round-trip mismatch: {key}")
    print(f"Checked {len(expected)} {input_source} case/path rows; no historical timing claim")


def check_results(out):
    expected = {(case_id, path): (raw, layout)
                for case_id, raw, layout, source in fixtures() for path in PATHS}
    check_rows(out, expected, "synthetic")


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/aibrix-layout"))
    parser.add_argument("--repeats", type=positive, default=11)
    parser.add_argument("--warmups", type=positive, default=2)
    parser.add_argument("--input-npy", type=Path, help="own fixed-width 2-D payload; never labeled model-captured")
    parser.add_argument("--check", type=Path, metavar="RESULT_DIR")
    args = parser.parse_args()
    if args.check:
        check_results(args.check)
        return
    inputs = list(fixtures())
    if args.input_npy:
        array = np.load(args.input_npy, allow_pickle=False)
        if array.ndim != 2 or array.dtype.hasobject or array.dtype.fields:
            raise ValueError("expected a non-object, unstructured 2-D array")
        layout = Layout(*array.shape, array.itemsize)
        inputs = [("supplied_array", np.ascontiguousarray(array).tobytes(), layout, "user_supplied_npy")]
    rows, samples, manifest = measure(inputs, args.repeats, args.warmups)
    write_results(args.out_dir, rows, samples, manifest)
    print(f"Wrote {len(rows)} byte-exact rows to {args.out_dir}")


if __name__ == "__main__":
    main()
