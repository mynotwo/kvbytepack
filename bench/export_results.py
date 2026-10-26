#!/usr/bin/env python3
"""Export the pinned public aggregate; no model downloads or third-party imports."""
import argparse
import csv
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import statistics as st

try:
    from .summarize import collect_rows
except ImportError:
    from summarize import collect_rows

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/kv_aggregate_qwen.json"
SOURCE_SHA = "f466824433e8b2de6e71f873a958dde6c896d0af63764fd0ed9dd3c9162606c8"
FORMATS = ("bf16", "fp16", "fp32", "fp8_e4m3", "fp8_e5m2", "int8", "int4", "int2")
METHODS = ("raw_zstd-1", "raw_zstd-9", "raw_zstd-19", "raw_lz4", "raw_snappy",
           "raw_lzma", "raw_brotli", "raw_gzip", "transpose+zstd-1",
           "bytesplit+zstd-1", "pipeline_zstd-1", "pipeline_zstd-9",
           "pipeline_zstd-19", "pipeline_lz4", "pipeline_lzma")


def validate(data):
    expected = set(itertools.product(("Qwen2.5-7B", "Qwen3-8B"),
                   (512, 1024, 2048, 4096, 8192),
                   ("english", "code", "math", "chinese", "json")))
    seen = set()
    for cell_id, cell in data.items():
        key = (cell["model"], cell["context_length"], cell["input_type"])
        if key in seen or cell_id != "_".join(map(str, key)):
            raise ValueError(f"duplicate or mismatched cell: {cell_id}")
        seen.add(key)
        if set(cell["formats"]) != set(FORMATS):
            raise ValueError(f"format coverage: {cell_id}")
        for scopes in cell["formats"].values():
            if set(scopes) != {"K", "V", "KV"}:
                raise ValueError(f"scope coverage: {cell_id}")
            for ratios in scopes.values():
                if set(ratios) != set(METHODS):
                    raise ValueError(f"method coverage: {cell_id}")
                if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                       for v in ratios.values()):
                    raise ValueError(f"invalid ratio: {cell_id}")
    if seen != expected:
        raise ValueError("model/context/input grid coverage")


def csv_bytes(header, rows):
    out = io.StringIO(newline="")
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return out.getvalue().encode()


def export(source=SOURCE):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("archive SHA256 mismatch; do not silently replace the source")
    data = json.loads(raw)
    validate(data)
    rows = []
    for cell_id, cell in sorted(data.items()):
        for fmt, scopes in sorted(cell["formats"].items()):
            for scope, ratios in sorted(scopes.items()):
                for method, ratio in sorted(ratios.items()):
                    rows.append((cell_id, cell["model"], cell["context_length"],
                                 cell["input_type"], "repeated_paragraph", fmt,
                                 scope, method, ratio))
    measurements = csv_bytes(("cell_id", "model", "context_length", "input_type",
                              "input_protocol", "format", "scope", "method",
                              "payload_ratio"), rows)
    summaries = []
    for (scope, fmt), acc in sorted(collect_rows(c["formats"] for c in data.values()).items()):
        summaries.append((scope, fmt, len(acc["best"]), st.median(acc["best"]),
                          min(acc["best"]), max(acc["best"]),
                          st.median(acc["raw_zstd-1"]),
                          st.median(acc["bytesplit+zstd-1"]),
                          st.median(acc["pipeline_zstd-1"]),
                          st.median(acc["lift"]) * 100))
    summary = csv_bytes(("scope", "format", "cells", "best_median", "best_min",
                         "best_max", "raw_zstd1_median", "bytesplit_zstd1_median",
                         "pipeline_zstd1_median", "best_vs_best_raw_median_pct"), summaries)
    files = {"measurements.csv": measurements, "summary.csv": summary}
    manifest = {"schema_version": 1, "source": "results/kv_aggregate_qwen.json",
                "source_commit": "83a478d51d3d12a17d511977ef2af9c5a02bae6d",
                "source_sha256": SOURCE_SHA, "cells": len(data),
                "measurement_rows": len(rows), "summary_rows": len(summaries),
                "ratio_definition": "uncompressed bytes / compressed payload bytes; excludes plane headers",
                "input_protocol": "repeated paragraphs, not independent real-text documents",
                "scope": "historical aggregate statistics; no fresh model capture",
                "absent_evidence": ["original tensor bytes", "per-layer compressed byte counts",
                                    "capture environment and model revisions", "timings",
                                    "per-blob round-trip verification records"],
                "files": {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
    files["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--check", action="store_true", help="fail on missing or changed exports")
    args = parser.parse_args()
    files = export()
    if args.check:
        for name, content in files.items():
            path = args.out_dir / name
            if not path.exists() or path.read_bytes() != content:
                raise SystemExit(f"export mismatch: {path}")
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            (args.out_dir / name).write_bytes(content)
    print(f"{'Checked' if args.check else 'Exported'} 50 cells, 18000 ratios, 8 summaries")


if __name__ == "__main__":
    main()
