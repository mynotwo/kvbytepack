#!/usr/bin/env python3
"""Headline numbers from benchmark output.

For each scope (KV, act) and format: median / range of the best ratio over all
cells, median ratio of raw zstd-1 / bytesplit+zstd-1 / pipeline_zstd-1, and the
median lift of the best method over the best raw (layout-free) codec.

Accepts either a directory of run_bench.py JSON files or the aggregate file in
results/ (dict of cell -> {"formats": {fmt: {scope: {method: ratio}}}}).
"""
import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict


def load_cells(path):
    if os.path.isdir(path):
        for f in sorted(glob.glob(os.path.join(path, "*.json"))):
            d = json.load(open(f))
            yield {fmt: c["aggregate"] for fmt, c in d["formats"].items()}
    else:
        for d in json.load(open(path)).values():
            yield d["formats"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default="results/raw")
    args = ap.parse_args()

    rows = defaultdict(lambda: defaultdict(list))
    for cell in load_cells(args.path):
        for fmt, scopes in cell.items():
            for scope in ("KV", "act"):
                r = scopes.get(scope)
                if not r:
                    continue
                best_raw = max(v for m, v in r.items() if m.startswith("raw_"))
                best = max(r.values())
                acc = rows[(scope, fmt)]
                acc["best"].append(best)
                acc["lift"].append(best / best_raw - 1)
                for m in ("raw_zstd-1", "bytesplit+zstd-1", "pipeline_zstd-1"):
                    if m in r:
                        acc[m].append(r[m])

    print(f"{'scope':<5} {'fmt':<9} {'n':>3} {'best(med)':>9} {'range':>13} "
          f"{'zstd-1':>7} {'bs+zstd':>7} {'pipe':>7} {'lift':>6}")
    for (scope, fmt), a in sorted(rows.items()):
        med = lambda k: f"{st.median(a[k]):.3f}" if a[k] else "-"  # noqa: E731
        rng = f"{min(a['best']):.2f}-{max(a['best']):.2f}"
        print(f"{scope:<5} {fmt:<9} {len(a['best']):>3} {med('best'):>9} {rng:>13} "
              f"{med('raw_zstd-1'):>7} {med('bytesplit+zstd-1'):>7} {med('pipeline_zstd-1'):>7} "
              f"{st.median(a['lift']) * 100:>5.1f}%")


if __name__ == "__main__":
    main()
