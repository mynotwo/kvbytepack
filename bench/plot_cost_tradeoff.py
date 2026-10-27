"""Export/check a serial-transfer cost model; optionally redraw the published figure."""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/cpu_decode"
OUT = ROOT / "results/cost_model"
FIGURE = ROOT / "docs/assets/kv-cost-tradeoff"
PATHS = ("raw_zstd1_control", "layout_record")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def calculate():
    source = (DATA / "measurements.csv").read_bytes()
    manifest = json.loads((DATA / "manifest.json").read_text())
    if digest(source) != manifest["files"]["measurements.csv"]:
        raise ValueError("published measurement checksum mismatch")
    rows = list(csv.DictReader(io.StringIO(source.decode())))
    groups = {}
    for row in rows:
        if row["variant"] != "after" or row["path"] not in PATHS:
            continue
        key = (int(row["tokens"]), int(row["round"]), row["path"])
        cases = groups.setdefault(key, {})
        if row["case_id"] in cases:
            raise ValueError("duplicate model input")
        cpu = (float(row["encode_median_ms"]) + float(row["decode_median_ms"])) / 1000
        if not math.isfinite(cpu) or cpu <= 0 or row["byte_exact"] != "True":
            raise ValueError("invalid time or byte check")
        cases[row["case_id"]] = (int(row["stored_bytes"]), cpu, row["input_sha256"])
    if len(groups) != 2 * manifest["rounds"] * len(PATHS):
        raise ValueError("missing context/round/path")
    summaries, curves = [], []
    for tokens in (128, 512):
        paired = []
        for index in range(manifest["rounds"]):
            baseline, candidate = (groups[tokens, index, p] for p in PATHS)
            if len(baseline) != 48 or baseline.keys() != candidate.keys():
                raise ValueError("incomplete or unpaired inputs")
            if any(baseline[k][2] != candidate[k][2] for k in baseline):
                raise ValueError("input hashes differ")
            bbytes = sum(v[0] for v in baseline.values())
            cbytes = sum(v[0] for v in candidate.values())
            bcpu = sum(v[1] for v in baseline.values())
            ccpu = sum(v[1] for v in candidate.values())
            if not bbytes > cbytes > 0 or not ccpu > bcpu > 0:
                raise ValueError("this frozen study must have byte savings and added CPU cost")
            paired.append((bbytes, cbytes, bcpu, ccpu))
        limits = [(b - c) / (ct - bt) / 1e9 for b, c, bt, ct in paired]
        summaries.append({"tokens": tokens, "payloads": 48, "rounds": len(paired),
                          "baseline_stored_bytes": paired[0][0],
                          "candidate_stored_bytes": paired[0][1],
                          "baseline_cpu_ms": st.median(p[2] * 1000 for p in paired),
                          "candidate_cpu_ms": st.median(p[3] * 1000 for p in paired),
                          "threshold_GB_s_median": st.median(limits),
                          "threshold_GB_s_min": min(limits), "threshold_GB_s_max": max(limits)})
        if any(p[:2] != paired[0][:2] for p in paired):
            raise ValueError("stored bytes changed across rounds")
        for step in range(301):
            bandwidth = 10 ** (-3 + step / 60)
            ratios = [(ct + c / (bandwidth * 1e9)) / (bt + b / (bandwidth * 1e9))
                      for b, c, bt, ct in paired]
            curves.append({"tokens": tokens, "bandwidth_GB_s": bandwidth,
                           "ratio_median": st.median(ratios),
                           "ratio_min": min(ratios), "ratio_max": max(ratios)})
    return summaries, curves


def csv_bytes(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def exports(summaries, curves):
    files = {"summary.csv": csv_bytes(summaries), "curves.csv": csv_bytes(curves)}
    manifest = {"source": "results/cpu_decode/measurements.csv", "variant": "after",
                "source_sha256": digest((DATA / "measurements.csv").read_bytes()),
                "paths": list(PATHS), "formula": "(candidate_cpu_s + candidate_bytes/B) / (baseline_cpu_s + baseline_bytes/B)",
                "bandwidth_unit": "GB/s decimal; B is bytes/second", "rounds": 12,
                "aggregation": "per round sum of 48 per-payload median encode+decode times; median/min/max of 12 round cost ratios",
                "limits": "serial non-overlapped warm CPU encode-transfer-decode model; baseline is a bare Zstd frame without KBL1 metadata or its SHA256 check; candidate includes that envelope and validation; not measured network/serving performance; range is not a confidence interval",
                "files": {name: digest(data) for name, data in files.items()}}
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    return files


def plot(summaries, curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "svg.hashsalt": "kvbytepack-cost-model-v1"})
    fig, ax = plt.subplots(figsize=(9.2, 5.8), layout="constrained")
    ax.axhspan(.8, 1, color="#e9f3ef")
    ax.axhspan(1, 2.3, color="#f7f0eb")
    for tokens, color in ((128, "#256a8a"), (512, "#ae5d35")):
        subset = [r for r in curves if r["tokens"] == tokens]
        x = [r["bandwidth_GB_s"] for r in subset]
        ax.plot(x, [r["ratio_median"] for r in subset], color=color,
                linewidth=2.4, label=f"{tokens}-token prefix")
        ax.fill_between(x, [r["ratio_min"] for r in subset],
                        [r["ratio_max"] for r in subset], color=color, alpha=.16)
        s = next(r for r in summaries if r["tokens"] == tokens)
        threshold = s["threshold_GB_s_median"]
        ax.vlines(threshold, .8, 1, color=color, linestyle=":", linewidth=1.5)
        ax.annotate(f"{threshold:.3f} GB/s", xy=(threshold, 1),
                    xytext=(-72 if tokens == 128 else 14, 24 if tokens == 128 else 8),
                    textcoords="offset points", color=color, fontsize=10,
                    arrowprops={"arrowstyle": "-", "color": color})
    ax.axhline(1, color="#444444", linewidth=1, linestyle="--")
    ax.text(16, .92, "Layout record lower cost", ha="right", color="#376b55", fontsize=10)
    ax.text(16, 1.09, "Raw Zstd control lower cost", ha="right", color="#855941", fontsize=10)
    ax.set(xscale="log", xlim=(.001, 100), ylim=(.8, 2.3),
           xlabel="Assumed link bandwidth (GB/s, decimal)",
           ylabel="Modelled serial time / raw-Zstd-control time")
    ax.set_title("When do saved KV bytes offset codec work?", loc="left", fontsize=16, pad=25)
    ax.text(0, 1.02, "Qwen2.5-0.5B BF16 · complete layout record vs bare Zstd level-1 control",
            transform=ax.transAxes, fontsize=10, color="#555555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=.17)
    ax.legend(loc="upper left", frameon=False)
    fig.text(.01, -.10,
             "48 K/V payloads per prefix · Apple M4 Pro · warm CPU calls · shaded: 12-round range, not a confidence interval\n"
             "Candidate includes KBL1 metadata, SHA256 and strict checks; the bare-Zstd control has a different validation contract.\n"
             "Serial encode + transfer + decode, without overlap or queueing. Cost model, not serving measurements.",
             fontsize=9, color="#555555")
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(FIGURE.with_suffix(".svg"), bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--plot", action="store_true", help="redraw PNG/SVG using optional matplotlib")
    args = parser.parse_args()
    summaries, curves = calculate()
    files = exports(summaries, curves)
    if args.check:
        saved = json.loads((OUT / "manifest.json").read_text())
        expected_manifest = json.loads(files["manifest.json"])
        if {k: v for k, v in saved.items() if k != "files"} != {k: v for k, v in expected_manifest.items() if k != "files"}:
            raise ValueError("cost-model provenance differs")
        for name in ("summary.csv", "curves.csv"):
            actual = (OUT / name).read_bytes()
            if digest(actual) != saved["files"][name]:
                raise ValueError(f"derived export checksum differs: {name}")
            a = list(csv.DictReader(io.StringIO(actual.decode())))
            b = list(csv.DictReader(io.StringIO(files[name].decode())))
            if len(a) != len(b) or list(a[0]) != list(b[0]):
                raise ValueError("derived export coverage/columns differ")
            if any(not math.isclose(float(x[k]), float(y[k]), rel_tol=1e-12, abs_tol=1e-12)
                   for x, y in zip(a, b) for k in x):
                raise ValueError(f"derived cost values differ: {name}")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, data in files.items():
            (OUT / name).write_bytes(data)
    if args.plot:
        plot(summaries, curves)
    for row in summaries:
        print(f"{row['tokens']} tokens: median model threshold {row['threshold_GB_s_median']:.6f} GB/s")
    print("Verified derived cost exports" if args.check else "Exported cost-model inputs; no serving claim")


if __name__ == "__main__":
    main()
