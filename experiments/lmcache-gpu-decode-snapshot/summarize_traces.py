"""Recompute stage summaries from raw Kineto traces and benchmark JSON."""
from __future__ import annotations

import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def median(values: list[float]) -> float:
    return statistics.median(values)


def main() -> None:
    profile = json.loads((ROOT / "profile-result.json").read_text())
    old = json.loads((ROOT / "previous-baseline.json").read_text())
    exact = json.loads((ROOT / "original-method-rerun.json").read_text())
    assert old["source_sha256"] == exact["source_sha256"] == profile["source_sha256"]
    assert old["corpus_sha256"] == exact["corpus_sha256"] == profile["corpus_sha256"]
    record_hashes = lambda run: [r["raw_sha256"] for r in run["records"]]
    assert record_hashes(old) == record_hashes(exact) == record_hashes(profile)
    profile_med = {mode: median(profile["uninstrumented_wall_ms"][mode]) for mode in ("cpu", "gpu")}
    old_med = {"cpu": median(old["cpu_samples_ms"]), "gpu": median(old["gpu_samples_ms"])}
    exact_med = {"cpu": median(exact["cpu_samples_ms"]), "gpu": median(exact["gpu_samples_ms"])}

    stage = {}
    for mode in ("gpu", "cpu"):
        groups = [s for s in profile["event_only_samples"] if s["mode"] == mode]
        host_values: dict[str, list[float]] = defaultdict(list)
        span_values: dict[str, list[float]] = defaultdict(list)
        for group in groups:
            for name, value in group["stage_host_wall_ms"].items():
                host_values[name].append(value)
            by_name: dict[str, float] = defaultdict(float)
            for span in group["stage_cuda_event_spans_ms"]:
                by_name[span["stage"]] += span["elapsed_ms"]
            for name, value in by_name.items():
                span_values[name].append(value)
        stage[mode] = {
            "event_only_wall_ms_samples": [g["wall_ms"] for g in groups],
            "event_only_wall_ms_median": median([g["wall_ms"] for g in groups]),
            "host_call_ms_medians": {k: median(v) for k, v in host_values.items()},
            "stream_span_ms_medians_by_group": {k: median(v) for k, v in span_values.items()},
        }

    active = {}
    for mode in ("gpu", "cpu"):
        trace = json.loads((ROOT / f"{mode}-trace.json").read_text())
        copies = [e for e in trace["traceEvents"] if e.get("cat") == "gpu_memcpy"]
        h2d = [e for e in copies if "HtoD" in e["name"]]
        non_h2d = [e for e in copies if "HtoD" not in e["name"]]
        if len(non_h2d):
            raise ValueError(f"unexpected non-HtoD copy events for {mode}: {[e['name'] for e in non_h2d]}")
        payload = [e for e in copies if e["args"]["bytes"] > 10_000]
        small = [e for e in copies if e["args"]["bytes"] <= 10_000]
        expected_payload = sorted(r["compressed_bytes"] for r in profile["records"]) if mode == "gpu" else sorted(r["raw_bytes"] for r in profile["records"])
        if sorted(e["args"]["bytes"] for e in payload) != expected_payload:
            raise ValueError(f"payload H2D sizes do not match records for {mode}")
        if mode == "gpu" and sorted(e["args"]["bytes"] for e in small) != [768, 768, 1536, 1536, 1536, 1536]:
            raise ValueError("nvCOMP descriptor H2D sizes differ from this fixture's expected arrays")
        active[mode] = {
            "memcpy": {
                "h2d_count": len(h2d),
                "h2d_bytes": sum(e["args"]["bytes"] for e in h2d),
                "h2d_active_us": sum(e["dur"] for e in h2d),
                "payload_bytes": sum(e["args"]["bytes"] for e in payload),
                "payload_active_us": sum(e["dur"] for e in payload),
                "small_descriptor_bytes": sum(e["args"]["bytes"] for e in small),
                "small_descriptor_active_us": sum(e["dur"] for e in small),
                "d2h_or_d2d_count": len(non_h2d),
            },
        }
        if mode == "gpu":
            memset = [e for e in trace["traceEvents"] if e.get("cat") == "gpu_memset"]
            kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
            by_name: dict[str, dict[str, float | int]] = {}
            for event in kernels:
                item = by_name.setdefault(event["name"], {"count": 0, "active_us": 0.0})
                item["count"] += 1
                item["active_us"] += event["dur"]
            zstd = {k: v for k, v in by_name.items() if "zstd::" in k}
            merge = {k: v for k, v in by_name.items() if "direct_copy_kernel_cuda" in k}
            main_decode = next(v["active_us"] for k, v in zstd.items() if "decompression_kernel" in k)
            active[mode]["memset"] = {
                "count": len(memset),
                "bytes": sum(e.get("args", {}).get("bytes", 0) for e in memset),
                "active_us": sum(e["dur"] for e in memset),
            }
            active[mode]["kernels"] = {
                "zstd_kernel_active_us": sum(v["active_us"] for v in zstd.values()),
                "main_zstd_decompression_kernel_active_us": main_decode,
                "zstd_setup_kernel_active_us": sum(v["active_us"] for v in zstd.values()) - main_decode,
                "zstd_by_name": zstd,
                "byte_merge_kernel_active_us": sum(v["active_us"] for v in merge.values()),
                "byte_merge_kernel_count": sum(v["count"] for v in merge.values()),
                "all_kernel_active_us": sum(e["dur"] for e in kernels),
            }

    kineto_wall = {}
    for result in profile["profiled_samples"]:
        mode = result["mode"]
        kineto_wall[mode] = {
            "profiled_wall_ms": result["profiled_wall_ms"],
            "profile_helper_uninstrumented_median_ms": profile_med[mode],
            "wall_instrumentation_overhead_pct": (result["profiled_wall_ms"] / profile_med[mode] - 1) * 100,
        }

    result = {
        "helper_source_sha256": profile["source_sha256"],
        "corpus_sha256": profile["corpus_sha256"],
        "profile_script_sha256": hashlib.sha256((ROOT / "profile_helper.py").read_bytes()).hexdigest(),
        "original_benchmark_sha256": hashlib.sha256((ROOT / "bench_codec_core.executed.py").read_bytes()).hexdigest(),
        "old_exact_benchmark_medians_ms": old_med,
        "old_exact_cpu_over_gpu_ratio": old_med["cpu"] / old_med["gpu"],
        "current_exact_original_script_medians_ms": exact_med,
        "current_exact_cpu_over_gpu_ratio": exact_med["cpu"] / exact_med["gpu"],
        "profile_helper_uninstrumented_medians_ms": profile_med,
        "profile_helper_cpu_over_gpu_ratio": profile_med["cpu"] / profile_med["gpu"],
        "first_call_ms": profile["uninstrumented_first_group_ms"],
        "profile_stage_observations": stage,
        "kineto_device_activity": active,
        "kineto_wall_instrumentation": kineto_wall,
        "sampling": {"warmups": 2, "timed_repetitions": 12, "profile_event_groups_per_mode": 4},
    }
    (ROOT / "derived-summary.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
