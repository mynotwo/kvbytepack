"""Profile frozen byte-split helper stages without changing its source."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
import zstandard
from torch.profiler import ProfilerActivity, profile, record_function

TASK = Path("/home/audreymao/kvbytepack_mp_proto_20261103/codec_pr_test")
OLD = Path("/home/audreymao/kvbytepack_mp_proto_20261103/lmcache-source/lmcache")
CORPUS = TASK / "kv_payloads.npz"
sys.path.insert(0, str(TASK))
import lmcache

lmcache.__path__.append(str(OLD))
import nvidia

nvidia.__path__.append(str(Path(sys.prefix) / "lib/python3.11/site-packages/nvidia"))
from nvidia import nvcomp

from lmcache import torch_dev
from lmcache.v1.distributed.compress_adapters.byte_split_zstd import (
    decode_zstd_byte_split_to_lease,
    merge_byte_planes,
    split_bf16_byte_planes,
)
from lmcache.v1.distributed.compress_adapters.device import DeviceExecutionContext
from lmcache.v1.distributed.compress_adapters.lease import DeviceOutputLeaseManager
from lmcache.v1.platform import current_device_spec


def build_records():
    data = np.load(CORPUS)
    raw_records = [("ctx128", np.stack([
        np.stack([data[f"ctx128_K_{i:02d}"] for i in range(24)]),
        np.stack([data[f"ctx128_V_{i:02d}"] for i in range(24)]),
    ]).tobytes(order="C"))]
    for label, sl in (("ctx512_first256", slice(0, 256)), ("ctx512_last256", slice(256, 512))):
        raw_records.append((label, np.stack([
            np.stack([data[f"ctx512_K_{i:02d}"][sl] for i in range(24)]),
            np.stack([data[f"ctx512_V_{i:02d}"][sl] for i in range(24)]),
        ]).tobytes(order="C")))
    compressor = zstandard.ZstdCompressor(level=1)
    result = []
    for name, raw in raw_records:
        planes = split_bf16_byte_planes(raw)
        frames = tuple(tuple(compressor.compress(p[i:i + 16384]) for i in range(0, len(p), 16384)) for p in planes)
        sizes = tuple(tuple(min(16384, len(p) - i) for i in range(0, len(p), 16384)) for p in planes)
        result.append((name, raw, frames, sizes))
    return result


class Context:
    def __init__(self, output):
        self.output, self.device = output, output.device
        self.stream = current_device_spec.current_stream(self.device)

    def get_temp_object_group_buffer(self, batch_idx, object_group_idx):
        assert (batch_idx, object_group_idx) == (0, 0)
        return self.output


def sync_stream(stream):
    current_device_spec.synchronize_stream(stream)


def gpu_snapshot():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ], text=True).strip()
    return [line.strip() for line in raw.splitlines() if line.strip()]


def main():
    device = torch.device("cuda", torch_dev.current_device())
    records = build_records()
    outputs = []
    streams = []
    leases = []
    for _, raw, _, _ in records:
        out = torch.empty(len(raw) // 2, dtype=torch.bfloat16, device=device)
        ctx = Context(out)
        streams.append(ctx.stream)
        execution = DeviceExecutionContext.from_cache_context(ctx)
        manager = DeviceOutputLeaseManager.for_execution_context(execution)
        leases.append((manager, manager.acquire(batch_idx=0, object_group_idx=0, byte_length=len(raw))))
        outputs.append(out)
    for _, lease in leases:
        pass
    results = {
        "corpus_sha256": hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256((TASK / "lmcache/v1/distributed/compress_adapters/byte_split_zstd.py").read_bytes()).hexdigest(),
        "records": [{"name": n, "raw_bytes": len(r), "raw_sha256": hashlib.sha256(r).hexdigest(), "compressed_bytes": sum(map(len, f[0] + f[1])), "frame_count": sum(map(len, f))} for n, r, f, _ in records],
        "gpu": torch.cuda.get_device_name(device),
        "uuid": os.environ["GPU_UUID"],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nvcomp_version": __import__("importlib.metadata", fromlist=["version"]).version("nvidia-nvcomp-cu12"),
        "profile_mode": "torch.profiler Kineto CPU+CUDA, same CUDA stream; profiled calls separate from wall baseline",
        "snapshots": {"before": gpu_snapshot()},
        "samples": [],
    }

    # Original, uninstrumented helper wall reference; match the prior 3-record endpoint.
    first_wall = {}
    cpu = zstandard.ZstdDecompressor()

    def cpu_restore(rec, out, stream):
        _, raw, frames, _ = rec
        planes = tuple(b"".join(cpu.decompress(frame) for frame in batch) for batch in frames)
        merged = merge_byte_planes(planes)
        pinned = torch.frombuffer(bytearray(merged), dtype=torch.uint8).pin_memory()
        with torch_dev.device(out.device):
            out.view(torch.uint8).reshape(-1).copy_(pinned, non_blocking=True)
        sync_stream(stream)

    def group(mode):
        if mode == "gpu":
            for rec, (manager, lease) in zip(records, leases, strict=True):
                decode_zstd_byte_split_to_lease(rec[2], rec[3], lease)
        else:
            for rec, out, stream in zip(records, outputs, streams, strict=True):
                cpu_restore(rec, out, stream)

    def verify(mode):
        for rec, out in zip(records, outputs, strict=True):
            actual = out.view(torch.uint8).reshape(-1).cpu().numpy().tobytes()
            assert hashlib.sha256(actual).digest() == hashlib.sha256(rec[1]).digest(), (mode, rec[0])

    for mode in ("cpu", "gpu"):
        t = time.perf_counter_ns(); group(mode); first_wall[mode] = (time.perf_counter_ns() - t) / 1e6
        verify(mode)
    baseline = {"cpu": [], "gpu": []}
    for i in range(2):
        for mode in (("cpu", "gpu") if i % 2 == 0 else ("gpu", "cpu")):
            group(mode)
            verify(mode)
    for i in range(12):
        for mode in (("cpu", "gpu") if i % 2 == 0 else ("gpu", "cpu")):
            t = time.perf_counter_ns(); group(mode); baseline[mode].append((time.perf_counter_ns() - t) / 1e6)
            verify(mode)
    results["uninstrumented_first_group_ms"] = first_wall
    results["uninstrumented_wall_ms"] = baseline

    # Warm profiler and kernel/module loading before collecting active events.
    for _ in range(2):
        group("gpu")
        verify("gpu")
    torch.cuda.synchronize()

    stage_wall_ns = {}
    cuda_spans = []
    original_copy = torch.Tensor.copy_
    original_as_arrays = nvcomp.as_arrays
    original_get_frame_parameters = zstandard.get_frame_parameters
    original_empty = torch.empty
    original_frombuffer = torch.frombuffer
    original_pin_memory = torch.Tensor.pin_memory
    original_sync_stream = current_device_spec.synchronize_stream
    original_torch_sync = torch.cuda.synchronize
    original_merge = merge_byte_planes
    original_decompressor_class = zstandard.ZstdDecompressor

    def timed(name, fn, *args, cuda_span=False, **kwargs):
        stream = torch.cuda.current_stream(device)
        start_event = end_event = None
        if cuda_span:
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
        started = time.perf_counter_ns()
        try:
            return fn(*args, **kwargs)
        finally:
            stage_wall_ns[name] = stage_wall_ns.get(name, 0) + time.perf_counter_ns() - started
            if cuda_span:
                end_event.record(stream)
                cuda_spans.append((name, start_event, end_event))

    class CodecProxy:
        def __init__(self, *args, **kwargs):
            self._codec = original_codec(*args, **kwargs)

        def get_required_decompression_alignment(self):
            return timed("alignment_query_host", self._codec.get_required_decompression_alignment)

        def decompression_config(self, *args, **kwargs):
            return timed("config_host_call", self._codec.decompression_config, *args, cuda_span=True, **kwargs)

        def decode(self, *args, **kwargs):
            return timed("decode_host_call", self._codec.decode, *args, cuda_span=True, **kwargs)

    original_codec = nvcomp.Codec
    nvcomp.Codec = CodecProxy
    nvcomp.as_arrays = lambda *a, **kw: timed("array_descriptor_host", original_as_arrays, *a, **kw)
    zstandard.get_frame_parameters = lambda *a, **kw: timed("frame_header_host", original_get_frame_parameters, *a, **kw)
    torch.empty = lambda *a, **kw: timed("torch_empty_host", original_empty, *a, **kw)
    torch.frombuffer = lambda *a, **kw: timed("torch_frombuffer_host", original_frombuffer, *a, **kw)
    torch.Tensor.pin_memory = lambda self, *a, **kw: timed("pin_memory_host", original_pin_memory, self, *a, **kw)

    def copy_wrapper(self, src, *args, **kwargs):
        name = "h2d_copy_submit" if self.device.type == "cuda" and getattr(src, "device", torch.device("cpu")).type == "cpu" else "d2d_merge_copy_submit"
        return timed(name, original_copy, self, src, *args, cuda_span=True, **kwargs)

    torch.Tensor.copy_ = copy_wrapper
    current_device_spec.synchronize_stream = lambda stream: timed("sync_host_wait", original_sync_stream, stream)
    torch.cuda.synchronize = lambda *a, **kw: timed("sync_host_wait", original_torch_sync, *a, **kw)

    class TimedDecompressor:
        def __init__(self, *a, **kw):
            self._decoder = original_decompressor_class(*a, **kw)

        def decompress(self, frame):
            return timed("cpu_zstd_decode", self._decoder.decompress, frame)

    # Replace the decoder created before instrumentation was installed.
    cpu = TimedDecompressor()

    zstandard.ZstdDecompressor = TimedDecompressor
    # CPU byte merge is called through this global in cpu_restore.
    globals()["merge_byte_planes"] = lambda *a, **kw: timed("cpu_byte_merge", original_merge, *a, **kw)

    profile_results = []
    event_only_results = []
    try:
        # Event-only runs separate stream spans from Kineto's active-time trace.
        # They retain the original helper and add only a small, fixed set of events.
        for repetition in range(4):
            order = ("gpu", "cpu") if repetition % 2 == 0 else ("cpu", "gpu")
            for mode in order:
                stage_wall_ns.clear()
                cuda_spans.clear()
                t = time.perf_counter_ns()
                group(mode)
                wall_ms = (time.perf_counter_ns() - t) / 1e6
                verify(mode)
                event_only_results.append({
                    "mode": mode,
                    "wall_ms": wall_ms,
                    "stage_host_wall_ms": {k: v / 1e6 for k, v in stage_wall_ns.items()},
                    "stage_cuda_event_spans_ms": [
                        {"stage": n, "elapsed_ms": a.elapsed_time(b)} for n, a, b in cuda_spans
                    ],
                })
        for mode in ("gpu", "cpu"):
            stage_wall_ns.clear()
            cuda_spans.clear()
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False, profile_memory=True) as prof:
                t = time.perf_counter_ns()
                group(mode)
                wall_ms = (time.perf_counter_ns() - t) / 1e6
            verify(mode)
            events = []
            for e in prof.events():
                if getattr(e.device_type, "name", "") == "CUDA":
                    events.append({"name": e.name, "cuda_us": getattr(e, "device_time_total", 0.0), "cpu_launch_us": getattr(e, "cpu_time_total", 0.0)})
            kernels = [e for e in events if "Memcpy" not in e["name"] and "memcpy" not in e["name"].lower()]
            copies = [e for e in events if "Memcpy" in e["name"] or "memcpy" in e["name"].lower()]
            profile_results.append({
                "mode": mode, "profiled_wall_ms": wall_ms,
                "stage_host_wall_ms": {k: v / 1e6 for k, v in stage_wall_ns.items()},
                "stage_cuda_event_span_ms": [
                    {"stage": n, "elapsed_ms": a.elapsed_time(b)} for n, a, b in cuda_spans
                ],
                "cuda_memcpy_active_us_sum": sum(e["cuda_us"] for e in copies),
                "cuda_kernel_active_us_sum": sum(e["cuda_us"] for e in kernels),
                "cuda_events": events,
                "profiler_key_averages": prof.key_averages().table(sort_by="cuda_time_total", row_limit=40),
                "trace_file": f"{mode}-trace.json",
            })
            prof.export_chrome_trace(f"{mode}-trace.json")
    finally:
        nvcomp.Codec = original_codec
        nvcomp.as_arrays = original_as_arrays
        zstandard.get_frame_parameters = original_get_frame_parameters
        torch.empty = original_empty
        torch.frombuffer = original_frombuffer
        torch.Tensor.pin_memory = original_pin_memory
        torch.Tensor.copy_ = original_copy
        current_device_spec.synchronize_stream = original_sync_stream
        torch.cuda.synchronize = original_torch_sync
        zstandard.ZstdDecompressor = original_decompressor_class
        globals()["merge_byte_planes"] = original_merge
    results["profiled_samples"] = profile_results
    results["event_only_samples"] = event_only_results
    results["snapshots"]["after"] = gpu_snapshot()
    for manager, lease in leases:
        lease.release(); manager.close()
    Path("result.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps({"source_sha256": results["source_sha256"], "corpus_sha256": results["corpus_sha256"], "uninstrumented_first_group_ms": first_wall, "profiled": [{k: x[k] for k in ("mode", "profiled_wall_ms", "cuda_memcpy_active_us_sum", "cuda_kernel_active_us_sum", "trace_file")} for x in profile_results], "event_only": [{"mode": x["mode"], "wall_ms": x["wall_ms"]} for x in event_only_results], "gpu_uuid": results["uuid"], "snapshots": results["snapshots"]}, sort_keys=True))


if __name__ == "__main__":
    main()
