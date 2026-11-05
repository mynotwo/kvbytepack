"""Measure the experimental codec core on the checked-in Qwen KV payloads."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import zstandard

TASK_ROOT = Path("/home/audreymao/kvbytepack_mp_proto_20261103/codec_pr_test")
OLD_LMCACHE = Path("/home/audreymao/kvbytepack_mp_proto_20261103/lmcache-source/lmcache")
CORPUS = Path("/home/audreymao/kvbytepack_mp_proto_20261103/codec_pr_test/kv_payloads.npz")
sys.path.insert(0, str(TASK_ROOT))
import lmcache
lmcache.__path__.append(str(OLD_LMCACHE))
import nvidia
nvidia.__path__.append(str(Path(sys.prefix) / "lib/python3.11/site-packages/nvidia"))

from lmcache import torch_dev
from lmcache.v1.distributed.compress_adapters.byte_split_zstd import (
    decode_zstd_byte_split_to_lease,
    merge_byte_planes,
    split_bf16_byte_planes,
)
from lmcache.v1.distributed.compress_adapters.device import DeviceExecutionContext
from lmcache.v1.distributed.compress_adapters.lease import DeviceOutputLeaseManager
from lmcache.v1.platform import current_device_spec


class CacheContext:
    """Minimal context using the public lease adapter over a real CUDA tensor."""

    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.device = output.device
        self.stream = current_device_spec.current_stream(self.device)

    def get_temp_object_group_buffer(self, batch_idx: int, object_group_idx: int) -> torch.Tensor:
        if (batch_idx, object_group_idx) != (0, 0):
            raise ValueError("unexpected staging slot")
        return self.output


def build_records() -> list[tuple[str, bytes, tuple[tuple[bytes, ...], tuple[bytes, ...]], tuple[tuple[int, ...], tuple[int, ...]]]]:
    """Build three 2x24xT x128 BF16 records and independent 16KiB Zstd frames."""
    data = np.load(CORPUS)
    raw_records = []
    raw_records.append(("ctx128", np.stack([
        np.stack([data[f"ctx128_K_{i:02d}"] for i in range(24)]),
        np.stack([data[f"ctx128_V_{i:02d}"] for i in range(24)]),
    ]).tobytes(order="C")))
    for label, selection in (("ctx512_first256", slice(0, 256)), ("ctx512_last256", slice(256, 512))):
        raw_records.append((label, np.stack([
            np.stack([data[f"ctx512_K_{i:02d}"][selection] for i in range(24)]),
            np.stack([data[f"ctx512_V_{i:02d}"][selection] for i in range(24)]),
        ]).tobytes(order="C")))
    compressor = zstandard.ZstdCompressor(level=1)
    result = []
    for label, raw in raw_records:
        plane_bytes = split_bf16_byte_planes(raw)
        frames = tuple(tuple(
            compressor.compress(plane[start:start + 16 * 1024])
            for start in range(0, len(plane), 16 * 1024)
        ) for plane in plane_bytes)
        sizes = tuple(tuple(
            min(16 * 1024, len(plane) - start)
            for start in range(0, len(plane), 16 * 1024)
        ) for plane in plane_bytes)
        result.append((label, raw, frames, sizes))
    return result


def cpu_restore(record, output: torch.Tensor, stream: object, decompressor: object) -> None:
    """CPU Zstd-decode, public byte merge, pinned H2D, and ready sync."""
    _, raw, frames, _ = record
    planes = tuple(b"".join(decompressor.decompress(frame) for frame in batch) for batch in frames)
    restored = merge_byte_planes(planes)
    host = torch.frombuffer(bytearray(restored), dtype=torch.uint8).pin_memory()
    with torch_dev.device(output.device), torch_dev.stream(stream):
        output.view(torch.uint8).reshape(-1).copy_(host, non_blocking=True)
    current_device_spec.synchronize_stream(stream)


def main() -> int:
    import nvidia.nvcomp
    records = build_records()
    device = torch.device("cuda", torch_dev.current_device())
    contexts = []
    for _, raw, _, _ in records:
        output = torch.empty(len(raw) // 2, dtype=torch.bfloat16, device=device)
        context = CacheContext(output)
        execution_context = DeviceExecutionContext.from_cache_context(context)
        manager = DeviceOutputLeaseManager.for_execution_context(execution_context)
        lease = manager.acquire(batch_idx=0, object_group_idx=0, byte_length=len(raw))
        contexts.append((output, context.stream, manager, lease))
    for output, stream, _, _ in contexts:
        current_device_spec.synchronize_stream(stream)
    results = {
        "records": [],
        "cpu_samples_ms": [],
        "gpu_samples_ms": [],
        "cpu_first_group_ms": None,
        "gpu_first_group_ms": None,
        "warmup": 2,
        "repetitions": 12,
    }
    for record, (output, stream, _, lease) in zip(records, contexts, strict=True):
        label, raw, frames, sizes = record
        results["records"].append({
            "name": label,
            "raw_bytes": len(raw),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "frames_per_plane": [len(batch) for batch in frames],
            "compressed_bytes": sum(len(frame) for batch in frames for frame in batch),
        })
    cpu_decoder = zstandard.ZstdDecompressor()
    def restore_group(mode: str) -> None:
        for record, (output, stream, _, lease) in zip(records, contexts, strict=True):
            if mode == "cpu":
                cpu_restore(record, output, stream, cpu_decoder)
            else:
                _, _, frames, sizes = record
                decode_zstd_byte_split_to_lease(frames, sizes, lease)

    def verify_group(mode: str) -> None:
        for record, (output, _, _, _) in zip(records, contexts, strict=True):
            raw = record[1]
            actual = output.view(torch.uint8).reshape(-1).cpu().numpy().tobytes()
            if hashlib.sha256(actual).digest() != hashlib.sha256(raw).digest():
                raise AssertionError(f"{mode} restore mismatch for {record[0]}")

    for mode in ("cpu", "gpu"):
        started = time.perf_counter_ns()
        restore_group(mode)
        results[f"{mode}_first_group_ms"] = (time.perf_counter_ns() - started) / 1e6
        verify_group(mode)
    for warmup_index in range(2):
        order = ("cpu", "gpu") if warmup_index == 0 else ("gpu", "cpu")
        for mode in order:
            restore_group(mode)
            verify_group(mode)
    for index in range(12):
        order = ("cpu", "gpu") if index % 2 == 0 else ("gpu", "cpu")
        for mode in order:
            started = time.perf_counter_ns()
            restore_group(mode)
            elapsed_ms = (time.perf_counter_ns() - started) / 1e6
            results[f"{mode}_samples_ms"].append(elapsed_ms)
            # The byte checks are outside the timed interval and cover every record.
            verify_group(mode)
    results["corpus_sha256"] = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    results["source_sha256"] = hashlib.sha256(
        (TASK_ROOT / "lmcache/v1/distributed/compress_adapters/byte_split_zstd.py")
        .read_bytes()
    ).hexdigest()
    results["torch"] = torch.__version__
    results["nvcomp"] = nvidia.nvcomp.__file__
    results["gpu"] = torch.cuda.get_device_name(device)
    results["gpu_uuid"] = os.environ.get("GPU_UUID", "recorded by outer monitor")
    results["cpu_workers"] = 1
    print(json.dumps(results, sort_keys=True))
    for _, _, manager, lease in contexts:
        lease.release()
        manager.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
