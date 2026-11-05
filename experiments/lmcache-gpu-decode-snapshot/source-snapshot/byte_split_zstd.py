# SPDX-License-Identifier: Apache-2.0
"""Experimental BF16 byte-split helpers for Zstd GPU decompression.

This module is deliberately not registered as a portable-record codec. The
current shared record contract has no Zstd codec or byte-merge transform.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence

# Third Party
import torch

# First Party
from lmcache import torch_dev
from lmcache.v1.distributed.compress_adapters.device import DeviceIdentity
from lmcache.v1.distributed.compress_adapters.lease import (
    DeviceOutputLease,
)
from lmcache.v1.platform import current_device_spec


class ByteSplitZstdError(ValueError):
    """Raised when byte-plane inputs or the leased output are inconsistent."""


def _validate_zstd_frame_size(
    frame: bytes,
    expected_size: int,
    frame_index: int,
) -> None:
    """Check a Zstd frame header against its trusted output-size metadata.

    Args:
        frame: One complete Zstd frame to be sent to nvCOMP.
        expected_size: Caller-provided uncompressed byte count.
        frame_index: Flattened frame index for error reporting.

    Returns:
        None when the frame declares exactly ``expected_size`` output bytes.

    Raises:
        RuntimeError: If the optional ``zstandard`` package is unavailable.
        ByteSplitZstdError: If the frame does not declare a known output size
            or the declared size differs from ``expected_size``.
        zstandard.ZstdError: If the frame header is malformed.
    """
    try:
        # Third Party
        import zstandard
    except ImportError as exc:
        raise RuntimeError(
            "the optional zstandard package is required to validate frame headers"
        ) from exc

    content_size = zstandard.get_frame_parameters(frame).content_size
    if content_size in (
        zstandard.CONTENTSIZE_UNKNOWN,
        zstandard.CONTENTSIZE_ERROR,
    ):
        raise ByteSplitZstdError(
            f"Zstd frame {frame_index} must declare its uncompressed size"
        )
    if content_size != expected_size:
        raise ByteSplitZstdError(
            f"Zstd frame {frame_index} declares {content_size} bytes, "
            f"expected {expected_size}"
        )


def split_bf16_byte_planes(raw: bytes) -> tuple[bytes, bytes]:
    """Split contiguous BF16 storage into its low-byte and high-byte planes.

    The operation treats the payload as uninterpreted bytes and never converts
    values to floating point, so NaN payloads and signed zero remain exact.

    Args:
        raw: Contiguous BF16 bytes. Its length must be a multiple of two.

    Returns:
        Two byte strings, each containing byte zero or byte one from every
        consecutive two-byte element.

    Raises:
        TypeError: If ``raw`` is not bytes.
        ByteSplitZstdError: If the byte count is odd.
    """
    if not isinstance(raw, bytes):
        raise TypeError(f"raw must be bytes, got {type(raw).__name__}")
    if len(raw) % 2:
        raise ByteSplitZstdError("BF16 payload byte count must be even")
    return raw[0::2], raw[1::2]


def merge_byte_planes(planes: Sequence[bytes]) -> bytes:
    """Interleave two equal byte planes into the original BF16 byte order.

    Args:
        planes: Exactly two byte strings in the order returned by
            :func:`split_bf16_byte_planes`.

    Returns:
        Contiguous interleaved bytes, without interpreting BF16 values.

    Raises:
        TypeError: If an item is not bytes.
        ByteSplitZstdError: If there are not two equal-length planes.
    """
    if len(planes) != 2:
        raise ByteSplitZstdError(f"expected two byte planes, got {len(planes)}")
    low, high = planes
    if not isinstance(low, bytes) or not isinstance(high, bytes):
        raise TypeError("byte planes must be bytes")
    if len(low) != len(high):
        raise ByteSplitZstdError(
            f"byte-plane lengths differ: {len(low)} and {len(high)}"
        )
    merged = bytearray(len(low) * 2)
    merged[0::2] = low
    merged[1::2] = high
    return bytes(merged)


def decode_zstd_byte_split_to_lease(
    frames_by_plane: tuple[tuple[bytes, ...], tuple[bytes, ...]],
    expected_sizes_by_plane: tuple[tuple[int, ...], tuple[int, ...]],
    output_lease: DeviceOutputLease,
) -> None:
    """Decode Zstd frame batches and merge them into an active output lease.

    This experimental CUDA helper consumes frame bytes and caller-supplied
    uncompressed frame sizes, then checks sizes in each Zstd frame header. It
    does not parse a portable record, verify a
    record checksum, or publish a KV-ready event. The caller must provide
    complete Zstd frames with declared output sizes and keep ownership of
    ``output_lease`` until this blocking call returns. The helper does not
    release the lease.

    Args:
        frames_by_plane: Two non-empty frame batches, one for each byte plane.
        expected_sizes_by_plane: Per-frame output sizes corresponding to each
            frame batch. Each plane's sizes must sum to the same value.
        output_lease: Active public LMCache output lease whose byte range is
            exactly twice the uncompressed size of either plane.

    Returns:
        None after decode, byte merge, and stream synchronization complete.

    Raises:
        TypeError: If inputs have the wrong types or the lease is invalid.
        ValueError: If frame counts, sizes, alignment, device, or lease bounds
            do not satisfy the documented contract.
        RuntimeError: If CUDA or the optional ``nvidia.nvcomp`` binding is not
            available, or if nvCOMP rejects a frame or decode request.

    Notes:
        nvCOMP is imported only when this function is called, so importing the
        module does not require a GPU or nvCOMP installation. The execution
        context's CUDA stream is used for input transfer, nvCOMP, and merge.
        The caller must supply trusted, complete frames. Binding and runtime
        exceptions propagate; this helper does not inspect per-frame numeric
        status or verify payload checksums.
    """
    if not isinstance(output_lease, DeviceOutputLease):
        raise TypeError("output_lease must be a DeviceOutputLease")
    if not output_lease.is_active:
        raise ByteSplitZstdError("output lease is not active")
    if len(frames_by_plane) != 2 or len(expected_sizes_by_plane) != 2:
        raise ByteSplitZstdError(
            "exactly two frame batches and size batches are required"
        )

    normalized_sizes: list[tuple[int, ...]] = []
    for plane_index, (frames, sizes) in enumerate(
        zip(frames_by_plane, expected_sizes_by_plane, strict=True)
    ):
        if len(frames) == 0 or len(frames) != len(sizes):
            raise ByteSplitZstdError(
                f"plane {plane_index} frame and size counts must match and be nonzero"
            )
        for frame_index, (frame, size) in enumerate(zip(frames, sizes, strict=True)):
            if not isinstance(frame, bytes) or not frame:
                raise TypeError(
                    f"plane {plane_index} frame {frame_index} must be nonempty bytes"
                )
            if type(size) is not int or size <= 0:
                raise ByteSplitZstdError(
                    f"plane {plane_index} frame {frame_index} size must be positive"
                )
            _validate_zstd_frame_size(
                frame,
                size,
                sum(len(previous) for previous in frames_by_plane[:plane_index])
                + frame_index,
            )
        normalized_sizes.append(tuple(sizes))

    plane_sizes = tuple(sum(sizes) for sizes in normalized_sizes)
    if plane_sizes[0] != plane_sizes[1]:
        raise ByteSplitZstdError(
            f"byte-plane output sizes differ: {plane_sizes[0]} and {plane_sizes[1]}"
        )

    buffer_range = output_lease.buffer_range
    output_size = plane_sizes[0] * 2
    if buffer_range.byte_length != output_size:
        raise ValueError(
            f"leased output has {buffer_range.byte_length} bytes, "
            f"expected {output_size}"
        )
    if buffer_range.byte_offset % 2:
        raise ValueError("leased output byte offset must preserve BF16 alignment")

    execution_context = output_lease.owner.execution_context
    if (
        execution_context.device.device_type != "cuda"
        or execution_context.device.backend_name != "cuda"
    ):
        raise ValueError("nvCOMP byte-split decode requires a CUDA execution context")
    stream = execution_context.stream
    device = torch.device("cuda", execution_context.device.device_index)
    device_spec = current_device_spec
    stream_handle = device_spec.get_stream_handle(stream)
    if getattr(stream, "device", None) != device:
        raise ValueError("execution context stream device does not match its device")

    tensor = buffer_range.tensor
    if not tensor.is_contiguous():
        raise ValueError("leased output tensor must remain contiguous")
    if tensor.dtype is not torch.bfloat16:
        raise ValueError("leased output tensor must use BF16 storage")
    if tensor.data_ptr() != buffer_range.base_address:
        raise ValueError("leased output tensor address changed after reservation")
    if tensor.numel() * tensor.element_size() != buffer_range.capacity:
        raise ValueError("leased output tensor capacity changed after reservation")
    if DeviceIdentity.from_device(tensor.device) != execution_context.device:
        raise ValueError("leased output tensor device does not match execution context")
    if buffer_range.device != execution_context.device:
        raise ValueError("leased output range device does not match execution context")
    if buffer_range.byte_offset + buffer_range.byte_length > buffer_range.capacity:
        raise ValueError("leased output range exceeds its tensor capacity")

    try:
        # Third Party
        from nvidia import nvcomp
    except ImportError as exc:
        raise RuntimeError(
            "nvCOMP Python bindings are required for GPU decode"
        ) from exc

    with torch_dev.device(device):
        decoder = nvcomp.Codec(
            algorithm="Zstd",
            device_id=execution_context.device.device_index,
            uncomp_chunk_size=65536,
            bitstream_kind=nvcomp.BitstreamKind.RAW,
            cuda_stream=stream_handle,
        )
        alignment = decoder.get_required_decompression_alignment()
        if type(alignment) is not int or alignment <= 0:
            raise RuntimeError("nvCOMP returned an invalid decompression alignment")
        frame_offsets: list[int] = []
        frame_sizes: list[int] = []
        frame_plane_indices: list[int] = []
        packed_length = 0
        for plane_index, frames in enumerate(frames_by_plane):
            for frame in frames:
                packed_length = (packed_length + alignment - 1) // alignment * alignment
                frame_offsets.append(packed_length)
                frame_sizes.append(len(frame))
                frame_plane_indices.append(plane_index)
                packed_length += len(frame)

        host_bytes = bytearray(packed_length)
        frame_cursor = 0
        for frames in frames_by_plane:
            for frame in frames:
                offset = frame_offsets[frame_cursor]
                host_bytes[offset : offset + len(frame)] = frame
                frame_cursor += 1
        host_input = torch.frombuffer(host_bytes, dtype=torch.uint8).pin_memory()
        compressed_device = torch.empty(packed_length, dtype=torch.uint8, device=device)
        expected_sizes = tuple(size for sizes in normalized_sizes for size in sizes)
        output_plane_offsets = [
            0,
            (plane_sizes[0] + alignment - 1) // alignment * alignment,
        ]
        output_offsets: list[int] = []
        output_plane_cursors = [0, 0]
        for plane_index, size in zip(
            frame_plane_indices,
            expected_sizes,
            strict=True,
        ):
            offset = (
                output_plane_offsets[plane_index] + output_plane_cursors[plane_index]
            )
            if offset % alignment:
                raise ByteSplitZstdError(
                    "non-final decoded frame sizes must preserve nvCOMP alignment"
                )
            output_offsets.append(offset)
            output_plane_cursors[plane_index] += size
        output_storage_size = output_plane_offsets[1] + plane_sizes[1]
        output_storage = torch.empty(
            output_storage_size, dtype=torch.uint8, device=device
        )
        if compressed_device.data_ptr() % alignment:
            raise ValueError("nvCOMP input allocation does not meet alignment")
        if output_storage.data_ptr() % alignment:
            raise ValueError("nvCOMP output allocation does not meet alignment")
        if any(
            (compressed_device.data_ptr() + offset) % alignment
            for offset in frame_offsets
        ):
            raise ValueError("nvCOMP input frame view is not aligned")
        if any(
            (output_storage.data_ptr() + offset) % alignment
            for offset in output_offsets
        ):
            raise ValueError("nvCOMP output frame view is not aligned")

        queued_work = False
        try:
            with torch_dev.stream(stream):
                queued_work = True
                compressed_device.copy_(host_input, non_blocking=True)
                input_batch = nvcomp.as_arrays(
                    compressed_device,
                    frame_offsets,
                    frame_sizes,
                    cuda_stream=stream_handle,
                )
                output_batch = nvcomp.as_arrays(
                    output_storage,
                    output_offsets,
                    list(expected_sizes),
                    cuda_stream=stream_handle,
                )
                config = decoder.decompression_config(input_batch)
                decoder.decode(
                    input_batch,
                    out=output_batch,
                    decompression_config=config,
                )
                queued_work = True

                destination = (
                    tensor.reshape(-1)
                    .view(torch.uint8)
                    .reshape(-1)
                    .narrow(
                        0,
                        buffer_range.byte_offset,
                        buffer_range.byte_length,
                    )
                )
                interleaved = destination.view(-1, 2)
                low_plane = output_storage.narrow(0, 0, plane_sizes[0])
                high_plane = output_storage.narrow(
                    0,
                    output_plane_offsets[1],
                    plane_sizes[1],
                )
                interleaved[:, 0].copy_(low_plane)
                interleaved[:, 1].copy_(high_plane)
            device_spec.synchronize_stream(stream)
        except BaseException:
            if queued_work:
                device_spec.synchronize_stream(stream)
            raise
