"""Pack NumPy arrays or PyTorch tensors into KVB2 records and back, bit-exactly.

The last axis is treated as the channel axis; all leading axes are flattened
into rows. For a KV block ``[layers, heads, tokens, head_dim]`` that groups
``head_dim`` channels; for ``[tokens, heads * head_dim]`` it groups all heads.
Put the axis whose values share a scale last for the best ratio.
"""
import numpy as np

from . import legacy, record

DEFAULT_LAYOUT = "channel-planes"
DEFAULT_CODEC = "zstd:1"


def _to_bytes(x):
    if isinstance(x, np.ndarray):
        arr = x if x.flags.c_contiguous else x.copy(order="C")  # ascontiguousarray would 1-D a 0-d array
        return arr.tobytes(), arr.shape, arr.dtype.itemsize, "numpy:" + arr.dtype.str
    mod = type(x).__module__
    if mod.startswith("torch"):
        import torch
        t = x.detach().cpu().contiguous()
        flat = t.reshape(-1).view(torch.uint8) if t.numel() else torch.empty(0, dtype=torch.uint8)
        name = str(t.dtype).removeprefix("torch.")
        return flat.numpy().tobytes(), tuple(t.shape), t.element_size(), "torch:" + name
    if isinstance(x, (bytes, bytearray, memoryview)):
        raise TypeError("raw buffers need an explicit geometry: use kvbytepack.record.encode")
    raise TypeError(f"unsupported tensor type {type(x).__name__}")


def _from_bytes(raw, shape, tag):
    kind, _, dtype = tag.partition(":")
    if kind == "numpy":
        return np.frombuffer(raw, dtype=np.dtype(dtype)).reshape(shape).copy()
    if kind == "torch":
        import torch
        dt = getattr(torch, dtype)
        if not raw:
            return torch.empty(shape, dtype=dt)
        return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(dt).reshape(shape)
    raise ValueError(f"record holds opaque bytes (tag {tag!r}); use kvbytepack.record.decode")


def pack(x, *, layout=DEFAULT_LAYOUT, codec=DEFAULT_CODEC):
    """Compress an array/tensor into a self-describing KVB2 record (``bytes``)."""
    raw, shape, width, tag = _to_bytes(x)
    return record.encode(raw, shape, width, tag=tag, layout=layout, codec=codec)


def unpack(blob):
    """Restore the array/tensor packed by :func:`pack` (also reads legacy KVZ1 blobs).

    PyTorch-origin records come back as CPU tensors.
    """
    head = bytes(blob[:4])
    if head == record.MAGIC:
        h, raw = record.decode(blob)
        return _from_bytes(raw, h.shape, h.tag)
    if head == legacy.MAGIC:
        meta, raw = legacy.decode(blob)
        if meta["kind"] == "numpy":
            return _from_bytes(raw, tuple(meta["shape"]), "numpy:" + np.dtype(meta["dtype"]).str)
        return _from_bytes(raw, tuple(meta["shape"]), "torch:" + meta["dtype"])
    raise ValueError("not a kvbytepack record")
