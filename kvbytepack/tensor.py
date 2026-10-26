"""Compress a whole tensor in one call: zip_tensor / unzip_tensor.

The last dimension is the channel axis D and every leading dimension is
flattened into tokens S, so a [layers, heads, tokens, head_dim] KV block
works as is. The blob records method, dtype and shape, so decoding needs
nothing but the blob. Accepts numpy arrays and torch tensors (bfloat16
included); returns the same kind it was given.
"""
import json
import struct

import numpy as np

from .codec import decode, encode

MAGIC = b"KVZ1"
# torch dtypes numpy lacks travel as same-width integers.
_TORCH_VIEW = {"bfloat16": "int16", "float8_e4m3fn": "uint8", "float8_e5m2": "uint8"}


def _as_numpy(x):
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x), str(x.dtype), "numpy"
    import torch

    name = str(x.dtype).removeprefix("torch.")
    t = x.detach().cpu().contiguous()
    if name in _TORCH_VIEW:
        t = t.view(getattr(torch, _TORCH_VIEW[name]))
    return t.numpy(), name, "torch"


def zip_tensor(x, method="pipeline_zstd-1"):
    arr, dtype, kind = _as_numpy(x)
    shape = list(arr.shape) or [1]
    D = shape[-1]
    S = arr.size // D if D else 0
    head = json.dumps({"m": method, "dtype": dtype, "shape": list(arr.shape),
                       "kind": kind}).encode()
    body = encode(arr.tobytes(), S, D, arr.itemsize, method)
    return MAGIC + struct.pack("<I", len(head)) + head + body


def unzip_tensor(blob):
    if blob[:4] != MAGIC:
        raise ValueError("not a kvzip tensor blob")
    (n,) = struct.unpack_from("<I", blob, 4)
    meta = json.loads(blob[8:8 + n])
    shape, dtype = meta["shape"], meta["dtype"]
    store = np.dtype(_TORCH_VIEW.get(dtype, dtype))
    D = shape[-1] if shape else 1
    count = int(np.prod(shape)) if shape else 1
    S = count // D if D else 0
    raw = decode(blob[8 + n:], S, D, store.itemsize, meta["m"])
    arr = np.frombuffer(raw, dtype=store).reshape(shape).copy()
    if meta["kind"] == "numpy":
        return arr
    import torch

    t = torch.from_numpy(arr)
    return t.view(getattr(torch, dtype)) if dtype in _TORCH_VIEW else t
