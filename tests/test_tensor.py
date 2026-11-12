import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import kvbytepack
from kvbytepack import layout

LEGACY = Path(__file__).parent / "fixtures" / "legacy_kvz1"


@pytest.mark.parametrize("dtype", ["<f2", "<f4", ">f4", "<f8", "|u1", "<i2", "|V2"])
@pytest.mark.parametrize("lay", layout.LAYOUTS)
def test_numpy_round_trip(dtype, lay):
    raw = np.random.default_rng(0).integers(0, 256, 4 * 6 * 5 * 8, dtype=np.uint8).tobytes()
    arr = np.frombuffer(raw[: 4 * 6 * 5 * np.dtype(dtype).itemsize], dtype=dtype).reshape(4, 6, 5)
    out = kvbytepack.unpack(kvbytepack.pack(arr, layout=lay))
    assert out.dtype == arr.dtype and out.shape == arr.shape and out.tobytes() == arr.tobytes()
    assert out.flags.writeable


def test_non_contiguous_input_packs_logical_order():
    arr = np.arange(60, dtype="<f4").reshape(6, 10)[:, ::2]
    assert np.array_equal(kvbytepack.unpack(kvbytepack.pack(arr)), arr)


def test_special_float_bits_survive():
    bits = np.array([0, 0x8000, 1, 0x7C00, 0xFC00, 0x7E01, 0x7D55, 0xFFFF], dtype="<u2")
    arr = bits.view("<f2")
    assert kvbytepack.unpack(kvbytepack.pack(arr)).view("<u2").tolist() == bits.tolist()


def test_raw_buffers_need_geometry():
    with pytest.raises(TypeError):
        kvbytepack.pack(b"\0\0")


def test_layout_shrinks_real_bf16_kv():
    path = Path(__file__).parents[1] / "data" / "qwen2.5-0.5b-bf16-kv.npz"
    with np.load(path) as z:
        k = z["ctx512_K_05"]
    plain = len(kvbytepack.pack(k, layout="none"))
    laid = len(kvbytepack.pack(k))
    assert laid < 0.9 * plain


def test_reads_every_legacy_kvz1_fixture():
    index = json.loads((LEGACY / "index.json").read_text())["blobs"]
    assert len(index) == 32
    for name, want in index.items():
        out = kvbytepack.unpack((LEGACY / name).read_bytes())
        assert str(out.dtype) == want["dtype"] and list(out.shape) == want["shape"], name
        assert hashlib.sha256(out.tobytes()).hexdigest() == want["sha256"], name


@pytest.mark.parametrize("name", ["bfloat16", "float16", "float32", "float8_e4m3fn", "int8"])
def test_torch_round_trip(name):
    torch = pytest.importorskip("torch")
    dt = getattr(torch, name)
    src = torch.randint(0, 256, (3, 4, 8), dtype=torch.uint8)
    width = torch.empty(0, dtype=dt).element_size()
    x = src.reshape(-1)[: 96 // width * width].view(dt) if width > 1 else src.reshape(-1).view(dt)
    x = x.reshape(-1, 4)
    y = kvbytepack.unpack(kvbytepack.pack(x))
    assert isinstance(y, torch.Tensor) and y.dtype == x.dtype and y.shape == x.shape
    assert torch.equal(y.view(torch.uint8), x.view(torch.uint8))
