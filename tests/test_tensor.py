import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from kvzip import unzip_tensor, zip_tensor  # noqa: E402


def _kv_like(shape, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    return x * rng.uniform(0.1, 8, shape[-1]).astype(np.float32)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
@pytest.mark.parametrize("method", ["raw_zstd-1", "pipeline_zstd-1"])
def test_numpy_round_trip(dtype, method):
    x = _kv_like((2, 4, 33, 16)).astype(dtype)
    y = unzip_tensor(zip_tensor(x, method))
    assert y.dtype == x.dtype and y.shape == x.shape
    assert y.tobytes() == x.tobytes()


def test_pipeline_beats_raw_on_kv_like_data():
    x = _kv_like((64, 128)).astype(np.float16)
    assert len(zip_tensor(x, "pipeline_zstd-1")) < len(zip_tensor(x, "raw_zstd-1"))


def test_rejects_foreign_blob():
    with pytest.raises(ValueError):
        unzip_tensor(b"nope" + bytes(8))


def test_torch_bfloat16_round_trip():
    torch = pytest.importorskip("torch")
    x = torch.from_numpy(_kv_like((3, 40, 32))).to(torch.bfloat16)
    y = unzip_tensor(zip_tensor(x))
    assert y.dtype == torch.bfloat16 and y.shape == x.shape
    assert torch.equal(x.view(torch.int16), y.view(torch.int16))
