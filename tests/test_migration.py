import importlib

import numpy as np
import pytest

import kvbytepack
import kvzip


@pytest.mark.parametrize("name", ["codec", "tensor", "formats", "inputs"])
def test_old_submodule_is_canonical_module(name):
    assert importlib.import_module("kvzip." + name) is importlib.import_module("kvbytepack." + name)


def test_old_and_new_api_decode_each_others_blobs():
    array = np.array([0, 0x8000, 1, 0x7C00, 0xFC00, 0x7E01], dtype="<u2").view("<f2")
    assert kvzip.__all__ == kvbytepack.__all__
    for encode, decode in ((kvzip.zip_tensor, kvbytepack.unzip_tensor),
                           (kvbytepack.zip_tensor, kvzip.unzip_tensor)):
        blob = encode(array)
        assert blob[:4] == b"KVZ1"  # Existing wire format remains readable.
        restored = decode(blob)
        assert restored.dtype == array.dtype
        assert restored.tobytes() == array.tobytes()
