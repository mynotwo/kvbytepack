"""CPU-only byte-exact smoke check. Synthetic fixtures, not model measurements."""
import hashlib
import json

import numpy as np

from . import decode, encode, unzip_tensor, zip_tensor


def main():
    # Include signed zero, infinities, subnormals and different NaN payloads.
    bits = np.array([0, 0x8000, 1, 0x3F80, 0x7F80, 0xFF80,
                     0x7FC1, 0x7FFF], dtype="<u2").reshape(2, 4)
    raw = bits.tobytes()
    methods = ("raw_zstd-1", "transpose+zstd-1", "bytesplit+zstd-1",
               "pipeline_zstd-1")
    for method in methods:
        blob = encode(raw, 2, 4, 2, method)
        if decode(blob, 2, 4, 2, method) != raw:
            raise RuntimeError(f"byte mismatch: {method}")
        # Interpreted as float16 to check representation, not numerical equality.
        array = bits.view("<f2")
        restored = unzip_tensor(zip_tensor(array, method))
        if restored.dtype != array.dtype or restored.tobytes() != raw:
            raise RuntimeError(f"tensor mismatch: {method}")
    print(json.dumps({"fixture": "synthetic-16bit-patterns-v1",
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "checks": 8, "byte_exact": True,
                      "model_weights": False, "gpu": False}, sort_keys=True))


if __name__ == "__main__":
    main()
