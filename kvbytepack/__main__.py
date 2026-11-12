"""Command line: ``python -m kvbytepack {selftest,bench,inspect}``."""
import argparse
import hashlib
import json
import sys

import numpy as np

from . import bench, codecs, layout, record, tensor


def _selftest(_args):
    """Round-trip edge-case bit patterns through every layout and codec."""
    bits = np.array([0, 0x8000, 1, 0x7C00, 0xFC00, 0x7E01, 0x7FFF, 0xFFFF], dtype="<u2")
    rng = np.random.default_rng(0)
    cases = {
        "f16-specials": bits.view("<f2").reshape(2, 4),
        "f32-scaled": (rng.standard_normal((64, 48)) * rng.uniform(0.1, 8, 48)).astype("<f4"),
        "u8": rng.integers(0, 256, (5, 7), dtype=np.uint8),
        "f64-scalar": np.array(-0.0),
        "empty": np.zeros((0, 3), dtype="<f2"),
    }
    checks = 0
    for arr in cases.values():
        for lay in layout.LAYOUTS:
            for name in codecs.available():
                out = tensor.unpack(tensor.pack(arr, layout=lay, codec=name))
                if out.dtype != arr.dtype or out.shape != arr.shape or out.tobytes() != arr.tobytes():
                    raise SystemExit(f"FAIL {lay} {name} {arr.dtype}{arr.shape}")
                checks += 1
    print(json.dumps({"byte_exact": True, "checks": checks, "codecs": codecs.available(),
                      "native_kernel": layout.native_available()}, sort_keys=True))


def _bench(args):
    arrays = bench.load_arrays(args.paths)
    configs = [tuple(c.split("/", 1)) for c in args.config] if args.config else bench.DEFAULT_CONFIGS
    results = bench.measure(arrays, configs, repeats=args.repeats, threads=args.threads)
    print(f"{len(arrays)} arrays from {', '.join(args.paths)}; {args.threads} thread(s)")
    print(bench.report(results, args.bandwidth))
    if args.json:
        rows = [m._asdict() for m in results]
        with open(args.json, "w") as f:
            json.dump({"arrays": len(arrays), "results": rows}, f, indent=1)


def _inspect(args):
    for path in args.records:
        with open(path, "rb") as f:
            blob = f.read()
        h = record.read_header(blob)
        d = h._asdict()
        d.update(path=path, raw_bytes=h.raw_bytes, ratio=round(h.raw_bytes / h.record_bytes, 4),
                 sha256=hashlib.sha256(blob).hexdigest())
        print(json.dumps(d))


def main(argv=None):
    p = argparse.ArgumentParser(prog="kvbytepack", description=__doc__)
    sub = p.add_subparsers(dest="cmd")
    sub.add_parser("selftest", help="bit-exact round trips, no data needed")
    b = sub.add_parser("bench", help="measure layouts/codecs on .npy/.npz KV arrays")
    b.add_argument("paths", nargs="+")
    b.add_argument("--config", action="append", metavar="LAYOUT/CODEC",
                   help="e.g. channel-planes/zstd:1 (repeatable; default: a standard set)")
    b.add_argument("--repeats", type=int, default=5)
    b.add_argument("--threads", type=int, default=1, help="encode/decode blocks on N threads")
    b.add_argument("--bandwidth", type=float, nargs="+", default=[1, 10, 25, 100],
                   metavar="GBPS", help="link speeds for the pipelined table")
    b.add_argument("--json", help="also write raw measurements here")
    i = sub.add_parser("inspect", help="print the header of KVB2 record files")
    i.add_argument("records", nargs="+")
    args = p.parse_args(argv)
    {"bench": _bench, "inspect": _inspect}.get(args.cmd, _selftest)(args)


if __name__ == "__main__":
    sys.exit(main())
