import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from kvbytepack import bench, costmodel, record
from kvbytepack.__main__ import main
from kvbytepack.integrations.aibrix import make_marshaller

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))
from aibrix import BaseMarshaller, ZstdCompressor  # noqa: E402  (unmodified upstream modules)

M = costmodel.Measurement


def test_serial_crossover_matches_direct_timing():
    base = costmodel.uncompressed(1000)
    cand = M("c", 1000, 400, 2e-6, 1e-6)
    c = costmodel.crossover(cand, base)
    assert c.wins == "below" and c.bandwidth == pytest.approx(600 / 3e-6)
    for bw in (c.bandwidth * 0.5, c.bandwidth * 2):
        faster = costmodel.serial_seconds(cand, bw) < costmodel.serial_seconds(base, bw)
        assert faster == (bw < c.bandwidth)


@pytest.mark.parametrize("cand,wins", [
    (M("x", 10, 5, 0, 0), "always"),
    (M("x", 10, 12, 1, 1), "never"),
    (M("x", 10, 10, 0, 0), "never"),
])
def test_crossover_edge_cases(cand, wins):
    assert costmodel.crossover(cand, costmodel.uncompressed(10)).wins == wins


def test_crossover_above():
    base = M("slow-small", 100, 50, 2.0, 0)
    fast_big = M("fast-big", 100, 80, 1.0, 0)
    c = costmodel.crossover(fast_big, base)
    assert c.wins == "above" and c.bandwidth == pytest.approx(30.0)


def test_pipelined_goodput_is_min_stage():
    m = M("c", 100, 50, 1.0, 0.5)      # enc 100 B/s, dec 200 B/s
    assert costmodel.pipelined_goodput(m, 10) == 20     # link-bound: 10 * ratio 2
    assert costmodel.pipelined_goodput(m, 1e6) == 100   # encoder-bound
    assert costmodel.pipelined_goodput(costmodel.uncompressed(100), 7) == 7
    assert math.isinf(costmodel.pipelined_ceiling(costmodel.uncompressed(1)))


def test_bench_sizes_are_complete_records():
    arrays = {"a": np.arange(512, dtype="<u2").reshape(32, 16)}
    (m,) = bench.measure(arrays, [("channel-planes", "zstd:1")], repeats=1)
    assert m.stored_bytes == len(record.encode(arrays["a"].tobytes(), (32, 16), 2))
    assert m.encode_s > 0 and m.decode_s > 0
    assert "channel-planes zstd:1" in bench.report([m])


def test_cli(tmp_path, capsys):
    main(["selftest"])
    assert json.loads(capsys.readouterr().out)["byte_exact"] is True
    np.save(tmp_path / "k.npy", np.arange(1024, dtype="<u2").reshape(64, 16))
    main(["bench", str(tmp_path / "k.npy"), "--repeats", "1", "--config", "none/zstd:1",
          "--json", str(tmp_path / "r.json")])
    assert "none zstd:1" in capsys.readouterr().out
    assert json.loads((tmp_path / "r.json").read_text())["results"][0]["raw_bytes"] == 2048
    (tmp_path / "x.kvb").write_bytes(record.encode(bytes(64), (4, 8), 2))
    main(["inspect", str(tmp_path / "x.kvb")])
    assert json.loads(capsys.readouterr().out)["shape"] == [4, 8]


def test_aibrix_marshaller_chains_with_upstream_class():
    raw = np.random.default_rng(0).integers(0, 4, 16 * 64 * 2, dtype=np.uint8).tobytes()
    m = make_marshaller(BaseMarshaller, tokens=16, channels=64, width=2)
    rec = m.marshal(raw)
    assert rec[:4] == record.MAGIC and m.unmarshal(rec) == raw
    # Upstream compressor wrapping ours, as AIBrix would chain them.
    outer = ZstdCompressor(m)
    assert outer.unmarshal(outer.marshal(raw)) == raw
    with pytest.raises(ValueError):
        m.marshal(raw[:-2])
    other = make_marshaller(BaseMarshaller, tokens=32, channels=32, width=2)
    with pytest.raises(ValueError):
        other.unmarshal(rec)
