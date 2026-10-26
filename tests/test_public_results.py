import copy
import csv
import io
import json
from pathlib import Path
import statistics

import pytest

from bench.export_results import SOURCE, export, validate


def test_export_has_exact_source_rows_and_independent_summary():
    source = json.loads(SOURCE.read_text())
    files = export()
    rows = list(csv.DictReader(io.StringIO(files["measurements.csv"].decode())))
    assert len(rows) == 18000
    keys = set()
    for row in rows:
        key = (row["cell_id"], row["format"], row["scope"], row["method"])
        assert key not in keys
        keys.add(key)
        assert float(row["payload_ratio"]) == source[key[0]]["formats"][key[1]][key[2]][key[3]]
    summary = list(csv.DictReader(io.StringIO(files["summary.csv"].decode())))
    assert len(summary) == 8
    for row in summary:
        ratios = [c["formats"][row["format"]]["KV"] for c in source.values()]
        assert int(row["cells"]) == 50
        assert float(row["best_median"]) == statistics.median(max(r.values()) for r in ratios)
        assert float(row["best_vs_best_raw_median_pct"]) == statistics.median(
            max(r.values()) / max(v for m, v in r.items() if m.startswith("raw_")) - 1
            for r in ratios) * 100


@pytest.mark.parametrize("failure", ["cell", "duplicate", "format", "scope", "method", "nan", "zero"])
def test_incomplete_or_invalid_archive_rejected(failure):
    data = copy.deepcopy(json.loads(SOURCE.read_text()))
    cell = next(iter(data.values()))
    ratios = cell["formats"]["bf16"]["KV"]
    if failure == "cell":
        data.pop(next(iter(data)))
    elif failure == "duplicate":
        data["duplicate"] = cell
    elif failure == "format":
        cell["formats"].pop("fp16")
    elif failure == "scope":
        cell["formats"]["bf16"].pop("K")
    elif failure == "method":
        ratios.pop("raw_zstd-1")
    else:
        ratios["raw_zstd-1"] = float("nan") if failure == "nan" else 0
    with pytest.raises(ValueError):
        validate(data)


def test_changed_source_rejected(tmp_path):
    altered = tmp_path / "altered.json"
    altered.write_bytes(SOURCE.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA256"):
        export(altered)


def test_committed_exports_match():
    for name, content in export().items():
        assert (Path(SOURCE.parent) / name).read_bytes() == content
