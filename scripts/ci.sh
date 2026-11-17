#!/usr/bin/env bash
# The CI gates. Each function is one step of .github/workflows/test.yml, which
# runs them in this order and decides there which Python job runs which gate.
#
#   scripts/ci.sh GATE...   run the named gates with the active python/pip
#   scripts/ci.sh           repeat the CI job for the active Python version:
#                           every gate, in order, in a throwaway environment
#
# Gates change the environment they run in: `torch` installs PyTorch (and
# `entry` requires it to be absent), `fallback` and `wheel` replace the
# editable install with built wheels. Hence the fresh environment for a full
# run; after running those gates by hand, run `install` again.
set -e
cd "$(dirname "$0")/.."

gate_results() {  # Recompute archived results without third-party dependencies
    python bench/export_results.py --check
}

gate_install() {
    pip install -e '.[test,codecs]'
}

gate_entry() {  # CPU entry without PyTorch
    python -c "import importlib.util; assert importlib.util.find_spec('torch') is None"
    python -m kvbytepack
    python -c "from kvbytepack.codec import _native_merge; assert _native_merge is not None"
}

gate_evidence() {  # AIBrix isolated contract and complete-record experiment
    python -m examples.aibrix_layout.benchmark --check results/aibrix_layout
    python -m examples.aibrix_layout.capture --check results/aibrix_layout/model_capture
    python bench/compare_cpu.py --check results/cpu_decode
    git fetch origin 98f56cc16fe46195d70730e99ac64b754d5f75ad
    git worktree add --detach "$RUNNER_TEMP/kvbytepack-before" FETCH_HEAD
    python bench/compare_cpu.py --baseline-dir "$RUNNER_TEMP/kvbytepack-before" --rounds 2 --repeats 3 --warmups 1 --out-dir "$RUNNER_TEMP/cpu-comparison"
    python -m examples.aibrix_layout.benchmark --repeats 3 --out-dir "$RUNNER_TEMP/aibrix-layout"
}

gate_torch() {
    pip install torch --index-url https://download.pytorch.org/whl/cpu
}

gate_test() {
    python -m pytest -q tests
}

gate_fallback() {  # Verify no-compiler wheel and source fallback (of the committed HEAD)
    git archive --format=tar HEAD -o "$RUNNER_TEMP/kvbytepack-source.tar"
    mkdir "$RUNNER_TEMP/kvbytepack-fallback"
    tar -xf "$RUNNER_TEMP/kvbytepack-source.tar" -C "$RUNNER_TEMP/kvbytepack-fallback"
    CC=false python -m pip wheel "$RUNNER_TEMP/kvbytepack-fallback" --no-deps --wheel-dir "$RUNNER_TEMP/fallback-dist"
    python -m pip install --force-reinstall --no-deps "$RUNNER_TEMP"/fallback-dist/kvbytepack-*.whl
    cd "$RUNNER_TEMP"
    python -c "from kvbytepack.codec import _native_merge; assert _native_merge is None"
    python -m kvbytepack
    python "$RUNNER_TEMP/kvbytepack-fallback/bench/compare_cpu.py" --check "$RUNNER_TEMP/kvbytepack-fallback/results/cpu_decode"
}

gate_wheel() {  # Build and test the installed wheel outside the checkout
    pip wheel . --no-deps --wheel-dir dist
    pip install --force-reinstall --no-deps dist/kvbytepack-*.whl
    cd "$RUNNER_TEMP"
    python -m kvbytepack
    python -c "from kvbytepack.codec import _native_merge; assert _native_merge is not None"
    python -c "import kvbytepack, kvzip, kvzip.codec; assert kvzip.encode is kvbytepack.encode; assert kvzip.codec is kvbytepack.codec"
}

# GitHub provides RUNNER_TEMP in a fresh checkout. Elsewhere use a scratch
# directory and leave nothing behind: neither it, nor the baseline worktree
# `evidence` registers, nor the dist/ that `wheel` fills (a stale wheel there
# would break its install glob), unless dist/ already existed.
cleanup() {
    [ ! -d "$RUNNER_TEMP/kvbytepack-before" ] || git worktree remove --force "$RUNNER_TEMP/kvbytepack-before"
    rm -rf "$RUNNER_TEMP"
    [ -n "$keep_dist" ] || rm -rf dist
}
if [ -z "$RUNNER_TEMP" ]; then
    RUNNER_TEMP=$(mktemp -d)
    [ ! -e dist ] || keep_dist=1
    trap cleanup EXIT
fi

if [ $# -eq 0 ]; then
    python -m venv "$RUNNER_TEMP/venv"
    . "$RUNNER_TEMP/venv/bin/activate"
    # The workflow runs evidence, torch and fallback only in its Python 3.12
    # job: the pinned AIBrix fixture they load needs Python 3.10 or newer.
    if python -c "import sys; sys.exit(sys.version_info < (3, 10))"; then
        set -- results install entry evidence torch test fallback wheel
    else
        set -- results install entry test wheel
    fi
fi

for gate; do
    if ! declare -F "gate_$gate" > /dev/null; then
        echo "unknown gate: $gate" >&2
        exit 2
    fi
    echo "== $gate"
    ( "gate_$gate" )
done
