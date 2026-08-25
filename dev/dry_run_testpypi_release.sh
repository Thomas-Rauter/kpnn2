#!/usr/bin/env bash
set -euo pipefail

# Release helper for TestPyPI.
#
# Usage (run from project root):
#   chmod +x ./dev/dry_run_testpypi_release.sh
#   export TWINE_USERNAME="__token__"
#   export TWINE_PASSWORD="pypi-..."
#   ./dev/dry_run_testpypi_release.sh
#
# Notes:
# - TWINE_PASSWORD must be a TestPyPI API token, not a real PyPI token.
# - TestPyPI does not allow re-uploading the same version.
# - For test releases, use versions such as 0.1.0rc1, 0.1.0rc2, etc.
# - This script creates and deletes a temporary virtual environment.
# - The TestPyPI install bypasses pip's HTTP cache and retries, because a
#   just-uploaded version can be missing from a cached or lagging index.

TEST_ENV="$(mktemp -d /tmp/kpnn2-test.XXXXXX)"
START_DIR="$(pwd)"

cleanup() {
    cd "$START_DIR"
    rm -rf "$TEST_ENV"
}

trap cleanup EXIT

echo "Step 1: Running local quality checks"
ruff check .
ruff format --check .
mypy src
pytest

echo "Step 2: Removing old build artifacts"
rm -rf dist build *.egg-info src/*.egg-info

echo "Step 3: Installing build tools"
python -m pip install --upgrade build twine

echo "Step 4: Building package"
python -m build

echo "Step 5: Checking distribution"
python -m twine check dist/*

PACKAGE_VERSION="$(python - <<'PY'
import tomllib
from pathlib import Path

pyproject = tomllib.loads(Path("pyproject.toml").read_text())
declared = pyproject["project"]["version"]
wheels = list(Path("dist").glob("*.whl"))
if len(wheels) != 1:
    raise SystemExit(
        f"expected exactly one wheel in dist/, found {len(wheels)}"
    )
built = wheels[0].name.split("-", 2)[1]
if built != declared:
    raise SystemExit(
        f"wheel version {built!r} does not match pyproject.toml {declared!r}"
    )
print(built)
PY
)"
export PACKAGE_VERSION

echo "Step 6: Uploading to TestPyPI"
python -m twine upload --repository testpypi dist/*

echo "Step 7: Creating clean install test environment"
python -m venv "$TEST_ENV"

# shellcheck disable=SC1091
source "$TEST_ENV/bin/activate"

python -m pip install --upgrade --no-cache-dir pip

echo "Step 8: Installing kpnn2 ${PACKAGE_VERSION} from TestPyPI"
pip_install_from_testpypi() {
    python -m pip install \
        --no-cache-dir \
        --pre \
        --index-url https://test.pypi.org/simple/ \
        --extra-index-url https://pypi.org/simple/ \
        "kpnn2==${PACKAGE_VERSION}"
}

attempt=1
max_attempts=12
until pip_install_from_testpypi; do
    if (( attempt == max_attempts )); then
        echo "error: TestPyPI never listed kpnn2==${PACKAGE_VERSION}." >&2
        exit 1
    fi
    echo "TestPyPI does not list ${PACKAGE_VERSION} yet (attempt ${attempt}/${max_attempts}). Retrying in 15s..."
    attempt=$((attempt + 1))
    sleep 15
done

echo "Step 9: Running installed-package smoke test outside repository"
cd /tmp

python - <<'PY'
import os

import pandas as pd
import torch

import kpnn2
from kpnn2 import (
    GraphSpec,
    MaskedLinear,
    align_inputs,
    parse_edgelist,
)

expected = os.environ["PACKAGE_VERSION"]
installed = getattr(kpnn2, "__version__", "unknown")
print(installed)
if installed != expected:
    raise SystemExit(
        f"installed kpnn2 {installed!r}, expected {expected!r}"
    )

edgelist = pd.DataFrame(
    {
        "source": ["feature_a", "feature_b", "hidden"],
        "target": ["hidden", "hidden", "prediction"],
    }
)
spec = parse_edgelist(edgelist)
assert isinstance(spec, GraphSpec)
assert spec.input_nodes == ("feature_a", "feature_b")

data = pd.DataFrame(
    {
        "feature_b": [2.0, 4.0],
        "feature_a": [1.0, 3.0],
    }
)
x = align_inputs(data, spec)
layer = MaskedLinear(spec.masks[0])
with torch.no_grad():
    h = layer(x)

print("input_nodes:", spec.input_nodes)
print("aligned_shape:", tuple(x.shape))
print("hidden_shape:", tuple(h.shape))
print("TestPyPI install smoke test passed.")
PY

echo "Done. Temporary environment removed."
