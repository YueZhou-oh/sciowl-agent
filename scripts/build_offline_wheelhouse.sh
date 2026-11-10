#!/usr/bin/env bash
# Run on an online machine with the SAME OS/architecture/Python as target.
# This is the controller environment; do not replace your NPU serving stack with it.
set -euo pipefail
cd "$(dirname "$0")/.."
dest="${1:-wheelhouse}"
mkdir -p "$dest"
python3 -m pip wheel --wheel-dir "$dest" '.[test]'
python3 -m pip freeze > "$dest/build-environment.txt"
echo "On the offline target: python3 -m pip install --no-index --find-links '$dest' sciowl-agent"

