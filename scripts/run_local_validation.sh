#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m scripts.validate --output "${1:-validation/local}"

