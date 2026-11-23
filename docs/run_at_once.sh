#!/usr/bin/env bash
set -euo pipefail

[[ $# -eq 2 ]] || {
  echo 'Usage: bash docs/run_at_once.sh INPUT.cif "REQUEST"' >&2
  exit 2
}
: "${SCIOWL_BASE_URL:?Set SCIOWL_BASE_URL}"
: "${SCIOWL_MODEL:?Set SCIOWL_MODEL}"

PY="${SCIOWL_PYTHON:-python3}"
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="$(realpath -e -- "$1")"
REQUEST="$2"
mkdir -p "$ROOT/runs"
OUT="$(mktemp -d "$ROOT/runs/calculation-XXXXXXXX")"
REGISTRY="$("$PY" "$ROOT/scripts/prepare_registry.py" \
  --root "$ROOT" --output "$OUT/registry.json" --registry "${SCIOWL_REGISTRY:-}")"
cd -- "$ROOT"

printf 'Run directory: %s\n' "$OUT" >&2
"$PY" -m sciowl_agent.cli \
  --workspace "$OUT/workspace" --registry "$REGISTRY" run \
  --structure "$INPUT" \
  --protocol "${SCIOWL_PROTOCOL:-native}" \
  --max-tokens "${SCIOWL_MAX_TOKENS:-4096}" \
  --http-timeout "${SCIOWL_HTTP_TIMEOUT:-600}" \
  --max-steps 40 --max-jobs 8 \
  --wall-seconds 1800 --job-timeout 1800 \
  "Use the potential explicitly named in the request; do not substitute another if it is unavailable or incompatible. If none is named, choose a suitable registered potential and explain the choice. Check compatibility before calculations, retrieve completed results, and report actual convergence. User request: $REQUEST" \
  2> >(tee "$OUT/progress.log" >&2) | tee "$OUT/result.json"
