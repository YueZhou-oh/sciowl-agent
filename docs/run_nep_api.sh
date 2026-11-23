#!/usr/bin/env bash
set -euo pipefail
: "${SCIOWL_BASE_URL:?Set endpoint URL}"
: "${SCIOWL_MODEL:?Set served model name}"
python3 -m sciowl_agent.cli --registry "${SCIOWL_REGISTRY:-registry_nep.json}" run \
  --structure "${1:-scripts/Cu4_fcc_emt_demo.cif}" --potential nep89 \
  --max-tokens "${SCIOWL_MAX_TOKENS:-4096}" \
  --wall-seconds 1800 --job-timeout 1800 \
  "${2:-Calculate energy, atomic forces and stress at the input geometry. Do not relax. Retrieve the completed result.}"
