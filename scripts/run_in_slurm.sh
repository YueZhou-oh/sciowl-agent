#!/usr/bin/env bash
# Submit THIS SCRIPT with sbatch and administrator-selected partition/time/memory.
# Keeps controller + SQLite + workers on one allocated host; SciOWL can be remote.
set -euo pipefail
: "${SCIOWL_WORKSPACE:?Set node-local SCIOWL_WORKSPACE}"
: "${SCIOWL_SESSION:?Set SCIOWL_SESSION}"
: "${SCIOWL_BASE_URL:?Set SCIOWL_BASE_URL}"
python3 -m sciowl_agent.cli --workspace "$SCIOWL_WORKSPACE" worker --loop &
worker_pid=$!
trap 'kill "$worker_pid" 2>/dev/null || true' EXIT
python3 -m sciowl_agent.cli --workspace "$SCIOWL_WORKSPACE" agent \
  --session "$SCIOWL_SESSION" --request "${SCIOWL_REQUEST:-Resume the research task}" \
  --base-url "$SCIOWL_BASE_URL" --model "${SCIOWL_MODEL:-SciOWL-Instruct}" --follow --wall-seconds "${AGENT_WALL_SECONDS:-3600}"
echo "Sync node-local artifacts before the allocation ends."

