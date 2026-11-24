#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
INPUT="${1:-$ROOT/scripts/Cu4_fcc_emt_demo.cif}"
REQUEST="${2:-Use grace to calculate energy, atomic forces and stress at the input geometry. Do not relax. Retrieve results.}"
exec bash "$ROOT/docs/run_at_once.sh" "$INPUT" "$REQUEST"
