# Native NEP integration

Uses the official GPUMD `nep` executable in prediction mode, adapted from
`predict_nep89.py`. No mace-torch dependency is needed. Install the normal package
and supply your compiled executable, CUDA runtime, and trusted NEP model.

## Register once

From the repository root:

```bash
python3 scripts/register_nep.py \
  --model /absolute/path/nep89_20250409.txt \
  --executable /absolute/path/GPUMD/src/nep \
  --output registry_nep.json
```

For the GLIBCXX workaround, add `--runtime-library-dir /absolute/path/runtime/usr/lib/x86_64-linux-gnu`
using the actual directory containing `libstdc++.so.6`. Only the child executable's
LD_LIBRARY_PATH changes. Otherwise the executable inherits the normal environment,
including CUDA_VISIBLE_DEVICES. The registry pins both model and executable hashes,
and extracts element coverage from the model header. Re-register after changing either asset.
The generator refuses to overwrite an existing registry.

## API workflow

Configure SCIOWL_BASE_URL, SCIOWL_MODEL and SCIOWL_API_KEY as for MACE, then:

```bash
bash docs/run_nep_api.sh scripts/Cu4_fcc_emt_demo.cif
```

Or request fixed-cell relaxation:

```bash
bash docs/run_nep_api.sh scripts/Cu4_fcc_emt_demo.cif \
  "Relax atomic positions at fixed cell to maximum force below 0.3 eV/Angstrom, up to 200 steps. Retrieve results and report actual convergence."
```

These commands use the `run` CLI from the one-command workflow branch. They import
one frame, check compatibility, run the worker, and retrieve evidence automatically.
The model must still produce valid tool calls.

## Existing tools and batch predictions

NEP uses `evaluate_energy_forces`, `evaluate_structures`, `submit_relaxation`,
`submit_eos`, and `submit_md` with `potential_id=nep89`; no new LLM tool schema is needed.
For a dataset, use `import input.extxyz --all`, pass its ordered `structure_ids`
to `evaluate_structures`, run the worker, then `export-predictions --job JOB --output OUTPUT`.
As with MACE, global `--workspace` and `--registry` options precede the subcommand.
Batch evaluation launches ONE native process for all frames. Single-point uses one
process; relaxation/MD relaunch per changed geometry, which has substantial overhead
and is not a substitute for GPUMD's optimized native MD engine.

Predictions have the existing MACE-compatible JSONL fields: total energy in eV,
atomic forces in eV/Angstrom, stress in eV/Angstrom^3, ASE Voigt order xx yy zz yz xz xy.
Native energy per atom is multiplied by atom count. Native stress GPa is reordered
from xx yy zz xy yz zx and divided by 160.21766208, matching the supplied script.
Stress policies auto/required/skip retain their existing result behavior; the native
executable may still compute stress internally when skip is selected.

Input cells must be nonsingular (including nonperiodic systems; provide an explicit
vacuum cell). Existing limits remain 256 atoms/frame and 1000 frames/import.
Supported headers are standard nepN and nepN_zbl as handled by the supplied script;
other model families are rejected rather than guessed. Native input/output and
stdout/stderr logs stay under work/JOB/calculator/prediction-* for debugging.
Cancellation and worker deadlines are checked while the child runs; timeout_s in
the registry separately limits each executable invocation.

Tests use a fake executable to verify integration, units, batching, failure handling,
and ASE relaxation plumbing. Real NEP89 numerical agreement, stress sign and GPU
performance must be checked against your reference script on your installed GPUMD
version before research use. No real NEP checkpoint or GPU execution was available
in the development environment.
