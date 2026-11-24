# GRACE in a separate Python environment

The `grace` backend integrates the supplied `tensorpotential.calculator.TPCalculator`
example into the existing `evaluate_energy_forces`, `evaluate_structures`, and
`submit_relaxation` tools. Model inference uses
`/dpc-zhouy/zhouy/miniconda3/envs/GRACE/bin/python`; the agent and ASE optimizer
remain in the normal SciOWL environment. Do not install TensorPotential or
TensorFlow into the agent environment just for this backend.

## Register once on the calculation host

Run from the repository root using the normal SciOWL Python:

```bash
python3 scripts/register_grace.py --elements Cu
```

Defaults match the supplied example:

- Model directory: `/dpc-zhouy/zhouy/sciowl-agent-tools/demo/GRACE/GRACE-2L-SMAX-OMAT-large`
- Python: `/dpc-zhouy/zhouy/miniconda3/envs/GRACE/bin/python`
- Output: `registry/registry_grace.json`
- Potential ID: `grace`

`--elements Cu` registers only copper for this demo. Supply other verified elements
with `--elements Cu Al ...` when appropriate for your checkpoint; the helper does
not infer training coverage from the model name. `--model`, `--python`,
`--potential-id`, and `--output` can override the defaults. Registration refuses
to overwrite an existing file. It computes a deterministic SHA256 over the model
files and relative paths; changing any model file requires deliberate registration
of the updated model. Registration validates paths and hashes, not physical accuracy
or whether TensorPotential can load the model.

## Natural-language single-point or relaxation

With `SCIOWL_BASE_URL`, `SCIOWL_MODEL`, and `SCIOWL_API_KEY` already configured:

```bash
bash docs/run_grace_api.sh
bash docs/run_grace_api.sh scripts/Cu4_fcc_emt_demo.cif \
  "Use grace to relax below 0.3 eV/Angstrom, up to 5 steps. Retrieve results and report actual convergence."
```

The shared `prepare_registry.py` automatically includes `registry_grace.json`
alongside the MACE and NEP registries. Do not set `SCIOWL_PYTHON` to the GRACE
interpreter: that variable selects the agent's Python, while the registry's
`python_executable` selects GRACE's Python.

## Original standalone dataset example

The supplied example is preserved as `scripts/predict_grace.py`:

```bash
/dpc-zhouy/zhouy/miniconda3/envs/GRACE/bin/python scripts/predict_grace.py
```

It retains the input/output paths from the supplied file. Edit its `MODEL`, `INPUT`,
and `OUTPUT` constants when needed. It loads the model once and writes JSONL with
energy in eV, forces in eV/Angstrom, and stress in eV/Angstrom^3.

## Execution details and limits

The child process sets `TF_USE_LEGACY_KERAS=1` before importing TensorPotential,
then creates `TPCalculator(model_path, mode="diverse", enable_uq_if_available=False)`.
Input/output JSON files keep dependency logs separate from numerical results.
Child stdout and stderr are captured in each job's `calculator/prediction-*`
directory. Worker cancellation/deadlines and the registry's `timeout_s` terminate
a running child. Predictions must have finite energies, `(N, 3)` forces, and
six-component ASE Voigt stress (xx, yy, zz, yz, xz, xy) when available.

Batch `evaluate_structures` loads the model once per submitted batch. Relaxation
currently starts a fresh GRACE process for each new geometry evaluation; this
isolates dependencies but incurs model-loading overhead. Stress is requested for
periodic nonsingular cells; unavailable stress is handled by the existing
`auto`/`required` policy. Fixed-cell relaxation convergence is determined from the
actual forces, not job completion or the agent's prose.

Tests use a synthetic child process and a stub TPCalculator. They validate the
integration contract, not GRACE model accuracy. A live run with your checkpoint
and GRACE environment is still required on your machine.
