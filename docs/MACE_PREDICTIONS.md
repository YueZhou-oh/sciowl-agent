# MACE energy, forces, and stress

This extension uses `mace.calculators.MACECalculator` through the existing
SciOWL queue and immutable artifact store. A single-point call predicts at the
input coordinates; it does not relax the structure.

## Install and register the model

In the Python environment used by the worker:

```bash
python -m pip install -e '.[mlip]'
```

Use a compatible PyTorch CPU/CUDA installation. The planner may run on an NPU
endpoint while the worker runs MACE on a CPU or CUDA GPU; upstream MACE NPU
execution remains rejected by this project. No model download is performed.

Create a registry from the existing template:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

model = Path('./mace-omat-0-medium.model').resolve()
cfg = json.loads(Path('configs/registry_mace.example.json').read_text())['my_mace']
cfg.update(
    model_path=str(model),
    sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
    device='cuda',                 # or 'cpu', 'cuda:0'
    default_dtype='float32',       # matches the supplied prediction demo
    elements=['Li', 'Fe', 'P', 'O'],  # EDIT: allowed elements covered by your checkpoint
    revision=model.name,
    domain='Local MACE checkpoint; application accuracy not independently validated',
)
Path('registry_mace.json').write_text(json.dumps({'mace_omat': cfg}, indent=2))
PY
```

The example element list is an application allowlist, not a statement of the
checkpoint's full coverage. Replace it for your inputs and checkpoint. Keep
`demo_only` and `domain` accurate for your validation status. Existing registries
without `default_dtype` retain their previous `float64` behavior. Each submitted
job snapshots the registry, and the worker rechecks the model hash before loading.
Only the operator supplies model paths; agent tools accept registered IDs.

## One CIF or XYZ structure

From the repository root:

```bash
ws=runs/mace
sid=$(python -m sciowl_agent.cli --workspace "$ws" import target.cif |
  python -c 'import json,sys; print(json.load(sys.stdin)["structure_id"])')

jid=$(python -m sciowl_agent.cli --workspace "$ws" --registry registry_mace.json \
  tool evaluate_energy_forces \
  "{\"structure_id\":\"$sid\",\"potential_id\":\"mace_omat\",\"stress_policy\":\"required\"}" |
  python -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')

python -m sciowl_agent.cli --workspace "$ws" worker --progress --timeout 1800
python -m sciowl_agent.cli --workspace "$ws" tool get_result "{\"job_id\":\"$jid\"}"
python -m sciowl_agent.cli --workspace "$ws" export-predictions \
  --job "$jid" --output target_mace.jsonl
```

`import` retains its existing first-frame behavior. Use `--all` below to process
all frames. The worker command above executes one queued job; use a fresh
workspace or run `worker --loop --progress` in another terminal for a standing
queue. Export only succeeds after the requested job has completed.

## Every frame in XYZ, extXYZ, or CIF

```bash
ws=runs/mace_batch
python -m sciowl_agent.cli --workspace "$ws" import target_test.extxyz --all > imported.json

arguments=$(python - <<'PY'
import json
from pathlib import Path
ids = json.loads(Path('imported.json').read_text())['structure_ids']
print(json.dumps({'structure_ids': ids, 'potential_id': 'mace_omat', 'stress_policy': 'auto'}))
PY
)

jid=$(python -m sciowl_agent.cli --workspace "$ws" --registry registry_mace.json \
  tool evaluate_structures "$arguments" |
  python -c 'import json,sys; print(json.load(sys.stdin)["job_id"])')

python -m sciowl_agent.cli --workspace "$ws" worker --progress --timeout 1800
python -m sciowl_agent.cli --workspace "$ws" tool get_result "{\"job_id\":\"$jid\"}"
python -m sciowl_agent.cli --workspace "$ws" export-predictions \
  --job "$jid" --output target_test_mace.jsonl
```

This is one job for the entire ordered list. The worker constructs one calculator
and evaluates frames sequentially, including repeated structures. It reports
`frame=N/total`, formula, energy and stress availability to stderr. Each JSONL row
contains:

| Field | Meaning |
| --- | --- |
| `index` | Zero-based index in the submitted list |
| `structure_id` | Registered source structure |
| `formula` | ASE chemical formula |
| `energy_eV` | Total energy, eV |
| `forces_eV_per_A` | N × 3 forces, eV/Å |
| `stress_eV_per_A3` | Six Voigt components, eV/Å³, or `null` |
| `stress_available` | Whether stress was computed successfully |
| `stress_convention` | ASE tension-positive; xx, yy, zz, yz, xz, xy |
| `stress_unavailable_reason` | Present when stress is unavailable or skipped |

Single-point results retain `forces_id`, `max_force_eV_per_A`, and (when
available) `stress_voigt_eV_per_A3`/`stress_voigt_GPa`. Both prediction tools also
return a `predictions_id` pointing to the JSONL artifact. Large force arrays stay
in artifacts instead of being inserted into the planner's context.

## Stress policy

- `auto` (default): compute stress for periodic structures with a nonsingular
  cell. Return `null` with a reason for nonperiodic inputs or calculators that
  explicitly do not implement stress.
- `required`: fail the job if stress is unavailable.
- `skip`: calculate only energy and forces; stress is `null`.

Plain XYZ often lacks a cell and PBC. Use extended XYZ with lattice/PBC metadata
or CIF for periodic stress prediction. The importer does not invent a cell.
Nonfinite outputs, invalid shapes, and calculator runtime failures fail the job;
they are not converted to unavailable-stress results. For slabs or wires, the
reported stress uses the full simulation-cell volume, including vacuum.

## Use from the agent

Start a worker in another terminal, then pass the imported structure ID to the
planner using the same workspace and the registry:

```bash
python -m sciowl_agent.cli --workspace "$ws" --registry registry_mace.json \
  agent --session mace-single --protocol native --follow --wall-seconds 1800 \
  --request "Inspect structure $sid, check potential mace_omat, then evaluate energy, forces and stress with evaluate_energy_forces and stress_policy required. Retrieve the completed result and cite its evidence. Do not relax the structure."
```

For a small multi-frame input, provide the `structure_ids` list from
`imported.json` and ask for `evaluate_structures`. Both native tool calling and
the JSON protocol get the new schema automatically. For large lists, direct CLI
submission avoids spending the planner context budget on hundreds of IDs.

## Limits and validation

The existing 256-atoms-per-frame and geometry checks remain in force. An import
or batch allows at most 1000 frames; split larger datasets into smaller files.
The batch counts as one session job. Cooperative cancellation/timeouts are
checked between frames and after each prediction, not inside a GPU kernel.
A failed/cancelled batch has no completed predictions artifact; a partial
`work/<job_id>/predictions.jsonl` may remain for operator diagnosis. Successful
exports require a new destination outside the managed workspace and never
overwrite an existing file.

Run the normal suite with `python -m pytest -q`. The MACE constructor test uses
an adapter spy with real EMT calculations; it does not validate MACE physics.
To compare the full queue path with direct MACE inference on a real, trusted
checkpoint supporting Cu:

```bash
SCIOWL_TEST_MACE_MODEL=/absolute/path/mace-omat-0-medium.model \
SCIOWL_TEST_MACE_DEVICE=cuda \
python -m pytest -q tests/test_mace_predictions.py::test_optional_real_mace
```

The real-model test is skipped unless the weight path is supplied. It checks
implementation agreement, not scientific accuracy of the model.
