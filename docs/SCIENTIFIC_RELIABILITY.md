# Scientific execution contracts, recovery, and evidence assessments

This change targets three testable capabilities beyond context management.
They are implemented engineering features; claims of scientific novelty or
improved model accuracy require a separate comparative evaluation.

## 1. Enforce explicit execution contracts

`run --potential` now binds a persisted constraint, rather than only prompting
the model to use the chosen potential. `agent` also accepts `--potential`.
Both commands accept repeatable `--allowed-operation` flags. For example:

```bash
python -m sciowl_agent.cli --workspace runs/single --registry registry/registry_nep.json \
  run --structure scripts/Cu4_fcc_emt_demo.cif --potential nep89 \
  --allowed-operation evaluate_energy_forces \
  'Calculate energy, atomic forces and stress at the input geometry.'
```

The gateway checks calculation arguments against these constraints before
submission, including on resume. A model cannot substitute another potential or
submit a relaxation when only single-point evaluation is allowed. Constraints
are shown in native, JSON and schema prompts; they are validated in code, not
merely trusted to the model. Existing session constraints cannot be overwritten
by different flags. Omit the flags when resuming to retain saved constraints.

Natural-language restrictions are NOT automatically extracted into contracts.
For a hard restriction, set the explicit flags. These checks protect execution
within a session; they are not a security boundary against a user who controls
the workspace or intentionally submits under a different session.

Shared preflight checks reject invalid EOS periodicity/volume factors, MD
sampling configurations with fewer than three frames, and geometrically
undefined required stress before a job is queued. Existing geometry, element,
asset-integrity and calculator restrictions remain active. Workers repeat the
operation checks when executing stored jobs. Required stress can still fail at
runtime if the selected calculator does not implement it. Partial periodicity
retains the existing stress policy; EOS requires full periodicity.

## 2. Recover from endpoint failure without losing execution state

HTTP request errors and unsuccessful HTTP status responses now persist an
`endpoint_error` checkpoint with `last_error` containing an error class, HTTP
status (when available), and retryability classification. Timeouts, connection
errors, 408, 429 and server failures are marked potentially retryable. Other
HTTP client errors require configuration correction. No raw provider response,
URL or headers are copied into these diagnostics.

There is no automatic request retry or scientific-job resubmission. `--follow`
stops on `endpoint_error`. Fix the endpoint and explicitly resume the same
workspace and session with `agent --session SESSION --follow` and the same
registry/endpoint configuration. Each unsuccessful request consumes a decision
step, so repeated resumes remain bounded by `--max-steps`.

For the one-command `run` workflow, the returned `workspace` is the nested
workspace to use for resume and `session.session_id` identifies the session.
Do not rerun `run` to resume: it deliberately creates a new session/workspace.

The agent CLI prints valid JSON and returns exit code 1 for endpoint, context,
or decision-budget failures. Use `set -o pipefail` when piping to `tee`. The
`run` CLI already returns nonzero for non-completed execution.

Tested recovery includes a timeout after a calculation has completed: resuming
retrieves its existing result without duplicating the job. Stale worker recovery
is still explicit; this change does not retry failed or interrupted calculations.

## 3. Assess evidence separately from model commentary

Reports now contain `job_assessments`, generated deterministically from job
specifications and artifact-backed results. Finalization requires cited result
or failure evidence for **every** session job. It verifies result operation,
input IDs, potential ID, provenance-to-job/spec linkage and integrity of linked
provenance, forces, predictions, trajectories and output structures.

| Operation/state | Scientific outcome |
| --- | --- |
| Single-point / batch prediction | `calculated` |
| Relaxation | `converged` or `not_converged`, recomputed against the requested force threshold |
| EOS | `fit_checks_passed` or `fit_checks_failed`, from the stored numerical checks |
| MD | `provisional_trajectory`; no diffusion-convergence certification |
| Failed/cancelled/interrupted job | `no_successful_result` |

`completed` remains an execution status. It does not mean that convergence was
achieved or that the potential is scientifically accurate. Arbitrary model prose
remains explicitly `commentary_verified: false`; this is not a semantic verifier
of every sentence. Evidence assessments apply to newly finalized reports; old
saved reports are not retroactively rewritten.

## Verification and research evaluation

Run `python -m pytest -q`. Added tests exercise invalid requests without job
creation, immutable session constraints, every endpoint protocol, HTTP error
classification, explicit resume without duplicate jobs, nonzero CLI failure,
unconverged relaxation, omitted failed-job evidence, corrupt linked artifacts,
and mismatched provenance. Numerical tests use real CPU ASE EMT calculations;
endpoint failures are injected using mocked transports.

Suggested controlled ablations for a paper:

1. Prompt-only restrictions versus enforced contracts: forbidden-operation and
   potential-substitution rates under the same generated calls.
2. Exception-only failure handling versus resumable checkpoints: retained state,
   duplicate calculations and completion after injected outages.
3. Prose-only reporting versus deterministic assessments: omitted failure rate,
   convergence mislabeling and invalid evidence acceptance.

Do not claim these tests measure deployed SciOWL planning quality, MACE/NEP/GRACE
accuracy, or calibration under distribution shift. Those need separate model and
scientific validation datasets.
