# Calculate from a structure path and a natural-language request

Use `run` to import a local structure, inspect it, list registered potentials,
check an explicitly selected potential, and execute the agent and worker together.
No manual structure ID, JSON tool arguments, session name, or background worker
is needed. Endpoint URL/model/API key use the existing `SCIOWL_*` environment.

```bash
python3 -m sciowl_agent.cli --registry registry_mace.json run \
  --structure scripts/Cu4_fcc_emt_demo.cif --potential mace_omat \
  "Calculate total energy, atomic forces and stress at the input geometry. Do not relax or run MD."
```

For atomic relaxation instead:

```bash
python3 -m sciowl_agent.cli --registry registry_mace.json run \
  --structure target.cif --potential mace_omat \
  "Relax atomic positions at fixed cell to a maximum force below 0.3 eV/Angstrom, with at most 200 steps. Report actual convergence."
```

The installed console entry point `sciowl-agent` can replace
`python3 -m sciowl_agent.cli`. Quote paths containing spaces. `--structure` is
explicit so local file access does not depend on the model parsing a filename.
Only the first frame is imported; use existing batch tools for multiple frames.
If `--potential` is omitted, the model selects from registered potentials and
should check compatibility. The gateway always validates compatibility before
submitting calculations. The registry must permit the requested operation.

Every invocation gets an isolated workspace under
`<--workspace>/runs/run-<unique-id>` (default parent: `sciowl-workspace`).
Progress goes to stderr; stdout is a JSON object containing preflight checks,
session state, full calculation results and the evidence-backed report:

```bash
python3 -m sciowl_agent.cli --registry registry_mace.json run \
  --structure target.cif --potential mace_omat \
  "Calculate energy and forces without relaxation." \
  > result.json 2> progress.log
```

For endpoints supporting JSON-schema constrained output, use `--protocol schema`.
The default protocol is `SCIOWL_PROTOCOL` or `native`. The output budget defaults
to `SCIOWL_MAX_TOKENS` or 4096; `--max-tokens` overrides it. Native mode requires
appropriate server tool-call support. This wrapper does not fix a model that
cannot produce valid actions. It exits nonzero on preflight failure, exhausted
budgets, failed calculations or final answers without any calculation.

`--wall-seconds` bounds continuation between agent/worker calls; it is not a
hard interrupt for an in-flight HTTP or calculator call. HTTP timeout and
cooperative job timeout are separately configurable via `--http-timeout` and
`--job-timeout`. `--quiet` hides worker progress. Existing `agent`, `import`, and
`worker` commands retain their behavior.
