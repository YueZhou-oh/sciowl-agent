# Bounded tool discovery and compact potential catalogs

SciOWL uses a small stable core plus on-demand operation discovery. This is a
better initial fit than an LLM router or lossy history summarizer: selection is
local and deterministic, no additional model/service is needed, and exact
scientific observations and evidence IDs remain intact. This does not establish
that lexical routing is more accurate than other routers; benchmark your model.

## Behavior

Every harness decision exposes seven core tools: `inspect_structure`,
`list_potentials`, `check_potential`, `get_job_status`, `get_result`, `cancel_job`,
and `search_tools`. Up to three optional operations are ranked by overlap with
the original request. Exact operation names have priority; common scientific
aliases cover relaxation, MD, EOS, batch prediction and MSD.

The model can call:

```json
{"tool":"search_tools","arguments":{"query":"equation of state bulk modulus"}}
```

The response contains up to three names and short descriptions. Their full
argument schemas become available on the **next** model decision. Discovery
replaces the optional set rather than accumulating definitions. `offset` and
`next_offset` support browsing the rest of the catalog. Do not request an
unloaded operation in the same parallel-call turn as its discovery: it has not
been exposed yet. Unloaded calls are rejected before execution.

Selections are reconstructed from saved discovery observations after resume.
Existing persisted pending calls still replay through the normal validated,
idempotent gateway. Tools remain registered in `contracts.TOOLS`; selection does
not authorize arbitrary functions or change scientific backend validation.

Tool descriptions should include domain synonyms. Lexical retrieval is not
multilingual semantic retrieval and can select irrelevant candidates; discovery
and pagination are the fallback. There is no extra LLM router. Old call/result
pairs stay intact even when their optional schemas leave the current set.
Changing optional definitions may reduce provider prefix-cache reuse.

## Potential metadata

`list_potentials` now returns **compact, paginated summaries**, not full registry
entries. The no-argument call remains valid. Default page size is 10, maximum 20.
Each entry exposes backend, demo status, and a domain preview (240 characters).
An allowlist prevents unknown future configuration fields from leaking into the
prompt. Query matches potential IDs or backend names, case-insensitively.

```bash
python -m sciowl_agent.cli --registry registry/registry_nep.json \
  tool list_potentials '{"query":"nep","offset":0,"limit":10}'
```

Follow `next_offset` when non-null. Use `check_potential` with the actual
structure/potential IDs to validate compatibility. The compact domain is a
preview, not a complete scientific specification. Exact registry data, element
coverage, model hashes and runtime configuration remain in the backend and job
specifications. Compatibility still does not establish scientific accuracy.

The `run`/`docs/run_at_once.sh` preflight uses this compact catalog too. When
`run --potential` is supplied, the catalog is filtered by that ID. If downstream
code previously expected `list_potentials` to return elements/hashes, update it
to use administrator registry data; do not treat the catalog as a config export.

## Limits and diagnostics

- Selected schemas have a 24,000-byte ceiling. Oversized schemas stop the run
  with `context_limit` rather than being silently truncated.
- Endpoint requests have a default 60,000-byte ceiling, configurable with
  `SCIOWL_MAX_INPUT_BYTES` or `EndpointModel(max_input_bytes=...)`.
- The endpoint check measures compact UTF-8 serialization of the complete JSON
  request, including messages, native tool definitions, and JSON-schema response
  constraints. It runs before HTTP submission in native, JSON, and schema modes.
- This is a **byte budget, not an exact token budget**. It does not know the
  server's chat-template overhead or context window. Configure it conservatively
  for your deployment and reserve generation tokens separately. Raising
  `max_tokens` does not enlarge the server's context window.
- The existing message-character limit remains active. Growing histories and
  very large individual results can still cause `context_limit`; this change
  intentionally does not discard scientific observations to continue.
- Model-context events record selected names, schema bytes and message chars.
  No API keys or authorization headers are added to these events.

Both endpoint probes now use production schema selection by default. Pass
`--all-tools` for the old full-catalog comparison. Raw probe files also report
`selected_tools` and `schema_bytes`. These transport probes do not execute tools.
Trajectory exports retain the existing full-catalog training format; they are
not exact replicas of the dynamically selected per-decision inference schemas.

## Verification

Run `python -m pytest -q`. The added tests cover 1,000 extra tools, exact-name
retrieval, paginated discovery, replacement rather than accumulation, resume,
1,000 potential entries, metadata exclusion, unloaded-call rejection, and
request-size rejection before HTTP in all three protocols. An end-to-end test
uses a mock model HTTP transport but **real ASE EMT EOS calculations**, discovering
an initially unloaded operation and retrieving its evidence before finalizing.

Measured compact tool-schema sizes for an energy/forces request (synthetic extra
tools each have a long description; these are bytes, not token measurements):

| Extra tools | All schemas | Selected schemas | Reduction |
| ---: | ---: | ---: | ---: |
| 0 | 6,937 | 4,565 | 34.19% |
| 100 | 165,337 | 4,565 | 97.24% |
| 1,000 | 1,590,937 | 4,565 | 99.71% |

These measurements demonstrate bounded schema exposure, not improved SciOWL
reasoning accuracy. Live endpoint tool-selection accuracy must be evaluated
against your deployed model. Optional tests need local model/tokenizer assets.
