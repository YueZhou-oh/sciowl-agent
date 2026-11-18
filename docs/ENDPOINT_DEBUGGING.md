# Debug SciOWL tool calls with a vLLM endpoint

Run these commands from the repository root, in the controller's Python environment. Use a new output directory for each investigation. Set the URL to the address reachable **from the agent container**.

```bash
export SCIOWL_BASE_URL=http://127.0.0.1:8005/v1 SCIOWL_MODEL=sciowl
export SCIOWL_ENABLE_THINKING=0
d=$(mktemp -d)
python3 -m pip install -e '.[test]'
python3 -m pytest -q
```

## 1. Check generation independently

```bash
curl --noproxy '*' -sS -N --connect-timeout 10 --max-time 180 \
  "$SCIOWL_BASE_URL/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"sciowl","messages":[{"role":"user","content":"Reply with exactly OK"}],"temperature":0,"stream":true,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

An assistant delta followed by a stop and `[DONE]` establishes streaming generation for this request. The production adapter sends a non-streaming request and waits for the complete assistant turn.

## 2. Inspect a single non-executing call

```bash
for p in native json schema; do
  if ! python3 -m scripts.probe_sciowl_raw --protocol "$p" --max-tokens 512 \
    --timeout 180 --output "$d/$p.json"; then
    echo "Inspect $d/$p.json"
  fi
done
```

The expected action is exactly one `list_potentials` with `{}`. Inspect the saved HTTP status, full response, usage, finish reason and adapter output. With `--state path/to/session.json` the probe replays the first system/user pair and skips the list-specific score. Captures may include prompt or model content. Review before sharing.

| Observation | Interpretation and next check |
|---|---|
| HTTP 400 | Read the saved error body. For a context-length error, reserve space for both prompt and completion. For an unsupported schema, check backend support. |
| ReadTimeout while waiting for headers | Compare the short streaming check with the bounded non-streaming probe. Check request timing and generated tokens in vLLM. Worker timeout and agent follow time govern separate operations. |
| HTTP 200, finish_reason length | Inspect the preview. A schema echo or invented catalog indicates the model spent its budget on prose. Raising the limit alone can prolong the same failure. |
| Native response mentions a tool in content, with no tool_calls | The agent received prose. Verify the rendered chat template includes the tools and expects the same call format as the server parser. |
| JSON response selects list_potentials with structure_id | The tool name is correct but its argument contract requires an empty object. The probe fails this case. |
| Only a user turn in saved history | Inspect the HTTP capture and adapter error. A failure before a usable assistant message is returned can leave that turn absent. |

Native uses vLLM tool parsing. JSON asks for a text envelope. The optional schema protocol requests JSON-schema-constrained decoding with separate argument fields per tool and a final-answer branch. Backend support varies. A schema can enforce field shape while still allowing an incorrect tool or an invented identifier.

## 3. Check tool choice, argument values and a final answer

```bash
python3 -m scripts.probe_endpoint --protocol schema --max-tokens 512 \
  --timeout 180 --output "$d/interface.json"
```

Repeat with `native` or `json` to compare. Require all three probes to pass before trying that protocol in a live job. These checks exercise the adapter without dispatching a calculation. On a larger held-out set, measure exact tool and arguments, copied structure/job IDs and successful result retrieval. Persistent selection or value errors after verifying the template and decoding configuration are grounds to try targeted tool-use SFT.

## 4. Exercise the worker, then the full agent

Use the [MACE setup and direct CLI example](MACE_PREDICTIONS.md) to verify the checkpoint, registry, worker and export independently. The CLI accepts explicit arguments for a fixed workflow. The agent generates arguments and the gateway validates them before dispatch.

After the direct MACE run succeeds, use the same environment and registry for the full run:

```bash
export SCIOWL_REGISTRY="$PWD/registry_mace.json"
export SCIOWL_PROTOCOL=schema SCIOWL_MAX_TOKENS=512 SCIOWL_HTTP_TIMEOUT=180
bash docs/run_mace_api.sh scripts/Cu4_fcc_emt_demo.cif
```

Use the protocol that passed on your deployment. The script starts a worker, follows one single-point job, retrieves the result and exports predictions outside the managed workspace. Compare energy, atom count, forces and stress with the direct calculation for the same input and checkpoint.
