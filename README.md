# SciOWL Agent — executable reference implementation

This package implements the roadmap's constrained scientific-agent core. It includes a real LangGraph tool loop, SQLite session/job persistence, typed tools, content-addressed artifacts, ASE calculation workers, a potential registry, trajectory export, tokenizer-based loss masks, and validation scripts.

**Validation distinction:** local demonstrations use real ASE EMT calculations and a scripted expert policy. EMT is a classical potential, not an MLIP. Those results do not establish SciOWL capability or Ascend performance. SciOWL/NPU, optional MACE, Quantum ESPRESSO, Slurm, and MindSpeed must be validated in your environment. See `docs/STATUS.md`.

## 1. Install and run locally

Python 3.10+ on Linux. Use a separate environment from your NPU model server.

```bash
cd sciowl-agent
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[test,tokenizer]'
python3 -m sciowl_agent.cli --workspace runs/demo demo
bash scripts/run_local_validation.sh
```

The demo relaxes a perturbed periodic Cu cell, records tool observations, and writes `runs/demo/report.json` and `runs/demo/trajectory.jsonl`. Numerical results come from calculation, not hard-coded responses. The demo policy is explicitly scripted; it tests the harness and tools without an LLM.

Validation writes `validation/local/pytest.txt`, `pytest.xml`, `report.json`, and a replay workspace. All scientific examples are bounded demonstrations, not published benchmark scores.

## 2. Connect your SciOWL server

Export SciOWL to a supported Hugging Face format and validate export equivalence independently. In your matched vLLM-Ascend container:

```bash
export SCIOWL_MODEL_PATH=/path/to/exported/SciOWL-Instruct
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export TP_SIZE=4
export MAX_MODEL_LEN=8192
export SCIOWL_API_KEY=YOUR_INTERNAL_SERVICE_KEY
bash scripts/serve_ascend.sh
```

The script is a configurable starting command for Qwen3-style checkpoints; verify parser and flags against the exact installed serving version. It does not install CANN/vLLM or validate your checkpoint. Select TP/device allocation for your actual model and memory. Use private networking/authentication when another host connects.

On the controller host:

```bash
export SCIOWL_BASE_URL=http://YOUR_NPU_SERVER:8000/v1
export SCIOWL_API_KEY=YOUR_INTERNAL_SERVICE_KEY
python3 scripts/probe_endpoint.py --base-url "$SCIOWL_BASE_URL" --output endpoint_probe.json
```

No tool is executed by this probe. It tests a few basic decisions and arguments. If native function calling is not yet learned or parser-compatible, launch serving with `NATIVE_TOOL_CALLING=0` and use `--protocol json` on probe/agent commands. This adapter requests explicit JSON actions using ordinary chat roles.

The model client sends `chat_template_kwargs.enable_thinking=false` by default. Set `SCIOWL_ENABLE_THINKING=1` only after validating your thinking-mode serialization, reasoning parser, and output budget.

## 3. Import a structure and run the real agent

```bash
python3 -m sciowl_agent.cli import ./scripts/Cu4_fcc_emt_demo.cif
```

Copy the returned `structure_id` (replace `ARTIFACT_ID` below). Start a worker in a second terminal:

```bash
python3 -m sciowl_agent.cli worker --loop
```

Run the controller:

```bash
python3 -m sciowl_agent.cli agent \
  --session study-003 \
  --protocol json \
  --request 'Inspect structure a_d5e844e8e090c58b4611d1795637ea44f31f0b989d151a69e763f8119247bca7, check the EMT demonstration potential, then relax the atoms at fixed cell to a force tolerance of 0.03 eV/Angstrom. Report actual convergence.'
```

Only use EMT-covered example elements; Fe-containing systems are intentionally rejected. The agent may return `waiting` after submission. After the worker finishes, rerun the same command without `--request` to resume. Repeated invocations of a completed session return its saved result.

Add `--follow --wall-seconds 600` to follow worker completion automatically without calling the model repeatedly while jobs are still running. Internal endpoint requests ignore system proxies by default; set `SCIOWL_TRUST_ENV=1` if your deployment intentionally needs proxy settings (install httpx SOCKS support if relevant).

```bash
python3 -m sciowl_agent.cli --workspace runs/live agent \
  --session study-001 --base-url "$SCIOWL_BASE_URL" --model SciOWL-Instruct
```

Statuses include `thinking`, `waiting`, `completed`, `budget_exhausted`, and `context_limit`. A malformed model response persists a repair instruction; resume manually, bounded by the same step budget. HTTP errors remain visible; rerun after restoring service.

The harness rejects unobserved evidence IDs and pending-job completion claims. Numeric statements in the final qualitative summary are disallowed; evidence tables are populated from tool artifacts. Qualitative commentary is marked unverified. This does not constitute a general semantic proof of scientific correctness.

## 4. Use tools directly

Global `--workspace` and `--registry` flags must appear before the subcommand.

```bash
python3 -m sciowl_agent.cli --workspace runs/live tool list_potentials '{}'
python3 -m sciowl_agent.cli --workspace runs/live tool inspect_structure '{"structure_id":"ARTIFACT_ID"}'
python3 -m sciowl_agent.cli --workspace runs/live tool submit_eos '{"structure_id":"ARTIFACT_ID","potential_id":"emt_demo"}'
python3 -m sciowl_agent.cli --workspace runs/live worker
python3 -m sciowl_agent.cli --workspace runs/live tool get_result '{"job_id":"JOB_ID"}'
```

`worker` processes one queued job; `worker --loop` processes continuously. Calls are normalized and idempotent within a session: identical calculation requests reuse the job. To request an independent replica change the MD seed. To explicitly retry a failed identical calculation after fixing infrastructure, use a new session; review the previous worker first.

Implemented tools: inspection, potential listing/checking, single point, fixed-cell relaxation, fixed-fractional-coordinate EOS, fixed-cell NVE/NVT MD, job status/result/cancellation, and provisional species-resolved MSD analysis.

MD separates Langevin equilibration and production. It saves unwrapped Cartesian coordinates without wrapping, energy, temperature, symbols, and masses. Analysis checks fixed cell and even sampling; it never marks a diffusion fit scientifically converged. No NPT, variable-cell relaxation, phonons, NEB, or uncertainty-calibrated diffusion is claimed.

## 5. Register a real MLIP or DFT backend

### MACE

Install `.[mlip]` in a compatible worker environment; prepare trusted local weights. Copy `configs/registry_mace.example.json`, set actual elements, path, SHA256, device, and domain. The example is deliberately unvalidated and its hash must be replaced.

```bash
sha256sum /path/to/model.model
python3 -m sciowl_agent.cli --workspace runs/mlip --registry my_registry.json \
  tool check_potential '{"structure_id":"ARTIFACT_ID","potential_id":"my_mace"}'
```

The administrator registry is snapshotted into each job, so the worker uses the submitted configuration even if the registry later changes. Weight hashes are rechecked at execution. Upstream MACE NPU devices are intentionally rejected because this package has no validated MACE Ascend implementation. CPU/CUDA adapters are included but require actual-weight validation.

### Quantum ESPRESSO

`configs/registry_qe.example.json` configures ASE Espresso with an administrator-selected `pw.x` command, pseudopotentials, checksums, k-grid, and SCF settings. All examples require real local pseudopotentials and convergence studies. Use the same single-point/relaxation tools with that registry entry. QE runs in an isolated job directory; the model cannot supply executable commands. This adapter is not hardware-tested in the delivered validation.

### Active-learning acquisition

`scripts/select_active_learning.py` evaluates a trusted ensemble on registered candidates and ranks maximum per-atom force disagreement. It preserves member outputs. Supply at least two distinct members; disagreement is not calibrated uncertainty. DFT labeling uses the regular tools. Automatic model retraining/promotion is intentionally not implemented without a scientific training recipe and independent promotion suite.

## 6. Persistence and operational limits

- Atomic SQLite queue claiming and unique request keys prevent duplicate local submissions.
- Sessions are serialized using Linux file locks. Jobs persist independently of graph invocations.
- Stale workers are marked `interrupted`, never automatically resubmitted. Old owners cannot commit results afterward.
- `recover-stale --seconds 3600` is an operator action after checking the worker. It is not automatic job restart.
- Cancellation and the default 600-second limit are cooperative between calculation steps. A hung calculator/kernel requires process/scheduler termination; no hard preemption is claimed.
- SQLite/WAL and session locks are intended for one controller host with local storage. Do not place this DB on a multi-host NFS deployment. Use a service/transactional DB before distributed controllers.
- `scripts/run_in_slurm.sh` runs controller and worker on one allocated host. Configure partition/resources in `sbatch`, provide node-local workspace, and copy artifacts before allocation ends. It is an integration template, not a validated Slurm installation.
- The MVP limits systems to 256 atoms and bounds step counts. Administrators must validate before increasing limits.
- No arbitrary code tool or sandbox is enabled. This is a single-user trusted-workspace implementation, not a hardened multi-tenant web service.

## 7. Generate and export SFT data

```bash
python3 scripts/generate_dataset.py --output generated_dataset --structures 5
python3 -m sciowl_agent.cli --workspace runs/demo export \
  --session demo-relax --output demo_canonical.jsonl
```

The generator executes single-point, relaxation, and MD/MSD seed workflows and records real observations. It marks every record unreviewed. It groups the related Cu family together to prevent leakage; this small corpus cannot provide independent train/test evidence. Expand with different material families, tools, missing-information cases, and verified recovery scenarios.

After review, preserve canonical `tools`, `messages`, and metadata. Set `metadata.reviewed=true` only when review actually happened, or export with `--reviewed`. Failed assistant actions are marked `train=false`; observations remain context. Dataset validation rejects orphan observations and incomplete call chains.

```bash
python3 -m sciowl_agent.cli tokenize --input reviewed.jsonl --output tokenized.jsonl \
  --tokenizer /path/to/local/Qwen3-tokenizer --max-length 8192
```

The converter uses your tokenizer's actual chat template and fast-tokenizer offsets. It creates **one sample per supervised assistant decision**, retaining preceding observations as context. This handles Qwen3's different serialization of the last versus intermediate assistant turn. Only the target assistant span is labeled; preceding assistant turns and system/user/tool spans receive `-100`. Oversized decisions and prefix-unstable templates fail rather than silently corrupting training. The audit field contains rendered text, character spans, target message index, and supervised-token counts; inspect these before training. This reference uses empty thinking spans and does not synthesize chain-of-thought.

Optional ShareGPT export:

```bash
python3 -m sciowl_agent.cli --workspace runs/demo export --session demo-relax \
  --format sharegpt --output sharegpt.jsonl
```

This format requires checking role/tag mappings against your exact MindSpeed handler. Per-message loss-mask recovery records are rejected by this adapter because standard ShareGPT conversion cannot preserve those masks reliably.

## 8. MindSpeed indexed bins

Inside your existing MindSpeed environment with its actual Megatron and torch:

```bash
python3 scripts/build_mindspeed_bins.py --repo /path/to/MindSpeed-LLM \
  --input tokenized.jsonl --output-prefix /path/to/bins/sciowl_tools_train
```

This creates three indexed streams: `*_packed_input_ids_document`, `*_packed_attention_mask_document`, and `*_packed_labels_document`, and performs indexed readback. Signed int32 preserves `-100` labels. The builder refuses to overwrite files. Its integration cannot be executed here without your Megatron/NPU environment.

Use a copy of your validated full-SFT/LoRA launch recipe and point its SFT loader to the new data prefix according to that checkout. First run a tiny overfit/decoded-label audit. No fabricated universally compatible MindSpeed training launch command is supplied. Actual checkpoint conversion, training, and retention evaluation require your code revision and model assets.

## 9. Offline deployment

Run `scripts/build_offline_wheelhouse.sh` on an online machine matching the target's Python/OS/architecture. Transfer wheelhouse, source, tokenizer, and scientific assets. Install without index. The included `validation/environment.txt` records the local validation environment, not an aarch64/CANN compatibility lock. A controller wheelhouse and a matched NPU serving container are separate deliverables.

## 10. Validation and release scope

See `validation/local/report.json` for actual measurements and `docs/STATUS.md` for what is and is not validated. Rerun tests after dependency upgrades or backend changes. Demo success does not imply the roadmap's proposed 85% task success or 98% valid-call targets were achieved by SciOWL.

Evaluate the same small local suite with either policy:

```bash
python3 scripts/evaluate_agent.py --output validation/scripted_evaluation
python3 scripts/evaluate_agent.py --base-url "$SCIOWL_BASE_URL" --model SciOWL-Instruct --output validation/sciowl_evaluation
python3 scripts/audit_tokenizer.py --tokenizer /path/to/local/Qwen3-tokenizer --input runs/demo/trajectory.jsonl --output validation/tokenizer_audit.json
```

The related Cu cases are smoke tests, not an independent scientific benchmark. To enable the real-tokenizer pytest cases, set `SCIOWL_TEST_TOKENIZER=/path/to/local/Qwen3-tokenizer` when running validation. Without that variable those two tests are skipped explicitly.

Primary interfaces: [ASE](https://docs.ase-lib.org/), [LangGraph](https://docs.langchain.com/oss/python/langgraph/graph-api), [Qwen function calling](https://qwen.readthedocs.io/en/stable/framework/function_call.html), [vLLM-Ascend](https://github.com/vllm-project/vllm-ascend).

