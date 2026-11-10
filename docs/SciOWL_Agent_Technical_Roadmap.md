# SciOWL Agent: Technical Roadmap

Version 1.0 | 3 November 2026

## 1. Objective and scope

Build a reproducible scientific agent using SciOWL-Instruct as the decision model. The agent should interpret research objectives, select validated scientific tools, configure calculations, interpret observations, recover from supported failures, and produce evidence-linked reports.

The first release targets atomistic materials calculations: structure inspection, potential compatibility, single-point energy/forces, relaxation, and equation-of-state analysis. Subsequent releases add molecular dynamics, transport analysis, DFT escalation, and active learning. Molecular and life-science workflows become separate validated tool families.

The initial architecture is SciOWL inference on Ascend NPU, orchestration on CPU, and independent scientific workers. Scientific workers can use CPU, supported GPUs, or validated NPU backends. Full NPU acceleration is a separate workstream and is not a prerequisite for agent development.

All proposed budgets, sample counts, thresholds, and schedules in this document are engineering starting points. They are not measured SciOWL results or universal scientific standards. Repository interfaces are based on documentation reviewed in the preceding discussion; pin and verify exact revisions before implementation.

## 2. Core architecture and responsibility boundaries

| Component | Responsibility | Initial implementation |
| --- | --- | --- |
| SciOWL policy | Understand requests, propose actions, interpret observations, decide next steps | Existing SciOWL-Instruct checkpoint behind a compatible API |
| Harness | Build context, validate actions, persist workflow state, enforce budgets, route execution | LangGraph with typed state |
| Tool gateway | Validate schemas, resolve artifact IDs, dispatch approved operations | Python/Pydantic and HTTP where needed |
| Scientific workers | Execute tested calculation and analysis implementations | ASE/pymatgen plus selected calculators |
| Scheduler | Allocate resources and track long-running jobs | Local subprocess workers initially; Slurm adapter later |
| Evidence store | Preserve inputs, trajectories, logs, results, hashes, and claim provenance | Files/object storage plus SQLite initially |
| Evaluation service | Replay cases, grade numerical and operational outcomes | Versioned task suite and deterministic graders |
| Training pipeline | Convert verified trajectories into SFT records | Tool-aware conversion for the actual MindSpeed revision |

SciOWL proposes an action. The harness checks whether it is valid and executes it. Tool observations come from the execution layer. Successful process exit is distinct from scientific convergence.

Use one SciOWL controller initially. Deterministic validators remain outside the model. Multiple planner/reviewer roles should be introduced only after a controlled experiment demonstrates a benefit.

Recommended control states:

`intake -> inspect -> plan -> validate -> dispatch -> waiting -> analyze -> decide -> report`

Decision branches can request clarification, repeat an analysis with justified settings, launch a dependent job, or stop with an explicit limitation. Pending jobs are resumed by scheduler events or bounded polling rather than continuous model calls.

## 3. Framework and repository choices

| Project | Role | Decision |
| --- | --- | --- |
| LangGraph | Stateful orchestration and recovery | Primary runtime for constrained workflows |
| Deep Agents | Higher-level harness, filesystem and planning capabilities | Optional when broader coding/skills behavior is required |
| Qwen-Agent | Convenient tool-calling prototype | Useful baseline; not an additional mandatory runtime |
| AtomisticSkills | Scientific tools, tutorials, and workflows | Reuse selected interfaces and skills; validate each backend |
| Aitomia | Molecular computational chemistry | Separate molecular track |
| VASPilot / VASP Agent | VASP preparation, execution, and recovery | Reference or adapter for DFT escalation |
| ORNL matsim-agents | Materials agent patterns and MLIP/ASE workflows | Domain reference; verify implementation at pinned revision |
| ScienceDiscovery | Existing scientific workbench with Ascend execution integration | Alternative product foundation if adopting a full application |
| Ascend AgentSDK/Aura | Agent training/inference infrastructure | Evaluate after tool-use SFT and rewards are reliable |

Do not combine every framework. For the first build, choose LangGraph, a tool gateway, and ASE workers. Borrow domain code through well-defined adapters.

Generic harness compatibility through an API is different from documented native Ascend integration. None of these choices establishes that an MLIP runs on NPU. Current documented AtomisticSkills GPU environments use CUDA; containers do not automatically translate them to Ascend. The official VASP accelerator documentation does not list an Ascend backend.

## 4. SciOWL model serving on Ascend

1. Export the actual SciOWL architecture to a supported inference checkpoint format. If using LoRA, merge or load adapters only through a validated serving path.
2. Verify tokenizer, chat template, special tokens, output termination, and representative logits or outputs against the training-side implementation under controlled decoding.
3. Deploy with a matched vLLM-Ascend container if the architecture is supported. Record driver, firmware, CANN, PyTorch, torch_npu, vLLM, and vLLM-Ascend versions.
4. Expose a stable model name and an authenticated internal endpoint. Test ordinary chat before adding tool parsing.
5. Select one tool protocol. Prefer Qwen3-compatible Hermes-style serialization if that matches SciOWL training. Otherwise use a strict JSON action adapter as an initial baseline.
6. Test tool calls, tool observations, multiple turns, final answers, streaming if required, and context truncation.
7. Tune context length, TP/EP, batching, and KV-cache allocation against measured request patterns. Training parallelism settings are not serving defaults.

Keep model serving and scientific execution in separate environments. Begin with disjoint device allocations; do not assume training, serving, and MLIP workers can share NPU memory safely or efficiently.

Serving acceptance: a pinned checkpoint and container produce valid text and tool interactions without unexplained truncation, missing messages, or termination errors. Tool-use quality is evaluated separately from transport correctness.

## 5. Tool contracts and scientific registry

Implement the following initial contracts:

| Tool | Key inputs | Key outputs |
| --- | --- | --- |
| inspect_structure | structure_id | composition, atom count, cell, PBC, warnings |
| list_potentials | composition, workflow | candidate IDs and documented coverage |
| check_potential | structure_id, potential_id, workflow | compatibility evidence and limitations |
| evaluate_energy_forces | structure_id, potential_id | energy, forces artifact, supported stress |
| submit_relaxation | structure_id, potential_id, optimizer, tolerances | job_id and accepted configuration |
| get_job_status | job_id | queued/running/completed/failed/cancelled, diagnostics |
| get_result | job_id | result_id, artifacts, convergence, provenance |
| analyze_eos | result_ids, fit policy | fitted parameters, residuals, diagnostics |

Add submit_md, analyze_trajectory, submit_dft, and active-learning tools in later releases.

Use explicit units such as temperature_K, timestep_fs, and fmax_eV_per_A. Document whether energy is total or per atom, stress signs and units, and virial conventions. Reject invalid requests before scheduling.

The potential registry records elements, intended systems, training domain, validation datasets, PBC support, charge/spin assumptions, available observables, energy references, precision, code/weight hashes, licensing metadata, supported hardware, and measured performance.

Element coverage alone is insufficient. Validate the targeted phases, defects, temperatures, oxidation states, and properties. A solid-state potential is not automatically appropriate for proteins or LNPs.

Return small summaries to SciOWL and store large arrays as artifacts. Do not place full trajectories in model context. Resolve artifact IDs through the gateway; do not accept unrestricted model-generated paths or host commands as scientific tool arguments.

## 6. Execution, persistence, and reproducibility

Use separate states for computational execution and scientific validation. A job can be completed while its result is unconverged or scientifically unsuitable.

Every job record contains run_id, job_id, request_id, parent dependencies, input hashes, tool/schema versions, normalized settings, potential hash, environment digest, seed, resource allocation, timestamps, logs, output hashes, and validation status.

Provide an explicit idempotency key so a repeated request reuses a submission instead of launching duplicate expensive jobs. Resume a workflow by checking existing jobs and artifacts. Graph checkpointing alone does not guarantee that an external submission is executed exactly once.

Classify errors: schema error, unsupported scientific domain, unavailable resource, numerical instability, unconverged calculation, and infrastructure failure. Return machine-readable error codes with actionable summaries. Retry only under defined policies and bounded budgets.

Persist state after each consequential transition. Keep secrets in execution services, not model prompts. Expose bounded operations and workspace-scoped artifacts. Add a sandbox for optional generated scripts; maintain tested tools for core calculations.

Record traces locally when the deployment is offline. Package pinned containers, wheels, model weights, potential files, and dependency manifests before transfer. Test a complete run without external services; replace online database retrieval with curated local structures where needed.

## 7. Scientific workflow progression

### Release A: relaxation and EOS

Inspect the input; check the selected potential; run single-point diagnostics; relax using a documented fixed-cell or variable-cell policy; validate convergence; generate an EOS volume series; fit and inspect residuals; report results with artifacts.

Restrict initial energy ranking to scientifically comparable systems and consistent model references. Comparing unrelated compositions requires appropriate reference energies and thermodynamic treatment.

### Release B: MD and transport

Separate initialization, equilibration, and production. Validate geometry and density, timestep policy, thermostat/barostat settings, temperature/pressure behavior, and checkpoint creation. Run shorter stability checks before longer production jobs.

For diffusion, use appropriate coordinate unwrapping and drift treatment, inspect MSD fit windows, require evidence of a diffusive regime, and estimate uncertainty across replicas or suitable blocks. The relation MSD = 2dDt applies under the corresponding assumptions; use a tensor treatment when anisotropy matters.

An agent must be allowed to conclude that sampling is insufficient. For LiFePO4, establish potential validity for the chosen magnetic/oxidation/defect state and temperature. Begin with small validated cells, then scale. Do not make 100,000-atom runs the first agent acceptance test.

### Release C: DFT escalation and active learning

Use a validated CPU/GPU DFT backend through an HPC adapter. Version convergence settings and pseudopotentials. Compare like-for-like reference calculations.

An active-learning cycle is: explore with an MLIP, select configurations using a calibrated acquisition method, obtain reference labels, validate labels, retrain, evaluate on a fixed independent set, and promote a new model only after passing gates. Ensemble disagreement is an acquisition signal, not guaranteed calibrated uncertainty.

## 8. MLIP migration to Ascend

Treat each model/backend as a separate engineering project. Inspect CUDA-only extensions, graph construction, scatter/index operations, equivariant operations, automatic differentiation, and host-device transfers.

Preserve the same external calculator contract while changing implementation underneath. An ASE-compatible adapter should return documented energy, forces, and stress properties without leaking hardware-specific options into the research request.

Required validation:

| Check | Evidence |
| --- | --- |
| Device execution | Profiling demonstrates intended NPU operations and identifies CPU fallbacks |
| Numerical parity | Energy/force/stress agreement against a validated reference over representative systems |
| Derivatives | Finite-difference checks and supported higher derivatives where needed |
| Geometry | Neighbor lists, cutoffs, PBC, cell changes, and edge cases |
| Relaxation | Comparable convergence and final structures |
| Dynamics | Short NVE drift and trajectory statistics under an appropriate precision policy |
| Performance | End-to-end time, including graph construction and data movement |

Set tolerances before testing according to task and reference precision. Avoid bitwise identity as a universal requirement. Keep GPU/CPU fallback available until the NPU backend passes scientific validation and provides useful throughput.

## 9. Tool-use dataset construction

Store canonical JSONL with id, tools, messages, and metadata. Keep OpenAI-style tool-call IDs and explicit tool observations in source records. Render using the training template only at preprocessing time.

Each record should capture the actual tool descriptions and schemas available at that decision. Assistant calls must use only information available before the observation. Scientific outputs come from execution; label simulated observations and injected failures separately.

Pipeline: choose a task and input -> define admissible outcomes -> generate varied requests -> obtain expert actions -> execute tools -> record observations -> generate continuation -> verify schemas, execution, and scientific claims -> deduplicate -> split by source family -> render/tokenize.

Use teacher models for language and planning, existing scripts for expert baselines, and real tools for numerical evidence. Cache deterministic results and reuse historical logs with complete provenance to control cost.

Initial 10,000-conversation scientific pilot:

| Category | Count |
| --- | ---: |
| Single-tool selection and arguments | 2,000 |
| Dependent multi-step workflows | 3,000 |
| Observation interpretation | 1,500 |
| Failure recovery | 1,500 |
| Clarification | 1,000 |
| No-tool or unsupported-capability decisions | 1,000 |

Start with 500 manually reviewed examples and a preprocessing smoke test before scaling. Include multiple plausible candidate tools, varying schema descriptions, units, and missing information. Do not expand primarily by paraphrasing identical calculations.

Mix the scientific pilot with curated general function-calling data, such as xLAM or ToolACE, and retained scientific/general QA. A starting sampling mixture is 50% scientific tool trajectories, 25% general tool use, and 25% retained instruction data. Measure supervised token shares and tune the mixture against validation; these percentages are not a fixed prescription.

Split by structure/molecule family, original calculation, scenario, and augmentation parent before generating variants. Add held-out schemas, compositions, and workflow combinations. Keep benchmark prompts and reference answers outside teacher generation and training.

## 10. Training strategy with MindSpeed

Train from SciOWL-Instruct. Continued pretraining is not required merely to learn the interaction protocol. Revisit domain CPT only if controlled evaluations isolate a knowledge or representation bottleneck.

Keep canonical messages separate from framework-specific conversion. Check the actual MindSpeed checkout for tools fields, function/observation roles, and multi-turn behavior; older and newer handlers differ. Do not claim an untested preprocessing command is compatible.

Use assistant-only supervision for tool calls, clarifications, and final answers. Mask system messages, tool definitions, user messages, and tool observations from loss. Preserve all of them as context. If a recovery prefix contains a deliberately wrong assistant action, mask that action and supervise the correction.

Decode processed examples to verify labels and serialized content. Confirm that every desired assistant turn is supervised, observations are masked, schemas remain present, and packing/truncation does not corrupt dependencies. Split long trajectories at decision boundaries while preserving necessary observations and artifact summaries.

Run a small overfit test to establish that the pipeline can learn the format. Then compare prompting-only, LoRA, and full SFT when resources permit. Select learning rate, epochs, and checkpoint using held-out execution success and retained QA performance. Do not automatically reuse the settings from the previous 16B-token instruction run on a much smaller agent dataset.

After deployment, collect SciOWL failures, correct them with an expert or verified teacher, and retrain on validated corrections. This addresses the gap between teacher-generated states and states visited by SciOWL.

Consider DPO or RL only after robust rewards exist. Reward valid scientific completion, grounded reporting, and efficient action use. Penalize invalid calls, unsupported claims, duplicate expensive jobs, and excessive retries. Keep hard scientific constraints outside the reward function; token economy must not reward premature stopping.

## 11. Evaluation and proposed release gates

Evaluate model behavior, harness reliability, backend correctness, and scientific outcomes separately. Exact sequence matching is inappropriate when several valid plans exist.

| Layer | Metrics |
| --- | --- |
| Interface | Parse rate, schema validity, valid tool names, argument correctness |
| Decision quality | Appropriate tool/no-tool choice, clarification, dependency handling |
| Workflow | Completion rate, recovery rate, duplicate submission rate, resume correctness |
| Science | Reference agreement, convergence, physical checks, uncertainty and sampling adequacy |
| Grounding | Unsupported numerical claims, provenance completeness, correct units |
| Efficiency | Latency, tokens, tool calls, accelerator-hours per validated success |
| Retention | Scientific QA, instruction following, and selected general benchmarks |

Build 50 manually reviewed development tasks first. Expand to an independent 200-task suite after the initial contracts stabilize: 40 single-tool, 60 multi-step, 40 recovery, 30 clarification/no-tool, and 30 reporting/evidence tasks. Keep development and final test tasks distinct. Report confidence intervals and repeats for stochastic policies; avoid comparing differently composed suites as if they were identical.

Suggested gates, subject to task-owner agreement: at least 98% schema-valid calls, at least 85% validated success on the bounded MVP suite, and zero duplicate submissions across the defined retry/restart tests. Require evidence links for every reported calculated value and zero unsupported numerical claims in the audited release suite. These are proposed project targets, not observed performance or guarantees outside the suite.

Run controlled ablations: SciOWL without tools; deterministic expert workflow; SciOWL with tool descriptions; SciOWL with demonstrations; SciOWL after tool SFT; and a stronger reference model using identical tools and budgets. Add NPU versus reference-backend comparisons independently of the LLM study.

## 12. Repository and deliverables

Suggested project layout:

```text
sciowl-agent/
  configs/                 # versioned serving, tool, workflow, and budget settings
  sciowl_agent/
    model/                 # serving adapter and message conversion
    harness/               # graph, typed state, context, and recovery
    tools/                 # contracts and gateway adapters
    workers/               # ASE, MLIP, analysis, DFT execution
    registry/              # potentials and validated workflow policies
    provenance/            # job records and artifact manifests
  datasets/                # schemas, generation and conversion code; manifests
  evaluation/              # task definitions, graders, and replay runner
  deployment/              # pinned images, HPC and offline setup manifests
  tests/                   # meaningful interface, execution, and scientific checks
  docs/                    # workflow recipes and reproducibility instructions
```

Keep large trajectories and weights outside git; reference them by hashes. Keep data schemas and migrations versioned. Every release should provide the checkpoint identity, dependency manifests, tool schemas, evaluation task IDs, raw graded outcomes, and runnable workflow recipes.

## 13. Twelve-week implementation plan

Indicative schedule assuming one ML engineer and one scientific-computing engineer, with a domain reviewer. A smaller team should sequence the same milestones over a longer period.

| Phase | Time | Deliverable | Exit criterion |
| --- | --- | --- | --- |
| Scope and baseline | Week 1 | Selected materials task set, serving baseline, reference workflow | SciOWL export and API validated; first development cases specified |
| Scientific core | Weeks 2-3 | Registry, inspection, single-point and relaxation tools | Deterministic workflows pass numerical checks |
| Persistent harness | Weeks 4-5 | Graph, async jobs, budgets, provenance, recovery | Restart and duplicate-submission tests pass |
| Dataset pilot | Weeks 5-6 | 500 reviewed trajectories, then verified 10k pilot | Schemas, observations, splits, and loss masks audited |
| Tool-use SFT | Weeks 7-8 | SciOWL-Agent candidate checkpoint | Improves held-out execution with acceptable QA retention |
| MD extension | Weeks 8-10 | Equilibration, production, trajectory analysis | Validated small-system MD and insufficient-sampling cases |
| Release evaluation | Weeks 11-12 | Independent benchmark and deployment package | Agreed release gates met; limits documented |
| NPU MLIP port | Parallel, scope-dependent | One validated NPU calculator | Numerical and end-to-end performance checks pass |

Dependencies: define tool contracts before generating large training data; validate scientific tools before grading agent outcomes; establish reliable execution and rewards before agentic RL. DFT active learning and life-science expansion follow the MVP rather than blocking it.

## 14. First ten working days

1. Choose one target: validated relaxation of periodic solids with a known potential.
2. Select a manageable set of structures and scripted reference calculations.
3. Export and serve SciOWL; preserve and validate its chat/tool template.
4. Implement inspect, compatibility, submit, status, and result contracts.
5. Run the scripted workflow without an LLM to verify tools and provenance.
6. Add the SciOWL loop and compare against scripted outcomes.
7. Create initial success, clarification, failure, and no-tool examples.
8. Verify tokenization and loss masking on a tiny dataset.
9. Evaluate prompting versus a small SFT pilot.
10. Decide the next investment from observed failure categories rather than adding frameworks or data indiscriminately.

## 15. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Strong QA but weak tool use | Demonstrations, verified trajectory SFT, execution evaluation |
| Correct JSON but wrong science | Registry, deterministic prerequisites, reference calculations |
| Silent CUDA assumptions | Separate backend contract; profile and validate NPU implementations |
| Tool schema drift | Versioned definitions and regenerated/revalidated data |
| Expensive repeated jobs | Idempotency, budgets, durable scheduling |
| Fabricated conclusions | Evidence-linked claims and numerical graders |
| General capability regression | Retained instruction data and fixed retention tests |
| Dataset leakage | Group splits before augmentation; independent final benchmarks |
| Offline dependency failure | Pinned deployment bundle and disconnected rehearsal |

## 16. Primary references and implementation resources

These resources support component capabilities; their existence is not evidence that the complete SciOWL Agent has been implemented or validated.

- Qwen3 function-calling protocol: https://qwen.readthedocs.io/en/stable/framework/function_call.html
- vLLM-Ascend: https://github.com/vllm-project/vllm-ascend
- Qwen3 MoE serving guide: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3-30B-A3B.html
- LangGraph: https://github.com/langchain-ai/langgraph
- Deep Agents: https://github.com/langchain-ai/deepagents
- Qwen-Agent: https://github.com/QwenLM/Qwen-Agent
- ASE: https://docs.ase-lib.org/
- AtomisticSkills: https://github.com/learningmatter-mit/AtomisticSkills
- AtomisticSkills runtime setup: https://github.com/learningmatter-mit/AtomisticSkills/blob/main/docs/setup.md
- ORNL matsim-agents: https://github.com/ORNL/matsim-agents
- Aitomia: https://github.com/dralgroup/aitomia
- VASPilot: https://github.com/JiaxuanLiu-Arsko/VASPilot
- VASP Agent: https://github.com/Phoinikas03/VASP_Agent
- ScienceDiscovery: https://github.com/openJiuwen-ai/sciencediscovery
- ScienceDiscovery Ascend broker: https://sciencediscovery.github.io/docs/developer-docs/ascend-npu-runner.html
- AgentSDK/Aura: https://github.com/Ascend/AgentSDK
- dflow/DPDispatcher integration: https://github.com/deepmodeling/dflow
- xLAM dataset and verification pipeline: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
- ToolACE: https://huggingface.co/datasets/Team-ACE/ToolACE
- VASP accelerator support: https://vasp.at/wiki/OpenACC_GPU_port_of_VASP

