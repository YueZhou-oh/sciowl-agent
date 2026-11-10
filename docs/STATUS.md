# Implementation and validation boundary

## Implemented and locally exercised

- LangGraph orchestration, canonical tool messages, persistent sessions, pending-call replay, budgets, and evidence-linked reports.
- SQLite transactional queue, idempotent submissions, worker ownership, cancellation, and explicit stale-worker handling.
- Typed scientific contracts, potential element guards, checksummed artifacts, provenance, and session serialization.
- Real CPU ASE EMT single points/forces, fixed-cell relaxation, EOS, short NVE/NVT calculations, and provisional MSD analysis.
- Scripted baseline, endpoint transport mocks, dataset export, source-family splits, and tokenizer mask code.

## Implemented integration paths requiring site validation

- SciOWL OpenAI-compatible native/JSON model service and Ascend launch/probe scripts.
- Trusted local MACE model adapter (CPU/CUDA) and checksums.
- ASE Quantum ESPRESSO adapter with pinned pseudopotential checksums.
- Slurm single-host wrapper and matching-platform offline wheelhouse script.
- MindSpeed signed indexed-bin builder with readback; actual checkout required.
- Ensemble disagreement acquisition script; actual independent models required for scientific use.

## Not implemented as completed scientific capabilities

- MACE/UMA/DeePMD numerical kernels on Ascend.
- SciOWL checkpoint export, full SFT or RL runs, model-quality scores, and general-capability retention results.
- Automatic MLIP retraining/promotion, calibrated uncertainty, or a validated active-learning discovery campaign.
- NPT, variable-cell relaxation, NEB, phonons, general life-science tools, and validated transport coefficients.
- Multi-host transactional service, hard preemption, multi-tenant API security, or generated-code sandbox.

The supplied software implements an executable foundation and the local scientific core of the multi-stage roadmap. The items above require real hardware/model/scientific assets and further task-specific implementation; they are not represented as complete.

