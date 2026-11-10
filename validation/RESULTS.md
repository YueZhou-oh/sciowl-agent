# Validation results

## Executed locally

- 33 pytest cases passed, including two actual Qwen3 tokenizer/template tests. One ASE deprecation warning remains; it does not affect test outcomes.
- Actual HTTP loop: endpoint adapter -> LangGraph -> real ASE worker -> recorded evidence -> completed report, using a scripted server policy.
- Real perturbed-Cu fixed-cell relaxation: final maximum force 0.0278780266 eV/Angstrom against a requested 0.03 threshold.
- Cu EOS demonstration: fitted equilibrium volume 46.2614917 Angstrom^3, bulk modulus 134.387139 GPa, fit RMSE 3.95147e-6 eV. These are ASE EMT outputs, not experimental validation.
- Short NVE demonstration: 21 sampled frames over 20 fs; total-energy range 4.41316e-8 eV/atom in the recorded case. This is a numerical stability smoke test, not a transport-convergence test.
- Three related Cu relaxation evaluation cases passed under the scripted expert baseline. This is not a SciOWL benchmark.
- Six generated execution-trace records were audited using Qwen3-0.6B tokenizer files, yielding 32 separately supervised assistant decisions. Tokenizer and template hashes are in `tokenizer_audit.json`.
- Python source compilation, shell syntax checks, and wheel construction succeeded.

## Evidence files

- `local/pytest.txt`: complete pytest output.
- `local/pytest.xml`: machine-readable test results.
- `local/report.json`: numerical outputs, environment versions, and limitations.
- `local/workspace/`: job database and checksummed scientific artifacts.
- `scripted_evaluation/evaluation.json`: per-case execution metrics.
- `tokenizer_audit.json`: decision-level token counts and tokenizer hashes.
- `../examples/seed_dataset/`: six unreviewed seed records and execution artifacts.
- `../requirements-validation.txt`: dependency versions of this CPU validation environment.

## Not executed here

SciOWL inference, Ascend runtime/kernel tests, MACE model inference, Quantum ESPRESSO calculations, Slurm submission, MindSpeed bin generation/training, agentic RL, and scientific discovery validation require external model/hardware/software assets. Their included adapters and scripts must be exercised on the target system before claiming support.

The Qwen3 tokenizer itself is not bundled. Use your local SciOWL/Qwen3 tokenizer and set `SCIOWL_TEST_TOKENIZER` to reproduce the two template tests. Unreviewed seed records must not be silently promoted to training data.

