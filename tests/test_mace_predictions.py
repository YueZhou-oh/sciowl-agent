"""Real ASE queue/IO tests plus a MACE adapter spy (no simulated MACE accuracy claims)."""
import hashlib
import json
import os
import sys
import types

import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.emt import EMT
from ase.io import write

from sciowl_agent.cli import main
from sciowl_agent.contracts import tool_schemas
from sciowl_agent.gateway import Gateway
from sciowl_agent.science import (
    Registry, atoms_to_id, calculator, import_structure, import_structures, predict_frame,
)
from sciowl_agent.store import Store
from sciowl_agent.worker import run_one


def fetch_records(store, gateway, jid):
    response = gateway.call("get_result", {"job_id": jid})
    assert response.get("ready"), response
    result = response["result"]
    records = [json.loads(line) for line in store.artifact(result["predictions_id"], "predictions").read_text().splitlines()]
    return result, records


@pytest.mark.parametrize("suffix", ["xyz", "extxyz", "cif"])
def test_multiframe_import_and_prediction(tmp_path, monkeypatch, capsys, suffix):
    import sciowl_agent.science as science
    frames = [bulk("Cu", cubic=True), bulk("Al", cubic=True)]
    path = tmp_path / f"input.{suffix}"
    write(path, frames)
    store = Store(tmp_path / "workspace")
    ids = import_structures(store, path)
    assert len(ids) == 2
    assert science.load_atoms(store, import_structure(store, path)).get_chemical_formula() == "Cu4"
    made = []
    def factory(cfg, directory):
        made.append(cfg)
        return EMT()
    monkeypatch.setattr(science, "calculator", factory)
    gateway = Gateway(store)
    args = {"structure_ids": ids, "potential_id": "emt_demo", "stress_policy": "required"}
    queued = gateway.call("evaluate_structures", args)
    assert queued["ok"]
    assert gateway.call("evaluate_structures", args)["job_id"] == queued["job_id"]
    assert run_one(store, show_progress=True)
    result, records = fetch_records(store, gateway, queued["job_id"])
    assert len(made) == 1 and result["frames"] == 2
    assert "frame=2/2" in capsys.readouterr().err
    for index, (atoms, record) in enumerate(zip(frames, records)):
        atoms.calc = EMT()
        assert record["index"] == index and record["structure_id"] == ids[index]
        assert record["energy_eV"] == pytest.approx(atoms.get_potential_energy())
        np.testing.assert_allclose(record["forces_eV_per_A"], atoms.get_forces(), atol=1e-10)
        np.testing.assert_allclose(record["stress_eV_per_A3"], atoms.get_stress(), atol=1e-10)
    output = tmp_path / "output.jsonl"
    monkeypatch.setattr(sys, "argv", ["sciowl-agent", "--workspace", str(store.root), "export-predictions", "--job", queued["job_id"], "--output", str(output)])
    main()
    assert json.loads(capsys.readouterr().out)["predictions_id"] == result["predictions_id"]
    assert len(output.read_text().splitlines()) == 2
    with pytest.raises(FileExistsError):
        main()


def test_plain_xyz_stress_and_failed_batch(tmp_path):
    path = tmp_path / "molecule.xyz"
    path.write_text("2\nnonperiodic\nCu 0 0 0\nCu 2 0 0\n")
    store = Store(tmp_path / "workspace")
    ids = import_structures(store, path)
    gateway = Gateway(store)
    for policy in ("auto", "skip", "required"):
        jid = gateway.call("evaluate_structures", {"structure_ids": ids, "potential_id": "emt_demo", "stress_policy": policy})["job_id"]
        run_one(store)
        if policy == "required":
            failure = gateway.call("get_result", {"job_id": jid})
            assert failure["ok"] and failure["status"] == "failed" and not failure["ready"]
            assert failure["result"]["input_structure_ids"] == ids
        else:
            _, records = fetch_records(store, gateway, jid)
            assert records[0]["stress_eV_per_A3"] is None
            assert not records[0]["stress_available"]
            assert records[0]["stress_unavailable_reason"]


def test_singlepoint_legacy_fields_and_export(tmp_path):
    store = Store(tmp_path)
    gateway = Gateway(store)
    sid = atoms_to_id(store, bulk("Cu", cubic=True))
    jid = gateway.call("evaluate_energy_forces", {"structure_id": sid})["job_id"]
    run_one(store)
    result, records = fetch_records(store, gateway, jid)
    assert result["stress_voigt_eV_per_A3"] == records[0]["stress_eV_per_A3"]
    assert store.get_json(result["forces_id"])["forces_eV_per_A"] == records[0]["forces_eV_per_A"]


def test_mace_constructor_and_registry(tmp_path, monkeypatch):
    calls = []
    module = types.ModuleType("mace.calculators")
    module.MACECalculator = lambda **kwargs: calls.append(kwargs) or EMT()
    monkeypatch.setitem(sys.modules, "mace", types.ModuleType("mace"))
    monkeypatch.setitem(sys.modules, "mace.calculators", module)
    model = tmp_path / "model.model"
    model.write_bytes(b"adapter-test-placeholder-never-loaded")
    cfg = {"backend": "mace", "elements": ["Cu"], "model_path": str(model),
           "sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "device": "cuda",
           "default_dtype": "float32", "domain": "adapter test only", "demo_only": True}
    reg = Registry()
    reg.entries = {"mace_test": cfg}
    store = Store(tmp_path / "workspace")
    sid = atoms_to_id(store, bulk("Cu", cubic=True))
    gateway = Gateway(store, reg)
    jid = gateway.call("evaluate_structures", {"structure_ids": [sid, sid], "potential_id": "mace_test"})["job_id"]
    run_one(store)
    fetch_records(store, gateway, jid)
    assert calls == [{"model_paths": str(model), "device": "cuda", "default_dtype": "float32"}]
    calculator({k: v for k, v in cfg.items() if k != "default_dtype"})
    assert calls[-1]["default_dtype"] == "float64"
    cfg["default_dtype"] = "float16"
    assert not gateway.call("check_potential", {"structure_id": sid, "potential_id": "mace_test"})["ok"]
    cfg["default_dtype"] = "float32"
    jid = gateway.call("evaluate_structures", {"structure_ids": [sid], "potential_id": "mace_test"}, "tamper")["job_id"]
    model.write_bytes(b"changed-after-submission")
    run_one(store)
    assert store.job(jid)["status"] == "failed"
    assert "SHA256 mismatch" in store.job(jid)["error"]["message"]


def test_stress_errors_are_not_hidden():
    atoms = bulk("Cu", cubic=True)
    class Unsupported(EMT):
        def get_stress(self, atoms=None):
            raise PropertyNotImplementedError("not supported")
    atoms.calc = Unsupported()
    assert predict_frame(atoms)["stress_eV_per_A3"] is None
    with pytest.raises(ValueError, match="does not implement"):
        predict_frame(atoms, "required")
    class Broken(EMT):
        def get_stress(self, atoms=None):
            return np.full(6, np.nan)
    atoms.calc = Broken()
    with pytest.raises(ValueError, match="nonfinite stress"):
        predict_frame(atoms)
    class Crash(EMT):
        def get_stress(self, atoms=None):
            raise RuntimeError("calculator failure")
    atoms.calc = Crash()
    with pytest.raises(RuntimeError, match="calculator failure"):
        predict_frame(atoms)


def test_batch_validation_and_cancellation(tmp_path, monkeypatch):
    import sciowl_agent.science as science
    store = Store(tmp_path)
    gateway = Gateway(store)
    sid = atoms_to_id(store, bulk("Cu", cubic=True))
    iron = atoms_to_id(store, bulk("Fe", "bcc", a=2.8))
    for args in ({"structure_ids": []}, {"structure_ids": [sid] * 1001},
                 {"structure_ids": [sid, iron]}, {"structure_ids": [sid], "stress_policy": "bad"},
                 {"structure_ids": [sid], "model_path": "/anything"}):
        assert not gateway.call("evaluate_structures", {"potential_id": "emt_demo", **args})["ok"]
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0
    jid = gateway.call("evaluate_structures", {"structure_ids": [sid, sid], "potential_id": "emt_demo"})["job_id"]
    predict = science.predict_frame
    def cancel_after_first(*args):
        value = predict(*args)
        store.cancel(jid)
        return value
    monkeypatch.setattr(science, "predict_frame", cancel_after_first)
    run_one(store)
    assert store.job(jid)["status"] == "cancelled"
    assert not gateway.call("get_result", {"job_id": jid})["ready"]
    assert "evaluate_structures" in {tool["function"]["name"] for tool in tool_schemas()}


def test_optional_real_mace(tmp_path):
    """Opt-in CPU/CUDA comparison against direct MACE using user-supplied trusted weights."""
    model_path = os.environ.get("SCIOWL_TEST_MACE_MODEL")
    if not model_path:
        pytest.skip("Set SCIOWL_TEST_MACE_MODEL to a trusted local MACE model covering Cu")
    from pathlib import Path
    model = Path(model_path).resolve()
    cfg = {"backend": "mace", "elements": ["Cu"], "model_path": str(model),
           "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
           "device": os.environ.get("SCIOWL_TEST_MACE_DEVICE", "cpu"),
           "default_dtype": "float32", "domain": "integration test", "demo_only": True}
    store = Store(tmp_path)
    atoms = bulk("Cu", cubic=True)
    sid = atoms_to_id(store, atoms)
    reg = Registry(); reg.entries = {"mace": cfg}
    gateway = Gateway(store, reg)
    jid = gateway.call("evaluate_structures", {"structure_ids": [sid], "potential_id": "mace", "stress_policy": "required"})["job_id"]
    run_one(store)
    _, records = fetch_records(store, gateway, jid)
    atoms.calc = calculator(cfg)
    assert records[0]["energy_eV"] == pytest.approx(atoms.get_potential_energy(), abs=1e-4)
    np.testing.assert_allclose(records[0]["forces_eV_per_A"], atoms.get_forces(), atol=1e-5)
    np.testing.assert_allclose(records[0]["stress_eV_per_A3"], atoms.get_stress(), atol=1e-5)


def test_cli_import_all(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'input.xyz'
    write(path, [bulk('Cu'), bulk('Cu')])
    monkeypatch.setattr(sys, 'argv', ['sciowl-agent', '--workspace', str(tmp_path / 'ws'), 'import', str(path), '--all'])
    main()
    result = json.loads(capsys.readouterr().out)
    assert result['frames'] == 2 and len(set(result['structure_ids'])) == 2


def test_batch_agent_loop_and_budget(tmp_path):
    from sciowl_agent.harness import Harness
    from sciowl_agent.model import ScriptedModel
    store = Store(tmp_path)
    sid = atoms_to_id(store, bulk('Cu', cubic=True))
    class BatchPlanner(ScriptedModel):
        def next(self, messages, tools):
            # Reuse the baseline's inspect/check/result sequence while swapping its job.
            history = json.loads(json.dumps(messages))
            for item in history:
                for call in item.get('tool_calls', []):
                    if call['function']['name'] == 'evaluate_structures':
                        call['function']['name'] = 'submit_relaxation'
            message = super().next(history, tools)
            for call in message.get('tool_calls', []):
                if call['function']['name'] == 'submit_relaxation':
                    call['function'] = {'name': 'evaluate_structures', 'arguments': json.dumps({'structure_ids': [sid, sid], 'potential_id': 'emt_demo'})}
            return message
    state = Harness(store, Gateway(store), BatchPlanner(sid), inline_worker=True).run('batch', 'Predict both structures')
    assert state['status'] == 'completed' and len(state['jobs']) == 1
    facts = store.get_json(state['report_id'])['evidence']
    assert any(f['data'].get('frames') == 2 for f in facts)
    harness = Harness(store, Gateway(store), BatchPlanner(sid), max_jobs=1)
    state['pending'] = [{'id': 'budget', 'function': {'name': 'evaluate_structures', 'arguments': json.dumps({'structure_ids': [sid], 'potential_id': 'emt_demo'})}}]
    harness.act(state)
    assert 'budget exhausted' in state['messages'][-1]['content']
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 1
