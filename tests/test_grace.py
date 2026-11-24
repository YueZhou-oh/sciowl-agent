"""Synthetic subprocess tests; these do not validate a real GRACE checkpoint."""
import json
import sys
from pathlib import Path
import types

import numpy as np
import pytest
from ase.build import bulk
from sciowl_agent.grace import model_digest, predict_external
from sciowl_agent.science import Registry, atoms_to_id
from sciowl_agent.gateway import Gateway
from sciowl_agent.store import Store
from sciowl_agent.worker import run_one


def setup(tmp_path, body=None):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'saved_model.pb').write_bytes(b'fixture')
    exe = tmp_path / 'python-grace'
    exe.write_text(f'#!{sys.executable}\n' + (body or '''import sys, json, os
from pathlib import Path
assert sys.argv[1] == '-I'
assert Path(sys.argv[2]).name == 'grace_runner.py'
assert os.environ['TF_USE_LEGACY_KERAS'] == '1'
assert 'PYTHONPATH' not in os.environ
request = json.loads(Path(sys.argv[3]).read_text())
records = []
for frame in request['frames']:
    positions = frame['positions']
    records.append(dict(energy=sum(x*x/2 for row in positions for x in row),
                        forces=[[-x for x in row] for row in positions],
                        **({'stress': [1,2,3,4,5,6]} if frame['stress'] else {})))
Path(sys.argv[4]).write_text(json.dumps(records))
'''))
    exe.chmod(0o755)
    cfg = dict(backend='grace', elements=['Cu'], model_path=str(model),
               python_executable=str(exe), sha256=model_digest(model),
               timeout_s=10, domain='synthetic test')
    registry = Registry(); registry.entries = {'grace': cfg}
    store = Store(tmp_path / 'ws')
    gateway = Gateway(store, registry)
    sid = atoms_to_id(store, bulk('Cu', cubic=True))
    return store, gateway, sid, cfg


@pytest.mark.parametrize('operation', ['evaluate_energy_forces', 'evaluate_structures', 'submit_relaxation'])
def test_queue(tmp_path, operation):
    store, gateway, sid, cfg = setup(tmp_path)
    args = dict(potential_id='grace')
    args.update({'structure_ids': [sid, sid]} if operation == 'evaluate_structures' else {'structure_id': sid})
    if operation == 'submit_relaxation':
        args.update(max_steps=1, fmax_eV_per_A=0.001)
    queued = gateway.call(operation, args)
    assert queued['ok'], queued
    assert run_one(store)
    response = gateway.call('get_result', {'job_id': queued['job_id']})
    assert response['ready'], response
    result = response['result']
    if operation == 'submit_relaxation':
        assert result['converged'] is False
        assert result['steps'] == 1
    else:
        rows = [json.loads(s) for s in store.artifact(result['predictions_id']).read_text().splitlines()]
        assert len(rows) == (2 if operation == 'evaluate_structures' else 1)
        np.testing.assert_allclose(rows[0]['stress_eV_per_A3'], [1,2,3,4,5,6])
        assert len(list((store.root/'work'/queued['job_id']/'calculator').glob('prediction-*'))) == 1
    listed = gateway.call('list_potentials', {})['potentials']['grace']
    assert 'python_executable' not in listed and 'model_path' not in listed


@pytest.mark.parametrize('body,timeout,match', [
    ('raise SystemExit(7)', 10, 'exited with code 7'),
    ('import time; time.sleep(10)', 0.1, 'timed out'),
    ("import sys; from pathlib import Path; Path(sys.argv[4]).write_text('[]')", 10, 'frame count'),
    ("import sys; from pathlib import Path; Path(sys.argv[4]).write_text('[{\"energy\":NaN,\"forces\":[]}]')", 10, 'nonfinite'),
])
def test_failures(tmp_path, body, timeout, match):
    store, gateway, sid, cfg = setup(tmp_path, body)
    cfg['timeout_s'] = timeout
    jid = gateway.call('evaluate_energy_forces', {'structure_id': sid, 'potential_id': 'grace'})['job_id']
    run_one(store)
    assert store.job(jid)['status'] == 'failed'
    assert match in store.job(jid)['error']['message']


def test_deadline(tmp_path):
    store, gateway, sid, cfg = setup(tmp_path, 'import time; time.sleep(10)')
    jid = gateway.call('evaluate_energy_forces', {'structure_id': sid, 'potential_id': 'grace'})['job_id']
    run_one(store, timeout_s=0.1)
    assert 'time budget' in store.job(jid)['error']['message']


def test_asset_integrity_and_coverage(tmp_path):
    store, gateway, sid, cfg = setup(tmp_path)
    jid = gateway.call('evaluate_energy_forces', {'structure_id': sid, 'potential_id': 'grace'})['job_id']
    (Path(cfg['model_path']) / 'variables').mkdir()
    (Path(cfg['model_path']) / 'variables/weights').write_text('modified')
    run_one(store)
    assert 'SHA256 mismatch' in store.job(jid)['error']['message']
    cfg['elements'] = ['H']
    assert not gateway.call('check_potential', {'structure_id': sid, 'potential_id': 'grace'})['ok']


def test_standalone_runner_contract(tmp_path, monkeypatch):
    from sciowl_agent import grace_runner
    class FakeCalculator:
        def __init__(self, model, **kwargs):
            assert kwargs == dict(mode='diverse', enable_uq_if_available=False)
        def calculate(self, atoms, properties):
            assert properties == ['energy', 'forces', 'stress']
            self.results = dict(energy=-1, forces=np.zeros((len(atoms), 3)), stress=np.arange(6))
    package = types.ModuleType('tensorpotential')
    module = types.ModuleType('tensorpotential.calculator'); module.TPCalculator = FakeCalculator
    monkeypatch.setitem(sys.modules, 'tensorpotential', package)
    monkeypatch.setitem(sys.modules, 'tensorpotential.calculator', module)
    request, output = tmp_path/'in.json', tmp_path/'out.json'
    request.write_text(json.dumps(dict(model_path='fixture', frames=[dict(numbers=[29], positions=[[0,0,0]], cell=np.eye(3).tolist(), pbc=[True]*3, stress=True)])))
    monkeypatch.setattr(sys, 'argv', ['runner', str(request), str(output)])
    grace_runner.main()
    assert json.loads(output.read_text())[0]['stress'] == list(range(6))


def test_registration(tmp_path, monkeypatch):
    from scripts.register_grace import main
    store, gateway, sid, cfg = setup(tmp_path)
    out = tmp_path/'registered.json'
    monkeypatch.setattr(sys, 'argv', ['register_grace', '--model', cfg['model_path'],
                                    '--python', cfg['python_executable'], '--elements', 'Cu',
                                    '--output', str(out)])
    main()
    assert json.loads(out.read_text())['grace']['sha256'] == cfg['sha256']
    with pytest.raises(FileExistsError):
        main()


def test_skip_stress(tmp_path):
    store, gateway, sid, cfg = setup(tmp_path)
    jid = gateway.call('evaluate_structures', {'structure_ids': [sid], 'potential_id': 'grace', 'stress_policy': 'skip'})['job_id']
    run_one(store)
    result = gateway.call('get_result', {'job_id': jid})['result']
    row = json.loads(store.artifact(result['predictions_id']).read_text())
    assert row['stress_available'] is False
    request = next((store.root/'work'/jid/'calculator').glob('prediction-*/input.json'))
    assert json.loads(request.read_text())['frames'][0]['stress'] is False
