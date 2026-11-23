"""Adapter/queue tests with a fake executable, not NEP scientific validation."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pytest
from ase.build import bulk
from sciowl_agent.science import Registry, atoms_to_id
from sciowl_agent.gateway import Gateway
from sciowl_agent.store import Store
from sciowl_agent.worker import run_one


def setup(tmp_path, body=None):
    model = tmp_path / 'nep.txt'
    model.write_text('nep4 1 Cu\ncutoff 6 4\nn_max 4 4\nbasis_size 8 8\nl_max 4 2 1\nANN 50 0\n')
    exe = tmp_path / 'nep'
    exe.write_text(f'#!{sys.executable}\n' + (body or '''from ase.io import read
import numpy as np
from pathlib import Path
frames = read('train.xyz', index=':')
assert 'prediction 1' in Path('nep.in').read_text()
assert all(a.get_potential_energy() == 0 for a in frames)
np.savetxt('energy_train.out', [[-2, 0] for a in frames])
np.savetxt('force_train.out', [[0.1, 0.0, 0.0, 0, 0, 0] for a in frames for _ in a])
np.savetxt('stress_train.out', [list(range(1, 7)) + [0]*6 for a in frames])
'''))
    exe.chmod(0o755)
    cfg = dict(backend='nep', elements=['Cu'], model_path=str(model), nep_bin=str(exe),
               sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
               executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), domain='test fixture', timeout_s=5)
    reg = Registry(); reg.entries = {'nep89': cfg}
    store = Store(tmp_path / 'ws'); gateway = Gateway(store, reg)
    sid = atoms_to_id(store, bulk('Cu', cubic=True))
    return store, gateway, sid, cfg


@pytest.mark.parametrize('operation', ['evaluate_energy_forces', 'evaluate_structures', 'submit_relaxation'])
def test_native_queue(tmp_path, operation):
    store, gateway, sid, cfg = setup(tmp_path)
    args = {'potential_id': 'nep89'}
    args.update({'structure_ids': [sid, sid]} if operation == 'evaluate_structures' else {'structure_id': sid})
    if operation == 'submit_relaxation':
        args['fmax_eV_per_A'] = 0.3
    queued = gateway.call(operation, args)
    assert queued['ok'], queued
    assert run_one(store)
    response = gateway.call('get_result', {'job_id': queued['job_id']})
    assert response['ready'], response
    result = response['result']
    if operation == 'submit_relaxation':
        assert result['converged'] and result['max_force_eV_per_A'] == pytest.approx(0.1)
    else:
        rows = [json.loads(s) for s in store.artifact(result['predictions_id'], 'predictions').read_text().splitlines()]
        assert len(rows) == (2 if operation == 'evaluate_structures' else 1)
        assert rows[0]['energy_eV'] == -8
        np.testing.assert_allclose(rows[0]['stress_eV_per_A3'], np.array([1,2,3,5,6,4])/160.21766208)
    assert len(list((store.root/'work'/queued['job_id']/'calculator').glob('prediction-*'))) == 1


@pytest.mark.parametrize('body,match', [('raise SystemExit(3)', 'exited with code 3'), ('import time\ntime.sleep(10)', 'timed out'), ('pass', 'energy_train.out')])
def test_failures(tmp_path, body, match):
    store, gateway, sid, cfg = setup(tmp_path, body)
    cfg['timeout_s'] = 0.1
    jid = gateway.call('evaluate_energy_forces', {'structure_id': sid, 'potential_id': 'nep89'})['job_id']
    run_one(store)
    job = store.job(jid)
    assert job['status'] == 'failed'
    assert match in job['error']['message']


def test_tampered_assets(tmp_path):
    store, gateway, sid, cfg = setup(tmp_path)
    jid = gateway.call('evaluate_energy_forces', {'structure_id': sid, 'potential_id': 'nep89'})['job_id']
    Path(cfg['model_path']).write_text('changed')
    run_one(store)
    assert store.job(jid)['status'] == 'failed'
    assert 'SHA256 mismatch' in store.job(jid)['error']['message']


def test_worker_deadline_kills_child(tmp_path):
    store, gateway, sid, cfg = setup(tmp_path, 'import time\ntime.sleep(10)')
    jid = gateway.call('evaluate_energy_forces', {'structure_id': sid, 'potential_id': 'nep89'})['job_id']
    run_one(store, timeout_s=0.1)
    assert store.job(jid)['status'] == 'failed'
    assert 'time budget' in store.job(jid)['error']['message']


def test_nonfinite_output(tmp_path):
    body = "from pathlib import Path\nPath('energy_train.out').write_text('nan 0\\n')\nPath('force_train.out').write_text('0 0 0 0 0 0\\n'*4)"
    store, gateway, sid, cfg = setup(tmp_path, body)
    jid = gateway.call('evaluate_energy_forces', {'structure_id': sid, 'potential_id': 'nep89'})['job_id']
    run_one(store)
    assert store.job(jid)['status'] == 'failed'
    assert 'Nonfinite' in store.job(jid)['error']['message']
