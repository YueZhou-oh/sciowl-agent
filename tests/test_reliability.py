import json

import httpx
import pytest
from ase.build import bulk

from sciowl_agent.gateway import Gateway
from sciowl_agent.harness import Harness
from sciowl_agent.model import EndpointModel, ScriptedModel
from sciowl_agent.reliability import assess_job
from sciowl_agent.science import atoms_to_id
from sciowl_agent.store import Store
from sciowl_agent.worker import run_one


@pytest.fixture
def env(tmp_path):
    store = Store(tmp_path)
    atoms = bulk('Cu', cubic=True)
    atoms.positions[0, 0] += .08
    sid = atoms_to_id(store, atoms)
    return store, Gateway(store), sid


@pytest.mark.parametrize('operation,extra', [
    ('submit_eos', {'volume_factors': [1.] * 7}),
    ('submit_md', {'production_steps': 2, 'sample_interval': 5}),
])
def test_invalid_protocol_never_queued(env, operation, extra):
    store, gateway, sid = env
    out = gateway.call(operation, {'structure_id': sid, **extra})
    assert not out['ok']
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0


def test_invalid_geometry_in_any_batch_frame_never_queued(env):
    store, gateway, sid = env
    atoms = bulk('Cu', cubic=True)
    bad = atoms_to_id(store, atoms)
    data = store.get_json(bad)
    data['positions'][1] = data['positions'][0]
    bad = store.put_json(data, 'structure')
    out = gateway.call('evaluate_structures', {'structure_ids': [sid, bad], 'potential_id': 'emt_demo'})
    assert not out['ok'] and 'overlap' in out['error']['message']
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0


def test_constraints_persist_and_gateway_enforces(env):
    store, gateway, sid = env
    constraints = {'potential_id': 'emt_demo', 'allowed_operations': ['evaluate_energy_forces']}
    class Finish:
        def next(self, messages, tools):
            assert 'Enforced session constraints' in messages[0]['content']
            return {'role': 'assistant', 'content': '{"summary":"Ready","evidence_ids":[]}'}
    Harness(store, gateway, Finish(), constraints=constraints).run('locked', 'Calculate')
    assert not gateway.call('submit_relaxation', {'structure_id': sid}, 'locked')['ok']
    assert not gateway.call('evaluate_energy_forces', {'structure_id': sid, 'potential_id': 'other'}, 'locked')['ok']
    assert Gateway(store).call('evaluate_energy_forces', {'structure_id': sid}, 'locked')['ok']
    with pytest.raises(ValueError, match='Cannot change constraints'):
        Harness(store, gateway, Finish(), constraints={}).run('locked')


@pytest.mark.parametrize('error,retryable', [('timeout', True), (429, True), (503, True), (400, False), (401, False)])
def test_endpoint_error_checkpoint_and_explicit_resume(env, error, retryable):
    store, gateway, sid = env
    calls = []
    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            if error == 'timeout':
                raise httpx.ReadTimeout('secret-token-in-provider-error', request=request)
            return httpx.Response(error, json={'error': 'secret-token-in-provider-error'})
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': '{"summary":"Ready","evidence_ids":[]}'}}]})
    model = EndpointModel('http://test/v1', 'test', client=httpx.Client(transport=httpx.MockTransport(handle)))
    state = Harness(store, gateway, model).run('endpoint', 'Inspect')
    assert len(calls) == 1 and state['status'] == 'endpoint_error'
    assert state['last_error']['retryable'] is retryable
    assert state['steps'] == 1 and len(state['messages']) == 2
    assert 'secret-token' not in json.dumps(state)
    state = Harness(store, gateway, model).run('endpoint')
    assert state['status'] == 'completed' and state['last_error'] is None


def test_endpoint_failure_does_not_duplicate_completed_job(env):
    store, gateway, sid = env
    class Flaky(ScriptedModel):
        def next(self, messages, tools):
            if any(m['role'] == 'tool' and json.loads(m['content']).get('job_id') for m in messages):
                raise httpx.ReadTimeout('offline')
            return super().next(messages, tools)
    state = Harness(store, gateway, Flaky(sid), inline_worker=True).run('resume', 'Relax')
    assert state['status'] == 'endpoint_error' and len(state['jobs']) == 1
    state = Harness(store, gateway, ScriptedModel(sid)).run('resume')
    assert state['status'] == 'completed'
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 1


def test_unconverged_report_and_artifact_integrity(env):
    store, gateway, sid = env
    jid = gateway.call('submit_relaxation', {'structure_id': sid, 'max_steps': 1, 'fmax_eV_per_A': 1e-8})['job_id']
    run_one(store)
    out = gateway.call('get_result', {'job_id': jid})
    rid = out['result_id']
    assessment = assess_job(store, store.job(jid), [rid])
    assert assessment['execution_status'] == 'completed'
    assert assessment['scientific_outcome'] == 'not_converged'
    with pytest.raises(ValueError, match='every session job'):
        assess_job(store, store.job(jid), [])
    state = {'session_id': 'report', 'jobs': [jid], 'evidence': [rid]}
    Harness(store, gateway, None).finalize(state, {'content': json.dumps({'summary': 'Unverified model text', 'evidence_ids': [rid]})})
    report = store.get_json(state['report_id'])
    assert report['job_assessments'] == [assessment]
    assert not report['commentary_verified']
    store.artifact(out['result']['output_structure_id']).write_text('corrupt')
    with pytest.raises(ValueError, match='integrity'):
        assess_job(store, store.job(jid), [rid])


def test_failure_evidence_mandatory(env, monkeypatch):
    store, gateway, sid = env
    jid = gateway.call('evaluate_energy_forces', {'structure_id': sid})['job_id']
    def fail(*args, **kwargs):
        raise RuntimeError('calculator failed')
    monkeypatch.setattr('sciowl_agent.worker.execute', fail)
    run_one(store)
    with pytest.raises(ValueError, match='failure evidence'):
        assess_job(store, store.job(jid), [])
    failure = gateway.call('get_result', {'job_id': jid})
    assert assess_job(store, store.job(jid), [failure['result_id']])['scientific_outcome'] == 'no_successful_result'


@pytest.mark.parametrize('protocol', ['native', 'json', 'schema'])
def test_constraints_visible_in_all_protocols(env, protocol):
    store, gateway, _ = env
    def handle(request):
        payload = json.loads(request.content)
        assert 'Enforced session constraints' in payload['messages'][0]['content']
        assert 'evaluate_energy_forces' in payload['messages'][0]['content']
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': '{"summary":"Ready","evidence_ids":[]}'}}]})
    model = EndpointModel('http://test/v1', 'test', protocol,
                          client=httpx.Client(transport=httpx.MockTransport(handle)))
    state = Harness(store, gateway, model, constraints={'allowed_operations': ['evaluate_energy_forces']}).run('locked', 'Calculate')
    assert state['status'] == 'completed'


def test_evidence_cannot_omit_a_failed_job(env, monkeypatch):
    store, gateway, sid = env
    good = gateway.call('evaluate_energy_forces', {'structure_id': sid})['job_id']
    run_one(store)
    good_result = gateway.call('get_result', {'job_id': good})['result_id']
    bad = gateway.call('submit_relaxation', {'structure_id': sid})['job_id']
    monkeypatch.setattr('sciowl_agent.worker.execute', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('failed')))
    run_one(store)
    failure = gateway.call('get_result', {'job_id': bad})['result_id']
    state = {'session_id': 'all-jobs', 'jobs': [good, bad], 'evidence': [good_result, failure]}
    harness = Harness(store, gateway, None)
    with pytest.raises(ValueError, match='failure evidence'):
        harness.finalize(state, {'content': json.dumps({'summary': 'Done', 'evidence_ids': [good_result]})})
    harness.finalize(state, {'content': json.dumps({'summary': 'Partial result', 'evidence_ids': [good_result, failure]})})
    report = store.get_json(state['report_id'])
    assert [x['scientific_outcome'] for x in report['job_assessments']] == ['calculated', 'no_successful_result']


def test_mismatched_provenance_rejected(env):
    store, gateway, sid = env
    jid = gateway.call('evaluate_energy_forces', {'structure_id': sid})['job_id']
    run_one(store)
    job = store.job(jid)
    result = store.get_json(job['result']['result_id'])
    provenance = store.get_json(result['provenance_id'])
    provenance['job_id'] = 'j_other'
    result['provenance_id'] = store.put_json(provenance, 'provenance')
    rid = store.put_json(result, 'result')
    job['result']['result_id'] = rid
    with pytest.raises(ValueError, match='provenance'):
        assess_job(store, job, [rid])


def test_cli_endpoint_error_outputs_json_and_nonzero(env, monkeypatch, capsys):
    from sciowl_agent.cli import main
    store, _, _ = env
    class Offline:
        def __init__(self, *args, **kwargs):
            pass
        def next(self, *args):
            raise httpx.ConnectError('offline')
    monkeypatch.setattr('sciowl_agent.cli.EndpointModel', Offline)
    monkeypatch.setattr('sys.argv', ['sciowl-agent', '--workspace', str(store.root),
                                    'agent', '--session', 'cli', '--request', 'Inspect', '--follow'])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().out)['status'] == 'endpoint_error'
