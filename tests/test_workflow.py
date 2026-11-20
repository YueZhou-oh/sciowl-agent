import json
import re

import pytest

from sciowl_agent.cli import main


class SinglePointModel:
    def __init__(self, *args, **kwargs):
        pass

    def next(self, messages, tools):
        observations = [json.loads(m['content']) for m in messages if m['role'] == 'tool']
        result = next((o for o in observations if o.get('result_id')), None)
        if result:
            return {'role': 'assistant', 'content': json.dumps({
                'summary': 'Calculation completed.', 'evidence_ids': [result['result_id']]})}
        job = next((o['job_id'] for o in observations if o.get('job_id')), None)
        if job:
            name, args = 'get_result', {'job_id': job}
        else:
            sid = re.search(r'Imported input structure ID: (a_\w+)', messages[1]['content'])[1]
            name, args = 'evaluate_energy_forces', {'structure_id': sid, 'potential_id': 'emt_demo'}
        return {'role': 'assistant', 'content': None, 'tool_calls': [
            {'id': 'call', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}


def invoke(tmp_path, monkeypatch, capsys, model=SinglePointModel, potential='emt_demo', extra=()):
    from ase.build import bulk
    from ase.io import write
    source = tmp_path / 'input with spaces.cif'
    write(source, bulk('Cu', 'fcc', a=3.6, cubic=True))
    monkeypatch.setattr('sciowl_agent.cli.EndpointModel', model)
    monkeypatch.setattr('sys.argv', ['sciowl-agent', '--workspace', str(tmp_path / 'workspace'),
        'run', '--structure', str(source), '--potential', potential, *extra,
        'Calculate energy and forces without relaxation.'])
    with pytest.raises(SystemExit) as exc:
        main()
    captured = capsys.readouterr()
    return exc.value.code, json.loads(captured.out), captured.err


def test_one_command_real_emt(tmp_path, monkeypatch, capsys):
    code, out, progress = invoke(tmp_path, monkeypatch, capsys)
    assert code == 0
    assert out['status'] == 'completed'
    assert len(out['results']) == 1
    result = out['results'][0]['result']
    assert result['operation'] == 'evaluate_energy_forces'
    assert result['n_atoms'] == 4
    assert isinstance(result['energy_eV'], float)
    assert out['report']['evidence']
    assert 'completed result_id=' in progress


def test_preflight_failure_never_calls_model(tmp_path, monkeypatch, capsys):
    class NeverModel(SinglePointModel):
        def next(self, *args):
            raise AssertionError('Model should not be called')
    code, out, progress = invoke(tmp_path, monkeypatch, capsys, NeverModel, 'unknown_potential')
    assert code == 1 and out['status'] == 'preflight_failed'
    assert not progress


def test_malformed_model_cannot_claim_success(tmp_path, monkeypatch, capsys):
    class BadModel(SinglePointModel):
        def next(self, *args):
            return {'role': 'assistant', 'content': 'not JSON'}
    code, out, progress = invoke(tmp_path, monkeypatch, capsys, BadModel, extra=('--max-steps', '2'))
    assert code == 1 and out['status'] == 'budget_exhausted'
    assert out['session']['jobs'] == [] and out['results'] == []
    assert not progress


def test_final_without_calculation_is_not_success(tmp_path, monkeypatch, capsys):
    class EmptyModel(SinglePointModel):
        def next(self, *args):
            return {'role': 'assistant', 'content': '{"summary":"Done", "evidence_ids":[]}'}
    code, out, _ = invoke(tmp_path, monkeypatch, capsys, EmptyModel)
    assert code == 1 and out['status'] == 'no_calculation'


def test_failed_calculation_returns_failure_evidence(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise RuntimeError('Calculator unavailable')
    monkeypatch.setattr('sciowl_agent.worker.execute', fail)
    code, out, progress = invoke(tmp_path, monkeypatch, capsys)
    assert code == 1 and out['status'] == 'calculation_failed'
    assert out['results'][0]['status'] == 'failed'
    assert out['results'][0]['error']['message'] == 'Calculator unavailable'
    assert 'failed error=' in progress
