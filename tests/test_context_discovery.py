import json

import httpx
import pytest

from sciowl_agent.contracts import TOOLS, Empty, tool_schemas
from sciowl_agent.discovery import CORE_TOOLS, selected_schemas, schema_size
from sciowl_agent.gateway import Gateway
from sciowl_agent.harness import Harness
from sciowl_agent.model import EndpointModel, ContextLimitError, SYSTEM
from sciowl_agent.science import Registry
from sciowl_agent.store import Store


def messages(query='Calculate energy and forces'):
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': query}]


def names(schemas):
    return {t['function']['name'] for t in schemas}


def test_large_catalog_bounded_and_exact_discovery(monkeypatch, tmp_path):
    for i in range(1000):
        monkeypatch.setitem(TOOLS, f'extra_{i:04d}', (Empty, 'Specialized operation ' + 'documentation ' * 100))
    initial = messages()
    selected = selected_schemas(initial)
    assert len(selected) == len(CORE_TOOLS) + 3
    assert 'evaluate_energy_forces' in names(selected)
    assert schema_size(selected) < schema_size(tool_schemas()) / 10
    gateway = Gateway(Store(tmp_path))
    found = gateway.call('search_tools', {'query': 'extra_0999'})
    assert found['tools'][0]['name'] == 'extra_0999'
    initial.append({'role': 'tool', 'content': json.dumps(found)})
    assert 'extra_0999' in names(selected_schemas(initial))
    # A new search replaces, rather than unions with, the previous optional set.
    found = gateway.call('search_tools', {'query': 'submit_md'})
    initial.append({'role': 'tool', 'content': json.dumps(found)})
    selected = selected_schemas(initial)
    assert 'submit_md' in names(selected)
    assert 'extra_0999' not in names(selected)
    assert len(selected) == len(CORE_TOOLS) + 3
    assert names(selected_schemas(json.loads(json.dumps(initial)))) == names(selected)


def test_search_pagination_reaches_entire_catalog(tmp_path):
    gateway = Gateway(Store(tmp_path))
    found = set()
    offset = 0
    while offset is not None:
        page = gateway.call('search_tools', {'query': 'operation', 'offset': offset})
        found.update(t['name'] for t in page['tools'])
        offset = page['next_offset']
    assert found == set(TOOLS) - set(CORE_TOOLS)


def test_compact_registry_pagination_and_no_secrets(tmp_path):
    registry = Registry()
    registry.entries = {f'potential_{i:04d}': {
        'backend': 'mace', 'domain': 'long domain ' * 1000,
        'elements': ['H', 'C'], 'model_path': '/secret/model',
        'sha256': 'secret-checksum', 'new_private_field': 'secret',
    } for i in range(1000)}
    gateway = Gateway(Store(tmp_path), registry)
    page = gateway.call('list_potentials', {})
    assert page['total'] == 1000 and page['next_offset'] == 10
    assert len(page['potentials']) == 10
    assert len(json.dumps(page)) < 5000
    assert 'secret' not in json.dumps(page)
    second = gateway.call('list_potentials', {'offset': 10})
    assert not set(page['potentials']) & set(second['potentials'])
    exact = gateway.call('list_potentials', {'query': 'potential_0999'})
    assert list(exact['potentials']) == ['potential_0999']
    assert exact['next_offset'] is None
    assert not gateway.call('list_potentials', {'limit': 10000})['ok']
    assert registry.entries['potential_0000']['model_path'] == '/secret/model'


@pytest.mark.parametrize('protocol', ['native', 'json', 'schema'])
def test_full_request_budget_includes_definitions_before_http(protocol):
    requests = []
    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': '{"summary":"Done","evidence_ids":[]}'}}]})
    model = EndpointModel('http://test/v1', 'test', protocol,
                          max_input_bytes=15000,
                          client=httpx.Client(transport=httpx.MockTransport(handle)))
    model.next(messages(), selected_schemas(messages()))
    assert len(requests) == 1
    large = [{'type': 'function', 'function': {'name': 'large', 'description': '界' * 10000,
             'parameters': {'type': 'object', 'properties': {}}}}]
    with pytest.raises(ContextLimitError):
        model.next(messages(), large)
    assert len(requests) == 1


def test_context_failure_is_terminal_not_protocol_repair(tmp_path):
    class TooLarge:
        def next(self, messages, tools):
            raise ContextLimitError('too big')
    store = Store(tmp_path)
    state = Harness(store, Gateway(store), TooLarge()).run('limit', 'Calculate energy')
    assert state['status'] == 'context_limit'
    assert len(state['messages']) == 2
    assert Harness(store, Gateway(store), TooLarge()).run('limit') == state


def test_discovery_harness_resume_executes_newly_loaded_tool(monkeypatch, tmp_path):
    # More than three optional tools ensures the target is not initially exposed.
    for i in range(10):
        monkeypatch.setitem(TOOLS, f'zz_custom_{i}', (Empty, 'custom scientific operation'))
    class Discover:
        def next(self, history, tools):
            assert 'zz_custom_9' not in names(tools)
            return {'role': 'assistant', 'tool_calls': [{'id': 'find', 'type': 'function',
                'function': {'name': 'search_tools', 'arguments': '{"query":"zz_custom_9"}'}}]}
    class Resume:
        def next(self, history, tools):
            assert 'zz_custom_9' in names(tools)
            return {'role': 'assistant', 'content': '{"summary":"Found tool","evidence_ids":[]}'}
    store = Store(tmp_path)
    harness = Harness(store, Gateway(store), Discover())
    state = {'session_id': 'resume', 'messages': messages(), 'steps': 0, 'pending': [],
             'jobs': [], 'evidence': [], 'status': 'thinking', 'report_id': None}
    state = harness.act(harness.think(state))
    result = Harness(store, Gateway(store), Resume()).run('resume')
    assert result['status'] == 'completed'


@pytest.mark.parametrize('protocol', ['native', 'json', 'schema'])
def test_discover_then_run_real_eos_through_endpoint(tmp_path, protocol):
    from ase.build import bulk
    from sciowl_agent.science import atoms_to_id
    store = Store(tmp_path)
    sid = atoms_to_id(store, bulk('Cu', 'fcc', a=3.6, cubic=True))
    calls = []
    def handle(request):
        payload = json.loads(request.content)
        step = len(calls)
        calls.append(payload)
        if protocol == 'native':
            offered = names(payload['tools'])
        elif protocol == 'schema':
            offered = {b['properties']['tool']['enum'][0] for b in
                       payload['response_format']['json_schema']['schema']['anyOf']
                       if 'tool' in b['properties']}
        else:
            offered = names(json.loads(payload['messages'][0]['content'].split('Available tools: ', 1)[1]))
        if step == 0:
            assert 'submit_eos' not in offered
            obj = {'tool': 'search_tools', 'arguments': {'query': 'submit_eos'}}
        elif step == 1:
            assert 'submit_eos' in offered
            obj = {'tool': 'submit_eos', 'arguments': {'structure_id': sid}}
        elif step == 2:
            with store.connect() as db:
                jid = db.execute('SELECT id FROM jobs').fetchone()[0]
            obj = {'tool': 'get_result', 'arguments': {'job_id': jid}}
        else:
            state = store.load_session('eos')
            obj = {'summary': 'EOS evidence retrieved.', 'evidence_ids': state['evidence']}
        message = {'role': 'assistant', 'content': json.dumps(obj)}
        if protocol == 'native' and 'tool' in obj:
            message = {'role': 'assistant', 'content': None, 'tool_calls': [{
                'id': 'call', 'type': 'function', 'function': {
                    'name': obj['tool'], 'arguments': json.dumps(obj['arguments'])}}]}
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': message}]})
    model = EndpointModel('http://test/v1', 'test', protocol,
                          client=httpx.Client(transport=httpx.MockTransport(handle)))
    state = Harness(store, Gateway(store), model, inline_worker=True).run('eos', 'Inspect structure')
    assert state['status'] == 'completed'
    assert len(calls) == 4
    assert store.get_json(state['evidence'][0])['validated']


def test_unloaded_tool_cannot_execute(tmp_path):
    class Invalid:
        def next(self, history, tools):
            assert 'submit_eos' not in names(tools)
            return {'role': 'assistant', 'tool_calls': [{'id': 'call', 'type': 'function',
                'function': {'name': 'submit_eos', 'arguments': '{}'}}]}
    store = Store(tmp_path)
    state = Harness(store, Gateway(store), Invalid()).run('invalid', 'Inspect structure')
    assert not state['jobs'] and not state['pending']
    assert 'Tool not loaded' in state['messages'][-1]['content']
