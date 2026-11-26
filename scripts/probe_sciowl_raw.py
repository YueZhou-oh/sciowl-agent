#!/usr/bin/env python3
"""Capture one SciOWL endpoint response without executing any scientific tools.

Run from an environment where sciowl_agent is importable. --state replays only
the first system/user pair, not the failed session's accumulated repair prompts.
The output contains response data and request settings, never request headers.
"""
import argparse
import json
import os
from pathlib import Path

from scripts.probe_endpoint import matches_call

import httpx
from sciowl_agent.contracts import tool_schemas
from sciowl_agent.discovery import selected_schemas, schema_size
from sciowl_agent.model import EndpointModel, SYSTEM


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default=os.environ.get('SCIOWL_BASE_URL'))
    p.add_argument('--model', default=os.environ.get('SCIOWL_MODEL', 'sciowl'))
    p.add_argument('--protocol', choices=['native', 'json', 'schema'], default='native')
    p.add_argument('--state', type=Path)
    p.add_argument('--all-tools', action='store_true', help='Compare against the full catalog instead of production discovery')
    p.add_argument('--max-tokens', type=int, help='Completion budget for this probe', default=512)
    p.add_argument('--timeout', type=float, default=180)
    p.add_argument('--output', type=Path, default=Path('sciowl_raw.json'))
    a = p.parse_args()
    if not a.base_url:
        p.error('--base-url or SCIOWL_BASE_URL is required')
    if a.timeout <= 0:
        p.error('--timeout must be positive')
    if a.max_tokens < 1:
        p.error('--max-tokens must be positive')
    if a.output.exists():
        p.error('Output already exists; choose a new filename')
    messages = [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': 'List the available registered potentials using list_potentials. Call the tool now.'},
    ]
    if a.state:
        messages = json.loads(a.state.read_text())['messages'][:2]
        if [m['role'] for m in messages] != ['system', 'user']:
            p.error('State must begin with system and user messages')
    schemas = tool_schemas() if a.all_tools else selected_schemas(messages)
    capture = {'protocol': a.protocol, 'notice': 'No tool was executed.',
               'selected_tools': [t['function']['name'] for t in schemas],
               'schema_bytes': schema_size(schemas)}
    with httpx.Client(timeout=a.timeout, trust_env=os.environ.get('SCIOWL_TRUST_ENV') == '1') as http:
        class CaptureClient:
            @property
            def timeout(self):
                return http.timeout

            def post(self, url, **kwargs):
                payload = kwargs['json']
                capture['settings'] = {k: payload.get(k) for k in
                                       ('model', 'max_tokens', 'temperature', 'tool_choice', 'chat_template_kwargs', 'response_format')}
                response = http.post(url, **kwargs)
                capture['http_status'] = response.status_code
                try:
                    capture['raw_response'] = response.json()
                except ValueError:
                    capture['raw_response_text'] = response.text
                return response
        adapter = EndpointModel(a.base_url, a.model, a.protocol, client=CaptureClient(), max_tokens=a.max_tokens)
        try:
            capture['adapter_response'] = adapter.next(messages, schemas)
            if not a.state:
                capture['matches_expected_call'] = matches_call(capture['adapter_response'], 'list_potentials', {})
        except Exception as exc:
            capture['adapter_error'] = f'{type(exc).__name__}: {exc}'
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as out:
        json.dump(capture, out, indent=2, ensure_ascii=False)
    raw = capture.get('raw_response', {})
    if not isinstance(raw, dict):
        raw = {}
    choice = (raw.get('choices') or [{}])[0]
    message = choice.get('message') or {}
    settings = dict(capture.get('settings', {}))
    if settings.get('response_format'):
        settings['response_format'] = settings['response_format']['type']
    print(json.dumps({
        'saved': str(a.output), 'settings': settings,
        'http_status': capture.get('http_status'), 'usage': raw.get('usage'),
        'finish_reason': choice.get('finish_reason'),
        'tool_calls': message.get('tool_calls'),
        'content_preview': (message.get('content') or '')[:1200],
        'reasoning_preview': (message.get('reasoning_content') or message.get('reasoning') or '')[:1200],
        'adapter_error': capture.get('adapter_error'),
        'matches_expected_call': capture.get('matches_expected_call'),
    }, indent=2, ensure_ascii=False))
    return 1 if 'adapter_error' in capture or capture.get('matches_expected_call') is False else 0


if __name__ == '__main__':
    raise SystemExit(main())
