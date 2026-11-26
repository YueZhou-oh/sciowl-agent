"""Bounded, deterministic tool discovery. No extra model or embedding service."""
import json
import re

CORE_TOOLS = (
    'inspect_structure', 'list_potentials', 'check_potential',
    'get_job_status', 'get_result', 'cancel_job', 'search_tools',
)
OPTIONAL_LIMIT = 3
SCHEMA_BYTES = 24000
ALIASES = {
    'submit_relaxation': 'relax optimize optimisation optimization minimise minimize',
    'submit_eos': 'eos bulk modulus equation state',
    'submit_md': 'md dynamics simulation temperature',
    'evaluate_energy_forces': 'single point energy force stress predict',
    'evaluate_structures': 'batch multiple both structures frames predict',
    'analyze_trajectory': 'diffusion msd trajectory analysis',
}


def terms(text):
    return set(re.findall(r'[a-z0-9]+', text.lower().replace('_', ' ')))


def search_tools(query, limit=3, offset=0):
    from .contracts import TOOLS
    query_terms = terms(query)
    scored = []
    for name, (_, description) in TOOLS.items():
        if name in CORE_TOOLS:
            continue
        score = len(query_terms & terms(description + " " + ALIASES.get(name, ""))) + 3 * len(query_terms & terms(name))
        if name.lower() in query.lower():
            score += 100
        scored.append((-score, name))
    return [name for _, name in sorted(scored)[offset:offset+limit]]


def selected_schemas(messages):
    from .contracts import tool_schemas
    # Use the original task, not generated prose or raw numerical observations.
    query = next((m.get('content', '') for m in messages if m['role'] == 'user'), '')
    if "User request:\n" in query:
        query = query.rsplit("User request:\n", 1)[1]
    names = search_tools(query)
    # Discovery replaces the optional set rather than accumulating definitions.
    # Derive from persisted observations so resume does not need hidden state.
    for message in messages:
        if message['role'] != 'tool':
            continue
        try:
            observation = json.loads(message['content'])
        except (ValueError, TypeError):
            continue
        if observation.get('ok') and observation.get('discovery') == 'tools':
            names = [item['name'] for item in observation['tools']][:OPTIONAL_LIMIT]
    return tool_schemas(names=list(CORE_TOOLS) + names)


def schema_size(schemas):
    return len(json.dumps(schemas, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
