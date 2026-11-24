#!/usr/bin/env python3
"""Select an explicit registry or merge the repository's configured potentials."""
import argparse
import json
import sys
from pathlib import Path


def prepare_registry(root, output, explicit=None):
    if explicit:
        path = Path(explicit).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f'Registry is not a file: {path}')
        return path
    merged, sources = {}, []
    for name in ('registry_mace.json', 'registry_nep.json', 'registry_grace.json',
                 'registry/registry_mace.json', 'registry/registry_nep.json', 'registry/registry_grace.json'):
        path = root / name
        if not path.is_file():
            continue
        entries = json.loads(path.read_text())
        if not isinstance(entries, dict):
            raise ValueError(f'Registry must be a JSON object: {path}')
        for potential, cfg in entries.items():
            if potential in merged and merged[potential] != cfg:
                raise ValueError(
                    f'Conflicting potential {potential!r} in {path}. '
                    'Set SCIOWL_REGISTRY to your combined registry.')
            merged[potential] = cfg
        sources.append(str(path))
    if not merged:
        raise ValueError('No local potential registry found. Set SCIOWL_REGISTRY to a configured registry JSON.')
    output.write_text(json.dumps(merged, indent=2) + '\n')
    print('Registered potentials: ' + ', '.join(merged), file=sys.stderr)
    print('Registry sources: ' + ', '.join(sources), file=sys.stderr)
    return output.resolve()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registry', default='')
    args = parser.parse_args()
    try:
        path = prepare_registry(args.root.resolve(), args.output, args.registry)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')
    print(path)


if __name__ == '__main__':
    main()
