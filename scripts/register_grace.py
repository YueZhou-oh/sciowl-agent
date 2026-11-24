#!/usr/bin/env python3
"""Register a trusted local GRACE SavedModel and its separate Python interpreter."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sciowl_agent.grace import model_digest, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('/dpc-zhouy/zhouy/sciowl-agent-tools/demo/GRACE/GRACE-2L-SMAX-OMAT-large'))
    parser.add_argument('--python', default='/dpc-zhouy/zhouy/miniconda3/envs/GRACE/bin/python')
    parser.add_argument('--elements', nargs='+', required=True,
                        help='Verified element coverage for this checkpoint; use Cu for the Cu demo')
    parser.add_argument('--potential-id', default='grace')
    parser.add_argument('--output', type=Path, default=Path('registry/registry_grace.json'))
    args = parser.parse_args()
    from ase.data import atomic_numbers
    if any(element not in atomic_numbers for element in args.elements):
        parser.error('Unknown chemical element')
    cfg = dict(backend='grace', model_path=str(args.model.resolve()),
               python_executable=str(Path(args.python).absolute()), sha256=model_digest(args.model),
               elements=sorted(set(args.elements)), timeout_s=600, max_atoms=256,
               demo_only=True, revision=args.model.name,
               domain='Local GRACE checkpoint; application accuracy not independently validated')
    cfg = validate_config(cfg, None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump({args.potential_id: cfg}, stream, indent=2)
        stream.write('\n')
    print(f'Created {args.output}')


if __name__ == '__main__':
    main()
