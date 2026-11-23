#!/usr/bin/env python3
"""Create a local NEP registry from trusted executable/model paths."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True, type=Path)
    p.add_argument('--executable', required=True, type=Path)
    p.add_argument('--runtime-library-dir', type=Path)
    p.add_argument('--output', type=Path, default=Path('registry_nep.json'))
    args = p.parse_args()
    model, exe = args.model.resolve(), args.executable.resolve()
    header = model.read_text().splitlines()[0].split()
    cfg = dict(backend='nep', model_path=str(model), nep_bin=str(exe),
               sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
               executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),
               elements=header[2:2+int(header[1])], batch_size=128, timeout_s=600,
               max_atoms=256, demo_only=True, revision=model.name,
               domain='Local NEP checkpoint; application accuracy not independently validated')
    if args.runtime_library_dir:
        cfg['runtime_library_dir'] = str(args.runtime_library_dir.resolve())
    with args.output.open('x') as stream:
        json.dump({'nep89': cfg}, stream, indent=2)
    print(f'Created {args.output}')


if __name__ == '__main__':
    main()
