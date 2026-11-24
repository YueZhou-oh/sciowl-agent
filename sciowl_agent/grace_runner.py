"""Standalone child entry point. Run only with the configured GRACE interpreter."""
import json
import os
import sys
from pathlib import Path


def main():
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import PropertyNotImplementedError
    from tensorpotential.calculator import TPCalculator

    request = json.loads(Path(sys.argv[1]).read_text())
    calculator = TPCalculator(request['model_path'], mode='diverse', enable_uq_if_available=False)
    records = []
    for frame in request['frames']:
        atoms = Atoms(numbers=frame['numbers'], positions=frame['positions'],
                      cell=frame['cell'], pbc=frame['pbc'])
        properties = ['energy', 'forces'] + (['stress'] if frame['stress'] else [])
        try:
            calculator.calculate(atoms=atoms, properties=properties)
        except PropertyNotImplementedError:
            if not frame['stress']:
                raise
            calculator.calculate(atoms=atoms, properties=['energy', 'forces'])
        result = calculator.results
        records.append({'energy': float(result['energy']),
                        'forces': np.asarray(result['forces'], dtype=float).tolist(),
                        **({'stress': np.asarray(result['stress'], dtype=float).reshape(6).tolist()}
                           if frame['stress'] and 'stress' in result else {})})
    Path(sys.argv[2]).write_text(json.dumps(records, allow_nan=False))


if __name__ == '__main__':
    main()
