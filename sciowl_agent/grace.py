"""ASE adapter for GRACE inference in a separately configured Python environment."""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
from ase.calculators.calculator import Calculator, all_changes


def model_digest(directory):
    """Hash sorted relative paths and file digests, including SavedModel variables."""
    root = Path(directory).resolve()
    if not root.is_dir():
        raise ValueError('GRACE model_path must be a directory')
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if not files:
        raise ValueError('GRACE model directory is empty')
    digest = hashlib.sha256()
    for path in files:
        file_hash = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                file_hash.update(block)
        digest.update(path.relative_to(root).as_posix().encode() + b'\0' + file_hash.digest())
    return digest.hexdigest()


def validate_config(cfg, atoms):
    cfg = dict(cfg)
    python = Path(cfg['python_executable']).absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('GRACE python_executable is missing or not executable')
    model = Path(cfg['model_path']).resolve()
    if model_digest(model) != cfg['sha256']:
        raise ValueError('GRACE model SHA256 mismatch')
    timeout = float(cfg.get('timeout_s', 600))
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError('GRACE timeout_s must be positive and finite')
    cfg.update(model_path=str(model), python_executable=str(python))
    return cfg


def predict_external(frames, cfg, directory, tick=lambda: None, stress=True):
    frames = list(frames)
    if not frames:
        raise ValueError('No structures supplied')
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    work = Path(mkdtemp(prefix='prediction-', dir=directory))
    payload = {'model_path': cfg['model_path'], 'frames': [
        {'numbers': a.numbers.tolist(), 'positions': a.positions.tolist(),
         'cell': a.cell.tolist(), 'pbc': a.pbc.tolist(),
         'stress': bool(stress and a.pbc.any() and abs(np.linalg.det(a.cell)) > 1e-8)}
        for a in frames]}
    request, response = work / 'input.json', work / 'output.json'
    request.write_text(json.dumps(payload, allow_nan=False))
    env = os.environ.copy()
    env['TF_USE_LEGACY_KERAS'] = '1'
    # Do not import packages from the parent interpreter's environment.
    env.pop('PYTHONHOME', None)
    env.pop('PYTHONPATH', None)
    command = [cfg['python_executable'], '-I', str(Path(__file__).with_name('grace_runner.py')),
               str(request), str(response)]
    started = time.monotonic()
    tick()
    with (work / 'stdout.log').open('w') as out, (work / 'stderr.log').open('w') as err:
        process = subprocess.Popen(command, cwd=work, env=env, stdout=out, stderr=err)
        try:
            while True:
                try:
                    code = process.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    tick()
                    if time.monotonic() - started > float(cfg.get('timeout_s', 600)):
                        raise TimeoutError('GRACE subprocess timed out')
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    tick()
    if code:
        raise RuntimeError(f'GRACE exited with code {code}; see {work}/stderr.log and stdout.log')
    records = json.loads(response.read_text())
    if not isinstance(records, list) or len(records) != len(frames):
        raise ValueError('GRACE returned an unexpected frame count')
    results = []
    for atoms, record in zip(frames, records):
        energy = float(record['energy'])
        forces = np.asarray(record['forces'], dtype=float)
        if not np.isfinite(energy) or forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
            raise ValueError('Invalid or nonfinite GRACE energy/forces')
        result = dict(energy=energy, forces=forces)
        if record.get('stress') is not None:
            value = np.asarray(record['stress'], dtype=float)
            if value.shape != (6,) or not np.isfinite(value).all():
                raise ValueError('Invalid or nonfinite GRACE stress')
            result['stress'] = value
        results.append(result)
    return results


class ExternalGRACE(Calculator):
    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, cfg, directory, tick=lambda: None):
        super().__init__()
        self.cfg, self.work, self.tick = cfg, directory, tick

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = predict_external([self.atoms], self.cfg, self.work, self.tick)[0]
