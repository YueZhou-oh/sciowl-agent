"""Native GPUMD NEP prediction backend; no shell execution or Python MLIP dependency."""
import hashlib
import os
import re
import shutil
import subprocess
from pathlib import Path
from tempfile import mkdtemp
import numpy as np
from ase.io import write
from ase.calculators.calculator import Calculator, all_changes

GPA_PER_EV_A3 = 160.21766208

def _header_values(lines: list[str], name: str) -> list[str]:
    for line in lines[:40]:
        fields = line.split()
        if fields and fields[0].lower() == name.lower():
            return fields[1:]
    raise ValueError(f"NEP model header is missing {name!r}")


def build_nep_in(model: Path, batch_size: int) -> str:
    """Build a prediction-only nep.in that matches the selected nep.txt."""
    lines = [line.strip() for line in model.read_text(encoding="utf-8").splitlines()]
    first = next((line.split() for line in lines if line), None)
    if first is None or len(first) < 3:
        raise ValueError(f"invalid NEP model header: {model}")

    match = re.fullmatch(r"nep(\d+)(?:_zbl)?", first[0].lower())
    if match is None:
        raise ValueError(f"cannot determine NEP version from {first[0]!r}")

    number_of_types = int(first[1])
    elements = first[2 : 2 + number_of_types]
    if len(elements) != number_of_types:
        raise ValueError("incomplete element list in NEP model header")

    rows = [
        f"type {number_of_types} {' '.join(elements)}",
        f"version {match.group(1)}",
    ]
    if first[0].lower().endswith("_zbl"):
        rows.append(f"zbl {_header_values(lines, 'zbl')[-1]}")

    rows.extend(
        [
            f"cutoff {' '.join(_header_values(lines, 'cutoff')[:2])}",
            f"n_max {' '.join(_header_values(lines, 'n_max')[:2])}",
            f"basis_size {' '.join(_header_values(lines, 'basis_size')[:2])}",
            f"l_max {' '.join(_header_values(lines, 'l_max')[:3])}",
            f"neuron {_header_values(lines, 'ANN')[0]}",
            f"batch {batch_size}",
            "prediction 1",
            "",
        ]
    )
    return "\n".join(rows)


def _prediction_only_frames(frames):
    """Create temporary NEP-readable frames without changing the source data."""
    from ase import Atoms

    clean = []
    for source in frames:
        atoms = Atoms(
            symbols=source.get_chemical_symbols(),
            positions=source.get_positions(),
            cell=source.cell,
            pbc=source.pbc,
        )
        # Native nep requires labels even in prediction mode. These temporary
        # placeholders do not enter the model calculation.
        atoms.info["energy"] = 0.0
        atoms.arrays["forces"] = np.zeros((len(atoms), 3), dtype=float)
        clean.append(atoms)
    return clean


def _load_rows(path: Path, expected_rows: int, minimum_columns: int) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape[0] != expected_rows or rows.shape[1] < minimum_columns:
        raise RuntimeError(
            f"unexpected shape for {path.name}: {rows.shape}; "
            f"expected ({expected_rows}, >={minimum_columns})"
        )
    return rows



def validate_config(cfg, atoms):
    cfg = dict(cfg)
    for key, digest in (("model_path", "sha256"), ("nep_bin", "executable_sha256")):
        path = Path(cfg[key]).resolve()
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != cfg[digest]:
            raise ValueError(f"NEP {key} missing or SHA256 mismatch")
        cfg[key] = str(path)
    if not os.access(cfg["nep_bin"], os.X_OK):
        raise ValueError("NEP executable is not executable")
    build_nep_in(Path(cfg["model_path"]), 1)
    header = Path(cfg["model_path"]).read_text().splitlines()[0].split()
    if not set(atoms.get_chemical_symbols()).issubset(header[2:2 + int(header[1])]):
        raise ValueError("NEP model header does not cover these elements")
    if int(cfg.get("batch_size", 128)) < 1:
        raise ValueError("NEP batch_size must be positive")
    timeout = float(cfg.get("timeout_s", 600))
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("NEP timeout_s must be positive and finite")
    if cfg.get("runtime_library_dir"):
        directory = Path(cfg["runtime_library_dir"]).resolve()
        if not (directory / "libstdc++.so.6").is_file():
            raise ValueError("NEP runtime_library_dir must contain libstdc++.so.6")
        cfg["runtime_library_dir"] = str(directory)
    return cfg


def predict_native(frames, cfg, directory, tick=lambda: None):
    frames = list(frames)
    if not frames:
        raise ValueError("No structures supplied")
    for atoms in frames:
        if abs(np.linalg.det(atoms.cell)) < 1e-8:
            raise ValueError("Native NEP requires a nonsingular simulation cell, including for molecules")
        header = Path(cfg["model_path"]).read_text().splitlines()[0].split()
        if not set(atoms.get_chemical_symbols()).issubset(header[2:2 + int(header[1])]):
            raise ValueError("NEP model header does not cover these elements")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    work = Path(mkdtemp(prefix="prediction-", dir=directory))
    write(work / "train.xyz", _prediction_only_frames(frames), format="extxyz", write_results=False)
    shutil.copy2(cfg["model_path"], work / "nep.txt")
    (work / "nep.in").write_text(build_nep_in(Path(cfg["model_path"]), min(int(cfg.get("batch_size", 128)), len(frames))))
    env = os.environ.copy()
    if cfg.get("runtime_library_dir"):
        env["LD_LIBRARY_PATH"] = cfg["runtime_library_dir"] + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    tick()
    # Poll while running so cancellation and worker time budgets remain effective.
    import time
    started = time.monotonic()
    with (work / "stdout.log").open("w") as out, (work / "stderr.log").open("w") as err:
        process = subprocess.Popen([cfg["nep_bin"]], cwd=work, env=env, stdout=out, stderr=err)
        try:
            while True:
                try:
                    code = process.wait(timeout=0.2)
                    break
                except subprocess.TimeoutExpired:
                    tick()
                    if time.monotonic() - started > float(cfg.get("timeout_s", 600)):
                        raise TimeoutError("Native NEP subprocess timed out")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    tick()
    if code:
        raise RuntimeError(f"Native NEP exited with code {code}; see {work}/stderr.log and stdout.log")
    counts = np.array([len(a) for a in frames])
    energies = _load_rows(work / "energy_train.out", len(frames), 2)[:, 0] * counts
    forces = _load_rows(work / "force_train.out", int(counts.sum()), 6)[:, :3]
    stress_path = work / "stress_train.out"
    stresses = (_load_rows(stress_path, len(frames), 12)[:, [0, 1, 2, 4, 5, 3]] / GPA_PER_EV_A3
                if stress_path.exists() else None)
    if not np.isfinite(energies).all() or not np.isfinite(forces).all() or (stresses is not None and not np.isfinite(stresses).all()):
        raise ValueError("Nonfinite NEP predictions")
    offsets = np.concatenate(([0], np.cumsum(counts)))
    return [dict(energy=float(energies[i]), forces=forces[offsets[i]:offsets[i+1]],
                 **({"stress": stresses[i]} if stresses is not None else {})) for i in range(len(frames))]


class NativeNEP(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def __init__(self, cfg, directory, tick=lambda: None):
        super().__init__()
        self.cfg, self.work, self.tick = cfg, directory, tick

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = predict_native([self.atoms], self.cfg, self.work, self.tick)[0]
