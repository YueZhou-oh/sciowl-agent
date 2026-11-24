#!/usr/bin/env python3
"""Predict GRACE energies, forces, and stresses for target_test.extxyz."""

import json
import os
from pathlib import Path

import numpy as np


MODEL = Path("/dpc-zhouy/zhouy/sciowl-agent-tools/demo/GRACE/GRACE-2L-SMAX-OMAT-large")
INPUT = Path("/dpc-zhouy/zhouy/sciowl-agent-tools/demo/target_test.extxyz")
OUTPUT = Path("/dpc-zhouy/zhouy/sciowl-agent-tools/demo/GRACE/target_test_grace.jsonl")


def predict_dataset(*, frames, calculator, output: Path) -> int:
    """Run official TPCalculator inference and write MACE-compatible JSONL."""
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for index, atoms in enumerate(frames):
            calculator.calculate(
                atoms=atoms,
                properties=["energy", "forces", "stress"],
            )
            record = {
                "index": index,
                "formula": atoms.get_chemical_formula(),
                "energy_eV": float(calculator.results["energy"]),
                "forces_eV_per_A": np.asarray(
                    calculator.results["forces"], dtype=float
                ).tolist(),
                "stress_eV_per_A3": np.asarray(
                    calculator.results["stress"], dtype=float
                ).reshape(6).tolist(),
            }
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            count += 1
            print(index + 1, flush=True)
    return count


def main() -> int:
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

    from ase.io import iread
    from tensorpotential.calculator import TPCalculator

    if not MODEL.is_dir():
        raise FileNotFoundError(MODEL)
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    calculator = TPCalculator(
        str(MODEL),
        mode="diverse",
        enable_uq_if_available=False,
    )
    count = predict_dataset(
        frames=iread(INPUT, index=":"),
        calculator=calculator,
        output=OUTPUT,
    )
    print(f"Wrote {count} predictions to {OUTPUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
