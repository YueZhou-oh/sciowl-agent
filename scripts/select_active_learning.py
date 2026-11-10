#!/usr/bin/env python3
"""Select diverse-to-label candidates by force disagreement from a trusted ensemble.

This is acquisition only, not calibrated uncertainty or automatic MLIP retraining.
Uses existing registered structures; numerical member results are preserved.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sciowl_agent.store import Store
from sciowl_agent.science import Registry,load_atoms,calculator,finite_values


def main():
    p=argparse.ArgumentParser();p.add_argument("--workspace",required=True);p.add_argument("--registry",required=True)
    p.add_argument("--structures",required=True,help="JSON array of artifact IDs")
    p.add_argument("--potentials",nargs='+',required=True);p.add_argument("--top-k",type=int,default=10)
    p.add_argument("--output",required=True);a=p.parse_args()
    if len(set(a.potentials))<2:p.error("At least two distinct ensemble members are required")
    s=Store(a.workspace);reg=Registry(a.registry);rows=[]
    for aid in json.loads(Path(a.structures).read_text()):
        atoms=load_atoms(s,aid);forces=[];energies=[]
        for pid in a.potentials:
            cfg=reg.checked(atoms,pid)
            if cfg["backend"]=="espresso":raise ValueError("Acquisition script accepts surrogate calculators only")
            atoms.calc=calculator(cfg)
            energy,f=finite_values(atoms);forces.append(f);energies.append(energy)
        score=float(np.linalg.norm(np.std(forces,axis=0),axis=1).max())
        evidence=s.put_json({"structure_id":aid,"members":a.potentials,"energies_eV":energies,
            "forces_eV_per_A":np.array(forces).tolist()},"ensemble")
        rows.append({"structure_id":aid,"force_disagreement_eV_per_A":score,"evidence_id":evidence})
    rows.sort(key=lambda r:r["force_disagreement_eV_per_A"],reverse=True)
    result={"selected":rows[:a.top_k],"all":rows,"calibrated_uncertainty":False,
        "note":"No automatic promotion/retraining. Label selected structures and evaluate new models on independent data."}
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=="__main__":main()

