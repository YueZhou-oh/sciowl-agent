#!/usr/bin/env python3
"""Run numerical checks and the scripted harness; produces machine-readable evidence."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
import numpy as np
from ase.build import bulk
from sciowl_agent.cli import demo
from sciowl_agent.store import Store
from sciowl_agent.science import atoms_to_id
from sciowl_agent.gateway import Gateway
from sciowl_agent.worker import drain


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="validation/local");a=p.parse_args()
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    proc=subprocess.run([sys.executable,"-m","pytest","-q","--junitxml="+str(out/"pytest.xml")],text=True,capture_output=True)
    (out/"pytest.txt").write_text(proc.stdout+proc.stderr)
    if proc.returncode:
        print(proc.stdout+proc.stderr);raise SystemExit(proc.returncode)
    d=demo(out/"workspace")
    s=Store(out/"workspace");g=Gateway(s)
    sid=atoms_to_id(s,bulk("Cu","fcc",a=3.6,cubic=True))
    numerical={"relaxation":d["report"]["evidence"][0]["data"]}
    for name,kwargs in [("submit_eos",{}),("submit_md",{"production_steps":100,"equilibration_steps":10,
                "timestep_fs":0.2,"temperature_K":50.,"sample_interval":5})]:
        j=g.call(name,{"structure_id":sid,**kwargs},"validation")
        drain(s)
        r=g.call("get_result",{"job_id":j["job_id"]})
        if not r.get("ready"):raise RuntimeError(r)
        numerical[name]=r["result"]
    md=numerical["submit_md"]
    numerical["msd"]=g.call("analyze_trajectory",{"trajectory_id":md["trajectory_id"],"element":"Cu"})
    report={"status":"passed","policy":"Scripted baseline, NOT SciOWL",
        "hardware":"CPU","calculator":"ASE EMT classical demonstration, NOT an MLIP",
        "not_validated":["Ascend hardware","SciOWL checkpoint","MACE weights","Quantum ESPRESSO executable",
                         "Slurm cluster","MindSpeed training","physical diffusion convergence"],
        "python":platform.python_version(),"versions":{k:importlib.metadata.version(k) for k in
                   ["ase","numpy","scipy","pydantic","httpx","langgraph","pytest"]},"numerical":numerical}
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({"status":"passed","report":str(out/"report.json"),"tests":str(out/"pytest.txt")},indent=2))


if __name__=="__main__":main()

