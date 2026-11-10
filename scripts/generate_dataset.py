#!/usr/bin/env python3
"""Execute expert-scripted workflows; does not invent tool observations or CoT.

Generates a seed corpus, not the proposed diverse 10k production dataset.
All exports are marked unreviewed. Structures are grouped before augmentation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from ase.build import bulk
from sciowl_agent.store import Store
from sciowl_agent.science import atoms_to_id
from sciowl_agent.gateway import Gateway
from sciowl_agent.harness import Harness
from sciowl_agent.dataset import record_from_session,write_records,split_records


class Expert:
    def __init__(self,aid,operation):self.aid,self.operation=aid,operation
    def next(self,messages,tools):
        names=[c["function"]["name"] for m in messages for c in m.get("tool_calls",[])]
        obs=[json.loads(m["content"]) for m in messages if m["role"]=="tool"]
        result=next((o for o in obs if o.get("result_id")),None)
        if "inspect_structure" not in names:name,args="inspect_structure",{"structure_id":self.aid}
        elif "check_potential" not in names:name,args="check_potential",{"structure_id":self.aid,"potential_id":"emt_demo"}
        elif self.operation not in names:
            name,args=self.operation,{"structure_id":self.aid,"potential_id":"emt_demo"}
            if name=="submit_md":args.update(production_steps=100,equilibration_steps=10,timestep_fs=0.2,temperature_K=50.)
        elif result is None:
            name,args="get_result",{"job_id":next(o["job_id"] for o in obs if o.get("job_id"))}
        elif self.operation=="submit_md" and "analyze_trajectory" not in names:
            name,args="analyze_trajectory",{"trajectory_id":result["result"]["trajectory_id"],"element":"Cu"}
        else:
            evidence=[o[k] for o in obs for k in ["result_id","analysis_id"] if o.get(k)]
            return {"role":"assistant","content":json.dumps({"summary":"Classical EMT demonstration finished. Numerical outputs and validation limits are recorded in the evidence.","evidence_ids":evidence})}
        return {"role":"assistant","content":None,"tool_calls":[{"id":"expert","type":"function","function":{"name":name,"arguments":json.dumps(args)}}]}


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="generated_dataset")
    p.add_argument("--structures",type=int,default=5);args=p.parse_args()
    if not 1<=args.structures<=100:p.error("seed generation supports 1..100 structures")
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    store=Store(out/"workspace");records=[]
    for i in range(args.structures):
        atoms=bulk("Cu","fcc",a=3.58+i*0.002,cubic=True)
        atoms.positions[0,0]+=0.02+(i%5)*0.01
        aid=atoms_to_id(store,atoms,{"source_family":"Cu-fcc-seed"})
        for operation in ["evaluate_energy_forces","submit_relaxation","submit_md"]:
            sid=f"seed-{i}-{operation}"
            request=f"Inspect registered Cu structure {aid}, check the EMT demo potential, then execute {operation}. Treat the outputs as a classical workflow demonstration."
            s=Harness(store,Gateway(store),Expert(aid,operation),inline_worker=True).run(sid,request)
            if s["status"]!="completed":raise RuntimeError(s)
            r=record_from_session(store,sid)
            # These are related Cu examples; keep the ENTIRE family together.
            r["metadata"].update(source_group="Cu-fcc-seed-family",policy="scripted_expert",task_family=operation)
            records.append(r)
    write_records(records,out/"canonical.jsonl")
    splits=split_records(records)
    for name,rs in splits.items():write_records(rs,out/(name+".jsonl"))
    manifest={"records":len(records),"reviewed":False,"observation_source":"real_execution",
        "policy":"scripted expert; not SciOWL","calculator":"ASE EMT classical demo",
        "splits":{k:len(v) for k,v in splits.items()},
        "warning":"Single related Cu family: grouped split may be empty. Add independent source families for useful validation/test sets.",
        "sha256":hashlib.sha256((out/"canonical.jsonl").read_bytes()).hexdigest()}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))


if __name__=="__main__":main()

