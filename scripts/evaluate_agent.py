#!/usr/bin/env python3
"""Evaluate scripted or endpoint policy on a SMALL generated Cu demo suite.

Not the proposed independent 200-task scientific benchmark. Uses real tools.
"""
import argparse
import json
import time
from pathlib import Path
from ase.build import bulk
from sciowl_agent.store import Store
from sciowl_agent.science import atoms_to_id
from sciowl_agent.gateway import Gateway
from sciowl_agent.harness import Harness
from sciowl_agent.model import EndpointModel,ScriptedModel


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",default="evaluation_run")
    p.add_argument("--base-url");p.add_argument("--model",default="SciOWL-Instruct")
    p.add_argument("--protocol",choices=["native","json"],default="native")
    p.add_argument("--cases",type=int,default=3);a=p.parse_args()
    if not 1<=a.cases<=20:p.error("cases must be 1..20")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=[]
    for i in range(a.cases):
        s=Store(out/f"case_{i}");atoms=bulk("Cu","fcc",a=3.59+0.01*i,cubic=True)
        atoms.positions[0,0]+=0.04+0.01*i
        aid=atoms_to_id(s,atoms)
        model=EndpointModel(a.base_url,a.model,a.protocol) if a.base_url else ScriptedModel(aid)
        h=Harness(s,Gateway(s),model,inline_worker=True,max_steps=20,max_jobs=3)
        started=time.monotonic()
        request=f"Inspect structure {aid}, check emt_demo compatibility, relax at fixed cell with force tolerance 0.03 eV/Angstrom, then retrieve the result. Report its actual convergence and evidence."
        error=None
        try:
            state=h.run('evaluation',request)
            while state['status']=='thinking':state=h.run('evaluation')
            success=False;calculation=None
            if state['report_id']:
                report=s.get_json(state['report_id'])
                calculation=next((e['data'] for e in report['evidence'] if e['data'].get('operation')=='submit_relaxation'),None)
                success=bool(calculation and calculation['converged'] and calculation['max_force_eV_per_A']<0.03)
            observations=[json.loads(m['content']) for m in state['messages'] if m['role']=='tool']
            invalid=sum(not o.get('ok',False) for o in observations)
            rows.append({'case':i,'status':state['status'],'validated_success':success,'tool_calls':len(observations),
                         'invalid_requests':invalid,'model_steps':state['steps'],'wall_s':time.monotonic()-started,
                         'calculation':calculation,'report_id':state['report_id']})
        except Exception as exc:
            rows.append({'case':i,'validated_success':False,'error':str(exc)})
    result={'policy':a.model if a.base_url else 'scripted expert baseline','backend':'CPU ASE EMT classical demo',
        'cases':rows,'success_rate':sum(r['validated_success'] for r in rows)/len(rows),
        'scope':'Small related Cu relaxation suite; not independent evidence of broad scientific performance'}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

