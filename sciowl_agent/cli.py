import argparse
import json
import os
import time
from pathlib import Path
from ase.build import bulk
from .store import Store,canonical
from .science import Registry,atoms_to_id,import_structure,import_structures
from .gateway import Gateway
from .model import EndpointModel,ScriptedModel
from .harness import Harness
from .worker import run_one,drain
from .dataset import record_from_session,write_records,sharegpt,tokenize_decisions


def demo(root):
    store=Store(root)
    atoms=bulk("Cu","fcc",a=3.6,cubic=True)
    atoms.positions[0,0]+=0.08
    sid=atoms_to_id(store,atoms,{"source":"perturbed-Cu-demo"})
    harness=Harness(store,Gateway(store),ScriptedModel(sid),inline_worker=True)
    state=harness.run("demo-relax",f"Inspect and relax structure {sid} with the classical EMT demo potential.")
    if state["status"]!="completed":
        raise RuntimeError("Demo did not complete: "+state["status"])
    record=record_from_session(store,"demo-relax",reviewed=False)
    write_records([record],store.root/"trajectory.jsonl")
    report=store.get_json(state["report_id"],"report")
    (store.root/"report.json").write_text(json.dumps(report,indent=2))
    return {"structure_id":sid,"session_status":state["status"],"report_id":state["report_id"],"report":report}


def main():
    p=argparse.ArgumentParser(description="SciOWL scientific harness; use --help on subcommands")
    p.add_argument("--workspace",default="sciowl-workspace")
    p.add_argument("--registry",help="Administrator-controlled potential registry JSON")
    sub=p.add_subparsers(dest="cmd",required=True)
    sub.add_parser("demo",help="Real EMT calculation with a scripted baseline; no SciOWL required")
    imp=sub.add_parser("import");imp.add_argument("path")
    imp.add_argument("--all", action="store_true", help="Import all XYZ/extXYZ/CIF frames (up to 1000); default is the first frame")
    pred=sub.add_parser("export-predictions", help="Export a completed single-point/batch job to JSONL")
    pred.add_argument("--job", required=True);pred.add_argument("--output", required=True)
    tool=sub.add_parser("tool");tool.add_argument("name");tool.add_argument("arguments",help="JSON object");tool.add_argument("--session",default="cli")
    worker=sub.add_parser("worker");worker.add_argument("--loop",action="store_true");worker.add_argument("--timeout",type=float,default=600)
    worker.add_argument("--progress", action="store_true", help="Print job progress to stderr")
    rec=sub.add_parser("recover-stale");rec.add_argument("--seconds",type=float,default=3600)
    agent=sub.add_parser("agent");agent.add_argument("--session",required=True);agent.add_argument("--request")
    agent.add_argument("--base-url",default=os.environ.get("SCIOWL_BASE_URL","http://10.1.0.232:8005/v1"))
    agent.add_argument("--model",default=os.environ.get("SCIOWL_MODEL","sciowl"))
    agent.add_argument("--protocol",choices=["native","json"],default="native")
    agent.add_argument("--max-steps",type=int,default=20);agent.add_argument("--max-jobs",type=int,default=8)
    agent.add_argument("--follow",action="store_true",help="Resume automatically after worker completion")
    agent.add_argument("--wall-seconds",type=float,default=600)
    exp=sub.add_parser("export");exp.add_argument("--session",required=True);exp.add_argument("--output",required=True)
    exp.add_argument("--reviewed",action="store_true",help="Assert human/expert review was actually completed")
    exp.add_argument("--format",choices=["canonical","sharegpt"],default="canonical")
    tok=sub.add_parser("tokenize");tok.add_argument("--input",required=True);tok.add_argument("--output",required=True)
    tok.add_argument("--tokenizer",required=True);tok.add_argument("--max-length",type=int,default=8192)
    args=p.parse_args()
    store=Store(args.workspace)
    if args.cmd=="demo":
        print(json.dumps(demo(args.workspace),indent=2));return
    if args.cmd=="import":
        if args.all:
            ids=import_structures(store,args.path)
            out={"structure_ids":ids,"frames":len(ids)}
        else:
            out={"structure_id":import_structure(store,args.path)}
    elif args.cmd=="export-predictions":
        job=store.job(args.job)
        if job["status"] != "completed":
            raise ValueError("Predictions can only be exported from a completed job")
        result=store.get_json(job["result"]["result_id"], "result")
        if "predictions_id" not in result:
            raise ValueError("Job has no predictions JSONL artifact")
        source=store.artifact(result["predictions_id"], "predictions")
        target=Path(args.output).resolve()
        if target.is_relative_to(store.root):
            raise ValueError("Export outside the managed workspace to preserve job artifacts")
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("xb") as output:
            output.write(source.read_bytes())
        out={"output":str(target),"predictions_id":result["predictions_id"]}
    elif args.cmd=="tool":
        out=Gateway(store,Registry(args.registry)).call(args.name,json.loads(args.arguments),args.session)
    elif args.cmd=="worker":
        while True:
            worked=run_one(store,args.timeout,show_progress=args.progress)
            if not args.loop:break
            if not worked:time.sleep(0.5)
        out={"last_iteration_worked":worked}
    elif args.cmd=="recover-stale":
        if args.seconds<1:p.error("seconds must be positive")
        out={"marked_interrupted":store.recover_stale(args.seconds)}
    elif args.cmd=="agent":
        if args.max_steps<1 or args.max_jobs<1:p.error("budgets must be positive")
        model=EndpointModel(args.base_url,args.model,args.protocol)
        harness=Harness(store,Gateway(store,Registry(args.registry)),model,args.max_steps,args.max_jobs)
        deadline=time.monotonic()+args.wall_seconds
        out=harness.run(args.session,args.request)
        while args.follow and out["status"] in ["thinking","waiting"] and time.monotonic()<deadline:
            time.sleep(0.5)
            out=harness.run(args.session)
    elif args.cmd=="export":
        record=record_from_session(store,args.session,args.reviewed)
        if args.format=="canonical":write_records([record],args.output)
        else:Path(args.output).write_text(canonical(sharegpt(record))+"\n")
        out={"output":args.output,"reviewed":args.reviewed}
    else:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,local_files_only=True,trust_remote_code=False)
        count=0
        with open(args.input) as source,open(args.output,'w') as target:
            for line in source:
                record=json.loads(line)
                if not record.get("metadata",{}).get("reviewed"):
                    raise ValueError("Training requires reviewed records; unreviewed demos are not training-ready")
                for sample in tokenize_decisions(record,tokenizer,args.max_length):
                    target.write(canonical(sample)+"\n");count+=1
        out={"tokenized_decisions":count}
    print(json.dumps(out,indent=2))


if __name__=="__main__":main()

