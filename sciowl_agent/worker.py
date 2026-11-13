"""Queue worker; cooperative cancellation and time limits between calculator steps."""
import importlib.metadata
import hashlib
import platform
import sys
import time
import traceback
from contextlib import suppress
from pathlib import Path
from .science import execute
from .store import canonical


def run_one(store, timeout_s=600., *, show_progress=False):
    job=store.claim()
    if job is None:
        return False
    started=time.monotonic()
    job_id = job["id"]
    operation = job["spec"]["operation"]

    def progress(message):
        if show_progress:
            elapsed = time.monotonic() - started
            with suppress(OSError):
                print(f"[{job_id} {operation} +{elapsed:.2f}s] {message}", file=sys.stderr, flush=True)

    def tick():
        if time.monotonic()-started>timeout_s:
            raise TimeoutError("Calculation exceeded cooperative time budget")
        store.heartbeat(job["id"],job["owner"])
    try:
        args = job["spec"]["arguments"]
        inputs = args.get("structure_id") or f"{len(args['structure_ids'])} frames"
        progress(f"started structure={inputs} potential={args['potential_id']}")
        result=execute(store,job,tick,progress=progress if show_progress else None)
        provenance={"job_id":job["id"],"spec":job["spec"],"python":platform.python_version(),
            "platform":platform.platform(),"elapsed_s":time.monotonic()-started,
            "implementation_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob('*.py')},
            "packages":{p:importlib.metadata.version(p) for p in ["ase","numpy","scipy","pydantic"]}}
        if job["spec"]["potential"]["backend"] == "mace":
            for package in ("mace-torch", "torch", "e3nn"):
                with suppress(importlib.metadata.PackageNotFoundError):
                    provenance["packages"][package] = importlib.metadata.version(package)
        result["provenance_id"]=store.put_json(provenance,"provenance")
        rid=store.put_json(result,"result",{"job_id":job["id"]})
        store.finish(job["id"],job["owner"],"completed",{"result_id":rid})
        outcome = f"completed result_id={rid}"
        if operation == "submit_relaxation":
            outcome += f" converged={result['converged']} steps={result['steps']}"
        progress(outcome)
    except Exception as exc:
        # Detailed traceback stays in trusted local work directory, outside model observations.
        (store.root/"work"/job["id"]).mkdir(exist_ok=True)
        (store.root/"work"/job["id"]/"error.log").write_text(traceback.format_exc())
        status="cancelled" if isinstance(exc,InterruptedError) else "failed"
        store.finish(job["id"],job["owner"],status,error={"code":type(exc).__name__,"message":str(exc)[:1000]})
        progress(f"{status} error={type(exc).__name__}: {str(exc)[:1000]!r}")
    return True


def drain(store):
    count=0
    while run_one(store):
        count+=1
    return count

