"""Queue worker; cooperative cancellation and time limits between calculator steps."""
import importlib.metadata
import hashlib
import platform
import time
import traceback
from pathlib import Path
from .science import execute
from .store import canonical


def run_one(store, timeout_s=600.):
    job=store.claim()
    if job is None:
        return False
    started=time.monotonic()
    def tick():
        if time.monotonic()-started>timeout_s:
            raise TimeoutError("Calculation exceeded cooperative time budget")
        store.heartbeat(job["id"],job["owner"])
    try:
        result=execute(store,job,tick)
        provenance={"job_id":job["id"],"spec":job["spec"],"python":platform.python_version(),
            "platform":platform.platform(),"elapsed_s":time.monotonic()-started,
            "implementation_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path(__file__).parent.glob('*.py')},
            "packages":{p:importlib.metadata.version(p) for p in ["ase","numpy","scipy","pydantic"]}}
        result["provenance_id"]=store.put_json(provenance,"provenance")
        rid=store.put_json(result,"result",{"job_id":job["id"]})
        store.finish(job["id"],job["owner"],"completed",{"result_id":rid})
    except Exception as exc:
        # Detailed traceback stays in trusted local work directory, outside model observations.
        (store.root/"work"/job["id"]).mkdir(exist_ok=True)
        (store.root/"work"/job["id"]/"error.log").write_text(traceback.format_exc())
        status="cancelled" if isinstance(exc,InterruptedError) else "failed"
        store.finish(job["id"],job["owner"],status,error={"code":type(exc).__name__,"message":str(exc)[:1000]})
    return True


def drain(store):
    count=0
    while run_one(store):
        count+=1
    return count

