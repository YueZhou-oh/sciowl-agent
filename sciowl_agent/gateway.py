import hashlib
from pydantic import ValidationError
from .contracts import TOOLS
from .science import Registry,load_atoms,inspect,analyze_msd
from .store import canonical


class Gateway:
    def __init__(self,store,registry=None):
        self.store=store
        self.registry=registry or Registry()

    def call(self,name,arguments,session_id="cli"):
        try:
            if name not in TOOLS:
                raise ValueError("Tool not available")
            args=TOOLS[name][0].model_validate(arguments).model_dump()
            if name=="list_potentials":
                result={"potentials":{k:{x:v for x,v in cfg.items() if x not in ["model_path","command","pseudo_dir"]}
                                      for k,cfg in self.registry.entries.items()}}
            elif name=="inspect_structure":
                result=inspect(load_atoms(self.store,args["structure_id"]))
            elif name=="check_potential":
                cfg=self.registry.checked(load_atoms(self.store,args["structure_id"]),args["potential_id"])
                result={"compatible":True,"demo_only":cfg.get("demo_only",True),"domain":cfg["domain"],
                        "warning":"Compatibility is not independent scientific validation"}
            elif name in ["get_job_status","get_result","cancel_job"]:
                j=self.store.cancel(args["job_id"]) if name=="cancel_job" else self.store.job(args["job_id"])
                result={"job_id":j["id"],"status":j["status"],"error":j["error"]}
                if name=="get_result":
                    if j["status"]!="completed":
                        result["ready"]=False
                        if j["status"] in ["failed","cancelled","interrupted"]:
                            failure={"job_id":j["id"],"status":j["status"],"error":j["error"],
                                     "input_structure_id":j["spec"]["arguments"]["structure_id"],
                                     "notice":"No successful numerical result exists for this job"}
                            result["result_id"]=self.store.put_json(failure,"result")
                            result["result"]=failure
                    else:
                        result.update(ready=True,**j["result"])
                        result["result"]=self.store.get_json(j["result"]["result_id"],"result")
            elif name=="analyze_trajectory":
                data=analyze_msd(self.store,**args)
                aid=self.store.put_json(data,"analysis")
                result={"analysis_id":aid,**data}
            else:
                cfg=self.registry.checked(load_atoms(self.store,args["structure_id"]),args["potential_id"])
                spec={"operation":name,"arguments":args,"potential":cfg,"schema_version":1}
                key=hashlib.sha256((session_id+canonical(spec)).encode()).hexdigest()
                j=self.store.submit(spec,key)
                result={"job_id":j["id"],"status":j["status"],"idempotency_key":key}
            return {"ok":True,**result}
        except (ValueError,ValidationError,KeyError) as exc:
            return {"ok":False,"error":{"code":"INVALID_REQUEST","message":str(exc)[:2000]}}

