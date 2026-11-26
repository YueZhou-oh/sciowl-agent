"""LangGraph loop with SQLite checkpoints and recoverable pending calls."""
import json
import time
import hashlib
import fcntl
import httpx
from typing import TypedDict
from langgraph.graph import StateGraph,START,END
from .model import SYSTEM, ContextLimitError
from .discovery import selected_schemas, schema_size, SCHEMA_BYTES
from .store import canonical
from .reliability import CALCULATIONS, assess_job


class State(TypedDict):
    session_id: str
    messages: list
    steps: int
    pending: list
    jobs: list
    evidence: list
    status: str
    report_id: str | None
    constraints: dict
    last_error: dict | None


class Harness:
    def __init__(self,store,gateway,model,max_steps=20,max_jobs=8,max_context_chars=60000,inline_worker=False,constraints=None):
        self.store,self.gateway,self.model=store,gateway,model
        self.max_steps,self.max_jobs=max_steps,max_jobs
        self.max_context_chars=max_context_chars
        self.inline_worker=inline_worker
        self.constraints=constraints
        if constraints is not None:
            if set(constraints) - {"potential_id", "allowed_operations"}:
                raise ValueError("Unknown session constraint")
            if "potential_id" in constraints and not isinstance(constraints["potential_id"], str):
                raise ValueError("potential_id constraint must be a string")
            if "allowed_operations" in constraints and (not isinstance(constraints["allowed_operations"], list) or
                    any(op not in CALCULATIONS for op in constraints["allowed_operations"])):
                raise ValueError("Invalid allowed_operations constraint")
        graph=StateGraph(State)
        graph.add_node("model",self.think)
        graph.add_node("tools",self.act)
        graph.add_edge(START,"model")
        graph.add_conditional_edges("model",lambda s:"tools" if s["status"]=="acting" else END)
        graph.add_conditional_edges("tools",lambda s:"model" if s["status"]=="thinking" else END)
        self.graph=graph.compile()

    def persist(self,state):
        self.store.save_session(state["session_id"],state)
        return state

    def think(self,s):
        if s["pending"]:
            s["status"]="acting"
            return s
        if s["steps"]>=self.max_steps:
            s["status"]="budget_exhausted"
            return self.persist(s)
        # Fail closed instead of silently truncating prerequisite observations.
        if len(canonical(s["messages"]))>self.max_context_chars:
            s["status"]="context_limit"
            return self.persist(s)
        try:
            schemas=selected_schemas(s["messages"])
            if schema_size(schemas)>SCHEMA_BYTES:
                raise ContextLimitError("Selected tool schemas exceed 24000 bytes")
            self.store.event(s["session_id"], {"model_context": {
                "tools": [t["function"]["name"] for t in schemas],
                "schema_bytes": schema_size(schemas),
                "message_chars": len(canonical(s["messages"]))}})
            msg=self.model.next(s["messages"],schemas)
            s["last_error"]=None
            if msg.get("role")!="assistant":
                raise ValueError("Expected assistant message")
            calls=msg.get("tool_calls",[])
            if len(calls)>4:
                raise ValueError("At most four calls per decision")
            seen=set()
            for i,c in enumerate(calls):
                if c.get("type")!="function" or not isinstance(c["function"]["name"],str):
                    raise ValueError("Invalid tool-call envelope")
                if c["function"]["name"] not in {t["function"]["name"] for t in schemas}:
                    raise ValueError("Tool not loaded; use search_tools to discover the required operation")
                if c.get("id") in seen:
                    raise ValueError("Duplicate tool-call ID")
                seen.add(c.get("id"))
                # IDs need only correlate within a turn; normalize to avoid collisions on resume.
                c["id"]=f"turn_{s['steps']}_call_{i}"
            if calls:
                s["pending"]=calls.copy();s["status"]="acting"
            else:
                self.finalize(s,msg)
            s["messages"].append(msg)
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            retryable = status is None or status in (408, 429) or status >= 500
            # Never store raw provider errors/URLs: they may contain secrets.
            s["last_error"]={"code":type(exc).__name__, "http_status":status,
                             "retryable":retryable, "message":"Endpoint request failed. Inspect server logs and explicitly resume this session after correction."}
            s["status"]="endpoint_error"
            s["steps"]+=1
            self.store.event(s["session_id"], {"endpoint_error":s["last_error"]})
            return self.persist(s)
        except ContextLimitError as exc:
            s["status"]="context_limit"
            self.store.event(s["session_id"], {"context_limit":str(exc)})
            return self.persist(s)
        except (ValueError,KeyError,TypeError) as exc:
            s["messages"].append({"role":"user","content":"Protocol validation failed: "+str(exc)[:500]+". Return a valid action or final JSON."})
            s["status"]="thinking"
            # Return to caller on malformed model response; resume is bounded by steps.
        s["steps"]+=1
        return self.persist(s)

    def act(self,s):
        while s["pending"]:
            call=s["pending"][0]
            try:
                args=json.loads(call["function"]["arguments"])
                name=call["function"]["name"]
                if name in ["evaluate_energy_forces","evaluate_structures","submit_relaxation","submit_eos","submit_md"] and len(s["jobs"])>=self.max_jobs:
                    raise ValueError("Session calculation budget exhausted")
                out=self.gateway.call(name,args,s["session_id"])
            except (ValueError,TypeError,KeyError) as exc:
                out={"ok":False,"error":{"code":"INVALID_ACTION","message":str(exc)[:500]}}
            if out.get("job_id") and out["job_id"] not in s["jobs"]:
                s["jobs"].append(out["job_id"])
            for field in ["result_id","analysis_id"]:
                if out.get(field) and out[field] not in s["evidence"]:
                    s["evidence"].append(out[field])
            s["messages"].append({"role":"tool","tool_call_id":call["id"],"content":canonical(out)})
            self.store.event(s["session_id"],{"tool":call["function"]["name"],"observation":out})
            s["pending"].pop(0)
            # Commit each observation. A crash before this commit replays an idempotent action.
            self.persist(s)
        if self.inline_worker:
            from .worker import drain
            drain(self.store)
        active=any(self.store.job(j)["status"] in ["queued","running"] for j in s["jobs"])
        s["status"]="waiting" if active else "thinking"
        return self.persist(s)

    def finalize(self,s,msg):
        obj=json.loads(msg.get("content") or "")
        if set(obj)!={"summary","evidence_ids"} or not isinstance(obj["summary"],str) or not isinstance(obj["evidence_ids"],list):
            raise ValueError("Final JSON requires summary and evidence_ids")
        if any(not isinstance(x,str) or x not in s["evidence"] for x in obj["evidence_ids"]):
            raise ValueError("Evidence ID was not observed in this session")
        if s["jobs"] and not obj["evidence_ids"]:
            raise ValueError("Calculation report requires observed evidence")
        if any(self.store.job(j)["status"] in ["queued","running"] for j in s["jobs"]):
            raise ValueError("Cannot finalize with pending jobs")
        # Commentary is unverified; authoritative values remain in evidence artifacts.
        facts=[{"evidence_id":eid,"data":self.store.get_json(eid)} for eid in obj["evidence_ids"]]
        assessments=[assess_job(self.store,self.store.job(j),obj["evidence_ids"]) for j in s["jobs"]]
        report={"session_id":s["session_id"],"job_assessments":assessments,"model_commentary":obj["summary"],
                "commentary_verified":False,"evidence":facts,
                "notice":"Numerical evidence is tool-generated. Model commentary requires scientific review."}
        s["report_id"]=self.store.put_json(report,"report")
        s["status"]="completed"

    def run(self,sid,request=None):
        lock_path=self.store.root/("session_"+hashlib.sha256(sid.encode()).hexdigest()+".lock")
        with lock_path.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            return self._run_locked(sid,request)

    def _run_locked(self,sid,request=None):
        s=self.store.load_session(sid)
        if s is None:
            if not request:
                raise ValueError("New session requires a request")
            s={"session_id":sid,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":request}],
               "steps":0,"pending":[],"jobs":[],"evidence":[],"status":"thinking","report_id":None,
               "constraints":self.constraints or {},"last_error":None}
            if s["constraints"]:
                s["messages"][0]["content"] += "\nEnforced session constraints: " + canonical(s["constraints"])
            self.persist(s)
        elif self.constraints is not None and self.constraints != s.get("constraints", {}):
            raise ValueError("Cannot change constraints of an existing session")
        elif s["status"] in ["completed","budget_exhausted","context_limit"]:
            return s
        elif any(self.store.job(j)["status"] in ["queued","running"] for j in s["jobs"]):
            s["status"]="waiting"
            return self.persist(s)
        s["status"]="thinking"
        return self.graph.invoke(s,config={"recursion_limit":max(50,4*self.max_steps)})

