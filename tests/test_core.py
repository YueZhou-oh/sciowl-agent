import concurrent.futures
import json
import sqlite3
import time
import numpy as np
import pytest
from ase.build import bulk
from sciowl_agent.store import Store
from sciowl_agent.science import atoms_to_id,load_atoms,calculator,DEFAULT_REGISTRY
from sciowl_agent.gateway import Gateway
from sciowl_agent.worker import drain,run_one
from sciowl_agent.harness import Harness
from sciowl_agent.model import ScriptedModel,EndpointModel
from sciowl_agent.dataset import record_from_session,validate_record,split_records,tokenize_record


@pytest.fixture
def setup(tmp_path):
    s=Store(tmp_path)
    a=bulk("Cu","fcc",a=3.6,cubic=True)
    a.positions[0,0]+=0.08
    aid=atoms_to_id(s,a)
    return s,Gateway(s),aid


def calculate(setup,name,**kw):
    s,g,aid=setup
    out=g.call(name,{"structure_id":aid,**kw})
    assert out["ok"],out
    drain(s)
    res=g.call("get_result",{"job_id":out["job_id"]})
    assert res.get("ready"),res
    return res["result"]


def test_relax_real(setup):
    r=calculate(setup,"submit_relaxation")
    assert r["converged"]
    assert r["max_force_eV_per_A"]<0.03
    assert r["demo_only"]


def test_singlepoint_force_finite_difference(setup):
    s,g,aid=setup
    r=calculate(setup,"evaluate_energy_forces")
    f=s.get_json(r["forces_id"])["forces_eV_per_A"][0][0]
    a=load_atoms(s,aid);a.calc=calculator(DEFAULT_REGISTRY["emt_demo"])
    x=a.positions[0,0];h=1e-5
    a.positions[0,0]=x+h;ep=a.get_potential_energy()
    a.positions[0,0]=x-h;em=a.get_potential_energy()
    assert abs(f+(ep-em)/(2*h))<1e-6


def test_eos_real(setup):
    s,g,aid=setup
    a=bulk("Cu","fcc",a=3.6,cubic=True)
    aid=atoms_to_id(s,a)
    r=calculate((s,g,aid),"submit_eos")
    assert r["validated"]
    assert 50<r["bulk_modulus_GPa"]<250
    assert r["fit_rmse_eV"]<0.01


def test_md_real_and_provisional_msd(setup):
    s,g,aid=setup
    r=calculate(setup,"submit_md",production_steps=100,equilibration_steps=10,sample_interval=5,temperature_K=50.,timestep_fs=0.2)
    assert r["frames"]==21
    assert r["energy_range_eV_per_atom"]<0.001
    out=g.call("analyze_trajectory",{"trajectory_id":r["trajectory_id"],"element":"Cu"})
    assert out["ok"] and out["validated_diffusion"] is False
    assert len(out["msd_A2"])==10


@pytest.mark.parametrize("arguments",[
    {"temperature_K":-1.}, {"production_steps":10001}, {"seed":"1"},
    {"timestep_fs":float('nan')}, {"shell":"rm -rf /"}, {"production_steps":True}
])
def test_reject_bad_schema(setup,arguments):
    s,g,aid=setup
    assert not g.call("submit_md",{"structure_id":aid,**arguments})["ok"]


def test_unknown_tool_and_path_escape(setup):
    s,g,aid=setup
    assert not g.call("run_shell",{"command":"anything"})["ok"]
    assert not g.call("inspect_structure",{"structure_id":"../../etc/passwd"})["ok"]


def test_unsupported_elements(setup):
    s,g,_=setup
    aid=atoms_to_id(s,bulk("Fe","bcc",a=2.8))
    assert not g.call("submit_relaxation",{"structure_id":aid})["ok"]


def test_concurrent_idempotency(setup):
    s,g,aid=setup
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(lambda _:g.call("submit_relaxation",{"structure_id":aid}),range(12)))
    assert len({r["job_id"] for r in results})==1
    with s.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]==1


def test_claim_exclusive_and_stale(setup):
    s,g,aid=setup
    g.call("submit_relaxation",{"structure_id":aid})
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        claimed=list(pool.map(lambda _:s.claim(),range(3)))
    assert sum(j is not None for j in claimed)==1
    job=next(j for j in claimed if j)
    with s.connect() as db:db.execute("UPDATE jobs SET heartbeat=?",(time.time()-100,))
    assert s.recover_stale(10)==1
    assert s.job(job["id"])["status"]=="interrupted"
    assert s.claim() is None
    with pytest.raises(RuntimeError):s.finish(job["id"],job["owner"],"completed",{})


def test_cancel_queued(setup):
    s,g,aid=setup
    j=g.call("submit_relaxation",{"structure_id":aid})["job_id"]
    assert s.cancel(j)["status"]=="cancelled"
    assert not run_one(s)


def test_completed_unconverged_not_success(setup):
    r=calculate(setup,"submit_relaxation",max_steps=1,fmax_eV_per_A=1e-8)
    assert r["converged"] is False


def test_integrity_detection(setup):
    s,g,aid=setup
    path=s.artifact(aid)
    path.write_text('{}')
    with pytest.raises(ValueError,match="integrity"):s.artifact(aid)


def test_harness_resume_and_export(setup):
    s,g,aid=setup
    h=Harness(s,g,ScriptedModel(aid))
    state=h.run("resume","Relax the registered Cu structure")
    assert state["status"]=="waiting"
    assert Harness(s,g,ScriptedModel(aid)).run("resume")["status"]=="waiting"
    drain(s)
    state=Harness(s,g,ScriptedModel(aid)).run("resume")
    assert state["status"]=="completed"
    r=record_from_session(s,"resume")
    validate_record(r)
    assert r["metadata"]["observation_source"]=="real_execution"
    assert not r["metadata"]["reviewed"]
    assert len(state["jobs"])==1


def test_pending_action_replay_no_duplicate(setup):
    s,g,aid=setup
    h=Harness(s,g,ScriptedModel(aid),inline_worker=True)
    state=h.run("replay","Relax")
    submit=next(m for m in state["messages"] if any(c["function"]["name"]=="submit_relaxation" for c in m.get("tool_calls",[])))
    state["pending"]=submit["tool_calls"].copy()
    state["status"]="acting"
    h.act(state)
    with s.connect() as db:assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]==1


def test_invented_evidence_rejected(setup):
    s,g,aid=setup
    class Bad:
        def next(self,messages,tools):
            return {"role":"assistant","content":json.dumps({"summary":"Done","evidence_ids":["fake"]})}
    state=Harness(s,g,Bad(),max_steps=2).run("bad","Relax")
    assert state["status"]!="completed"
    assert state["report_id"] is None


def test_budget_enforced(setup):
    s,g,aid=setup
    class Loop:
        def next(self,messages,tools):
            return {"role":"assistant","content":None,"tool_calls":[{"id":"c","type":"function","function":{"name":"list_potentials","arguments":"{}"}}]}
    state=Harness(s,g,Loop(),max_steps=3).run("budget","Loop")
    assert state["status"]=="budget_exhausted" and state["steps"]==3


def test_http_adapter_native_and_truncation():
    import httpx
    def handle(req):
        body=json.loads(req.content)
        assert body["tools"]==[] and body["model"]=="SciOWL"
        return httpx.Response(200,json={"choices":[{"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}]})
    model=EndpointModel("http://test/v1","SciOWL",client=httpx.Client(transport=httpx.MockTransport(handle)))
    assert model.next([],[])["content"]=="ok"
    model.client=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json={"choices":[{"finish_reason":"length"}]})))
    with pytest.raises(ValueError,match="truncated"):model.next([],[])


def test_http_json_protocol():
    import httpx
    response={"choices":[{"finish_reason":"stop","message":{"content":'{"tool":"list_potentials","arguments":{}}'}}]}
    model=EndpointModel("http://test/v1","SciOWL",protocol="json",client=httpx.Client(transport=httpx.MockTransport(lambda r:httpx.Response(200,json=response))))
    assert model.next([],[])["tool_calls"][0]["function"]["name"]=="list_potentials"


def test_group_split():
    rs=[{"id":str(i),"metadata":{"source_group":"same-structure"}} for i in range(4)]
    split=split_records(rs)
    assert sorted(map(len,split.values()))==[0,0,4]


def test_orphan_observation_rejected():
    with pytest.raises(ValueError,match="Orphan"):
        validate_record({"tools":[],"messages":[{"role":"tool","tool_call_id":"missing","content":"{}"}]})

