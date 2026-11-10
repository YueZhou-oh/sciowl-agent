import json
import os
import hashlib
import threading
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import numpy as np
import pytest
from ase.build import bulk
from sciowl_agent.store import Store
from sciowl_agent.science import Registry,atoms_to_id,calculator,analyze_msd
from sciowl_agent.gateway import Gateway
from sciowl_agent.worker import run_one,drain
from sciowl_agent.model import EndpointModel,ScriptedModel
from sciowl_agent.harness import Harness
from sciowl_agent.dataset import record_from_session,tokenize_decisions,sharegpt


def test_real_http_loop(tmp_path):
    s=Store(tmp_path)
    aid=atoms_to_id(s,bulk("Cu","fcc",a=3.6,cubic=True))
    expert=ScriptedModel(aid)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            data=json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            assert self.path=="/v1/chat/completions"
            msg=expert.next(data["messages"],data["tools"])
            payload=json.dumps({"choices":[{"finish_reason":"stop","message":msg}]}).encode()
            self.send_response(200);self.send_header("Content-Type","application/json");self.end_headers();self.wfile.write(payload)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        endpoint=EndpointModel(f"http://127.0.0.1:{server.server_port}/v1","scripted-transport-test")
        state=Harness(s,Gateway(s),endpoint,inline_worker=True).run("http","Relax the structure")
        assert state["status"]=="completed"
    finally:
        server.shutdown();server.server_close();thread.join()


def test_running_cancellation_and_timeout(tmp_path):
    s=Store(tmp_path);g=Gateway(s);aid=atoms_to_id(s,bulk("Cu","fcc",a=3.6,cubic=True))
    j=g.call("submit_md",{"structure_id":aid})["job_id"]
    job=s.claim();s.cancel(j)
    with pytest.raises(InterruptedError):s.heartbeat(j,job["owner"])
    s.finish(j,job["owner"],"cancelled")
    j=g.call("submit_md",{"structure_id":aid},session_id="timeout")["job_id"]
    assert run_one(s,timeout_s=0.)
    assert s.job(j)["status"]=="failed"
    out=g.call("get_result",{"job_id":j})
    assert out["ready"] is False and out["result"]["status"]=="failed"


def test_bad_qe_asset_rejected(tmp_path):
    registry=Registry();registry.entries={"qe":{"backend":"espresso","elements":["Cu"],"pseudo_dir":str(tmp_path),
        "pseudopotential_sha256":{"missing.UPF":"fake"},"pseudopotentials":{"Cu":"missing.UPF"}}}
    with pytest.raises(ValueError,match="Pseudopotential"):
        registry.checked(bulk("Cu"),"qe")


def test_qe_adapter_construction(tmp_path):
    cfg={"backend":"espresso","command":"pw.x","pseudo_dir":str(tmp_path),
         "pseudopotentials":{"Cu":"Cu.UPF"},"kpts":[2,2,2],"input_data":{"system":{"ecutwfc":60}}}
    calc=calculator(cfg,tmp_path/"qe")
    assert str(calc.directory)==str(tmp_path/"qe")
    # Construction only; no fake executable or numerical DFT result.


def test_mace_npu_not_silently_claimed(tmp_path):
    model=tmp_path/"model.model";model.write_bytes(b'not-loaded')
    reg=Registry();reg.entries={"m":{"backend":"mace","elements":["Cu"],"model_path":str(model),
        "sha256":hashlib.sha256(model.read_bytes()).hexdigest(),"device":"npu:0"}}
    with pytest.raises(ValueError,match="not validated"):reg.checked(bulk("Cu"),"m")


def test_msd_constant_velocity_not_certified(tmp_path):
    import io
    s=Store(tmp_path);buf=io.BytesIO();t=np.arange(20,dtype=float)
    pos=np.zeros((20,2,3));pos[:,0,0]=t;pos[:,1,0]=-t
    np.savez_compressed(buf,positions_A=pos,time_fs=t,cell_A=np.tile(np.eye(3)*100,(20,1,1)),
                        symbols=np.array(['Cu','Cu']),masses=np.array([63.,63.]))
    aid=s.put_bytes(buf.getvalue(),'trajectory','.npz')
    out=analyze_msd(s,aid,'Cu')
    assert not out['validated_diffusion']
    np.testing.assert_allclose(out['msd_A2'],np.arange(1,11)**2)


@pytest.fixture
def tokenizer():
    path=os.environ.get("SCIOWL_TEST_TOKENIZER")
    if not path:pytest.skip("Set SCIOWL_TEST_TOKENIZER to real local Qwen3 tokenizer for template validation")
    transformers=pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(path,local_files_only=True)


def test_real_qwen_all_assistant_decisions_and_masks(tmp_path,tokenizer):
    s=Store(tmp_path);aid=atoms_to_id(s,bulk("Cu","fcc",a=3.6,cubic=True))
    Harness(s,Gateway(s),ScriptedModel(aid),inline_worker=True).run('mask','Relax')
    record=record_from_session(s,'mask')
    outputs=tokenize_decisions(record,tokenizer)
    assistants=[i for i,m in enumerate(record['messages']) if m['role']=='assistant']
    assert len(outputs)==len(assistants)==5
    for sample in outputs:
        (a,b),=sample['audit']['assistant_spans']
        encoded=tokenizer(sample['audit']['text'],add_special_tokens=False,return_offsets_mapping=True)
        for label,(start,end) in zip(sample['labels'],encoded['offset_mapping']):
            if start<a:assert label==-100
            if label!=-100:assert a<=start<end<=b
        assert sample['audit']['supervised_tokens']>0
        if sample['audit']['target_message_index']!=assistants[-1]:
            assert '<tool_call>' in sample['audit']['text'][a:b]


def test_real_qwen_failed_action_masked_and_length_guard(tmp_path,tokenizer):
    s=Store(tmp_path);aid=atoms_to_id(s,bulk("Cu","fcc",a=3.6,cubic=True))
    Harness(s,Gateway(s),ScriptedModel(aid),inline_worker=True).run('mask','Relax')
    record=record_from_session(s,'mask')
    record['messages'][2]['train']=False
    outputs=tokenize_decisions(record,tokenizer)
    assert len(outputs)==4
    assert all(o['audit']['target_message_index']!=2 for o in outputs)
    with pytest.raises(ValueError,match="max length"):tokenize_decisions(record,tokenizer,max_length=10)
    with pytest.raises(ValueError,match="masking"):sharegpt(record)

