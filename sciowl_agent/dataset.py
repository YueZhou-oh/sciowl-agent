"""Canonical trajectory export and conservative tokenizer-template loss masking."""
import hashlib
import json
import copy
from pathlib import Path
from .contracts import TOOLS,tool_schemas
from .store import canonical


def record_from_session(store,sid,reviewed=False):
    s=store.load_session(sid)
    if not s or s["status"]!="completed":
        raise ValueError("Only completed sessions may be exported")
    user=next(m["content"] for m in s["messages"] if m["role"]=="user")
    rid=s["report_id"]
    report=store.get_json(rid,"report")
    group=next((e["data"].get("input_structure_id") for e in report["evidence"] if e["data"].get("input_structure_id")),sid)
    messages=copy.deepcopy(s["messages"])
    for i,m in enumerate(messages):
        if m["role"]!="assistant" or not m.get("tool_calls"):
            continue
        ids={c["id"] for c in m["tool_calls"]}
        observations=[json.loads(x["content"]) for x in messages[i+1:] if x["role"]=="tool" and x["tool_call_id"] in ids]
        if any(not x.get("ok",False) for x in observations):
            m["train"]=False
    record={"id":sid,"tools":tool_schemas(),"messages":messages,
        "metadata":{"source_group":group,"observation_source":"real_execution","reviewed":reviewed,
                    "report_id":rid,"policy":"scripted_or_endpoint; see generation manifest","schema_version":1}}
    validate_record(record)
    return record


def validate_record(record):
    schemas={t["function"]["name"] for t in record["tools"]}
    pending=set()
    supervised=0
    for m in record["messages"]:
        role=m["role"]
        if role=="assistant":
            if pending:
                raise ValueError("Assistant continued before tool observations")
            supervised+=1
            for call in m.get("tool_calls",[]):
                name=call["function"]["name"]
                if m.get("train",True) and (name not in schemas or name not in TOOLS):
                    raise ValueError("Unknown tool")
                if m.get("train",True):
                    TOOLS[name][0].model_validate(json.loads(call["function"]["arguments"]))
                if call["id"] in pending:
                    raise ValueError("Duplicate call ID")
                pending.add(call["id"])
        elif role=="tool":
            if m["tool_call_id"] not in pending:
                raise ValueError("Orphan observation")
            pending.remove(m["tool_call_id"])
            json.loads(m["content"])
        elif role not in ["system","user"]:
            raise ValueError("Unsupported role")
    if pending or supervised==0:
        raise ValueError("Incomplete record")


def write_records(records,path):
    with Path(path).open('w') as f:
        for r in records:
            validate_record(r)
            f.write(canonical(r)+'\n')


def split_records(records):
    # Group must be assigned before augmentation; fractions stable across exports.
    out={"train":[],"validation":[],"test":[]}
    for r in records:
        group=r["metadata"]["source_group"]
        value=int(hashlib.sha256(group.encode()).hexdigest()[:8],16)%100
        out["test" if value<10 else "validation" if value<20 else "train"].append(r)
    return out


def sharegpt(record):
    """LLaMA-Factory/MindSpeed-style source adapter; actual handler must be audited."""
    converted=[]
    for m in record["messages"]:
        if m.get("train") is False:
            raise ValueError("ShareGPT adapter cannot preserve per-message masking; use canonical tokenizer path for recovery data")
        if m["role"]=="system":
            continue
        if m.get("tool_calls"):
            converted.append({"from":"function_call","value":canonical([
                {"name":c["function"]["name"],"arguments":json.loads(c["function"]["arguments"])} for c in m["tool_calls"]])})
        else:
            role={"user":"human","assistant":"gpt","tool":"observation"}[m["role"]]
            converted.append({"from":role,"value":m.get("content") or ""})
    return {"conversations":converted,"system":"\n".join(m["content"] for m in record["messages"] if m["role"]=="system"),
            "tools":canonical([t["function"] for t in record["tools"]])}


def tokenize_record(record,tokenizer,max_length=8192,target_index=None):
    """Use the local tokenizer's OWN template; reject ambiguous prefix rendering.

    Fast-tokenizer offsets map assistant spans to labels. This never trains user or
    tool observations. No truncation: discard or redesign oversized trajectories.
    """
    validate_record(record)
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer required for audited offset masks")
    messages=[]
    for source in record["messages"]:
        m={k:v for k,v in source.items() if k!="train"}
        if m.get("tool_calls"):
            m["tool_calls"]=json.loads(json.dumps(m["tool_calls"]))
            for c in m["tool_calls"]:
                c["function"]["arguments"]=json.loads(c["function"]["arguments"])
        messages.append(m)
    def render(ms):
        return tokenizer.apply_chat_template(ms,tools=record["tools"],tokenize=False,
                                             add_generation_prompt=False,enable_thinking=False)
    if target_index is None:
        target_index=max(i for i,m in enumerate(record["messages"]) if m["role"]=="assistant" and m.get("train") is not False)
    if messages[target_index]["role"]!="assistant" or record["messages"][target_index].get("train") is False:
        raise ValueError("Target must be a supervised assistant turn")
    # Render one decision at a time. Qwen3 adds empty thinking tags only to the
    # last assistant turn, so rendering full conversation prefixes is NOT stable.
    text=render(messages[:target_index+1])
    before=render(messages[:target_index])
    if not text.startswith(before):
        raise ValueError("Template is not prefix-stable at this decision; audit a template-specific adapter")
    spans=[(len(before),len(text))]
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    ids=encoded["input_ids"]
    if len(ids)>max_length:
        raise ValueError("Trajectory exceeds max length; no silent truncation")
    labels=[tok if end>start and any(start>=a and end<=b for a,b in spans) else -100
            for tok,(start,end) in zip(ids,encoded["offset_mapping"])]
    if all(x==-100 for x in labels):
        raise ValueError("No supervised tokens")
    return {"input_ids":ids,"attention_mask":[1]*len(ids),"labels":labels,
            "audit":{"id":record["id"],"target_message_index":target_index,"assistant_spans":spans,"text":text,
                     "supervised_tokens":sum(x!=-100 for x in labels)}}


def tokenize_decisions(record,tokenizer,max_length=8192):
    """Every supervised assistant decision becomes one training sample."""
    return [tokenize_record(record,tokenizer,max_length,i) for i,m in enumerate(record["messages"])
            if m["role"]=="assistant" and m.get("train") is not False]

