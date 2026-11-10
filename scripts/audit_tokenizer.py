#!/usr/bin/env python3
"""Audit masks with your actual local tokenizer without marking records reviewed."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from sciowl_agent.dataset import tokenize_decisions


def main():
    p=argparse.ArgumentParser();p.add_argument('--tokenizer',required=True);p.add_argument('--input',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
    audits=[]
    for line in Path(a.input).read_text().splitlines():
        record=json.loads(line)
        outputs=tokenize_decisions(record,tok)
        for sample in outputs:
            offsets=tok(sample['audit']['text'],add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
            a0,b0=sample['audit']['assistant_spans'][0]
            assert all(label==-100 or (a0<=start<end<=b0) for label,(start,end) in zip(sample['labels'],offsets))
        audits.append({'record':record['id'],'decisions':len(outputs),'supervised_tokens':[x['audit']['supervised_tokens'] for x in outputs],
                       'total_tokens':[len(x['input_ids']) for x in outputs]})
    hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(a.tokenizer).glob('*.json')}
    result={'passed':True,'records':audits,'tokenizer_sha256':hashes,'template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest(),
            'notice':'Template/mask audit only; does not certify scientific correctness or review status'}
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()

