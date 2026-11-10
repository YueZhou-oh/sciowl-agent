#!/usr/bin/env python3
"""Build indexed SFT bins from audited pretokenized records in your MindSpeed env.

Requires the actual local Megatron IndexedDatasetBuilder and torch installation.
No training code is downloaded. Verify your checkout's SFT data loader and naming.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument("--repo",required=True)
    p.add_argument("--input",required=True);p.add_argument("--output-prefix",required=True)
    args=p.parse_args();sys.path.insert(0,str(Path(args.repo).resolve()))
    import torch
    from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder,IndexedDataset
    prefix=Path(args.output_prefix).resolve();prefix.parent.mkdir(parents=True,exist_ok=True)
    keys=["input_ids","attention_mask","labels"]
    bases={k:str(prefix)+"_packed_"+k+"_document" for k in keys}
    if any(Path(b+ext).exists() for b in bases.values() for ext in [".bin",".idx"]):
        raise ValueError("Output exists; choose a new prefix")
    builders={k:IndexedDatasetBuilder(bases[k]+".bin",dtype=np.int32) for k in keys}
    count=0;supervised=0
    with open(args.input) as f:
        for line in f:
            row=json.loads(line)
            if not len(row["input_ids"])==len(row["labels"])==len(row["attention_mask"]):
                raise ValueError("Mismatched token/label lengths")
            if all(x==-100 for x in row["labels"]):raise ValueError("No supervised tokens")
            for k,b in builders.items():
                b.add_item(torch.tensor(row[k],dtype=torch.int32));b.end_document()
            count+=1;supervised+=sum(x!=-100 for x in row["labels"])
    if not count:raise ValueError("Empty input")
    for k,b in builders.items():b.finalize(bases[k]+".idx")
    for k in keys:
        dataset=IndexedDataset(bases[k])
        if len(dataset)!=count:raise RuntimeError("Indexed readback length mismatch")
        if k=="labels" and not any(np.any(dataset[i]==-100) for i in range(min(count,10))):
            raise RuntimeError("Expected masked context missing after readback")
    print(json.dumps({"prefix":str(prefix),"records":count,"supervised_tokens":supervised,
        "note":"Confirm naming and document-boundary behavior against your exact MindSpeed loader before training."},indent=2))


if __name__=="__main__":main()

