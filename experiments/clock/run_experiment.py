#!/usr/bin/env python3
"""Interleave preregistered three arms on a shared worker pool. Does not support ponder.
Arguments are clock_match.py arguments; --out is a new parent directory.
Controls default to event / 10. A=black, same config/eval both sides required.
Example: python3 -B run_experiment.py --config scaled.json --out experiment --games 6000 --parallel 40 --cpu-budget 40
"""
import concurrent.futures
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from clock_match import arguments, prepare, play, save_summary

def main():
    base=arguments()
    if base.a_color!='black' or base.rule!='none' or base.a_option or base.arm!='default':
        raise SystemExit('matrix fixes A=black and its three arms; omit arm/intervention arguments')
    root=Path(base.out)
    root.mkdir(parents=True,exist_ok=False)
    arms={}
    for arm in ['default','cap30','slow70']:
        a=copy.deepcopy(base)
        a.arm=arm
        a.out=str(root/arm)
        if arm=='cap30':
            a.rule='cap'
            a.cap_ms=base.cap_ms
        if arm=='slow70':
            a.a_option=['SlowMover=70']
        out,config,lines=prepare(a)
        if config['A']!=config['B']:
            raise SystemExit('pre-registration requires the same engine/eval/options in A/B config')
        arms[arm]=(a,out,config,lines)
    order=[]
    rng=random.Random(base.seed ^ 180)
    for i in range(base.games):
        names=list(arms)
        rng.shuffle(names)
        order.extend((i,arm) for arm in names)
    meta={'order':order,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'parallel':base.parallel,'cpu_budget':base.cpu_budget,'ponder':False}
    (root/'matrix.json').write_text(json.dumps(meta,indent=2)+'\n')
    rows={arm:[] for arm in arms}
    def task(pair):
        i,arm=pair
        a,out,config,lines=arms[arm]
        return arm,play(i,a,config,lines)
    started=time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=base.parallel) as pool:
        for arm,row in pool.map(task,order):
            rows[arm].append(row)
    for arm,(a,out,config,lines) in arms.items():
        save_summary(rows[arm],out,time.monotonic()-started)
if __name__=='__main__':
    main()
