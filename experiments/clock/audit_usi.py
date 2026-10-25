#!/usr/bin/env python3
"""Compare recorded nonponder sente go parameters against source TM, and charged elapsed."""
import argparse
from collections import defaultdict
import datetime
import json
import hashlib
from pathlib import Path
import re
from analyze_logs import quant, tm
from clock_rule import position_ply
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
root=Path(a.data_root)/'denryusen'
paths=list(root.glob('xub/book1/nobook/logs/usi-*.log'))+list(root.glob('xub/run1/*/logs/usi-*.log'))+list(root.glob('floodgate/**/logs/usi-*.log'))
rows=[]
for path in paths:
    pending=None
    position=None
    for raw in path.open():
        m=re.match(r'\[([^]]+)\].*?: ([<>]) (.*)',raw)
        if not m: continue
        stamp=datetime.datetime.fromisoformat(m[1]).timestamp()
        direction,line=m[2],m[3].strip()
        if direction=='>':
            if line.startswith('position '): position=line
            elif line.startswith('go '):
                pending=None
                if not position or ' ponder ' in line: continue
                ply,color=position_ply(position)
                if color!=0: continue
                t=line.split()
                if not all(k in t for k in ['btime','binc']): continue
                bt=int(t[t.index('btime')+1]);inc=int(t[t.index('binc')+1])
                pending={'path':str(path),'ply':ply,'btime':bt,'inc':inc,
                         'pred':tm(bt,inc,ply),'start':stamp,'engine_time':None}
            elif line=='stop': pending=None
        elif pending:
            if line.startswith('info ') and ' time ' in line:
                x=re.search(r' time (\d+)',line)
                if x: pending['engine_time']=int(x[1])
            elif line.startswith('bestmove '):
                pending['elapsed_ms']=round((stamp-pending.pop('start'))*1000,3)
                rows.append(pending);pending=None
bygroup=defaultdict(list)
for r in rows:
    cat='stage2_nobook' if '/book1/' in r['path'] else ('floodgate' if '/floodgate/' in r['path'] else 'local_run1')
    bygroup[cat].append(r)
summary={}
for k,v in bygroup.items():
    ratios=[r['elapsed_ms']/r['pred']['optimum'] for r in v if r['pred']['optimum']>0]
    diffs=[r['elapsed_ms']-r['engine_time'] for r in v if r['engine_time'] is not None]
    summary[k]={'nonponder_sente_go':len(v),'elapsed_over_optimum':quant(ratios),
                'elapsed_minus_engine_ms':quant(diffs),'exceed_tm_maximum':sum(r['elapsed_ms']>r['pred']['maximum']+50 for r in v),
                'max_excess_ms':max(r['elapsed_ms']-r['pred']['maximum'] for r in v)}
with open(a.out,'x') as f: json.dump({'summary':summary,'rows':rows,'inputs':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}},f,indent=2)
print(json.dumps(summary,indent=2))
