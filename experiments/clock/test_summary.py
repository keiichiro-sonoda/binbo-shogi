#!/usr/bin/env python3
"""Nonzero paired/cluster bootstrap oracle and strict missing-data checks."""
import argparse
import copy
import csv
import json
from pathlib import Path
from paired_summary import summarize
p=argparse.ArgumentParser()
p.add_argument('--template',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
root=Path(a.out)
root.mkdir(parents=True,exist_ok=False)
template=json.loads(Path(a.template).read_text())
scores={'default':[0,.4,1,0], 'candidate':[.4,1,1,.4]}
for arm,values in scores.items():
 d=root/arm;d.mkdir()
 meta=copy.deepcopy(template)
 meta['args']['arm']=arm;meta['args']['games']=4
 (d/'run.json').write_text(json.dumps(meta))
 for i,score in enumerate(values):
  r={'seed':i,'arm':arm,'result':'1-0' if score==1 else '1/2-1/2' if score==.4 else '0-1',
     'black':'A','white':'B','black_score':score,'opening_id':str(i//2),
     'opening':str(i//2),'reason':'fixture'}
  with (d/f'{i}.tsv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=list(r),delimiter='\t');w.writeheader();w.writerow(r)
output=summarize([root/'default',root/'candidate'],20000,180)
r=output['candidate']
assert abs(r['delta']-.35)<1e-12 and r['pairs']==4 and r['opening_clusters']==2
assert all(abs(x-y)<1e-12 for x,y in zip(r['ci95'],[.2,.5])),r
(root/'oracle.json').write_text(json.dumps(output,indent=2)+'\n')
(root/'candidate/3.tsv').unlink()
try:
 summarize([root/'default',root/'candidate'],100,180)
except ValueError as exc:
 assert 'incomplete run' in str(exc)
else:
 raise AssertionError('missing row accepted')
print('PASS: paired delta .35; cluster CI [.2,.5]; incomplete arm rejected')
