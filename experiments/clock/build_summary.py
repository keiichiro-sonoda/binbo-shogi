#!/usr/bin/env python3
"""Rebuild Task 180 machine-readable summary from the retained evidence."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import math
import re
import sys
from analyze_logs import tm

p=Path(__file__).resolve().parent
data=Path(sys.argv[1]).resolve()
source=data/'engines/yaneuraou/source'
files=[source/'timeman.cpp',source/'timeman.h',source/'search.h',source/'usi.cpp',
       source/'engine/yaneuraou-engine/yaneuraou-search.cpp',source/'YaneuraOu-by-gcc',
       data/'eval_dirs/task154_R_result30/nn.bin']
manifest={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
rows=[]
nps=[]
for f in (p/'smoke10').glob('*.tsv'):
 with f.open() as fd: rows+=list(csv.DictReader(fd,delimiter='\t'))
for f in (p/'smoke10').glob('*.usi.log'):
 last=None
 for line in f.read_text().splitlines():
  if ' > go ' in line:last=None
  if ' < info ' in line:
   n=re.search(r' nodes (\d+)',line);t=re.search(r' time (\d+)',line)
   if n and t and int(t[1]):last=int(n[1])*1000/int(t[1])
  if ' < bestmove' in line and last:nps.append(last)
power=[]
z=statistics.NormalDist().inv_cdf(.9875)+statistics.NormalDist().inv_cdf(.8)
for decisive in [.5,.2]:
 for draw in [0,.1]:
  mean=(1-draw)*decisive+.4*draw
  variance=(1-draw)*decisive+.16*draw-mean**2
  for elo in [20,40]:
   prob=1/(1+(1-decisive)/decisive*10**(-elo/400))
   delta=(1-draw)*(prob-decisive)
   power.append({'decisive_p':decisive,'draw_fraction':draw,'elo':elo,'score_delta':delta,
                 'pairs_per_arm_approx':math.ceil(z*z*2*variance/delta**2),
                 'paired_variance_assumed':2*variance})
prereg={'date':'2026-10-24','status':'NOT_RUN','eval':'task154_R_result30',
        'seed':2026180000,'pairs_per_arm':10000,'games_total':30000,
        'arms':{'default':{},'cap30':{'rule':'cap','cap_ms':3000,'k':15},
                'slow70':{'a_option':['SlowMover=70']}},
        'control_ms':{'black_base':18000,'white_base':60000,'black_inc':200,'white_inc':200},
        'threads':1,'hash_mb':256,'ponder':False,'opening_plies':4,'max_plies':512,
        'network_margin_ms':0,'charge_quantum_ms':0,'opening_source':'seeded uniform legal moves after buoy',
        'primary_metric':'mean(sente_score_arm - sente_score_default); W=1 D=.4 L=0',
        'inference':'opening-cluster paired percentile bootstrap; 20000 reps; seed 180; two arms Bonferroni 97.5% CIs',
        'stopping':'fixed 10000 pairs; no significance peeking; missing/INVALID invalidates analysis',
        'followup':'fresh openings, event clock and options including ponder; separate authorization/run'}
(p/'preregistration.json').write_text(json.dumps(prereg,indent=2)+'\n')
summary={'task':180,'date':'2026-10-24','large_experiment_run':False,
         'engine_source_revision':'820940ee31762ad598a9156a615b4b896ab8b321',
         'inputs_sha256':manifest,'scripts_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(p.glob('*.py'))},
         'analysis':json.loads((p/'analysis_complete/aggregate.json').read_text()),
         'analysis_by_own_move':'analysis_complete/by_own_move.tsv',
         'analysis_games_manifest':'analysis_complete/games.json',
         'usi_audit':json.loads((p/'usi_audit_complete.json').read_text())['summary'],
         'formula_initial_ms':tm(180000,2000,5),
         'formula_initial_no_ponder_ms':tm(180000,2000,5,ponder=False),
         'formula_cap30_ms':tm(30000,2000,5),
         'formula_scaled_initial_ms':tm(18000,200,5,ponder=False,minimum=200,nd=12,nd2=112,rounding=False),
         'smoke10':json.loads((p/'smoke10/summary.json').read_text()),
         'smoke_details':{'median_total_plies':statistics.median(int(r['plies']) for r in rows),
                          'sum_game_wall_seconds':sum(float(r['wall_seconds']) for r in rows),
                          'nps_samples':len(nps),'nps_median':statistics.median(nps),
                          'nps_q25_q75':statistics.quantiles(nps,n=4)[::2]},
         'memory_probe':{'engine_rss_kib':444592,'hash_mib':256,'engine_threads_option':1,
                         'resident_engines_at_parallel40':80,'estimated_rss_gib_at_parallel40':444592*80/1024**2},
         'tests':{'test_clock':{'passed':8,'log':'tests_verified.log'},
                  'forced_timeout':'tests_verified/timeout', 'fourfold_game':'tests_verified/cycle',
                  'summary_oracle':'summary_verified.log','paired_pipeline':'matrix_check/paired.json',
                  'additional_truncated_engine_games':12},
         'power_assumptions':{'power':.8,'family_alpha_two_sided':.05,'comparisons':2,
                              'null_variance_approximation':True,'pair_correlation_assumed':0},
         'power':power,'preregistration':prereg}
(p/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
print('wrote summary.json and preregistration.json')
