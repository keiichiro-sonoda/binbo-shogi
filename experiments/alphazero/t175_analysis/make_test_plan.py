#!/usr/bin/env python3
"""Validate existing histories and write USI test inputs; never launch an engine."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,hashlib
from pathlib import Path
from collections import Counter
import cshogi
OUT=Path(__file__).resolve().parent

def dump(name,x):
    with (OUT/name).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    source=json.loads((OUT/'repetition_fixtures.json').read_text());valid=[];reject=[]
    for i,f in enumerate(source['selected']):
        b=cshogi.Board();counts=Counter()
        moves=f['position'].split(' moves ',1)[1].split()
        for u in moves:
            counts[' '.join(b.sfen().split()[:3])]+=1
            assert b.is_legal(b.move_from_usi(u));b.push_usi(u)
        counts[' '.join(b.sfen().split()[:3])]+=1
        assert b.sfen()==f['sfen'] and b.turn==f['turn']
        statuses=[]
        for u in f['draw_moves']:
            assert b.is_legal(b.move_from_usi(u));b.push_usi(u)
            assert counts[' '.join(b.sfen().split()[:3])]==3
            statuses.append(b.is_draw());b.pop()
        if any(s!=cshogi.REPETITION_DRAW for s in statuses):reject.append(dict(index=i,statuses=statuses));continue
        f['id']=f"rep_{i:02d}";f['cshogi_draw_statuses']=statuses;f['history_legal']=True;valid.append(f)
    assert Counter(f['turn'] for f in valid)[0]>=4 and Counter(f['turn'] for f in valid)[1]>=4
    nodes=10000
    # Minimal setup is text only. An interactive harness MUST wait for usiok,
    # readyok and bestmove; blindly piping quit can abort a search.
    setup=['usi','setoption name Threads value 1','setoption name USI_Hash value 16',
       'setoption name USI_OwnBook value false','setoption name BookFile value no_book',
       'setoption name USI_Ponder value false','setoption name MultiPV value 1',
       'setoption name MaxMovesToDraw value 512','setoption name ResignValue value 99999']
    cases=[]
    for f in valid:
        cases.append(dict(fixture=f['id'],setup=setup,screen_continuation=[
          'setoption name DrawValueBlack value -2','setoption name DrawValueWhite value -2',
          'usinewgame','isready',f['position'],
          f"go nodes {nodes} searchmoves "+' '.join(f['other_legal_moves'])],
          caveat='root immediate fourfold moves excluded, but later repetition may remain; inspect PV and reject it as a play-on control if it draws later'))
    plan=dict(task=175,executed=False,engine_runs=0,training_runs=0,fixtures=valid,rejected=reject,
       settings=dict(baseline=[-2,-2],tournament_initial_candidate=[-60,60],depth9_candidate=[-260,290],
                     units='USI cp, root-color option; internal=int(option*90/100); allow ~2cp draw jitter'),
       screening_cases=cases,nodes_per_search=nodes,threads=1,
       stages=[
         'Screen 24 validated history roots at 10k nodes with immediate draw moves excluded; continuation score c is not the logged root score near -2.',
         'Keep up to 4 roots per color: two with c in the predicted switch band and one on each side outside it (>=10cp margin). No coverage means INCONCLUSIVE; do not manufacture evidence.',
         'Fresh process / same eval / book off / same 10k nodes, compare baseline (-2,-2) to (-60,+60), both unrestricted and draw-move-only. Wait bestmove before each subsequent command; repeat once.',
         'Black: expected draw->play only if -60<c<-2. White: expected play->draw only if -2<c<+60. Within jitter margin classify undecided. Outside-band changes fail the local stability check, inspect PV and node/depth effects.',
         'For two roots per color: change only inactive-color option and confirm exact same bestmove/score with fresh process. Replay same board via bare SFEN without history as negative control; absent newly reached repetition, any change fails this control.',
         'A single root continuation PV can still reach repetition. Restrict to the best non-repeating root move and verify completed PVs before treating c as the continuation alternative.',
         '512-move cap controls: take a legal, nonchecking fixture SFEN without history and set its move-number to 512 (one legal move reaches game_ply 513); compare cap512 vs cap0 under both settings. Cap0 disables this rule. Also inspect 511/513 and exclude mates/perpetual checks. Under cap512 all surviving cap-draw branches can change value; this is an intended documented exception to the repetition-only check.',
         'Record eval/engine SHA, full history, options, go, final exact score/depth/nodes/PV/bestmove, legality, c, forced-draw score. These are behavioral checks, not an Elo or policy-improvement estimate.'
       ],budget=dict(screen_searches=len(valid),screen_nodes=len(valid)*nodes,
             main_roots=8,main_unrestricted_searches=32,main_forced_draw_searches=16,
             controls_upper_searches=24,total_searches_upper=96,total_nodes_upper=960000,
             followup='If main continuation PV is uncertain, double only that root to 20k; no whole-game matches needed'),
       source_checks=dict(history_and_fourfold_positions=len(valid),continuous_check_excluded=True),
       limitation='Cannot guarantee zero changes outside a band in unrestricted selective search, or when 512 cap/MAX_PLY is reachable. The proposed check tests selected roots with explicit coverage and margins.')
    dump('fixed_position_test_plan.json',plan)
    print('validated fixtures',len(valid),'rejected',len(reject))

if __name__=='__main__':main()
