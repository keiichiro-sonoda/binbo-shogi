#!/usr/bin/env python3
"""Same 2% shard sample: repetition/collapse detail and actual result30 labels."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,time
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
import cshogi
import analyze_duplicates as m

def main():
    start=time.monotonic();base=json.loads((m.OUT/'duplicates.json').read_text());n=base['sample']['rows']
    x=np.load(m.DATA/'caches/shared_64m_qs129/train_indices.npy',mmap_mode='r')
    y=np.load(m.DATA/'work/task129/labels/train_dlshogi_evals.npy',mmap_mode='r')
    mixed=np.load(m.DATA/'work/task172/A_result30.npy',mmap_mode='r')
    w=np.load(m.DATA/'caches/shared_64m_qs129/train_wdl.npy',mmap_mode='r')
    # Only duplicate groups need retain root metadata; one source shard at a time.
    groups={};deltas=[];firstlast=[];gap_q=[];gap_r=[];origins=Counter();dups=Counter();mixed_variance=[];mixed_tvar=[];mixed_conflicts=0
    for si,s in enumerate(base['shards']):
        rd=m.DATA/'teacher'/s['run'];raw=np.fromfile(rd/f"shard_{s['index']:04d}.hcpe",dtype=cshogi.HuffmanCodedPosAndEval)
        meta=[]
        with (rd/f"shard_{s['index']:04d}.expected.tsv").open() as stream:
            next(stream)
            for line in stream:
                name=line.split('\t',2)[1];g,p=map(int,name[1:].split('_p'));meta.append((g,p))
        off=s['offset']
        for i,(g,p) in enumerate(meta):
            row=off+i;key=x[row].tobytes();r=raw[i]['hcp'].tobytes()
            rec=(r,si,g,p,int(y[row]),int(mixed[row]),int(w[row]),row)
            if key in groups:groups[key].append(rec)
            else:groups[key]=[rec]
    near=[];counts_wdl=Counter();allmix=[];members=0;mix_zero=0;keptbias=0
    for group in groups.values():
        if len(group)<2:continue
        members+=len(group);first=group[0];rawroots={};mixes=np.array([r[5] for r in group],float)
        mixed_variance.append(len(group)*float(mixes.var()));targets=1/(1+np.exp(-np.clip(mixes,-2000,2000)/600));mixed_tvar.append(len(group)*float(targets.var()))
        mixed_conflicts+=np.ptp(mixes)>0;firstlast.append(group[-1][5]-first[5]);allmix.extend(mixes.tolist())
        keptbias+=len(group)*(first[5]-float(mixes.mean()))
        for k,r in enumerate(group):
            root,si,g,p,yy,mix,ww,row=r
            if k:
                delta=mix-first[5];deltas.append(delta);counts_wdl[f'{first[6]},{ww}']+=1
                samegame=(si,g)==(first[1],first[2]);same_root=root in rawroots
                kind='source_repeat' if same_root else 'qs_collapse';origins[f'{kind}_{"same_game" if samegame else "different_game"}']+=1
                if same_root and (si,g)==(rawroots[root][1],rawroots[root][2]):gap_r.append(p-rawroots[root][3])
                elif not same_root and samegame:gap_q.append(p-first[3])
                dups[kind]+=1
                if samegame and not same_root:near.append(p-first[3])
            rawroots.setdefault(root,r)
    d=np.abs(deltas)
    result=dict(task=175,part='B-followup',sample_rows=n,duplicate_excess=len(deltas),origins=origins,
       raw_repeat_within_game_ply_gap=m.stat(gap_r),qs_collapse_within_game_ply_gap=m.stat(gap_q),
       qs_collapse_gap_le4=int(np.sum(np.array(gap_q)<=4)),qs_collapse_same_game_pairs=len(gap_q),
       result30_labels=dict(source=str(m.DATA/'work/task172/A_result30.npy'),absolute_difference=m.stat(d),
       identical_pairs=int(np.sum(d==0)),pairs_over100=int(np.sum(d>100)),groups_with_different_labels=int(mixed_conflicts),
       irreducible_raw_label_mse_all_rows=float(sum(mixed_variance)/n),irreducible_sigmoid_target_mse_all_rows=float(sum(mixed_tvar)/n),
       first_minus_group_mean_weighted_cp=float(keptbias/n),first_last_difference=m.stat(firstlast),wdl_pair_counts=counts_wdl),
       interpretation='pure WS-T labels agree; outcome mixing can add conflicting labels when identical leaf appears in different games or a repeated root cycle. WDL=0 is unknown or draw, not proven draw.',
       runtime_seconds=time.monotonic()-start)
    m.dump('duplicates_followup.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
