#!/usr/bin/env python3
"""Read-only clustered sample; outputs are exclusive-created in this directory."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import sys, json, time, hashlib, resource
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import cshogi

OUT=Path(__file__).resolve().parent
DATA=Path(os.environ['TEACHER_DIR']).parent
REPO=Path(os.environ['TASK175_REPO'])
sys.path.insert(0,str(REPO))
from trainer import kp
SEED=20261750

def dump(name,x):
    with (OUT/name).open('x') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist())
        f.write('\n')

def stat(x):
    a=np.asarray(x,dtype=float)
    return dict(n=len(a),mean=float(a.mean()) if len(a) else None,
                quantiles=dict(zip(['min','p25','p50','p75','p90','p95','p99','max'],np.percentile(a,[0,25,50,75,90,95,99,100]).tolist())) if len(a) else {})

def phase(sfen):
    board,turn,hand=sfen.split()
    h=sum(int(n or '1') for n in __import__('re').findall(r'(\d*)[PLNSGBRplnsgbr]',hand))
    return board.count('+')+h

def binply(p):
    return '006-029' if p<30 else '030-059' if p<60 else '060-099' if p<100 else '100-149' if p<150 else '150-199' if p<200 else '200-255'

def binphase(p):
    return 'promoted+hand=0' if p==0 else '1-4' if p<5 else '5-9' if p<10 else '10+'

def main():
    start=time.monotonic()
    meta=json.loads((DATA/'caches/shared_64m_qs129/meta.json').read_text())
    features=np.load(DATA/'caches/shared_64m_qs129/train_indices.npy',mmap_mode='r')
    labels=np.load(DATA/'work/task129/labels/train_dlshogi_evals.npy',mmap_mode='r')
    original_labels=np.load(DATA/'caches/shared_64m_qs129/train_evals.npy',mmap_mode='r')
    wdl=np.load(DATA/'caches/shared_64m_qs129/train_wdl.npy',mmap_mode='r')
    changed=np.load(DATA/'work/task129/changed.npy',mmap_mode='r')
    rng=np.random.default_rng(SEED)
    # 128 uniformly sampled train shards, each exactly 10,000 rows. This preserves
    # entire game sequences (except shard-end truncation) and local QS collapse.
    catalog=[];offset=0; runs=[]
    for ri,rd in enumerate([meta['run_dir'],*meta['extra_runs']]):
        rd=Path(rd); manifest=json.loads((rd/'manifest.json').read_text())
        shards=[s for s in manifest['shards'] if s['done']]
        if ri==0:shards=[s for s in shards if s['index'] in meta['train_shards']]
        runs.append(dict(name=rd.name,base_seed=manifest['meta']['base_seed'],shards=len(shards),offset=offset,
                         seed_range=[min(s['seed'] for s in shards),max(s['seed'] for s in shards)]))
        for s in shards:
            assert s['count']==10000
            catalog.append((rd,s,offset,ri));offset+=s['count']
    assert offset==len(features)==63950000
    selected=np.sort(rng.choice(len(catalog),128,replace=False))
    # Fields: global row, shard sample id, run, game id, ply, activity, raw key,
    # label, original inherited label, WDL, changed, feature row key.
    all_rows=[];shardstats=[];examples=[];checked=0;raw_seen={};leaf_seen={}
    raw_duplicated=[]; leaf_duplicated=[]; local_flags=[]
    all_game_sequences=defaultdict(list)
    for si,ci in enumerate(selected):
        rd,s,off,ri=catalog[ci]
        raw=np.fromfile(rd/s['file'],dtype=cshogi.HuffmanCodedPosAndEval)
        text=(rd/s['file']).with_suffix('.expected.tsv')
        rows=[]
        with text.open() as stream:
            next(stream)
            for i,line in enumerate(stream):
                fields=line.rstrip().split('\t')
                _,name,sfen=fields[:3];g,p=map(int,name[1:].split('_p'))
                assert int(fields[0])==i and bytes(raw[i]['hcp']).hex()==fields[8]
                rows.append((g,p,phase(sfen)))
                all_game_sequences[(si,g)].append(raw[i]['hcp'].tobytes())
        assert len(rows)==len(raw)==10000
        x=np.asarray(features[off:off+10000]);lab=np.asarray(labels[off:off+10000]);w=np.asarray(wdl[off:off+10000])
        c=np.asarray(changed[off:off+10000]);inh=np.asarray(original_labels[off:off+10000])
        localraw=set();localleaf=set();ldup=0;rf=0
        for i,(g,p,a) in enumerate(rows):
            rawkey=raw[i]['hcp'].tobytes();leafkey=x[i].tobytes();j=len(all_rows)
            all_rows.append((off+i,si,ri,g,p,a,rawkey,int(lab[i]),int(inh[i]),int(w[i]),bool(c[i]),leafkey))
            raw_duplicated.append(rawkey in raw_seen);leaf_duplicated.append(leafkey in leaf_seen)
            local_flags.append(leafkey in localleaf)
            if rawkey in localraw:rf+=1
            if leafkey in localleaf:ldup+=1
            raw_seen.setdefault(rawkey,j);leaf_seen.setdefault(leafkey,[]).append(j)
            localraw.add(rawkey);localleaf.add(leafkey)
        # Source alignment: direct HCP -> features on unchanged sample, and
        # inverse leaf -> HCP -> features checks. No engine or QS is executed.
        b=cshogi.Board()
        for i in rng.choice(10000,8,replace=False):
            if not c[i]:
                b.set_hcp(raw[i]['hcp']);assert np.array_equal(kp.canonical_shared_row(b),x[i]);checked+=1
            sf=kp.canonical_shared_row_to_sfen(x[i]);b.set_sfen(sf)
            assert np.array_equal(kp.canonical_shared_row(b),x[i]);checked+=1
        shardstats.append(dict(run=rd.name,index=s['index'],offset=off,seed=s['seed'],raw_duplicate_excess=rf,
                               leaf_duplicate_excess=ldup,changed=int(c.sum()),rows=10000))
        if (si+1)%16==0:print(f'B sample shards {si+1}/128',flush=True)
    n=len(all_rows);leafdup=np.asarray(leaf_duplicated);rawdup=np.asarray(raw_duplicated)
    group_sizes=[];excess_origins=Counter();group_origins=Counter();cross=Counter();dist=[];label_delta=[];inherit_delta=[]
    conflicting_wdl=0;known_conflict=0;zero_mixed=0;group_ranges=[];group_variances=[];targetvar=[];examplegroups=[]
    member=np.zeros(n,bool); excess_by_ply=Counter();member_by_ply=Counter();all_by_ply=Counter()
    excess_by_phase=Counter();member_by_phase=Counter();all_by_phase=Counter();changed_excess=Counter()
    perrun=defaultdict(Counter);crosshalf=0;crossgame=0;crossshard=0;samegame=0;pair_count=0
    # Duplicate classification: one row per distinct raw root in a leaf group
    # can already be present; subsequent new raw roots are QS coalescences;
    # repeated raw roots are source duplicates. Decomposition sums exactly.
    for key,ids in leaf_seen.items():
        group_sizes.append(len(ids))
        if len(ids)<2:continue
        member[ids]=True
        roots=set();origin=Counter();ys=[];ws=[];groupshards=set();games=set();halves=set()
        for k,j in enumerate(ids):
            row,si,ri,g,p,a,r,y,inh,w,c,f=all_rows[j]
            ys.append(y);ws.append(w);groupshards.add(si);games.add((si,g));halves.add(int(row>=31950000))
            if k:
                kind='same_raw_root' if r in roots else 'different_raw_roots_same_leaf'
                origin[kind]+=1;excess_origins[kind]+=1
                prev=all_rows[ids[0]]
                cross['same_game' if (si,g)==(prev[1],prev[3]) else 'different_game']+=1
                cross['same_shard' if si==prev[1] else 'different_shard']+=1
                cross['same_half' if int(row>=31950000)==int(prev[0]>=31950000) else 'different_half']+=1
                if (si,g)==(prev[1],prev[3]):dist.append(p-prev[4])
                label_delta.append(y-prev[7]);inherit_delta.append(inh-prev[8]);changed_excess[str(c)]+=1
                if len(examples)<12:
                    examples.append(dict(rows=[prev[0],row],plies=[prev[4],p],labels=[prev[7],y],wdl=[prev[9],w],
                                         same_root=r==prev[6],same_game=(si,g)==(prev[1],prev[3]),changed=[prev[10],c]))
            roots.add(r)
        group_origins['mixed' if len(origin)>1 else next(iter(origin))]+=1
        if len(set(ws))>1:conflicting_wdl+=1
        if -1 in ws and 1 in ws:known_conflict+=1
        if 0 in ws and any(ws):zero_mixed+=1
        yr=np.asarray(ys,float);group_ranges.append(float(np.ptp(yr)));group_variances.append(len(ids)*float(yr.var()))
        tar=1/(1+np.exp(-np.clip(yr,-2000,2000)/600));targetvar.append(len(ids)*float(tar.var()))
        if len(examplegroups)<8 and np.ptp(yr)>100:
            examplegroups.append(dict(rows=[all_rows[j][0] for j in ids],labels=ys,wdl=ws,range=float(np.ptp(yr))))
    for j,r in enumerate(all_rows):
        pb=binply(r[4]);ab=binphase(r[5]);all_by_ply[pb]+=1;all_by_phase[ab]+=1
        perrun[runs[r[2]]['name']]['rows']+=1
        if leafdup[j]:excess_by_ply[pb]+=1;excess_by_phase[ab]+=1;perrun[runs[r[2]]['name']]['leaf_excess']+=1
        if rawdup[j]:perrun[runs[r[2]]['name']]['raw_excess']+=1
        if member[j]:member_by_ply[pb]+=1;member_by_phase[ab]+=1
    def distribution(total,excess,memb):
        return {k:dict(rows=v,duplicate_excess=excess[k],duplicate_member=memb[k],rate=excess[k]/v) for k,v in sorted(total.items())}
    # Cluster bootstrap over the sampled shards; estimates local within-shard
    # collapse, not the few cross-shard overlaps in this sample.
    z=np.array([s['leaf_duplicate_excess']/10000 for s in shardstats]);zr=np.array([s['raw_duplicate_excess']/10000 for s in shardstats])
    boot=rng.integers(0,len(z),(2000,len(z)));ci=np.percentile(z[boot].mean(1),[2.5,97.5]).tolist()
    game_keys=Counter(tuple(v[:16]) for v in all_game_sequences.values() if len(v)>=16)
    repeated_prefix=sum(v-1 for v in game_keys.values())
    counts=np.asarray(group_sizes,dtype=float);dupgroups=int(np.sum(counts>1));dup_excess=n-len(leaf_seen)
    # First occurrence in cache order equals shard order and row order here.
    absdelta=np.abs(label_delta);absinh=np.abs(inherit_delta)
    exact_a=json.loads((DATA/'work/task172/audit.json').read_text())
    result=dict(task=175,part='B',method='128 uniformly sampled complete train shards; exact 77-column byte keys; no full rescan',seed=SEED,
      source=str(DATA/'caches/shared_64m_qs129'),sample=dict(shards=128,population_shards=len(catalog),rows=n,fraction=n/63950000,
      leaf_duplicate_excess=dup_excess,leaf_duplicate_rate=dup_excess/n,raw_duplicate_excess=n-len(raw_seen),raw_duplicate_rate=(n-len(raw_seen))/n,
      local_leaf_duplicate_rate=float(z.mean()),local_leaf_rate_cluster_ci95=ci,local_raw_duplicate_rate=float(zr.mean()),
      cross_shard_additional_excess=dup_excess-int(np.sum(local_flags)),duplicate_member_rows=int(member.sum()),duplicate_groups=dupgroups),
      exact_audit=dict(rows=63950000,duplicate_excess=15095305,unique=63950000-15095305,removed_fraction=15095305/63950000,
                       same_steps_unique_reweighting=63950000/(63950000-15095305),validation_excluded_rows=610),
      runs=runs,by_run=perrun,shards=shardstats,origins_excess=excess_origins,origins_group=group_origins,
      first_representative_relation=cross,changed_excess=changed_excess,within_game_root_ply_gap=stat(dist),
      group_size=stat(group_sizes),max_multiplicity=int(max(group_sizes)),kish_size_sample=float(n*n/np.sum(counts**2)),
      phase_definition='root promoted-piece count + pieces in hand (activity proxy, not a learned phase)',
      by_ply=distribution(all_by_ply,excess_by_ply,member_by_ply),by_phase=distribution(all_by_phase,excess_by_phase,member_by_phase),
      labels=dict(unit='WS-T internal raw/600 training target, not USI cp',reference='first occurrence in sampled cache order',
      absolute_difference=stat(absdelta),difference=stat(label_delta),inherited_gikou_absolute_difference=stat(absinh),
      identical_duplicate_labels=int(np.sum(np.asarray(label_delta)==0)),different_duplicate_labels=int(np.sum(np.asarray(label_delta)!=0)),
      abs_difference_over_100=int(np.sum(absdelta>100)),group_range=stat(group_ranges),
      conflicting_wdl_groups=conflicting_wdl,known_win_loss_conflict_groups=known_conflict,unknown_known_mixed_groups=zero_mixed,
      irreducible_label_mse_all_rows=float(sum(group_variances)/n),irreducible_sigmoid_target_mse_all_rows=float(sum(targetvar)/n),
      note='averaging targets removes within-position target variance; dropping first/last alone does not guarantee cleaner labels'),
      repeated_game_prefixes=dict(games_16plus=sum(len(v)>=16 for v in all_game_sequences.values()),excess_identical_first16_hcp=repeated_prefix),
      checks=dict(source_row_roundtrips=checked,source_hcp_hex_all_sample_rows=True,hash_collisions='no hashes used; exact bytes',
                  origin_partition_sum=sum(excess_origins.values()),expected=dup_excess),examples=examples,label_conflict_examples=examplegroups,
      runtime_seconds=time.monotonic()-start,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    assert sum(excess_origins.values())==dup_excess
    dump('duplicates.json',result)
    print(json.dumps({k:result[k] for k in ['sample','origins_excess','first_representative_relation','labels','by_ply','runtime_seconds','peak_rss_mib']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
