#!/usr/bin/env python3
"""Observational WDL calibration, <=1 BLAS thread; no engine invocation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json, sys, time, re, resource, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import cshogi

OUT=Path(__file__).resolve().parent
DATA=Path(os.environ['TEACHER_DIR']).parent
SEED=20261751
LIMIT=600
def dump(name,x):
    with (OUT/name).open('x') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,default=lambda a:a.item() if isinstance(a,np.generic) else a.tolist())
        f.write('\n')

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-35,35)))

def design(cp,turn,ply,draw=False):
    x=cp/200.;p=(ply-80)/80.;t=turn.astype(float)
    if draw:return np.array([np.ones(len(x)),t,p,p*p,np.abs(x),np.abs(x)*p]).T
    return np.array([np.ones(len(x)),t,p,p*p,x,x*p,t*x,t*p]).T

def fit(X,y,g):
    """Binomial MLE and game-cluster sandwich covariance (not iid per ply)."""
    b=np.zeros(X.shape[1]);b[0]=np.log((y.sum()+.5)/(len(y)-y.sum()+.5))
    ridge=1e-7
    for i in range(40):
        p=sigmoid(X@b);w=np.maximum(p*(1-p),1e-8)
        H=X.T@(w[:,None]*X)+np.eye(X.shape[1])*ridge
        db=np.linalg.solve(H,X.T@(y-p)-b*ridge)
        # guard unreasonably large Newton steps in very sparse strata
        db*=min(1,5/max(5,np.max(np.abs(db))))
        b+=db
        if max(abs(db))<1e-8:break
    p=sigmoid(X@b);H=X.T@((p*(1-p))[:,None]*X)+np.eye(X.shape[1])*ridge
    _,inv=np.unique(g,return_inverse=True);s=np.zeros((inv.max()+1,len(b)))
    np.add.at(s,inv,X*(y-p)[:,None])
    hi=np.linalg.inv(H);cov=hi@(s.T@s)@hi
    ng=len(s)
    if ng>1:cov*=ng/(ng-1)
    return dict(beta=b,cov=cov,rows=len(y),games=ng,events=int(y.sum()),iterations=i+1,
                logloss=float(np.mean(np.logaddexp(0,X@b)-y*(X@b))))

def cutoff(m,turn,ply,d):
    x0=design(np.array([0.]),np.array([turn]),np.array([ply]))[0]
    dx=design(np.array([1.]),np.array([turn]),np.array([ply]))[0]-x0
    b=m['beta'];s=dx@b;v=(math.log(d/(1-d))-x0@b)/s
    grad=-(x0+v*dx)/s;se=float(np.sqrt(max(0,grad@m['cov']@grad)))
    return dict(cp=float(v),ci95_cp=[float(v-1.96*se),float(v+1.96*se)],se_cp=se,slope_per_cp=float(s))

def predictions(w,d,cp,turn,ply):
    q=float(sigmoid(design(np.array([cp]),np.array([turn]),np.array([ply]))@w['beta'])[0])
    r=float(sigmoid(design(np.array([cp]),np.array([turn]),np.array([ply]),True)@d['beta'])[0])
    return dict(win=(1-r)*q,draw=r,loss=(1-r)*(1-q))

def model(a):
    cp,turn,ply,result,gid,src=a.T
    X=design(cp,turn,ply);D=design(cp,turn,ply,True);known=result!=0
    w=fit(X[known],(result[known]>0).astype(float),gid[known]);d=fit(D,(result==0).astype(float),gid)
    return w,d

def packmodels(w,d):
    return dict(win_given_decisive=w,draw=d,cutoffs={str(p):{str(t):cutoff(w,t,p,.4 if t==0 else .6) for t in (0,1)} for p in (24,40,60,80,100,120,160,200,260)},
                wdl_grid=[dict(cp=c,turn=t,ply=p,**predictions(w,d,c,t,p)) for p in (40,80,120,200) for t in (0,1) for c in (-200,-100,-50,0,50,100,200)])

def fixtures(drawgames):
    """Find genuine history-dependent fourfold-draw choices by legal replay."""
    ans=[]
    for task,game in drawgames:
        b=cshogi.Board();hist=Counter();checks=[]
        for ply,mv in enumerate(game['moves']):
            root=b.sfen();key=' '.join(root.split()[:3]);hist[key]+=1
            if ply>=30 and ply>=len(game['moves'])-5:
                drawing=[];other=[]
                for move in list(b.legal_moves):
                    u=cshogi.move_to_usi(move);b.push(move);k=' '.join(b.sfen().split()[:3]);rep=hist[k]>=3
                    b.pop();(drawing if rep else other).append(u)
                if drawing and other:
                    row=next((r for r in game['rows'] if r['ply']==ply),None)
                    ans.append(dict(source=task,game_id=game['game_id'],ply=ply,turn=b.turn,sfen=root,
                       position='position startpos moves '+' '.join(game['moves'][:ply]),
                       draw_moves=drawing,other_legal_moves=other,recorded_cp=int(row['raw']) if row and not row['mate'] else None,
                       note='drawing move reaches fourth exact board/hands/turn occurrence; perpetual-check exclusion must be checked for selected line'))
            assert b.is_legal(b.move_from_usi(mv));b.push_usi(mv)
    # Limit to 12 per side, spread across games. Full fixture search is data-only.
    selected=[];seen=set();nt=Counter()
    for f in sorted(ans,key=lambda f:(f['ply'],f['source'],f['game_id'])):
        k=(f['source'],f['game_id'],f['turn'])
        if k in seen or nt[f['turn']]>=12:continue
        seen.add(k);nt[f['turn']]+=1;selected.append(f)
    dump('repetition_fixtures.json',dict(candidates=len(ans),selected=selected))
    return len(selected)

def floodgate():
    # Intentionally the ONLY denryusen path that is enumerated or opened.
    files=sorted((DATA/'denryusen/floodgate').glob('*/me/records/*.csa'))
    rows=[];games=[];terms=Counter();seen=set()
    for f in files:
        text=f.read_text();digest=__import__('hashlib').sha256(text.encode()).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        lines=text.splitlines();moves=[];term=None;names={}
        for line in lines:
            if line.startswith('N+'):names[0]=line[2:]
            if line.startswith('N-'):names[1]=line[2:]
            if re.match(r'^[+-]\d{4}[A-Z]{2}$',line):moves.append(dict(turn=int(line[0]=='-'),ply=len(moves),comments=[]))
            elif line.startswith("'") and moves:moves[-1]['comments'].append(line)
            elif line.startswith('%'):term=line
        terms[term]+=1
        if term=='%TORYO':winner=moves[-1]['turn']
        elif term in ('%SENNICHITE','%JISHOGI','%HIKIWAKE'):winner=-1
        else:winner=None
        count=0
        if winner is not None:
            for m in moves:
                if 'binbo' not in names.get(m['turn'],'').lower():continue
                comment='\n'.join(m['comments']);match=re.search(r"\*\*評価値=([+-]?\d+)",comment)
                if not match:continue
                blackcp=int(match[1]);cp=blackcp*(1 if m['turn']==0 else -1)
                rows.append((cp,m['turn'],m['ply'],0 if winner==-1 else 1 if winner==m['turn'] else -1,len(games),0));count+=1
        games.append(dict(path=str(f),winner=winner,terminal=term,moves=len(moves),scored_binbo_rows=count,names=names))
    a=np.asarray(rows,dtype=float);mask=(np.abs(a[:,0])<=LIMIT) if len(a) else np.array([],bool)
    result=dict(games=games,terminal_counts=terms,unique_files=len(games),rows=len(rows),near_rows=int(mask.sum()),
       perspective='CSA viewer comments absolute sente; negate for gote mover; score belongs to position before the immediately preceding move',
       note='small time-controlled external sample; calibration not pooled with depth9 selfplay')
    if mask.sum()>30:
        sub=a[mask];X=np.array([np.ones(len(sub)),sub[:,0]/200]).T
        f=fit(X,(sub[:,3]>0).astype(float),sub[:,4]);result['decisive_simple_logit']=f
        cuts=[]
        for d in (.4,.6):
            b=f['beta'];v=(math.log(d/(1-d))-b[0])*200/b[1];grad=np.array([-200/b[1],-v/b[1]])
            se=np.sqrt(max(0,grad@f['cov']@grad));cuts.append(dict(draw_credit=d,cp=v,ci95_cp=[v-1.96*se,v+1.96*se]))
        result['pooled_side_cutoffs']=cuts
    return result

def local_inventory():
    # Existing local test summary TSVs have results, opening moves and average
    # depths/nodes, but no per-ply score. They cannot identify WDL(score,ply).
    out=[]
    for task,pattern in [('task158','main_w*.tsv'),('task165','p1_w*.tsv')]:
        fs=sorted((DATA/'work'/task).glob(pattern));n=0;results=Counter();headers=[]
        for f in fs:
            with f.open() as stream:
                for line in stream:
                    fields=line.rstrip().split('\t')
                    if len(fields)>4 and fields[0].isdigit():n+=1;results[fields[3]]+=1
                    elif line.startswith('game\t'):headers=fields
        out.append(dict(task=task,files=[str(f) for f in fs],games=n,results=results,header=headers,used=False,
                        reason='game summaries only; no per-ply scores'))
    return out

def main():
    start=time.monotonic();rng=np.random.default_rng(SEED);allrows=[];stats={};drawgames=[];idsets={};filesread=0
    for si,task in enumerate(['task154','task159']):
        manifest=json.loads((DATA/'work'/task/'manifest.json').read_text());ranges=manifest['ranges']
        population=np.concatenate([np.arange(lo,hi) for lo,hi in ranges]);ids=np.sort(rng.choice(population,8192,replace=False));idsets[task]=ids
        reasons=Counter();winners=Counter();nr=0;bad=0;drawreason=Counter();scorechecks=0
        for i,gid in enumerate(ids):
            path=DATA/'work'/task/'games'/f'game_{gid:012d}.json';game=json.loads(path.read_text());filesread+=1
            reasons[game['reason']]+=1;winners[str(game['winner'])]+=1
            if game['reason']=='repetition_draw':drawgames.append((task,game))
            for r in game['rows']:
                if r['mate']:continue
                cp=int(r['raw']);internal=(abs(cp)*90+99)//100*(1 if cp>=0 else -1)
                assert internal==r['score'];scorechecks+=1
                assert r['result']==(0 if game['winner']==-1 else 1 if r['turn']==game['winner'] else -1)
                if abs(cp)>LIMIT:continue
                nr+=1
                if r['result']==0:drawreason[game['reason']]+=1
                allrows.append((cp,r['turn'],r['ply'],r['result'],int(gid)+si*150000,si))
            if (i+1)%2048==0:print(f'A {task} games {i+1}/8192',flush=True)
        stats[task]=dict(population=len(population),sample_games=len(ids),reasons=reasons,winners=winners,near_rows=nr,draw_rows_by_reason=drawreason,score_unit_checks=scorechecks,
                        manifest=str(DATA/'work'/task/'manifest.json'),eval=manifest['config']['engine']['eval_dir'])
    a=np.asarray(allrows,dtype=float)
    # Save compact numeric evidence, not redundant raw records.
    assert np.array_equal(np.load(OUT/'draw_sample.npy'),a.astype(np.int32))
    assert {k:v.tolist() for k,v in idsets.items()}==json.loads((OUT/'sample_game_ids.json').read_text())
    w,d=model(a);models={'pooled':packmodels(w,d)}
    for si,task in enumerate(['task154','task159']):
        wi,di=model(a[a[:,5]==si]);models[task]=packmodels(wi,di)
    # Whole-game holdout, generated deterministically from game id; never split
    # positions from one game across fit/evaluation.
    hold=(a[:,4].astype(np.int64)*2654435761 % 5)==0
    tw,td=model(a[~hold]);test=a[hold];q=sigmoid(design(test[:,0],test[:,1],test[:,2])@tw['beta']);r=sigmoid(design(test[:,0],test[:,1],test[:,2],True)@td['beta'])
    pred=np.array([(1-r)*q,r,(1-r)*(1-q)]).T;y=np.array([test[:,3]>0,test[:,3]==0,test[:,3]<0]).T.astype(float)
    calibration=[]
    for lo,hi in [(-600,-200),(-200,-100),(-100,0),(0,100),(100,200),(200,601)]:
        mask=(test[:,0]>=lo)&(test[:,0]<hi)
        calibration.append(dict(cp=[lo,hi],rows=int(mask.sum()),predicted_WDL=pred[mask].mean(0),observed_WDL=y[mask].mean(0)))
    holdout=dict(games=len(np.unique(test[:,4])),rows=len(test),wdl_brier=float(np.mean(np.sum((pred-y)**2,axis=1))),
                 wdl_logloss=float(-np.mean(np.sum(y*np.log(np.clip(pred,1e-12,1)),axis=1))),calibration=calibration)
    # Independent side/phase simple logits make the continuous-model assumption
    # visible; thresholds outside the score support are not recommendations.
    strata=[]
    for si in [-1,0,1]:
        for lo,hi in [(16,40),(40,80),(80,120),(120,200),(200,320)]:
            for t in (0,1):
                m=(a[:,1]==t)&(a[:,2]>=lo)&(a[:,2]<hi)&(a[:,3]!=0)&((a[:,5]==si) if si>=0 else True)
                z=a[m];X=np.array([np.ones(len(z)),z[:,0]/200]).T;f=fit(X,(z[:,3]>0).astype(float),z[:,4]);b=f['beta']
                credit=.4 if t==0 else .6;v=(math.log(credit/(1-credit))-b[0])*200/b[1];grad=np.array([-200/b[1],-v/b[1]])
                se=float(np.sqrt(max(0,grad@f['cov']@grad)))
                strata.append(dict(source='pooled' if si<0 else ['task154','task159'][si],ply=[lo,hi],turn=t,cp=v,ci95_cp=[v-1.96*se,v+1.96*se],fit=f))
    nf=fixtures(drawgames)
    # Model sensitivity to artificial 320-ply draws: exclude these games and
    # refit P(draw); the decisive conditional probability and cutoff stay equal.
    nocap=[]
    for task,ids in idsets.items():
        pass
    result=dict(task=175,part='A',seed=SEED,method='two-stage logistic WDL, game-cluster sandwich covariance, per-ply observational likelihood',
      score_unit='USI cp reconstructed from raw (not internal score: PawnValue=90)',score_support=[-LIMIT,LIMIT],sample=stats,
      columns=['usi_cp','turn_0_black_1_white','ply_0_based','outcome_stm','unique_game_id','source_0_t154_1_t159'],
      formulas=dict(win='(1-r)*q',draw='r',loss='(1-r)*(1-q)',q='sigmoid(X beta), conditional win given decisive result',
                    r='sigmoid(Z gamma)',X=['1','turn','(ply-80)/80','((ply-80)/80)^2','cp/200','cp/200*(ply-80)/80','turn*cp/200','turn*(ply-80)/80'],
                    Z=['1','turn','(ply-80)/80','((ply-80)/80)^2','abs(cp)/200','abs(cp)/200*(ply-80)/80'],
                    equality='d = Pwin + d*Pdraw iff q=Pwin/(1-Pdraw)=d; natural draw probability cancels if r<1'),
      models=models,stratified_checks=strata,holdout=holdout,floodgate=floodgate(),local_tests=local_inventory(),repetition_fixture_count=nf,
      limitations=['observational threshold is not a causal policy evaluation: baseline already avoids/accepts repetitions at -2',
       'depth9 selfplay, shallow-score temperature, opening randomization and eval differ from tournament opponents/time control',
       '320-ply cap draws and perpetual-check-as-draw convention differ from tournament 512-ply rule',
       'cluster CI quantifies game sampling, not source/policy/model uncertainty',
       'logistic draw component constrained approximation; inspect holdout calibration and raw counts',
       'no calibration beyond ply320; no engine or training executed'],
      runtime_seconds=time.monotonic()-start,files_read=filesread,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    dump('draw_value.json',result)
    print(json.dumps(dict(sample=stats,cutoffs=models['pooled']['cutoffs'],holdout=holdout,runtime_seconds=result['runtime_seconds']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
