#!/usr/bin/env python3
"""Authoritative fit: cached sample, monotone-likelihood Newton updates."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import json,math,time
import numpy as np
import analyze_draw as m

def fit(X,y,g):
    b=np.zeros(X.shape[1]);b[0]=np.log((y.sum()+.5)/(len(y)-y.sum()+.5));ridge=1e-7
    def loss(b):
        v=X@b
        return np.sum(np.logaddexp(0,v)-y*v)+ridge*(b@b)/2
    last=loss(b)
    for i in range(80):
        p=m.sigmoid(X@b);H=X.T@((p*(1-p))[:,None]*X)+np.eye(X.shape[1])*ridge
        grad=X.T@(y-p)-ridge*b
        db=np.linalg.solve(H,grad);scale=1.
        while scale>1e-9:
            candidate=b+scale*db;value=loss(candidate)
            if value<=last+1e-8:break
            scale/=2
        b=candidate;last=value
        if np.max(np.abs(scale*db))<1e-8:break
    p=m.sigmoid(X@b);H=X.T@((p*(1-p))[:,None]*X)+np.eye(X.shape[1])*ridge
    _,inv=np.unique(g,return_inverse=True);s=np.zeros((inv.max()+1,len(b)));np.add.at(s,inv,X*(y-p)[:,None])
    hi=np.linalg.inv(H);cov=hi@(s.T@s)@hi*len(s)/(len(s)-1)
    score_norm=float(np.max(np.abs(X.T@(y-p)-ridge*b))/len(y))
    assert score_norm<1e-7,(i,score_norm)
    return dict(beta=b,cov=cov,rows=len(y),games=len(s),events=int(y.sum()),iterations=i+1,
                logloss=float(np.mean(np.logaddexp(0,X@b)-y*(X@b))),score_norm=score_norm,
                fitted_mean=float(p.mean()),observed_mean=float(y.mean()))

def main():
    start=time.monotonic();a=np.load(m.OUT/'draw_sample.npy').astype(float);old=json.loads((m.OUT/'draw_value.json').read_text())
    m.fit=fit
    models={}
    for si,source in [(-1,'pooled'),(0,'task154'),(1,'task159')]:
        z=a if si<0 else a[a[:,5]==si];w,d=m.model(z);models[source]=m.packmodels(w,d)
        # Do not silently extrapolate unreliable late-ply slopes.
        for pc,turns in models[source]['cutoffs'].items():
            for t,c in turns.items():
                c['in_score_support']=abs(c['cp'])<=m.LIMIT and c['slope_per_cp']>0
                c['recommendation_eligible']=c['in_score_support'] and int(pc)<=120
    hold=(a[:,4].astype(np.int64)*2654435761%5)==0
    tw,td=m.model(a[~hold]);test=a[hold];q=m.sigmoid(m.design(test[:,0],test[:,1],test[:,2])@tw['beta']);r=m.sigmoid(m.design(test[:,0],test[:,1],test[:,2],True)@td['beta'])
    pred=np.array([(1-r)*q,r,(1-r)*(1-q)]).T;y=np.array([test[:,3]>0,test[:,3]==0,test[:,3]<0]).T.astype(float)
    calibration=[]
    for lo,hi in [(-600,-200),(-200,-100),(-100,0),(0,100),(100,200),(200,601)]:
        mask=(test[:,0]>=lo)&(test[:,0]<hi)
        calibration.append(dict(cp=[lo,hi],rows=int(mask.sum()),predicted_WDL=pred[mask].mean(0),observed_WDL=y[mask].mean(0)))
    # Only draw games must be opened to distinguish cap from repetition.
    caps=[];perpetual=[];draw_reasons={}
    for gid in np.unique(a[a[:,3]==0,4]).astype(int):
        si=int(gid>=150000);task=['task154','task159'][si];path=m.DATA/'work'/task/'games'/f'game_{gid-si*150000:012d}.json'
        game=json.loads(path.read_text());draw_reasons[str(gid)]=game['reason']
        if game['reason']=='max_ply_draw':caps.append(gid)
        else:
            b=m.cshogi.Board()
            for mv in game['moves']:b.push_usi(mv)
            if b.is_draw() in (m.cshogi.REPETITION_WIN,m.cshogi.REPETITION_LOSE):perpetual.append(gid)
    no_caps=a[~np.isin(a[:,4],caps+perpetual)]
    nw,nd=m.model(no_caps)
    old['models']=models
    old['holdout']=dict(games=len(np.unique(test[:,4])),rows=len(test),wdl_brier=float(np.mean(np.sum((pred-y)**2,axis=1))),
                       wdl_logloss=float(-np.mean(np.sum(y*np.log(np.clip(pred,1e-12,1)),axis=1))),calibration=calibration)
    old['draw_reason_game_ids']=draw_reasons
    old['excluded_cap_sensitivity']=dict(cap_games=len(caps),perpetual_check_games=len(perpetual),remaining_rows=len(no_caps),model=m.packmodels(nw,nd),
               note='q and its threshold unchanged: cap/perpetual games had no decisive rows')
    old['authoritative']=True;old['supersedes']='draw_value.json: draft draw-logit optimizer did not converge; final stdout serializer failed after JSON was saved'
    old['method']='two-stage logistic WDL, line-search Newton MLE verified score gradient, game-cluster sandwich covariance'
    old['refine_runtime_seconds']=time.monotonic()-start
    m.dump('draw_value_refined.json',old)
    print('Final fit complete',json.dumps(dict(rows=len(a),draw_games=len(draw_reasons),cap_games=len(caps),perpetual=len(perpetual),holdout_logloss=old['holdout']['wdl_logloss'],runtime=old['refine_runtime_seconds'])))

if __name__=='__main__':main()
