import json, os, random, statistics as st, sys
# Pre-tournament strategy review 10/23 (final judge agent): post-hoc "stop when the WS-T top-1 agrees with the search, give time to disagreements" on Task 167 probe13/main/report.json.
D=(sys.argv[1].rstrip('/')+'/') if len(sys.argv)>1 else os.path.expanduser('~/binbo-shogi-data/work/strategy/probe13/main/')
R=json.load(open(D+'report.json'))['roots']; P=json.load(open(D+'policy.json'))['roots']
rows=[]
for r in R:
    if not r['stable'] or r['mate']: continue
    s=json.load(open(D+'search/%s.json'%r['root_id']))
    nb=s['budgets']['100000']['bestmove']; tm=P[r['root_id']]['top_moves']
    rows.append(dict(ln=r['loss_n'],lu=r['loss_uniform'],l4=r['loss_4n'],ll=r.get('loss_live_4n'),agree=(tm[0]['move']==nb),p1=tm[0]['probability'],
                     agree3=(tm[0]['move']==nb and tm[0]['probability']>=0.3)))
def rule(rs,key='agree',big='l4'):
    a=[x for x in rs if x[key]]; d=[x for x in rs if not x[key]]
    n=len(rs)
    if not d: return 0,0,0,0
    q=(1.6*n-len(a)-1.6*len(d))/(2.4*len(d))  # fraction of disagree roots at 4N, rest 1.6N
    q=max(0,min(1,q))
    loss=(sum(x['ln'] for x in a)+sum(q*x[big]+(1-q)*x['lu'] for x in d))/n
    base=st.mean(x['lu'] for x in rs)
    agree_part=sum(x['lu']-x['ln'] for x in a)/n
    dis_part=sum(q*(x['lu']-x[big]) for x in d)/n
    return base-loss,agree_part,dis_part,q
for key in ('agree','agree3'):
  for big in ('l4','ll'):
    pt=rule(rows,key,big)
    random.seed(20271699); bs=[]
    for _ in range(2000):
        s=[random.choice(rows) for _ in rows]; bs.append(rule(s,key,big)[0])
    bs.sort()
    print(key,big,'n_agree',sum(x[key] for x in rows),'reduction %.2f [%.2f, %.2f]  agree-part %.2f  disagree-part %.2f  q4N %.3f'%(pt[0],bs[50],bs[1949],pt[1],pt[2],pt[3]))
# null: what if agree roots' N-vs-1.6N difference were zero
# Gate 0 robustness (Task 169): the same rule with the two roots that contribute most removed.
contrib=sorted(rows,key=lambda x:-abs((x['lu']-x['ln']) if x['agree'] else 0))
trim=contrib[2:]
for big in ('l4','ll'):
    pt=rule(trim,'agree',big); random.seed(20271698); bs=[]
    for _ in range(2000):
        s=[random.choice(trim) for _ in trim]; bs.append(rule(s,'agree',big)[0])
    bs.sort()
    print('trim2 agree',big,'reduction %.2f [%.2f, %.2f]'%(pt[0],bs[50],bs[1949]))
print('roots',len(rows))
