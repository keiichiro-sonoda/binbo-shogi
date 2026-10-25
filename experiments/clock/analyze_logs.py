#!/usr/bin/env python3
"""Read-only CSA clock audit; select canonical server records and deduplicate events."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np

BUOY_CSA = ['+5958OU','-5152OU','+5859OU','-5251OU']

def quant(values):
    return dict(zip(['q25','median','q75'],map(float,np.quantile(values,[.25,.5,.75])))) if values else None

def tm(btime, inc, ply, maxmoves=512, slow=100, ponder=True, minimum=2000, nd=120, nd2=1120, rounding=True):
    horizon = 200-min(ply,40) if inc == 0 else 180-min(ply,80)
    mtg = min(maxmoves-ply+2,horizon)//2
    remain = max(btime+inc-nd2,100 if rounding else 1)
    def rup(t):
        if rounding:
            v = max(((t+999)//1000)*1000,minimum)-nd
            if v<t:
                v+=1000
        else:
            v=max(t,minimum)-nd
        return min(v,remain)
    if mtg <= 0:
        return {'mtg':mtg,'minimum':500,'optimum':500,'maximum':500}
    if mtg == 1:
        return {'mtg':1,'minimum':remain,'optimum':remain,'maximum':remain}
    lo=max(minimum-nd,1000 if rounding else 1)
    est=max(0,btime+inc*mtg-((mtg+1)*1000 if rounding else 0))
    ratio=min(5,max(btime/60000,1)) if inc == 0 else 5
    opt=min(lo+est//mtg,remain)*slow//100
    if ponder:
        opt+=opt//4
    mx=min(lo+int(np.float32(est)*np.float32(ratio)/np.float32(mtg)),int(est*.3),remain)
    return {'mtg':mtg,'minimum':rup(lo),'optimum':min(opt,remain),'maximum':rup(mx)}

def read_csa(path):
    text=path.read_text(errors='replace')
    lines=text.splitlines()
    def match(pat,default=''):
        m=re.search(pat,text,re.M)
        return m[1] if m else default
    event=match(r'^\$EVENT:(.*)')
    base_inc=re.search(r'-(\d+)-(\d+)F(?:\+|$)',event)
    if not base_inc:
        return None
    base,inc=map(int,base_inc.groups())
    black=match(r'^N\+(.*)')
    moves=[]
    for i,l in enumerate(lines):
        if re.fullmatch('[+-][0-9]{4}[A-Z]{2}',l):
            if i+1>=len(lines) or not re.fullmatch('T[0-9]+',lines[i+1]):
                raise ValueError('missing time '+str(path))
            moves.append((l,int(lines[i+1][1:])))
    buoy_len = match(r"^'buoy game starting with ([0-9]+) moves")
    buoy = ([m for m,_ in moves[:4]] == BUOY_CSA and moves[0][1] == 422
            and (not buoy_len or int(buoy_len) == 4))
    if 'floodgate' in event:
        if not black.lower().startswith('binbo'):
            return None
        category='floodgate_book' if 'book' in black else 'floodgate_nobook'
    elif not buoy:
        return None
    elif '/book1/' in str(path):
        category='book_stage2_'+black
    else:
        category='local_buoy_'+('main' if '/xub/srv/' in str(path) else str(path).split('/xub/')[1].split('/')[0])
    rem=base
    seq=[]
    own=0
    for ply,(move,t) in enumerate(moves,1):
        if not move.startswith('+'):
            continue
        pre=rem
        rem+=inc-t
        if buoy and ply<=4:
            continue
        own+=1
        pred=tm(pre*1000,inc*1000,ply)
        seq.append({'own':own,'ply':ply,'time_s':t,'pre_s':pre,'remaining_s':rem,
                    'increment_only':pre<=inc,'zero_base':pre<=0,
                    'optimum_s':pred['optimum']/1000,'maximum_s':pred['maximum']/1000})
    return {'event':event.strip(),'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'category':category,'base_s':base,'increment_s':inc,'black':black,'moves':seq,
            'reason':match(r"^'summary:(.*)") or match(r'^(%.*)'), 'buoy':buoy}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    root=Path(a.data_root)/'denryusen'
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    # Existing local games have a server authority; records under two bridge dirs are copies.
    candidates=sorted(root.glob('xub/**/srv/**/*.csa'))+sorted(root.glob('xub/**/records/*.csa'))+sorted(root.glob('floodgate/**/records/*.csa'))
    selected={}
    authority_events=set()
    for path in candidates:
        event_match=re.search(r'^\$EVENT:(.*)',path.read_text(errors='replace'),re.M)
        event_key=event_match[1].strip() if event_match else str(path)
        if event_key in authority_events:
            continue
        authority_events.add(event_key)
        game=read_csa(path)
        if game:
            selected.setdefault(game['event'],game)
    groups=defaultdict(list)
    for g in selected.values():
        groups[g['category']].append(g)
    aggregate={}
    ownrows=[]
    for cat,games in sorted(groups.items()):
        allmoves=[m for g in games for m in g['moves']]
        byown=defaultdict(list)
        for g in games:
            for m in g['moves']:
                byown[m['own']].append(m)
        for own,ms in sorted(byown.items()):
            row={'category':cat,'own':own,'n':len(ms)}
            for k in ['time_s','remaining_s','optimum_s','maximum_s']:
                row.update({k+'_'+q:v for q,v in quant([m[k] for m in ms]).items()})
            row['increment_only_fraction']=sum(m['increment_only'] for m in ms)/len(ms)
            ownrows.append(row)
        first15=[sum(m['time_s'] for m in g['moves'][:15]) for g in games if len(g['moves'])>=15]
        aggregate[cat]={'games':len(games),'moves':len(allmoves),'first15_s':quant(first15),
                        'minimum_remaining_s':quant([min(m['remaining_s'] for m in g['moves']) for g in games if g['moves']]),
                        'increment_only_moves':sum(m['increment_only'] for m in allmoves),
                        'increment_only_games':sum(any(m['increment_only'] for m in g['moves']) for g in games),
                        'zero_base_moves':sum(m['zero_base'] for m in allmoves),
                        'pre_le_5s_moves':sum(m['pre_s']<=5 for m in allmoves),
                        'pre_le_10s_moves':sum(m['pre_s']<=10 for m in allmoves),
                        'min_le_5s_games':sum(any(m['remaining_s']<=5 for m in g['moves']) for g in games),
                        'moves_le_increment':sum(m['time_s']<=g['increment_s'] for g in games for m in g['moves']),
                        'reason_counts':{r:sum(g['reason']==r for g in games) for r in sorted({g['reason'] for g in games})},
                        'clock_conditions':sorted({(g['base_s'],g['increment_s']) for g in games})}
    with (out/'by_own_move.tsv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(ownrows[0]),delimiter='\t')
        w.writeheader();w.writerows(ownrows)
    (out/'games.json').write_text(json.dumps(list(selected.values()),indent=2)+'\n')
    (out/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    print(json.dumps(aggregate,indent=2))
if __name__=='__main__':
    main()
