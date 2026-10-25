#!/usr/bin/env python3
"""Deterministic USI test double; not an experimental opponent."""
import argparse
import sys
import time
import cshogi
p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['cycle','legal','resign','win','illegal','die'],default='cycle')
p.add_argument('--delay-ms',type=int,default=0)
a=p.parse_args()
b=cshogi.Board()
for raw in sys.stdin:
    line=raw.strip()
    if line=='usi':
        print('id name task180 test double')
        print('option name Threads type spin default 1 min 1 max 4')
        print('option name USI_Hash type spin default 1 min 1 max 256')
        print('option name USI_Ponder type check default false')
        print('option name SlowMover type spin default 100 min 1 max 1000')
        print('usiok',flush=True)
    elif line=='isready':
        print('readyok',flush=True)
    elif line.startswith('position '):
        b.set_position(line[9:])
    elif line.startswith('go'):
        time.sleep(a.delay_ms/1000)
        if a.mode in ('cycle','die'):
            move=['5i5h','5a5b','5h5i','5b5a'][(b.move_number-1)%4]
        elif a.mode=='legal':
            move=sorted(cshogi.move_to_usi(m) for m in b.legal_moves)[0]
        elif a.mode=='illegal':
            move='9a9i'
        else:
            move=a.mode
        print('bestmove '+move,flush=True)
        if a.mode=='die':
            sys.exit(0)  # dies while idle, after its answer
    elif line=='quit':
        break
