#!/usr/bin/env python3
"""Generate portable configs; source scripts/env.sh then pass --data-root "$DATA_ROOT"."""
import argparse
import json
from pathlib import Path
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--data-root',required=True)
p.add_argument('--out',required=True)
p.add_argument('--scale',type=float,default=.1)
a = p.parse_args()
r = Path(a.data_root).resolve()
s = a.scale
if not 0 < s <= 1:
    p.error('0 < scale <= 1')
options = {'EvalDir':str(r/'eval_dirs/task154_R_result30'), 'USI_OwnBook':'false',
           'BookFile':'no_book', 'EnteringKingRule':'CSARule27', 'MaxMovesToDraw':'512',
           'Stochastic_Ponder':'false', 'NetworkDelay':str(max(0,round(120*s))),
           'NetworkDelay2':str(max(0,round(1120*s))),
           'MinimumThinkingTime':str(max(1,round(2000*s))),
           'RoundUpToFullSecond':'true' if s == 1 else 'false', 'SlowMover':'100',
           'ResignValue':'99999', 'FV_SCALE':'16'}
spec = {'argv':[str(r/'engines/yaneuraou/source/YaneuraOu-by-gcc')],
        'cwd':str(r/'engines/yaneuraou/source'), 'options':options}
with open(a.out,'x') as f:
    json.dump({'A':spec,'B':spec},f,indent=2)
    f.write('\n')
