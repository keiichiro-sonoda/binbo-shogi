#!/usr/bin/env python3
"""Rule fixtures + end-to-end USI and proxy failure-path checks. No strong-engine search."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import unittest
import cshogi
from clock_match import Rules, make_opening, debit
from clock_rule import BUOY, rewrite
from analyze_logs import tm

WORK=None
HERE=Path(__file__).resolve().parent

class ClockTests(unittest.TestCase):
    def test_fourfold_includes_buoy(self):
        r=Rules()
        for i,move in enumerate(BUOY*3):
            result=r.push(move)
            self.assertEqual(result,(-1,'sennichite') if i==11 else None)
        # The upstream search API reports a repetition already at the first cycle.
        b=cshogi.Board()
        for m in BUOY:
            b.push_usi(m)
        self.assertNotEqual(b.is_draw(),cshogi.NOT_REPETITION)
    def test_perpetual_checker_loses(self):
        r=Rules('4k4/9/5R3/9/9/9/9/9/K8 b - 1')
        moves=['4c5c','5a4a','5c4c','4a5a']*3
        for i,m in enumerate(moves):
            result=r.push(m)
            self.assertEqual(result,(1,'perpetual_check') if i==11 else None)
    def test_27_point_boundaries(self):
        cases=[
          ('RRB5G/4K4/PPPPPPPPP/9/9/9/9/9/4k4 b 3Pb3g4s4n4l6p 1',True),
          ('RRB5G/4K4/PPPPPPPPP/9/9/9/9/9/4k4 b 2Pb3g4s4n4l7p 1',False),
          ('RRB6/4K4/PPPPPP3/6PPP/9/9/9/9/4k4 b 7Pb4g4s4n4l2p 1',False),
          ('RRB2s2G/4K4/PPPPPPPPP/9/9/9/9/9/4k4 b 3Pb3g3s4n4l6p 1',False),
          ('RRB5G/9/PPPPPPPPP/4K4/9/9/9/9/4k4 b 3Pb3g4s4n4l6p 1',False),
          ('4K4/9/9/9/9/9/ppppppppp/4k4/g5brr w B3G4S4N4L7P2p 1',True),
          ('4K4/9/9/9/9/9/ppppppppp/4k4/g5brr w B3G4S4N4L8Pp 1',False)]
        for sfen,expected in cases:
            r=Rules(sfen)
            self.assertEqual(r.declaration(1),expected)
            self.assertFalse(r.declaration(0))
    def test_rewrite_position_not_call_counter(self):
        go='go ponder btime 180000 wtime 600000 binc 2000 winc 2000'
        pos='position startpos moves '+' '.join(BUOY)
        self.assertIn('btime 30000',rewrite(go,pos,'cap'))
        self.assertEqual(rewrite(go,pos+' 7g7f','cap'),go)
        self.assertEqual(rewrite(go,'position sfen 9/9/9/9/9/9/9/9/9 b - 35','cap'),go)
        self.assertIn('btime 30000',rewrite(go,'position sfen 9/9/9/9/9/9/9/9/9 b - 33','cap'))
        self.assertEqual(rewrite('ponderhit',pos,'cap'),'ponderhit')
        self.assertEqual(rewrite('go infinite',pos,'cap'),'go infinite')
        self.assertIn('btime 0',rewrite(go,pos,'byoyomi'))
        self.assertEqual(rewrite(go,pos,'cap',cap_ms=200000),go)
    def test_clock_and_formula(self):
        self.assertEqual(debit(180000,2000,3000,120,0),(178880,3120))
        self.assertEqual(debit(180000,2000,3999,0,1000),(179000,3000))
        self.assertEqual(tm(180000,2000,5),{'mtg':87,'minimum':1880,'optimum':6171,'maximum':17880})
        self.assertEqual(tm(180000,2000,512)['mtg'],1)
    def test_opening_reproducible_and_legal(self):
        self.assertEqual(make_opening(180,4,None),make_opening(180,4,None))
        self.assertNotEqual(make_opening(180,4,None),make_opening(181,4,None))
        self.assertEqual(make_opening(180,2,['7g7f 3c3d']),BUOY+['7g7f','3c3d'])
        with self.assertRaises(ValueError):
            make_opening(180,1,['9a9i'])
    def test_real_process_paths(self):
        for mode,expected in [('cycle','sennichite'),('resign','resign'),('win','invalid_declaration'),
                              ('illegal','illegal_move'),('legal','max_plies'),('timeout','time_forfeit')]:
            spec={'argv':[sys.executable,'-B',str(HERE/'fake_engine.py'),'--mode',
                          'cycle' if mode=='timeout' else mode]}
            if mode=='timeout':
                spec['argv']+=['--delay-ms','1500']
            config=WORK/f'{mode}.json'
            config.write_text(json.dumps({'A':spec,'B':spec}))
            out=WORK/mode
            subprocess.run([sys.executable,'-B',str(HERE/'clock_match.py'),'--config',str(config),
                            '--out',str(out),'--games','1','--opening-plies','0',
                            '--max-plies','12' if mode!='cycle' else '512',
                            '--black-base-ms','300','--white-base-ms','300',
                            '--black-inc-ms','0','--white-inc-ms','0'],check=True,timeout=20)
            with next(out.glob('*.tsv')).open() as f:
                row=list(csv.DictReader(f,delimiter='\t'))[0]
            self.assertEqual(row['reason'],expected)
            if mode=='cycle':
                self.assertEqual(int(row['plies']),12)
            if mode=='timeout':
                self.assertEqual(row['result'],'0-1')
    def test_idle_engine_death_is_one_invalid_game(self):
        # Review 10/24: an engine that dies between its moves must give one INVALID row, not abort the run.
        die={'argv':[sys.executable,'-B',str(HERE/'fake_engine.py'),'--mode','die']}
        slow={'argv':[sys.executable,'-B',str(HERE/'fake_engine.py'),'--mode','cycle','--delay-ms','300']}
        config=WORK/'die.json'
        config.write_text(json.dumps({'A':die,'B':slow}))  # A is surely dead before its next position
        out=WORK/'die'
        p=subprocess.run([sys.executable,'-B',str(HERE/'clock_match.py'),'--config',str(config),
                          '--out',str(out),'--games','2','--opening-plies','0','--max-plies','12',
                          '--black-base-ms','3000','--white-base-ms','3000',
                          '--black-inc-ms','0','--white-inc-ms','0'],capture_output=True,text=True,timeout=30)
        rows=[]
        for path in sorted(out.glob('*.tsv')):
            with path.open() as f:
                rows+=list(csv.DictReader(f,delimiter='\t'))
        self.assertEqual(len(rows),2)
        self.assertTrue(all(r['result']=='INVALID' and r['reason']=='engine_error' for r in rows))
        self.assertNotIn('Traceback',p.stderr)
        self.assertIn('INVALID games',p.stderr)
    def test_proxy_end_to_end(self):
        p=subprocess.run([sys.executable,'-B',str(HERE/'clock_proxy.py'),'--rule','cap','--',
                          sys.executable,'-u','-c',
                          'import sys\nfor s in sys.stdin:\n print(s.strip(),flush=True)\n if s.strip()=="quit": break'],
                         input='position startpos moves '+' '.join(BUOY)+
                         '\ngo ponder btime 180000 wtime 600000 binc 2000 winc 2000\nponderhit\nquit\n',
                         text=True,capture_output=True,check=True,timeout=10)
        self.assertIn('go ponder btime 30000 wtime 600000 binc 2000 winc 2000',p.stdout)
        self.assertIn('\nponderhit\n',p.stdout)
        self.assertEqual(p.stderr,'')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    a,rest=ap.parse_known_args()
    WORK=Path(a.out)
    WORK.mkdir(parents=True,exist_ok=False)
    unittest.main(argv=[sys.argv[0]]+rest)
