#!/usr/bin/env python3
"""Real-clock USI matches (ponder disabled). Dependencies: stdlib + cshogi.
A is black by default. JSON config describes executable argv/cwd/options for A/B.
Opening file: one whitespace-separated USI continuation AFTER the buoy per line.
"""
import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import queue
import random
import re
import subprocess
import threading
import time
import cshogi
from clock_rule import BUOY, rewrite

class EngineError(Exception):
    pass

class Engine:
    def __init__(self, spec, log_path):
        self.q = queue.Queue()
        self.log = open(log_path, 'x', buffering=1)
        self.lock = threading.Lock()
        self.p = subprocess.Popen(spec['argv'], cwd=spec.get('cwd'), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
                                  env={**os.environ, 'OMP_NUM_THREADS':'1', 'OPENBLAS_NUM_THREADS':'1',
                                       'MKL_NUM_THREADS':'1'})
        def reader():
            for raw in self.p.stdout:
                stamp = time.monotonic_ns()
                line = raw.rstrip()
                self.record('<', line, stamp)
                self.q.put((stamp, line))
            self.q.put((time.monotonic_ns(), None))
        self.reader = threading.Thread(target=reader, daemon=True)
        self.reader.start()
        try:
            self.send('usi')
            options = self.wait('usiok', 60)[2]
            offered = {}
            for line in options:
                m = re.match(r'option name (.*?) type (\w+)(.*)', line)
                if m:
                    offered[m[1]] = (m[2], m[3])
            for name, value in spec.get('options', {}).items():
                if name not in offered:
                    raise EngineError('unknown option: ' + name)
                kind, tail = offered[name]
                if kind == 'spin':
                    bounds = re.search(r'min (-?\d+) max (-?\d+)', tail)
                    if bounds and not int(bounds[1]) <= int(value) <= int(bounds[2]):
                        raise EngineError('option out of range: ' + name)
                if kind == 'check' and str(value).lower() not in ['true', 'false']:
                    raise EngineError('bad check option: ' + name)
                self.send(f'setoption name {name} value {value}')
            self.send('isready')
            self.wait('readyok', 120)
            self.send('usinewgame')
        except BaseException:
            self.close()
            raise
    def record(self, direction, line, stamp=None):
        with self.lock:
            self.log.write(f'{stamp or time.monotonic_ns()} {direction} {line}\n')
    def send(self, line):
        self.record('>', line)
        try:
            self.p.stdin.write(line + '\n')
            self.p.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise EngineError('engine stdin closed') from exc
    def wait(self, prefix, timeout, start=None):
        start = start or time.monotonic_ns()
        deadline = start + int(timeout * 1e9)
        lines = []
        while True:
            try:
                stamp, line = self.q.get(timeout=max(0, (deadline-time.monotonic_ns())/1e9))
            except queue.Empty:
                raise TimeoutError(prefix)
            if stamp > deadline:
                raise TimeoutError(prefix)
            if line is None:
                raise EngineError('engine exited')
            if line.startswith(prefix):
                return stamp, line, lines
            lines.append(line)
    def close(self):
        if self.p.poll() is None:
            try:
                self.send('quit')
                self.p.wait(timeout=2)
            except (subprocess.TimeoutExpired, EngineError):
                self.p.kill()
                self.p.wait()
        self.reader.join(timeout=2)
        # A dead engine leaves unsent bytes buffered; closing stdin then raises BrokenPipeError (review 10/24).
        for stream in (self.p.stdout, self.p.stdin, self.log):
            try:
                stream.close()
            except (OSError, ValueError):
                pass

class Rules:
    """Exact state identity (SFEN excluding move number), not twofold is_draw()."""
    def __init__(self, sfen=None):
        self.board = cshogi.Board(sfen) if sfen else cshogi.Board()
        self.seen = {self.key(): [0]}
        self.checks = []  # (mover, gives_check)
    def key(self):
        return ' '.join(self.board.sfen().split()[:3])
    def push(self, usi):
        legal = {cshogi.move_to_usi(m): m for m in self.board.legal_moves}
        if usi not in legal:
            raise ValueError('illegal move: ' + usi)
        mover = self.board.turn
        self.board.push(legal[usi])
        self.checks.append((mover, self.board.is_check()))
        visits = self.seen.setdefault(self.key(), [])
        visits.append(len(self.checks))
        if len(visits) < 4:
            return None
        interval = self.checks[visits[-4]:]
        for side in [0, 1]:
            checks = [v for c, v in interval if c == side]
            if checks and all(checks):
                return (1-side, 'perpetual_check')
        return (-1, 'sennichite')
    def declaration(self, remaining_ms):
        return remaining_ms > 0 and self.board.is_nyugyoku()

def debit(remaining, increment, elapsed, margin, quantum):
    used = elapsed + margin
    if quantum:
        used = math.floor(used / quantum) * quantum
    return remaining + increment - used, used

def make_opening(seed, plies, lines):
    rng = random.Random(seed)
    rules = Rules()
    moves = BUOY.copy()
    for move in moves:
        assert rules.push(move) is None
    if lines:
        continuation = lines[rng.randrange(len(lines))].split()
        if len(continuation) < plies:
            raise ValueError('opening line shorter than --opening-plies')
        continuation = continuation[:plies]
    else:
        continuation = []
        for _ in range(plies):
            legal = sorted(cshogi.move_to_usi(m) for m in rules.board.legal_moves)
            if not legal:
                raise ValueError('terminal random opening')
            move = rng.choice(legal)
            if rules.push(move):
                raise ValueError('repeated random opening; change seed')
            continuation.append(move)
        return moves + continuation
    for move in continuation:
        if rules.push(move):
            raise ValueError('repetition in opening')
    return moves + continuation

def finish_value(winner):
    return ('1-0', 1., 0.) if winner == 0 else (('0-1', 0., 1.) if winner == 1 else ('1/2-1/2', .4, .6))

def play(index, a, config, lines):
    seed = a.seed + index
    moves = make_opening(seed, a.opening_plies, lines)
    opening = ' '.join(moves)
    opening_id = hashlib.sha256(opening.encode()).hexdigest()[:16]
    game_id = f'{a.arm}_{seed}'
    rules = Rules()
    for move in moves:
        rules.push(move)
    black = a.a_color == 'black'
    names = ['A', 'B'] if black else ['B', 'A']
    engines = []
    clocks = [a.black_base_ms, a.white_base_ms]
    inc = [a.black_inc_ms, a.white_inc_ms]
    times, remains = [[], []], [[], []]
    winner, reason = -1, 'max_plies'
    errors = ''
    out = Path(a.out)
    wall_start = time.monotonic()
    try:
        for name in names:
            spec = dict(config[name])
            opts = {**spec.get('options', {}), 'Threads': str(a.threads),
                    'USI_Hash': str(a.hash_mb), 'USI_Ponder':'false'}
            if name == 'A':
                for value in a.a_option:
                    k, v = value.split('=', 1)
                    if k in ['Threads', 'USI_Ponder', 'USI_Hash']:
                        raise ValueError('resource/ponder option must use harness arguments')
                    opts[k] = v
            spec['options'] = opts
            engines.append(Engine(spec, out / f'{game_id}_{name}.usi.log'))
        while rules.board.move_number - 1 < a.max_plies:
            side = rules.board.turn
            if not list(rules.board.legal_moves):
                winner, reason = 1-side, 'no_legal_move'
                break
            pos = 'position startpos moves ' + ' '.join(moves)
            go = f'go btime {max(0,int(clocks[0]))} wtime {max(0,int(clocks[1]))} binc {inc[0]} winc {inc[1]}'
            if names[side] == 'A':
                go = rewrite(go, pos, a.rule, a.k, a.cap_ms, a.byoyomi_ms)
            budget = clocks[side] + inc[side]
            # A floor-charging server can accept a fractional extra quantum.
            deadline_ms = max(0, budget - a.network_margin_ms + a.charge_quantum_ms)
            engine = engines[side]
            start = time.monotonic_ns()
            try:
                engine.send(pos)
                engine.send(go)
                stamp, answer, info = engine.wait('bestmove ', deadline_ms/1000, start)
                elapsed = (stamp-start)/1e6
            except TimeoutError:
                elapsed = max(deadline_ms, (time.monotonic_ns()-start)/1e6)
                clocks[side], charged = debit(clocks[side], inc[side], elapsed,
                                              a.network_margin_ms, a.charge_quantum_ms)
                times[side].append(round(elapsed, 3))
                remains[side].append(round(clocks[side], 3))
                winner, reason = 1-side, 'time_forfeit'
                break
            clocks[side], charged = debit(clocks[side], inc[side], elapsed,
                                          a.network_margin_ms, a.charge_quantum_ms)
            times[side].append(round(elapsed, 3))
            remains[side].append(round(clocks[side], 3))
            if clocks[side] < 0:
                winner, reason = 1-side, 'time_forfeit'
                break
            move = answer.split()[1]
            if move == 'resign':
                winner, reason = 1-side, 'resign'
                break
            if move == 'win':
                valid = rules.declaration(clocks[side])
                winner, reason = (side, 'declaration_27') if valid else (1-side, 'invalid_declaration')
                break
            try:
                terminal = rules.push(move)
            except ValueError:
                winner, reason = 1-side, 'illegal_move'
                errors = move
                break
            moves.append(move)
            # Event rule: 512th move mate is also a draw.
            if rules.board.move_number - 1 >= a.max_plies:
                winner, reason = -1, 'max_plies'
                break
            if terminal:
                winner, reason = terminal
                break
    except (EngineError, TimeoutError, OSError) as exc:
        # Infrastructure faults invalidate the observation; never silently score as chess strength.
        # TimeoutError here is engine start-up (usiok/readyok); a slow bestmove is the time forfeit above.
        reason, errors = 'engine_error', f'{type(exc).__name__}: {exc}'
    finally:
        for engine in engines:
            try:
                engine.close()
            except Exception:  # noqa: BLE001 - one engine's cleanup must not skip the other or abort the run
                pass
    result, sb, sw = finish_value(winner)
    if reason == 'engine_error':
        result, sb, sw = 'INVALID', '', ''
    row = dict(game_id=game_id, arm=a.arm, seed=seed, opening_id=opening_id, opening=opening,
               black=names[0], white=names[1], result=result, reason=reason,
               plies=rules.board.move_number-1, played_plies=len(moves)-len(opening.split()),
               black_score=sb, white_score=sw,
               black_ms=json.dumps(times[0],separators=(',',':')),
               white_ms=json.dumps(times[1],separators=(',',':')),
               black_remaining_ms=round(clocks[0],3), white_remaining_ms=round(clocks[1],3),
               black_remaining_trace=json.dumps(remains[0],separators=(',',':')),
               white_remaining_trace=json.dumps(remains[1],separators=(',',':')),
               moves=' '.join(moves), wall_seconds=round(time.monotonic()-wall_start,3), error=errors)
    with open(out / f'{game_id}.tsv', 'x', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row), delimiter='\t')
        w.writeheader()
        w.writerow(row)
    print(f'{game_id}: {result} {reason} {row["plies"]} plies', flush=True)
    return row

def arguments(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--games', type=int, default=10)
    ap.add_argument('--parallel', type=int, default=1)
    ap.add_argument('--cpu-budget', type=int, default=4)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--hash-mb', type=int, default=256)
    ap.add_argument('--seed', type=int, default=2026102400)
    ap.add_argument('--arm', default='default')
    ap.add_argument('--a-color', choices=['black','white'], default='black')
    ap.add_argument('--a-option', action='append', default=[])
    ap.add_argument('--rule', choices=['none','cap','byoyomi'], default='none')
    ap.add_argument('--cap-ms', type=int, default=3000)
    ap.add_argument('--byoyomi-ms', type=int, default=300)
    ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--black-base-ms', type=int, default=18000)
    ap.add_argument('--white-base-ms', type=int, default=60000)
    ap.add_argument('--black-inc-ms', type=int, default=200)
    ap.add_argument('--white-inc-ms', type=int, default=200)
    ap.add_argument('--network-margin-ms', type=float, default=0)
    ap.add_argument('--charge-quantum-ms', type=float, default=0)
    ap.add_argument('--opening-file')
    ap.add_argument('--opening-plies', type=int, default=4)
    ap.add_argument('--max-plies', type=int, default=512)
    a = ap.parse_args(argv)
    if min(a.parallel, a.threads, a.games, a.hash_mb) < 1 or a.parallel*a.threads > a.cpu_budget:
        ap.error('parallel * threads must fit --cpu-budget (default 4)')
    if min(a.black_base_ms, a.white_base_ms, a.black_inc_ms, a.white_inc_ms,
           a.network_margin_ms, a.charge_quantum_ms, a.opening_plies, a.k, a.cap_ms, a.byoyomi_ms) < 0:
        ap.error('negative time/opening parameter')
    if not 4 + a.opening_plies < a.max_plies <= 512:
        ap.error('max plies must exceed opening length and be <= 512')
    return a

def prepare(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(a.config).read_text())
    lines = [s.strip() for s in Path(a.opening_file).read_text().splitlines()
             if s.strip() and not s.lstrip().startswith('#')] if a.opening_file else None
    # Validate every opening before launching engines, including legality/history.
    for i in range(a.games):
        make_opening(a.seed+i, a.opening_plies, lines)
    assets = {}
    for spec in config.values():
        paths = [Path(spec['argv'][0])]
        if 'EvalDir' in spec.get('options', {}):
            paths.append(Path(spec['options']['EvalDir'])/'nn.bin')
        for path in paths:
            if str(path) not in assets and path.is_file():
                assets[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata = {'assets_sha256':assets, 'cshogi_version':importlib.metadata.version('cshogi'), 'args':vars(a), 'config':config, 'ponder':False,
                'clock':'monotonic_ns; timestamp when stdout reader receives line',
                'python':os.sys.version, 'cshogi_path':cshogi.__file__,
                'script_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), Path(__file__).with_name('clock_rule.py')]}}
    (out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return out, config, lines

def save_summary(rows, out, elapsed):
    valid = [r for r in rows if r['result'] != 'INVALID']
    summary = {'games':len(rows), 'valid':len(valid),
               'black_wdl':{'wins':sum(r['result']=='1-0' for r in valid),
                            'draws':sum(r['result']=='1/2-1/2' for r in valid),
                            'losses':sum(r['result']=='0-1' for r in valid)}, 'reasons':{k:sum(r['reason']==k for r in rows)
                for k in sorted({r['reason'] for r in rows})},
               'black_score':sum(r['black_score'] for r in valid)/len(valid) if valid else None,
               'white_score':sum(r['white_score'] for r in valid)/len(valid) if valid else None,
               'wall_seconds':elapsed}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if len(valid) != len(rows):
        raise SystemExit('INVALID games: inspect engine logs')
def main():
    a = arguments()
    out, config, lines = prepare(a)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.parallel) as pool:
        rows = list(pool.map(lambda i: play(i,a,config,lines), range(a.games)))
    save_summary(rows,out,time.monotonic()-started)

if __name__ == '__main__':
    main()
